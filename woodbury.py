"""
Woodbury Matrix Identity Optimization

This is how to mplement the Woodbury matrix identity to reduce the computational complexity
of the EKF innovation covariance inversion from O(m^3) to O(n^3), where
m >> n (sensor nodes >> degrees of freedom).

Mathematical Foundation (NanoPose, Eq. 10):
    S_inv = R_inv - R_inv @ H @ (P_inv + H.T @ R_inv @ H)^{-1} @ H.T @ R_inv

This identity is valid when R is diagonal (spatially independent thermal noise),
which is a physically motivated assumption for nanosensor arrays.

References:
    [1] Woodbury, M.A. (1950). Inverting modified matrices.
        Memorandum Rept. 42, Statistical Research Group, Princeton University.
    [2] NanoPose paper, Section IV-A, Eq. (10).
"""

import numpy as np
from numpy.linalg import solve

# Minimum diagonal ridge added before Cholesky to absorb floating-point
# negative eigenvalues (e.g. -1e-14) that arise from accumulated rounding.
# Too small to affect conditioning but large enough to guarantee factorisation.
_CHOL_RIDGE: float = 1e-12


def woodbury_innovation_inverse(
    P_prior: np.ndarray,
    H: np.ndarray,
    R_diag: np.ndarray,
) -> np.ndarray:
    """
    Compute S^{-1} using the Woodbury matrix identity.

    Standard EKF requires inverting S = H P H^T + R ∈ R^{m×m}, costing O(m^3).
    The Woodbury identity reformulates this as an inversion in R^{n×n}, costing
    O(n^3 + mn^2), which is orders of magnitude cheaper when m >> n.

    Args:
        P_prior:  A priori state covariance, shape (n, n).
        H:        Observation Jacobian, shape (m, n).
        R_diag:   Diagonal of measurement noise covariance R, shape (m,).
                  Assumed diagonal (spatially independent sensor noise).

    Returns:
        S_inv:    Inverse of innovation covariance S, shape (m, m).

    Complexity:
        Standard:  O(m^3) — dominated by inversion of S ∈ R^{m×m}.
        Woodbury:  O(n^3) + O(mn^2) — dominated by inversion of inner ∈ R^{n×n}.

    Notes:
        The diagonal assumption on R is physically motivated as the thermal and
        electronic noise at the nanoscale is spatially decorrelated across
        independent sensor nodes [NanoPose, Sec. IV-A].
    """
    m, n = H.shape
    assert P_prior.shape == (n, n), f"P_prior must be ({n},{n}), got {P_prior.shape}"
    assert R_diag.shape == (m,), f"R_diag must be ({m},), got {R_diag.shape}"

    R_inv_diag = 1.0 / R_diag                        # O(m)  — diagonal inverse

    # Inner matrix: (P^{-1} + H^T R^{-1} H) ∈ R^{n×n}
    P_inv = _stable_inverse(P_prior)                  # O(n^3)
    inner = P_inv + (H.T * R_inv_diag) @ H            # O(mn^2)
    inner = 0.5 * (inner + inner.T)                   # Enforce symmetry before Cholesky
    inner_inv = _stable_inverse(inner)                # O(n^3)

    # Woodbury correction: R_inv @ H @ inner_inv @ H^T @ R_inv
    # Applied efficiently without forming the full m×m matrix until the end
    RH = H * R_inv_diag[:, None]                      # (m, n),  O(mn)
    correction = RH @ inner_inv @ RH.T                # (m, m),  O(mn^2)

    S_inv = np.diag(R_inv_diag) - correction          # O(m^2)
    return S_inv


def woodbury_innovation_inverse_lowmem(
    P_prior: np.ndarray,
    H: np.ndarray,
    R_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient factored form of the Woodbury innovation inverse.

    Instead of materialising the full m×m matrix S^{-1}, this function returns
    the three lightweight factors (R_inv_diag, RH, inner_inv) that define it:

        S^{-1} v = R_inv_diag * v - RH @ (inner_inv @ (RH.T @ v))

    This reduces the spatial complexity of the Kalman gain computation from
    O(m^2) down to O(mn), making the filter scale to m = 5,000+ sensor nodes
    without materialising a 5000×5000 matrix.

    Specifically, the Kalman gain K = P H^T S^{-1} is computed as:

        PH_T   = P_prior @ H.T                   # (n, m)
        K      = PH_T * R_inv_diag               # broadcast row-wise
                 - (PH_T @ RH) @ inner_inv @ RH.T   # (n, m)

    Both terms are O(mn^2) — no m×m matrix is ever instantiated.

    Args:
        P_prior:  A priori state covariance, shape (n, n).
        H:        Observation Jacobian, shape (m, n).
        R_diag:   Diagonal of measurement noise covariance R, shape (m,).

    Returns:
        R_inv_diag:  shape (m,)  — diagonal of R^{-1}
        RH:          shape (m,n) — R^{-1} H
        inner_inv:   shape (n,n) — (P^{-1} + H^T R^{-1} H)^{-1}
    """
    m, n = H.shape
    R_inv_diag = 1.0 / R_diag
    P_inv = _stable_inverse(P_prior)
    inner = P_inv + (H.T * R_inv_diag) @ H
    # Enforce exact symmetry before Cholesky: floating-point accumulation in
    # (H.T * R_inv_diag) @ H can produce asymmetry at the ~1e-15 level, which
    # is enough to fail the Cholesky factorisation on a theoretically SPD matrix.
    inner = 0.5 * (inner + inner.T)
    inner_inv = _stable_inverse(inner)
    RH = H * R_inv_diag[:, None]
    return R_inv_diag, RH, inner_inv


def kalman_gain_lowmem(
    P_prior: np.ndarray,
    H: np.ndarray,
    R_diag: np.ndarray,
) -> np.ndarray:
    """
    Compute the Kalman gain K = P H^T S^{-1} without forming S^{-1} ∈ R^{m×m}.

    Uses the factored Woodbury representation to keep spatial complexity at
    O(mn^2) instead of O(m^2 n). For m = 5,000, n = 10 this is a 500× memory
    reduction — the critical enabler for deployment on embedded edge hardware.

    Derivation:
        K = P H^T S^{-1}
          = P H^T [R^{-1} - R^{-1} H (P^{-1} + H^T R^{-1} H)^{-1} H^T R^{-1}]
          = PH^T * R_inv_diag  -  (PH^T @ RH) @ inner_inv @ RH^T

    Args:
        P_prior:  A priori state covariance, shape (n, n).
        H:        Observation Jacobian, shape (m, n).
        R_diag:   Diagonal measurement noise, shape (m,).

    Returns:
        K: Kalman gain matrix, shape (n, m).
    """
    R_inv_diag, RH, inner_inv = woodbury_innovation_inverse_lowmem(P_prior, H, R_diag)
    PH_T = P_prior @ H.T                                           # (n, m)
    # R_inv_diag[None, :] forces explicit (1, m) → (n, m) broadcast,
    # making the intended column-wise scaling unambiguous regardless of
    # how PH_T was constructed or reshaped by a caller.
    K = PH_T * R_inv_diag[None, :] - (PH_T @ RH) @ inner_inv @ RH.T  # (n, m)
    return K


def flop_count(m: int, n: int) -> dict:
    """
    Theoretical FLOP estimates for standard vs Woodbury inversion.

    Based on NanoPose paper, Section IV-B, Eqs. (11) and (12).

    Args:
        m: Number of sensor observation nodes.
        n: Dimensionality of kinematic state space.

    Returns:
        Dictionary with 'standard' and 'woodbury' FLOP counts.
    """
    standard = m**3 + m**2 * n          # Eq. (11)
    woodbury  = n**3 + m * n**2         # Eq. (12)
    return {
        "standard":        standard,
        "woodbury":        woodbury,
        "reduction_ratio": standard / woodbury,
    }


def _stable_inverse(A: np.ndarray) -> np.ndarray:
    """
    Numerically stable inverse via ridge-regularised Cholesky decomposition.

    A tiny diagonal ridge (_CHOL_RIDGE) is added before factorisation to absorb
    floating-point negative eigenvalues (e.g. -1e-14) that arise from rounding
    in P_{t|t-1} accumulation. The ridge is small enough to leave conditioning
    unchanged for any well-posed filter problem, but guarantees the Cholesky
    factorisation never encounters a non-positive-definite matrix.

    Falls back to np.linalg.inv if Cholesky still fails (e.g. a genuinely
    singular matrix should not occur in a healthy filter).
    """
    try:
        A_reg = A + _CHOL_RIDGE * np.eye(len(A))
        L = np.linalg.cholesky(A_reg)
        return solve(L.T, solve(L, np.eye(len(A))))
    except np.linalg.LinAlgError:
        return np.linalg.inv(A)
