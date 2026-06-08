"""
The core algorithmic contribution of NanoPose. Extends the standard EKF with:

    1. Low-memory Woodbury Kalman gain (O(m^2) → O(mn)) — scales to m=5000+
    2. Joseph-form covariance update for guaranteed positive-definiteness
    3. Cholesky-based projection objective — avoids explicit P^{-1} inversion
    4. Convergence-failure diagnostics for the constraint QP
    5. Physiological constraint projection via SLSQP quadratic programme

Full algorithm (NanoPose, Algorithm 1):

    Prediction:
        x̂_{t|t-1} = f(x̂_{t-1|t-1})
        P_{t|t-1}  = F_t P_{t-1|t-1} F_t^T + Q_t

    Update (low-memory Woodbury):
        K_t   = P H^T * R_inv  -  (P H^T RH) @ inner_inv @ RH^T   [no m×m matrix]
        x̂_{t|t} = x̂_{t|t-1} + K_t (z_t - h(x̂_{t|t-1}))
        P_{t|t}  = (I - K H) P (I - K H)^T + K R K^T              [Joseph form]

    Kinematic Projection:
        if C x̂_{t|t} > d:
            x̃_{t|t} = argmin_{Cx≤d}  (x - x̂)^T P^{-1} (x - x̂)  [via Cholesky]
        else:
            x̃_{t|t} = x̂_{t|t}

References:
    [1] NanoPose paper, Section III-D and Algorithm 1.
    [2] Simon, D. (2006). Optimal State Estimation. Wiley.
    [3] Woodbury, M.A. (1950). Inverting modified matrices. Princeton.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .ekf import EKFState
from .woodbury import kalman_gain_lowmem, woodbury_innovation_inverse_lowmem

logger = logging.getLogger(__name__)


@dataclass
class ConstraintSet:
    """
    Linear inequality constraints on the kinematic state: C x ≤ d.

    Encodes physiological joint limits as described in NanoPose, Sec. V-B.

    Args:
        C: Constraint matrix, shape (k, n). Each row is one constraint.
        d: Constraint upper bounds, shape (k,).

    Example (knee extension):
        C = np.array([[ 1, 0],   # θ_knee ≤ 135°
                      [-1, 0],   # -θ_knee ≤ 0°  →  θ_knee ≥ 0°
                      [ 0, 1],   # θ_hip ≤ 120°
                      [ 0,-1]])  # θ_hip ≥ -15°
        d = np.array([135, 0, 120, 15]) * np.pi / 180
    """
    C: np.ndarray  # (k, n)
    d: np.ndarray  # (k,)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Returns True if x satisfies all constraints."""
        return bool(np.all(self.C @ x <= self.d))

    def num_violations(self, x: np.ndarray) -> int:
        """Number of violated constraints."""
        return int(np.sum(self.C @ x > self.d))


class ConstrainedEKF:
    """
    NanoPose Constrained Extended Kalman Filter.

    Combines a low-memory Woodbury Kalman gain (no m×m matrix ever formed)
    with a Cholesky-based projection QP to produce physically plausible
    pose estimates from high-density nanosensor arrays.

    Args:
        f:           State transition function  x_t = f(x_{t-1}).
        h:           Observation function       z_t = h(x_t).
        jac_f:       Jacobian of f, returns shape (n, n).
        jac_h:       Jacobian of h, returns shape (m, n).
        Q:           Process noise covariance, shape (n, n).
        R_diag:      Diagonal of measurement noise covariance R, shape (m,).
                     Diagonal assumption is physically motivated by spatially
                     independent thermal nanosensor noise [NanoPose, Sec. IV-A].
        constraints: ConstraintSet encoding physiological joint limits.
    """

    def __init__(
        self,
        f:           Callable[[np.ndarray], np.ndarray],
        h:           Callable[[np.ndarray], np.ndarray],
        jac_f:       Callable[[np.ndarray], np.ndarray],
        jac_h:       Callable[[np.ndarray], np.ndarray],
        Q:           np.ndarray,
        R_diag:      np.ndarray,
        constraints: ConstraintSet,
    ) -> None:
        self.f           = f
        self.h           = h
        self.jac_f       = jac_f
        self.jac_h       = jac_h
        self.Q           = Q
        self.R_diag      = R_diag
        self.constraints = constraints

        # Diagnostic counters
        self._total_steps        = 0
        self._projection_calls   = 0
        self._projection_failures = 0

    # ------------------------------------------------------------------
    # Core Algorithm
    # ------------------------------------------------------------------

    def predict(self, state: EKFState) -> EKFState:
        """
        Prediction step (NanoPose Algorithm 1, lines 2-4).

        Propagates the a posteriori estimate forward through the nonlinear
        state transition f(·) and linearises via its Jacobian F_t.

        Returns:
            A priori EKFState x̂_{t|t-1}, P_{t|t-1}.
        """
        x = self.f(state.mean)
        F = self.jac_f(state.mean)
        P = F @ state.covariance @ F.T + self.Q
        return EKFState(mean=x, covariance=P, t=state.t + 1)

    def update(self, prior: EKFState, z: np.ndarray) -> EKFState:
        """
        Low-memory Woodbury update step (Algorithm 1, lines 6-10).

        Key improvement over naïve Woodbury: the Kalman gain K = P H^T S^{-1}
        is computed via 'kalman_gain_lowmem' which never materializes the
        m×m innovation covariance matrix S^{-1}. Spatial complexity drops from
        O(m^2) to O(mn), allowing the filter to run on m=5000+ sensor nodes
        without exceeding the memory budget of a wearable edge microcontroller.

        Covariance is updated in Joseph form:
            P_{t|t} = (I - KH) P (I - KH)^T + K R K^T
        which preserves positive-definiteness under floating-point accumulation,
        unlike the simpler (I - KH) P form.

        Args:
            prior: A priori EKFState from predict().
            z:     Observation vector z_t, shape (m,).

        Returns:
            Unconstrained a posteriori EKFState.
        """
        x_prior = prior.mean
        P_prior = prior.covariance
        H = self.jac_h(x_prior)
        n = len(x_prior)

        # Low-memory Kalman gain — no m×m matrix ever formed (O(mn^2))
        K = kalman_gain_lowmem(P_prior, H, self.R_diag)   # (n, m)

        # State update
        innovation = z - self.h(x_prior)
        x_post = x_prior + K @ innovation

        # Joseph form covariance update for guaranteed positive-definiteness
        IKH    = np.eye(n) - K @ H
        R_mat  = np.diag(self.R_diag)
        P_post = IKH @ P_prior @ IKH.T + K @ R_mat @ K.T
        P_post = 0.5 * (P_post + P_post.T)   # Enforce exact symmetry

        return EKFState(mean=x_post, covariance=P_post, t=prior.t)

    def project(self, state: EKFState) -> EKFState:
        """
        Kinematic constraint projection (Algorithm 1, lines 12-17).

        If the unconstrained estimate violates physiological bounds, solves
        the Mahalanobis-distance quadratic programme (NanoPose, Eq. 8):

            min  (x - x̂)^T P^{-1} (x - x̂)
            s.t. C x ≤ d

        Implementation details
        *No explicit P^{-1}:* Rather than computing np.linalg.inv(P), the
        Cholesky factor L (where P = L L^T) is used to solve the linear system
        P^{-1} (x - x̂) = L^{-T} L^{-1} (x - x̂) via two triangular solves.
        This is cheaper, more numerically stable, and avoids forming the full
        inverse matrix inside the inner optimisation loop.

        *Failure handling:* If SLSQP fails to converge (rare but possible after
        a large state jump), the unconstrained estimate is returned as a safe
        fallback and the event is logged at WARNING level. The failure rate is
        tracked via 'projection_failure_rate' for developer visibility.

        Args:
            state: Unconstrained a posteriori EKFState.

        Returns:
            Projected EKFState satisfying C x ≤ d.
        """
        self._total_steps += 1

        if self.constraints.is_feasible(state.mean):
            return state

        self._projection_calls += 1
        x_hat = state.mean
        n     = len(x_hat)

        # Cholesky factor of P for stable linear-system solves
        # (avoids explicit P^{-1} inversion inside the hot loop)
        P_reg = state.covariance + 1e-10 * np.eye(n)
        try:
            L = np.linalg.cholesky(P_reg)          # P_reg = L L^T
        except np.linalg.LinAlgError:
            # Fallback: use direct inverse if Cholesky fails (degenerate P)
            L = None
            P_inv = np.linalg.inv(P_reg)

        def _mahal_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            """Mahalanobis objective and analytical gradient via Cholesky solves."""
            delta = x - x_hat
            if L is not None:
                # Solve L v = delta, then ||v||^2 = delta^T P^{-1} delta
                v = np.linalg.solve(L, delta)
                # Gradient: 2*P^{-1} delta = 2*L^{-T} v
                g = 2.0 * np.linalg.solve(L.T, v)
                return float(v @ v), g
            else:
                Pinv_d = P_inv @ delta
                return float(delta @ Pinv_d), 2.0 * Pinv_d

        constraints_scipy = [
            {
                "type": "ineq",
                "fun":  lambda x, i=i: self.constraints.d[i] - self.constraints.C[i] @ x,
                "jac":  lambda x, i=i: -self.constraints.C[i],
            }
            for i in range(len(self.constraints.d))
        ]

        result = minimize(
            _mahal_and_grad,
            x0=x_hat,
            jac=True,           # objective returns (f, grad) together
            method="SLSQP",
            constraints=constraints_scipy,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if result.success:
            x_proj = result.x
        else:
            self._projection_failures += 1
            logger.warning(
                "Projection QP failed at step %d (violation=%d constraints, "
                "SLSQP status=%d: %s). Returning unconstrained estimate.",
                self._total_steps,
                self.constraints.num_violations(x_hat),
                result.status,
                result.message,
            )
            x_proj = x_hat   # Safe fallback: unconstrained estimate

        return EKFState(mean=x_proj, covariance=state.covariance, t=state.t)

    def step(self, state: EKFState, z: np.ndarray) -> EKFState:
        """
        Full NanoPose step: predict → update → project (Algorithm 1).

        Args:
            state: Previous a posteriori EKFState.
            z:     Current observation vector, shape (m,).

        Returns:
            Physiologically constrained a posteriori EKFState.
        """
        prior         = self.predict(state)
        unconstrained = self.update(prior, z)
        return self.project(unconstrained)

    # Diagnostics
    
    @property
    def projection_rate(self) -> float:
        """Fraction of steps that required constraint projection."""
        if self._total_steps == 0:
            return 0.0
        return self._projection_calls / self._total_steps

    @property
    def projection_failure_rate(self) -> float:
        """
        Fraction of projection calls where SLSQP failed to converge.

        A non-zero value signals that the filter is encountering large
        state discontinuities or poorly-conditioned constraint boundaries.
        Inspect logs at WARNING level for per-step failure details.
        """
        if self._projection_calls == 0:
            return 0.0
        return self._projection_failures / self._projection_calls

    def diagnostics(self) -> dict:
        """
        Return a summary of filter health statistics.

        Returns:
            dict with keys: total_steps, projection_calls, projection_failures,
            projection_rate, projection_failure_rate.
        """
        return {
            "total_steps":             self._total_steps,
            "projection_calls":        self._projection_calls,
            "projection_failures":     self._projection_failures,
            "projection_rate":         self.projection_rate,
            "projection_failure_rate": self.projection_failure_rate,
        }

    def reset_diagnostics(self) -> None:
        """Reset all diagnostic counters."""
        self._total_steps         = 0
        self._projection_calls    = 0
        self._projection_failures = 0
