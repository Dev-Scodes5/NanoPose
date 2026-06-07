# NanoPose

**Constrained Nonlinear Filtering for High-Density Wearable Nanosensor Arrays**

> **Reference implementation** of the NanoPose architecture to be presented at [IEEE-NANO 2026]. 
> This library provides a fully documented, numerically verified Python implementation of the constrained
> Extended Kalman Filter described in the paper, with reproducible simulation results.

---

## The Problem

Standard optical pose estimation fails under occlusion, a frequent occurrence in physical rehabilitation when limbs overlap, equipment blocks the camera, or therapists obstruct the view. Wearable IMUs bypass this, but their rigid form factor and cumulative drift make them impractical for long-term outpatient monitoring.

**High-density piezoelectric nanosensor fabrics** embedded in clothing offer a third path through visually-independent, unobtrusive biomechanical tracking. The barrier is computational. A 1000-node sensor grid generates observation vectors in ℝ¹⁰⁰⁰, and the standard Extended Kalman Filter update requires inverting an innovation covariance matrix **S ∈ ℝ^{m×m}**, an O(m³) operation that exceeds the capacity of any wearable microcontroller.

## The Solution

NanoPose resolves this with two coupled innovations:

**1. Woodbury-optimised innovation inversion (O(m³) → O(n³))**

By exploiting the diagonal structure of thermal noise and the Woodbury matrix
identity, the inversion is reformulated in the *kinematic state space* (n ~ 10 DOF) rather than the *sensor space* (m ~ 1000 nodes):

$$S^{-1} = R^{-1} - R^{-1} H \left(P^{-1} + H^\top R^{-1} H\right)^{-1} H^\top R^{-1}$$

For m = 1000, n = 10, this yields a **~10⁶× reduction** in theoretical FLOPs.

**2. Physiological constraint projection (Quadratic Programming)**

The unconstrained EKF posterior is projected onto the physiologically feasible
subspace by solving a Mahalanobis-distance quadratic programme:

$$\tilde{x}_{t|t} = \arg\min_{x} \; (x - \hat{x})^\top P^{-1} (x - \hat{x}) \quad \text{s.t.} \quad Cx \leq d$$

This eliminates physically impossible artifact poses induced by high-dimensional
sensor noise, without requiring heuristic post-processing.

---

## Quickstart

```bash
pip install nanopose          # PyPI (coming soon)
# or from source:
git clone https://github.com/sheraz-arshad/nanopose
cd nanopose && pip install -e .
```

```python
import numpy as np
from nanopose import (
    ConstrainedEKF, ConstraintSet, EKFState,
    LowerBodyKinematics, PiezoelectricObservationModel,
    simulate_knee_extension, compute_rmse,
)

# 1. Build the observation model (200-node piezoelectric fabric, 2 joints)
obs_model = PiezoelectricObservationModel(n_nodes=200, n_joints=2)

# 2. Simulate a knee extension with σ=0.1V thermal noise
sim = simulate_knee_extension(obs_model, dt=0.01, duration=2.0, sigma_noise=0.1)

# 3. Configure the constrained EKF
kin = LowerBodyKinematics(dt=0.01, joints=2)
C, d = LowerBodyKinematics.default_constraints(joints=2)

ekf = ConstrainedEKF(
    f=kin.f, h=obs_model.h,
    jac_f=kin.jac_f, jac_h=obs_model.jac_h,
    Q=LowerBodyKinematics.process_noise(dt=0.01),
    R_diag=obs_model.noise_covariance_diag(sigma_v=0.1),
    constraints=ConstraintSet(C=C, d=d),
)

# 4. Run the filter
state = EKFState(mean=np.zeros(4), covariance=np.eye(4) * 0.01)
estimates = []
for z in sim.observations:
    state = ekf.step(state, z)
    estimates.append(state.mean[:2])   # Extract joint angles

theta_est = np.array(estimates)

# 5. Evaluate
rmse = compute_rmse(theta_est, sim.theta_true, in_degrees=True)
print(f"Knee RMSE: {rmse[0]:.2f}°  |  Hip RMSE: {rmse[1]:.2f}°")
# → Knee RMSE: 2.3°  |  Hip RMSE: 1.8°
```

---

## Results

### Computational Complexity (Figure 1)

| m (sensor nodes) | Standard EKF (FLOPs) | NanoPose Woodbury (FLOPs) | Reduction |
|:----------------:|:--------------------:|:-------------------------:|:---------:|
| 100              | 1.0 × 10⁶           | 1.1 × 10⁴               | ~91×      |
| 500              | 1.25 × 10⁸          | 2.6 × 10⁴               | ~4,800×   |
| 1000             | 1.0 × 10⁹           | 1.0 × 10⁵               | ~10,000×  |

### Tracking Accuracy (Figure 4)

Simulated seated knee extension (2s window, σ = 0.1V thermal noise):

| Metric                   | Value         |
|:-------------------------|:-------------:|
| Convergence time         | < 0.2 seconds |
| Mean Knee RMSE (steady)  | 2.3°          |
| Constraint violations    | 0             |

Reproduce these results:

```bash
python scripts/benchmark_flops.py
jupyter notebook notebooks/01_reproduce_fig4.ipynb
```