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