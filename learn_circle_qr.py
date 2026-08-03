#!/usr/bin/env python3
"""
Learning the Area of a Circle via LSSR -- QR / SVD version.

Same problem as learn_circle_normal.py, but we solve the least-squares step
using methods that DON'T form D^T D explicitly. This avoids squaring the
condition number, so c_lsq comes back with full double-precision fidelity --
no quantization "bruise."

The change is essentially one line:
    OLD:  c_lsq = np.linalg.inv(D.T @ D) @ (D.T @ b)
    NEW:  c_lsq, *_ = np.linalg.lstsq(D, b, rcond=None)

Under the hood, np.linalg.lstsq calls LAPACK's gelsd, which uses the SVD --
the same idea your friend CM mentioned in Discord.
"""

import numpy as np
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Parameters -- identical to the normal-equations file, including SEED, so the
# two scripts work on the exact same radii and the exact same b vector.
# -----------------------------------------------------------------------------
NUM_PARAMS  = 11
NUM_TRIALS  = 30
QUAD_POINTS = 100_000
RADIUS_MIN  = 1.0
RADIUS_MAX  = 5.0
SEED        = 42

def numeric_area(r, n_points=QUAD_POINTS):
    n = int(np.ceil(r * n_points))
    x = np.linspace(0.0, r, n)
    y = 4.0 * np.sqrt(np.maximum(r * r - x * x, 0.0))
    return np.trapezoid(y, x)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
radii = np.sort(rng.uniform(RADIUS_MIN, RADIUS_MAX, NUM_TRIALS))
b = np.array([numeric_area(r) for r in radii])
D = np.vander(radii, N=NUM_PARAMS, increasing=True)

cond_D = np.linalg.cond(D)
print(f"Condition number of D : {cond_D:.3e}")
print("(We never form D^T D, so we don't pay the squaring penalty.)\n")

# -----------------------------------------------------------------------------
# L: Least Squares via SVD (no normal equations).
#
# np.linalg.lstsq returns (solution, residuals, rank, singular_values).
# We capture all four so you can inspect the singular values -- their spread
# is what makes this problem hard, and SVD handles it gracefully.
# -----------------------------------------------------------------------------
c_lsq, residuals, rank, sing_vals = np.linalg.lstsq(D, b, rcond=None)

print("Least-squares coefficients (c_lsq) via SVD:")
for j, val in enumerate(c_lsq):
    print(f"  c[{j:2d}] (r^{j:<2d}) = {val: .8f}")

print(f"\nMatrix rank      : {rank}")
print(f"Singular values  : {sing_vals}")
print(f"Ratio max/min sv : {sing_vals.max() / sing_vals.min():.3e}")
print("(That ratio IS the condition number of D.)\n")

# -----------------------------------------------------------------------------
# S: Sparsify (same as before)
# -----------------------------------------------------------------------------
threshold = np.max(np.abs(c_lsq)) / 10.0
survivors = np.where(np.abs(c_lsq) >= threshold)[0]
print(f"Threshold = {threshold:.6f}")
print(f"Surviving coefficient indices: {survivors.tolist()}")

# -----------------------------------------------------------------------------
# R: Refine via BFGS on the surviving subspace
# -----------------------------------------------------------------------------
def loss_on_subspace(c_sub, survivors, D, b):
    c_full = np.zeros(NUM_PARAMS)
    c_full[survivors] = c_sub
    residual = D @ c_full - b
    return 0.5 * np.mean(residual ** 2)

c_sub_init = c_lsq[survivors]
result = minimize(
    loss_on_subspace,
    c_sub_init,
    args=(survivors, D, b),
    method="BFGS",
    options={"gtol": 1e-16, "xrtol": 1e-16, "maxiter": 10_000},
)

c_star = np.zeros(NUM_PARAMS)
c_star[survivors] = result.x

print("\nFinal sparse coefficients (c_star):")
for j, val in enumerate(c_star):
    marker = "  <-- nonzero!" if abs(val) > 1e-9 else ""
    print(f"  c[{j:2d}] (r^{j:<2d}) = {val: .12f}{marker}")

print(f"\nLearned coefficient on r^2 : {c_star[2]:.12f}")
print(f"Actual value of pi         : {np.pi:.12f}")
print(f"Absolute error             : {abs(c_star[2] - np.pi):.2e}")
