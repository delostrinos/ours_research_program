#!/usr/bin/env python3
"""
Learning the Area of a Circle via LSSR -- NORMAL EQUATIONS version.

This file uses the textbook formula  c = (D^T D)^-1 D^T b  explicitly, the same
formula we worked by hand and the same one the Ruby version used. Expect to see
the same "ill-conditioning bruise" we saw in Ruby: c_lsq comes back with low
effective precision because D^T D squares the already-bad condition number of
the Vandermonde matrix.

Compare against learn_circle_qr.py to see the difference numerics makes.
"""

import numpy as np
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Parameters (same as MATLAB / Ruby)
# -----------------------------------------------------------------------------
NUM_PARAMS  = 11        # degree-10 polynomial => 11 coefficients
NUM_TRIALS  = 30        # number of random radii (training samples)
QUAD_POINTS = 100_000   # points used in the numerical integral
RADIUS_MIN  = 1.0
RADIUS_MAX  = 5.0
SEED        = 42        # so the two scripts hit the same radii

# -----------------------------------------------------------------------------
# Input side: integral populating b, via the trapezoidal rule.
#
# In numpy this collapses to two lines instead of Ruby's hand loop -- this is
# the "vectorized power" you keep hearing about. np.linspace builds the grid,
# np.trapezoid sums the trapezoids.
# -----------------------------------------------------------------------------
def numeric_area(r, n_points=QUAD_POINTS):
    """Numerically integrate 4*sqrt(r^2 - x^2) from 0 to r."""
    n = int(np.ceil(r * n_points))
    x = np.linspace(0.0, r, n)
    y = 4.0 * np.sqrt(np.maximum(r * r - x * x, 0.0))
    return np.trapezoid(y, x)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
radii = np.sort(rng.uniform(RADIUS_MIN, RADIUS_MAX, NUM_TRIALS))

# Observation vector b: numerically integrated areas, one per radius.
b = np.array([numeric_area(r) for r in radii])

# Vandermonde matrix D: D[i, j] = radii[i]**j  for j in 0..10
# np.vander with increasing=True gives exactly this layout.
D = np.vander(radii, N=NUM_PARAMS, increasing=True)

# Diagnostic: how badly conditioned is this problem?
cond_D   = np.linalg.cond(D)
cond_DtD = np.linalg.cond(D.T @ D)
print(f"Condition number of D     : {cond_D:.3e}")
print(f"Condition number of D^T D : {cond_DtD:.3e}")
print("(Rule of thumb: log10(cond) digits of precision are lost.)\n")

# -----------------------------------------------------------------------------
# L: Least Squares via the NORMAL EQUATIONS  c = (D^T D)^-1 D^T b
# This is the formula from the paper and from our hand work. It is also the
# numerically dangerous one. We do it the dangerous way on purpose here.
# -----------------------------------------------------------------------------
DtD = D.T @ D
Dtb = D.T @ b
c_lsq = np.linalg.inv(DtD) @ Dtb

print("Least-squares coefficients (c_lsq) via normal equations:")
for j, val in enumerate(c_lsq):
    print(f"  c[{j:2d}] (r^{j:<2d}) = {val: .8f}")

# -----------------------------------------------------------------------------
# S: Sparsify -- zero out anything below 10% of the largest magnitude.
# -----------------------------------------------------------------------------
threshold = np.max(np.abs(c_lsq)) / 10.0
survivors = np.where(np.abs(c_lsq) >= threshold)[0]
print(f"\nThreshold = {threshold:.6f}")
print(f"Surviving coefficient indices: {survivors.tolist()}")

# -----------------------------------------------------------------------------
# R: Refine -- minimize the actual loss on the surviving subspace using a real
# nonlinear optimizer (scipy's BFGS), which is the Python analogue of MATLAB's
# fminunc. This matches the paper exactly, unlike the Ruby version where we
# substituted a sub-least-squares.
# -----------------------------------------------------------------------------
def loss_on_subspace(c_sub, survivors, D, b):
    """Empirical mean of 0.5 * (D_full @ c_full - b)^2, with c_full zero
    outside survivors."""
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

# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
print("\nFinal sparse coefficients (c_star):")
for j, val in enumerate(c_star):
    marker = "  <-- nonzero!" if abs(val) > 1e-9 else ""
    print(f"  c[{j:2d}] (r^{j:<2d}) = {val: .12f}{marker}")

print(f"\nLearned coefficient on r^2 : {c_star[2]:.12f}")
print(f"Actual value of pi         : {np.pi:.12f}")
print(f"Absolute error             : {abs(c_star[2] - np.pi):.2e}")
