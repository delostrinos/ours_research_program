#!/usr/bin/env python3
"""
Learning the Area of a Circle via LSSR (Least Squares -> Sparsify -> Refine).

Python translation of the MATLAB code from "Learning Mathematics Through the
Machine". The algorithm is NEVER told that area = pi * r^2. It only gets
(radius, numerically-integrated-area) pairs and discovers that the only
surviving coefficient is the r^2 term, and that it equals pi.

This file implements the least-squares step TWO ways so you can compare:
  - "normal"  : c = (D^T D)^{-1} D^T b   <- the paper's textbook formula
  - "qrsvd"   : solve D c = b via QR/SVD <- the numerically stable way
Both should recover pi; the difference shows up in conditioning, not here.
"""

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# Tunable parameters (mirror numP, numTrials, N in the MATLAB main script)
# ----------------------------------------------------------------------------
NUM_PARAMS = 11        # degree-10 polynomial => 11 coefficients c0..c10
NUM_TRIALS = 30        # number of random radii (training samples)
QUAD_POINTS = 100_000  # points used to numerically integrate each area
RADIUS_MIN = 1.0       # radii sampled uniformly from [RADIUS_MIN, RADIUS_MAX]
RADIUS_MAX = 5.0


# ----------------------------------------------------------------------------
# Step 1 (INPUT side): the integral that defines the "true" area.
#   g(r) = integral_0^r  4 * sqrt(r^2 - x^2) dx
# Computed with the trapezoidal rule, exactly like MATLAB's trapz.
# The computer does NOT know this equals pi*r^2; it just sums trapezoids.
# ----------------------------------------------------------------------------
def numeric_area(r):
    n = int(np.ceil(r * QUAD_POINTS))      # more points for bigger radii
    x = np.linspace(0.0, r, n)             # grid from 0 to r
    f = 4.0 * np.sqrt(np.clip(r * r - x * x, 0.0, None))  # clip guards tiny negs
    return np.trapezoid(f, x)              # trapezoidal rule


# ----------------------------------------------------------------------------
# The two least-squares solvers. This is the ONLY place the two approaches
# differ. Everything upstream (data, Vandermonde) and downstream
# (sparsify, refine) is shared.
# ----------------------------------------------------------------------------
def lstsq_normal(D, b):
    r"""Normal equations: c = (D^T D)^{-1} D^T b. Matches the paper's formula
    and MATLAB's `(R'*R)\(R'*g)`. Forms D^T D, which squares the condition
    number -- fine for this clean problem, dangerous at high degree."""
    return np.linalg.solve(D.T @ D, D.T @ b)


def lstsq_qrsvd(D, b):
    """Solve D c = b directly via SVD-based least squares (np.linalg.lstsq).
    Never forms D^T D, so it keeps the condition number from being squared.
    Same answer, computed the numerically honest way."""
    c, *_ = np.linalg.lstsq(D, b, rcond=None)
    return c


SOLVERS = {"normal": lstsq_normal, "qrsvd": lstsq_qrsvd}


# ----------------------------------------------------------------------------
# The loss function that Refine minimizes -- this is MATLAB's getObjective.
# It rebuilds the full coefficient vector from the surviving subspace, then
# measures the average squared gap between the polynomial and the true areas.
# ----------------------------------------------------------------------------
def make_objective(r_vec, areas, survivors, num_params):
    def objective(sub_coeffs):
        c = np.zeros(num_params)
        c[survivors] = sub_coeffs          # embed survivors back into full vector
        # polynomial evaluated at every radius: sum_j c_j * r^j
        powers = np.vander(r_vec, num_params, increasing=True)  # [1, r, r^2, ...]
        pred = powers @ c
        return 0.5 * np.mean((pred - areas) ** 2)
    return objective


# ----------------------------------------------------------------------------
# Sparsify + Refine -- this is MATLAB's sparsify function.
#   Sparsify: zero out any coefficient below 10% of the largest one.
#   Refine:   re-minimize the loss over ONLY the surviving coefficients.
# ----------------------------------------------------------------------------
def sparsify_and_refine(r_vec, areas, c_lsq, num_params):
    thresh = np.max(np.abs(c_lsq)) / 10.0           # 10% of the max coefficient
    c_sparse = np.where(np.abs(c_lsq) < thresh, 0.0, c_lsq)
    survivors = np.flatnonzero(c_sparse)            # indices that survived

    objective = make_objective(r_vec, areas, survivors, num_params)
    x0 = c_sparse[survivors]                        # start from the survivors
    result = minimize(objective, x0, method="BFGS",
                      options={"gtol": 1e-12, "maxiter": 10_000})

    c_star = np.zeros(num_params)
    c_star[survivors] = result.x                    # embed refined values back
    return c_star, thresh, survivors


# ----------------------------------------------------------------------------
# Main driver.
# ----------------------------------------------------------------------------
def run(method="normal", seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    r_vec = np.sort((RADIUS_MAX - RADIUS_MIN) * rng.random(NUM_TRIALS) + RADIUS_MIN)

    # observation vector b: numerically integrated area for each radius
    areas = np.array([numeric_area(r) for r in r_vec])

    # Vandermonde design matrix D: row k is [1, r_k, r_k^2, ..., r_k^10]
    D = np.vander(r_vec, NUM_PARAMS, increasing=True)

    # Step: Least squares (the swappable part)
    c_lsq = SOLVERS[method](D, areas)

    # Steps: Sparsify + Refine
    c_star, thresh, survivors = sparsify_and_refine(r_vec, areas, c_lsq, NUM_PARAMS)

    # Compare against the truth: only c_2 should be nonzero, equal to pi
    c_exact = np.zeros(NUM_PARAMS)
    c_exact[2] = np.pi
    error = np.linalg.norm(c_exact - c_star)

    if verbose:
        print(f"\n=== method = {method!r} (seed={seed}) ===")
        print(f"threshold used: {thresh:.6g}")
        print(f"surviving indices: {survivors.tolist()}")
        print(f"c_lsq[2]  (raw least squares) = {c_lsq[2]:.10f}")
        print(f"c_star[2] (after refine)      = {c_star[2]:.16f}")
        print(f"pi                            = {np.pi:.16f}")
        print(f"|c_star[2] - pi|              = {abs(c_star[2] - np.pi):.3e}")
        print(f"full-vector error ||c_exact - c_star|| = {error:.3e}")
    return c_lsq, c_star, error


if __name__ == "__main__":
    run("normal")
    run("qrsvd")
