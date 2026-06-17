#!/usr/bin/env python3
"""
Learning the solution of dx/dt = a*x, x(0)=1 via LSSR (Least Squares -> Sparsify
-> Refine).

This is the second problem from "Learning Mathematics Through the Machine",
built to mirror learn_circle.py as closely as possible. The algorithm is NEVER
told that the answer is exp(a*t). It only gets (a, t, x) triples -- where each x
is the true value exp(a*t) -- and it discovers that the only surviving
coefficients form the DIAGONAL of the coefficient matrix, with A[n,n] = 1/n!.

That diagonal IS the Maclaurin series for exp(a*t):
    x(a,t) = sum_n (a*t)^n / n! = sum_n (1/n!) * a^n * t^n.

Same five stages as the circle program, only lifted from one variable to two:
  - circle row was [1, r, r^2, ...]          (powers of ONE variable)
  - here  a row is [a^m * t^n for all m,n]   (products of TWO variables' powers)

Like before, least squares is implemented TWO ways for comparison:
  - "normal"  : c = (D^T D)^{-1} D^T b   <- the paper's textbook formula
  - "qrsvd"   : solve D c = b via QR/SVD <- the numerically stable way
"""

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# Tunable parameters. M, N are the max powers of a and t respectively.
#
# PROFESSOR'S EXPERIMENT: fix a = 1 and sample only t. This collapses the
# bivariate problem to a UNIVARIATE one -- exp(1*t) = e^t -- which is exactly
# the circle problem's shape. We do this by setting M_DEG = 0, so a appears
# only as a^0 = 1, and pinning a = 1 in the sampler. The target becomes the
# single row of Taylor coefficients 1/n! in t (no off-diagonal a-leakage
# possible). With N = 4 we can recover up to 1/4! = 1/24.
# ----------------------------------------------------------------------------
M_DEG = 0              # max power of a  => only a^0 = 1 (a is fixed out)
N_DEG = 10              # max power of t  => t^0 .. t^4
NUM_PARAMS = (M_DEG + 1) * (N_DEG + 1)   # 5 coefficients now, the flattened A

NUM_TRIALS = 1000000        # number of random t training samples

# With a fixed at 1, the exponent is just t, so e^t on [0,1] stays modest and a
# degree-4 polynomial can track it reasonably. a is no longer sampled.
A_FIXED = 1.0              # a is held constant at this value
T_MIN, T_MAX = 0.0, 1.0    # t sampled uniformly from [T_MIN, T_MAX]


# ----------------------------------------------------------------------------
# Step 1 (INPUT side): the "true" value of the solution.
# The circle program numerically integrated an integrand here. Per the paper,
# we are allowed to use the exact solution exp(a*t) instead -- so we do. The
# computer still does NOT know this is the answer; it is just a data source.
# ----------------------------------------------------------------------------
def true_x(a, t):
    return np.exp(a * t)


# ----------------------------------------------------------------------------
# Build ONE design-matrix row for a sample (a, t): every monomial a^m * t^n.
# This is the bivariate analog of the circle program's [1, r, r^2, ...].
# The column ordering is (m outer, n inner): index = m*(N_DEG+1) + n.
# ----------------------------------------------------------------------------
def design_row(a, t):
    a_powers = a ** np.arange(M_DEG + 1)      # [1, a, a^2, a^3, a^4]
    t_powers = t ** np.arange(N_DEG + 1)      # [1, t, t^2, t^3, t^4]
    return np.outer(a_powers, t_powers).ravel()  # all products a^m * t^n, flat


def design_matrix(a_vec, t_vec):
    """Stack one design_row per training sample -> D, shape (NUM_TRIALS, NUM_PARAMS)."""
    return np.array([design_row(a, t) for a, t in zip(a_vec, t_vec)])


# Helpers to move between the flat coefficient vector and the (M+1, N+1) matrix.
def vec_to_matrix(c):
    return c.reshape(M_DEG + 1, N_DEG + 1)


def matrix_to_vec(A):
    return A.ravel()


# ----------------------------------------------------------------------------
# The two least-squares solvers -- identical in spirit to learn_circle.py.
# This is the ONLY place the two approaches differ.
# ----------------------------------------------------------------------------
def lstsq_normal(D, b):
    r"""Normal equations: c = (D^T D)^{-1} D^T b. Matches the paper's formula.
    Forms D^T D, squaring the condition number -- fine here, dangerous at high
    degree."""
    return np.linalg.solve(D.T @ D, D.T @ b)


def lstsq_qrsvd(D, b):
    """Solve D c = b directly via SVD-based least squares (np.linalg.lstsq).
    Never forms D^T D, so it keeps the condition number from being squared."""
    c, *_ = np.linalg.lstsq(D, b, rcond=None)
    return c


SOLVERS = {"normal": lstsq_normal, "qrsvd": lstsq_qrsvd}


# ----------------------------------------------------------------------------
# The loss function that Refine minimizes -- the bivariate getObjective.
# Rebuild the full coefficient vector from the surviving subspace, then measure
# the average squared gap between the polynomial and the true values.
# ----------------------------------------------------------------------------
def make_objective(D, b, survivors, num_params):
    def objective(sub_coeffs):
        c = np.zeros(num_params)
        c[survivors] = sub_coeffs          # embed survivors back into full vector
        pred = D @ c                       # polynomial evaluated at every sample
        return 0.5 * np.mean((pred - b) ** 2)
    return objective


# ----------------------------------------------------------------------------
# Sparsify + Refine -- identical logic to learn_circle.py, just on the flat
# coefficient vector that represents the matrix A.
#   Sparsify: zero out any coefficient below 10% of the largest one.
#   Refine:   re-minimize the loss over ONLY the surviving coefficients.
# ----------------------------------------------------------------------------
def sparsify_and_refine(D, b, c_lsq, num_params):
    thresh = np.max(np.abs(c_lsq)) / 100000000000.0           # 10% of the max coefficient
    c_sparse = np.where(np.abs(c_lsq) < thresh, 0.0, c_lsq)
    survivors = np.flatnonzero(c_sparse)            # indices that survived

    objective = make_objective(D, b, survivors, num_params)
    x0 = c_sparse[survivors]                        # start from the survivors
    result = minimize(objective, x0, method="BFGS",
                      options={"gtol": 1e-12, "maxiter": 10_000})

    c_star = np.zeros(num_params)
    c_star[survivors] = result.x                    # embed refined values back
    return c_star, thresh, survivors


# ----------------------------------------------------------------------------
# Build the exact coefficient matrix for comparison: diagonal entries 1/n!,
# zeros elsewhere. This is the analog of "c_exact[2] = pi" from the circle code.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Build the exact coefficient matrix for comparison.
#
# In the FULL bivariate problem the truth is the diagonal A[n,n] = 1/n!.
# But in this a-fixed experiment (M_DEG = 0), a is pinned to 1, so the truth is
# the single row of Taylor coefficients of e^t: A[0, n] = 1/n! for every n.
# We detect the fixed-a case via M_DEG == 0 and build the right target so the
# error metric compares against the correct answer.
# ----------------------------------------------------------------------------
def exact_coeffs():
    from math import factorial
    A = np.zeros((M_DEG + 1, N_DEG + 1))
    if M_DEG == 0:
        # a fixed: every column n holds the t^n Taylor coefficient 1/n!
        for n in range(N_DEG + 1):
            A[0, n] = 1.0 / factorial(n)
    else:
        # full bivariate: diagonal entries only
        for n in range(min(M_DEG, N_DEG) + 1):
            A[n, n] = 1.0 / factorial(n)
    return A


# ----------------------------------------------------------------------------
# Main driver.
# ----------------------------------------------------------------------------
def run(method="normal", seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    a_vec = np.full(NUM_TRIALS, A_FIXED)   # a is held constant, not sampled
    t_vec = (T_MAX - T_MIN) * rng.random(NUM_TRIALS) + T_MIN

    # observation vector b: the true x = exp(a*t) for each sample
    b = np.array([true_x(a, t) for a, t in zip(a_vec, t_vec)])

    # design matrix D: row k is [a_k^m * t_k^n for all m, n]
    D = design_matrix(a_vec, t_vec)

    # Step: Least squares (the swappable part)
    c_lsq = SOLVERS[method](D, b)

    # Steps: Sparsify + Refine
    c_star, thresh, survivors = sparsify_and_refine(D, b, c_lsq, NUM_PARAMS)

    # Compare against the truth: only the diagonal A[n,n] = 1/n! should survive
    A_exact = exact_coeffs()
    A_star = vec_to_matrix(c_star)
    error = np.linalg.norm(A_exact - A_star)

    if verbose:
        np.set_printoptions(precision=8, suppress=True, linewidth=120)
        print(f"\n=== method = {method!r} (seed={seed}) ===")
        print(f"threshold used: {thresh:.6g}")
        print(f"surviving (m,n) indices: "
              f"{[(i // (N_DEG + 1), i % (N_DEG + 1)) for i in survivors]}")
        print("\nrecovered coefficient matrix A_star (rows=m, cols=n):")
        print(A_star)
        print("\nexact coefficient matrix (diagonal = 1/n!):")
        print(A_exact)
        print(f"\ndiagonal recovered: {np.diag(A_star)}")
        print(f"diagonal exact:     {np.diag(A_exact)}")
        if M_DEG == 0:
            print(f"\n[a fixed] t-coeffs recovered: {A_star[0]}")
            print(f"[a fixed] t-coeffs exact:     {A_exact[0]}")
        print(f"||A_exact - A_star|| = {error:.3e}")
    return c_lsq, c_star, error


if __name__ == "__main__":
    run("normal")
    run("qrsvd")
