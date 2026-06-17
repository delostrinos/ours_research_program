#!/usr/bin/env python3
"""
Sweep T_MAX for the a-fixed (univariate e^t) experiment.

For each sampling window t in [0, T_MAX], run the LSSR pipeline and record the
recovered t-coefficients. We expect: as the window widens, the high-order terms
(t^3, t^4) gain signal and sharpen toward 1/n! -- until T_MAX gets large enough
that a degree-4 polynomial can no longer track e^t and truncation error wrecks
everything.

This imports the machinery from learn_exp_afixed.py and only overrides the
sampling window, so the algorithm itself is unchanged.
"""
import numpy as np
from math import factorial
import learn_exp_afixed as L

N_DEG = L.N_DEG
EXACT = np.array([1.0 / factorial(n) for n in range(N_DEG + 1)])

def run_one(tmax, seed=0):
    """Replicate L.run's body but with a custom t-window and no printing."""
    rng = np.random.default_rng(seed)
    a_vec = np.full(L.NUM_TRIALS, L.A_FIXED)
    t_vec = tmax * rng.random(L.NUM_TRIALS)        # t in [0, tmax]
    b = np.array([L.true_x(a, t) for a, t in zip(a_vec, t_vec)])
    D = L.design_matrix(a_vec, t_vec)
    c_lsq = L.lstsq_qrsvd(D, b)                     # stable solver
    c_star, thresh, survivors = L.sparsify_and_refine(D, b, c_lsq, L.NUM_PARAMS)
    coeffs = c_star.reshape(L.M_DEG + 1, N_DEG + 1)[0]   # the single t-row
    err = np.linalg.norm(coeffs - EXACT)
    cond = np.linalg.cond(D)
    return coeffs, err, cond

print(f"{'exact':>40}: " + "  ".join(f"{v:8.5f}" for v in EXACT))
print("-" * 92)
header = "Tmax  | " + "  ".join(f"t^{n:<6}" for n in range(N_DEG + 1)) + " |   error    cond(D)"
print(header)
print("-" * 92)
for tmax in [1, 2, 3, 4, 5, 6, 8, 10]:
    coeffs, err, cond = run_one(tmax)
    row = "  ".join(f"{c:8.5f}" for c in coeffs)
    print(f"{tmax:4}  | {row} | {err:.2e}  {cond:.1e}")
print("-" * 92)
print(f"{'exact 1/n!':>5}  | " + "  ".join(f"{v:8.5f}" for v in EXACT))
