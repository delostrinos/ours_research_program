import numpy as np
from math import factorial
import learn_exp_afixed as L
N_DEG = L.N_DEG
EXACT = np.array([1.0/factorial(n) for n in range(N_DEG+1)])
def run_one(tmax, seed=0):
    rng = np.random.default_rng(seed)
    a_vec = np.full(L.NUM_TRIALS, L.A_FIXED)
    t_vec = tmax*rng.random(L.NUM_TRIALS)
    b = np.array([L.true_x(a,t) for a,t in zip(a_vec,t_vec)])
    D = L.design_matrix(a_vec,t_vec)
    c_lsq = L.lstsq_qrsvd(D,b)
    c_star,_,_ = L.sparsify_and_refine(D,b,c_lsq,L.NUM_PARAMS)
    coeffs = c_star.reshape(L.M_DEG+1,N_DEG+1)[0]
    return np.linalg.norm(coeffs-EXACT)
print("fine sweep near the low end:")
for tmax in [0.5,0.75,1.0,1.25,1.5,1.75,2.0]:
    print(f"  Tmax={tmax:4}:  error={run_one(tmax):.3e}")
