import numpy as np

def circle_area(r, npts=1000):
    """Numerically integrate the quarter-circle arc to get the area of a circle."""
    x = np.linspace(0, r, npts)
    y = 4 * np.sqrt(np.maximum(r**2 - x**2, 0))
    return np.trapezoid(y, x)

def stlsq(D, b, lam=0.1, max_iter=20):
    """
    Sequential Thresholded Least Squares (Brunton et al., SINDy).

    D : (N, K) design matrix of candidate basis functions
    b : (N,) target data vector
    lam : threshold fraction, relative to the largest surviving coefficient
    max_iter : maximum number of prune-and-refit rounds
    """
    c, *_ = np.linalg.lstsq(D, b, rcond=None)
    active = np.ones(D.shape[1], dtype=bool)
    history = []

    for it in range(max_iter):
        threshold = lam * np.max(np.abs(c[active]))
        small = np.abs(c) < threshold
        newly_killed = small & active
        active = active & ~small

        history.append((it, active.copy(), c.copy()))

        if not np.any(newly_killed):
            break  # active set has stabilized

        c_active, *_ = np.linalg.lstsq(D[:, active], b, rcond=None)
        c = np.zeros(D.shape[1])
        c[active] = c_active

    return c, active, history

# --- Generate data ---
np.random.seed(0)
r = np.random.uniform(1, 5, 30)
b = np.array([circle_area(ri) for ri in r])

# --- Build overcomplete Vandermonde library: columns 1, r, r^2, ..., r^10 ---
deg = 10
D = np.vander(r, deg + 1, increasing=True)

# --- Run STLSQ ---
c_final, active_final, history = stlsq(D, b, lam=0.1)

print("Iterations to converge:", len(history))
print("Active terms (indices):", np.where(active_final)[0])
for i in np.where(active_final)[0]:
    print(f"  c_{i} (r^{i}) = {c_final[i]:.6f}")
print("True value: c_2 should equal pi =", np.pi)
print("Error:", abs(c_final[2] - np.pi))