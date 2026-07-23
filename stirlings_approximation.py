import numpy as np
from scipy.special import gammaln

# --- Generate data ---
n = np.arange(5, 500)          # n = 5, 6, ..., 499
b = gammaln(n + 1)              # ln(n!) via the log-gamma function

# --- Build design matrix: columns [1, n ln n, n, ln n] ---
A = np.column_stack([
    np.ones_like(n, dtype=float),
    n * np.log(n),
    n.astype(float),
    np.log(n)
])

true = np.array([0.5 * np.log(2 * np.pi), 1.0, -1.0, 0.5])

# --- Condition numbers ---
U, S, Vt = np.linalg.svd(A, full_matrices=False)
cond_A = S[0] / S[-1]
cond_AtA = np.linalg.cond(A.T @ A)

print("Condition number of A:", cond_A)
print("Condition number of A^T A:", cond_AtA)
print("Ratio (should be about cond(A)^2):", cond_AtA / cond_A**2)

# --- QR solve ---
Q, R = np.linalg.qr(A)
x_qr = np.linalg.solve(R, Q.T @ b)

# --- SVD solve ---
x_svd = Vt.T @ np.diag(1 / S) @ (U.T @ b)

# --- Normal equations solve ---
AtA = A.T @ A
Atb = A.T @ b
x_normal = np.linalg.solve(AtA, Atb)

print("\nTrue coefficients:      ", true)
print("QR solution:             ", x_qr)
print("SVD solution:            ", x_svd)
print("Normal equations sol.:   ", x_normal)

print("\nQR error (L2 norm):     ", np.linalg.norm(x_qr - true))
print("SVD error (L2 norm):    ", np.linalg.norm(x_svd - true))
print("Normal eqns error (L2): ", np.linalg.norm(x_normal - true))