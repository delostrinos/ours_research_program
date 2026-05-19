"""Recover the disk-area formula g(r) = pi * r^2 from numerical "measurements".

Builds the two-panel "Visual of the Result" figure (slide 31):

    Right panel  -- J_N(c*) vs N for a single random radius per N.
                    The sparse coefficient vector c* = [0, 0, pi, 0, ..., 0]
                    matches the left-Riemann disk-area integrator, so the
                    squared loss decays to integrator precision (~ exp(-10))
                    by N = 100.

    Left panel   -- Coefficient comparison. An ordinary least-squares fit
                    of a degree-p polynomial to the same Riemann-sum data
                    spreads weight across every index, while c* concentrates
                    all of its mass on j = 2 at value pi.

The whole experiment is parameterized around the professor's snippet, which
evaluates the squared loss between a left-Riemann disk-area measurement and
a polynomial hypothesis at one random radius. See `loss` below.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def riemann_area(r, N):
    """Left-Riemann approximation of the disk area pi*r^2 using N samples."""
    dx = r / (N - 1)
    x = np.arange(0, r + dx / 2, dx)
    with np.errstate(invalid="ignore"):  # arange tail may slightly exceed r
        I = 4 * np.sqrt(r**2 - x**2)
    return np.sum(I[:N - 1]) * dx


def loss(c, r, N):
    """Squared loss between Anum(r; N) and the polynomial sum_j c[j] * r**j."""
    c = np.asarray(c, dtype=float)
    j = np.arange(c.size)
    Ahyp = np.sum(c * r**j)
    Anum = riemann_area(r, N)
    return (Anum - Ahyp) ** 2


# ---------------------------------------------------------------------------
# c_star: the sparse "right answer" -- pi sits at index 2, zeros elsewhere.
# ---------------------------------------------------------------------------
p = 10
c_star = np.zeros(p + 1)
c_star[2] = np.pi

# ---------------------------------------------------------------------------
# Panel 2 (right): J_N(c*) vs N, one random radius per N.
# ---------------------------------------------------------------------------
Ns = np.arange(5, 101)
J_vals = np.empty(Ns.size)
for k, N in enumerate(Ns):
    r = np.random.rand()
    J_vals[k] = loss(c_star, r, N)

# ---------------------------------------------------------------------------
# Panel 1 (left): build c_lsq from M Riemann-sum measurements.
# ---------------------------------------------------------------------------
M = 200
N_fit = 1000
rs = np.random.rand(M)
b = np.array([riemann_area(r, N_fit) for r in rs])
Phi = rs[:, None] ** np.arange(p + 1)[None, :]
c_lsq, *_ = np.linalg.lstsq(Phi, b, rcond=None)

# Slide's y-axis: clip |c| < 1 to 1 so zero entries land on the y = 0 line.
y_star = np.log(np.maximum(np.abs(c_star), 1))
y_lsq = np.log(np.maximum(np.abs(c_lsq), 1))

# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

js = np.arange(p + 1)
axL.axhline(0, color="gray", linestyle="--", linewidth=1)
axL.plot(js, y_lsq, "x", color="tab:blue", markersize=10,
         markeredgewidth=2, label=r"$c^{\mathrm{lsq}}$")
axL.plot(js, y_star, "o", color="tab:red", markersize=10,
         markerfacecolor="none", markeredgewidth=2, label=r"$c^{*}$")
axL.annotate(f"{c_star[2]:.6f}", xy=(2, y_star[2]),
             xytext=(8, 6), textcoords="offset points", color="tab:red")
axL.annotate(f"{c_lsq[2]:.6f}", xy=(2, y_lsq[2]),
             xytext=(8, -14), textcoords="offset points", color="tab:blue")
axL.set_xlabel("weight index j")
axL.set_ylabel("log(c)")
axL.set_xticks(js)
axL.legend(loc="upper right")

axR.plot(Ns, np.log(J_vals), "o", color="tab:blue",
         markerfacecolor="none", markeredgewidth=1.5)
axR.set_xlabel("N")
axR.set_ylabel(r"$\log J_N(c^{*})$")

fig.tight_layout()
fig.savefig("figures/visual_of_the_result.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Stdout summary.
# ---------------------------------------------------------------------------
print(f"c_lsq[2]            = {c_lsq[2]:.10f}")
print(f"c_star[2]           = {c_star[2]:.10f}")
print(f"J_N(c*) at N={Ns[-1]:>3} = {J_vals[-1]:.3e}")
print(f"y(c_star[2]) marker = {y_star[2]:.6f}  (ln(pi) = {np.log(np.pi):.6f})")
