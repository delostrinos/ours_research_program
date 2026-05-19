"""Recover the disk-area law g(r) = pi * r^2 from noisy numerical "measurements".

We pretend the only thing we can do is *measure* the area of a disk of radius r
by numerically integrating the quarter-arc f(x) = sqrt(r^2 - x^2) on [0, r] with
a LEFT Riemann sum (N points) and multiplying by 4. We then ask: given a
polynomial hypothesis A_hyp(r; c) = sum_j c_j r^j of degree p, which c best
fits the measurements?

Panel 1 shows that ordinary least squares (np.linalg.lstsq) spreads weight
across every monomial, while the true sparse vector c* = (0, 0, pi, 0, ..., 0)
is the *correct* generator. Panel 2 shows that the residual J_N(c*) under c*
shrinks toward zero at the rate of the Riemann sum itself as N grows --
evidence that c* is the right model and the apparent "error" is integrator
discretization, not model mis-specification.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

P = 10                       # polynomial degree
M = 200                      # number of radii
N_PANEL1 = 1000              # integration points for panel 1
N_RANGE = np.arange(5, 101)  # integration points sweep for panel 2
CLIP = 1e-13


def a_num(r, n):
    """Left Riemann sum approximation of the disk area for radii r."""
    r = np.asarray(r, dtype=float)
    dx = r / (n - 1)                                # (M,)
    i = np.arange(n - 1)                            # (n-1,) -> i = 0..n-2
    x = i[None, :] * dx[:, None]                    # (M, n-1)
    integrand = np.sqrt(np.clip(r[:, None] ** 2 - x ** 2, 0.0, None))
    return 4.0 * (integrand.sum(axis=1) * dx)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = 1.0 - np.random.uniform(0.0, 1.0, size=M)   # uniform in (0, 1]
    j = np.arange(P + 1)
    phi = r[:, None] ** j[None, :]                  # (M, P+1)

    b = a_num(r, N_PANEL1)
    c_lsq, *_ = np.linalg.lstsq(phi, b, rcond=None)

    c_star = np.zeros(P + 1)
    c_star[2] = np.pi

    j_n_star = np.empty(N_RANGE.size)
    a_hyp_star = phi @ c_star
    for k, n in enumerate(N_RANGE):
        a = a_num(r, int(n))
        j_n_star[k] = np.mean((a - a_hyp_star) ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    y_lsq = np.log10(np.maximum(np.abs(c_lsq), CLIP))
    y_star = np.log10(np.maximum(np.abs(c_star), CLIP))

    ax1.plot(j, y_lsq, "x", color="tab:blue", markersize=12,
             markeredgewidth=2, label=r"$c^{\,lsq}$")
    ax1.plot(j, y_star, "o", color="tab:red", markersize=10,
             markerfacecolor="none", markeredgewidth=2, label=r"$c^{*}$")
    ax1.axhline(np.log10(CLIP), linestyle="--", color="gray",
                linewidth=1, label="zero (clip)")
    ax1.set_xlabel("weight index j")
    ax1.set_ylabel(r"$\log(c)$")
    ax1.set_xticks(j)
    ax1.set_title("coefficients: sparse truth vs. ordinary least squares")
    ax1.legend(loc="lower left")
    ax1.annotate(f"$c^*_2 = \\pi = {np.pi:.6f}$",
                 xy=(2, y_star[2]), xytext=(3.2, y_star[2] + 0.6),
                 arrowprops=dict(arrowstyle="->", color="tab:red"),
                 color="tab:red")
    ax1.annotate(f"$c^{{lsq}}_2 = {c_lsq[2]:.6f}$",
                 xy=(2, y_lsq[2]), xytext=(3.2, y_lsq[2] - 1.2),
                 arrowprops=dict(arrowstyle="->", color="tab:blue"),
                 color="tab:blue")

    ax2.plot(N_RANGE, np.log10(j_n_star), "o", color="tab:blue",
             markersize=6, markerfacecolor="none", markeredgewidth=1.5)
    ax2.set_xlabel("N")
    ax2.set_ylabel(r"$\log J_N(c^*)$")
    ax2.set_title(r"residual under $c^*$ vs. integrator resolution")

    fig.tight_layout()
    out_path = out_dir / "visual_of_the_result.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"c_lsq[2]      = {c_lsq[2]:.12f}")
    print(f"c_star[2]     = {c_star[2]:.12f}  (pi)")
    print(f"J_N(c*) @ N={N_RANGE[-1]:<3d} = {j_n_star[-1]:.3e}")
    print(f"figure saved  -> {out_path}")


if __name__ == "__main__":
    main()
