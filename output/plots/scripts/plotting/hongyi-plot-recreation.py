"""
plot_hongyi_silicon_mr.py — plot the silicon_curve.ddat silicate curve on a mass-radius diagram.

Column 0 of the .ddat is total mass in Earth masses, column 1 is total radius in Earth radii.

Run:
    python "plot/script plots/plot_hongyi_silicon_mr.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SILICATE_CURVE = Path(SILICON_CURVE)
LUO_CURVES = ROOT / "luo_dorn_2024_mr.csv"   # M-R model curves from Luo, Dorn & Deng 2024
OUT = ROOT / "output/plots" / "hongyi_silicon_mr.png"


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def load_luo():
    """Luo, Dorn & Deng 2024 water-world M-R curves (scenario D). Returns list of
    (water_mass_frac, mass, radius) sorted by mass, one entry per curve."""
    d = np.genfromtxt(LUO_CURVES, delimiter=",", names=True)
    out = []
    for wmf in np.unique(d["water_mass_frac"]):
        sel = d["water_mass_frac"] == wmf
        mm, rr = d["mass_earth"][sel], d["radius_earth"][sel]
        o = np.argsort(mm)
        out.append((float(wmf), mm[o], rr[o]))
    return out


def main():
    m, r = load_silicate()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(m, r, "k-", lw=1.5, zorder=3, label="silicate line")

    # Edmondson et al. 2023 volatile (neptunian) branch: R = 0.97 * M^0.55.
    # Draw only where it is dominant, i.e. above the silicate crossing (where the
    # volatile radius first exceeds the silicate radius at fixed mass).
    def r_vol(mm):
        return 0.97 * mm ** 0.55

    diff = r_vol(m) - r                      # volatile minus silicate on the data grid
    cross = np.where(np.diff(np.sign(diff)))[0]
    if cross.size:
        i = cross[0]                          # linear interp for the crossing mass
        m_cross = m[i] - diff[i] * (m[i + 1] - m[i]) / (diff[i + 1] - diff[i])
    else:
        m_cross = m.min()

    m_ed = np.linspace(m_cross, m.max(), 400)
    r_ed = r_vol(m_ed)
    ax.plot(m_ed, r_ed, "-", color="tab:red", lw=1.5, zorder=4,
            label=r"Edmondson et al. 2023 volatile ($R=0.97\,M^{0.55}$)")

    # Intermediate dashed line between the silicate (black) and Edmondson (red)
    # lines, weighted much closer to the silicate line.
    W = 0.10   # 0 = on the black line, 1 = on the red line
    r_black_on_grid = np.interp(m_ed, m, r)
    r_mid = r_black_on_grid + W * (r_ed - r_black_on_grid)
    ax.plot(m_ed, r_mid, "--", color="0.35", lw=1.5, zorder=5,
            label="intermediate")

    # Dashed grey line just below the silicate line (very close to it)
    r_below = r * (1.0 - 0.02)   # 2% below the silicate radius
    ax.plot(m, r_below, "--", color="0.35", lw=1.5, zorder=5, label="sub-silicate")

    # Dashed grey line just above the Edmondson (red) line
    r_above_red = r_ed * (1.0 + 0.02)   # 2% above the Edmondson radius
    ax.plot(m_ed, r_above_red, "--", color="0.35", lw=1.5, zorder=5, label="super-volatile")

    # Shaded bands (all evaluated on the m_ed grid so the edges line up)
    r_below_on_ed = r_black_on_grid * (1.0 - 0.02)   # the sub-silicate line on m_ed
    # pink: between the top dashed line (super-volatile) and the intermediate line
    ax.fill_between(m_ed, r_mid, r_above_red, color="pink", alpha=0.5, zorder=1)
    # grey: between the two lower dashed lines (intermediate and sub-silicate)
    ax.fill_between(m_ed, r_below_on_ed, r_mid, color="0.5", alpha=0.5, zorder=1)

    # ---- scattered points: Gaussian around each line, clipped to its zone ----
    # (only between the crossing mass and the max mass, i.e. where the dashed
    # cut-off lines exist; points never spill past the band edges.)
    def scatter_zone(mu_fn, lo_grid, hi_grid, sigma, color, n=250, seed=0):
        # One point per sampled mass across the whole range, so even the thin
        # high-mass part of the band gets filled (Gaussian in radius, truncated
        # to the band; uniform fallback where the band is too thin to hit).
        rng = np.random.default_rng(seed)
        x = rng.uniform(m_cross, m.max(), n)
        lo = np.maximum(np.interp(x, m_ed, lo_grid), 0.5)
        hi = np.minimum(np.interp(x, m_ed, hi_grid), 2.35)
        sig = sigma(x) if callable(sigma) else np.full(n, sigma)
        mu = mu_fn(x)
        y = rng.normal(mu, sig)
        for _ in range(30):
            bad = (y < lo) | (y > hi)
            if not bad.any():
                break
            y[bad] = rng.normal(mu[bad], sig[bad])
        bad = (y < lo) | (y > hi)
        y[bad] = rng.uniform(lo[bad], hi[bad])
        ax.scatter(x, y, s=6, color=color, alpha=0.7, lw=0, zorder=6)

    # pink zone: red points centered on the Edmondson (red) line, with a wide
    # sigma so the spread reaches down to the black line (band edges clip it).
    scatter_zone(r_vol, r_mid, r_above_red, sigma=0.30, color="tab:red", n=450, seed=1)
    # grey zone: black points centered on the silicate (black) line, with sigma
    # set to the wider spacing above (black -> intermediate line) so it fills.
    sig_black = lambda x: np.interp(x, m_ed, r_mid) - np.interp(x, m, r)
    scatter_zone(lambda x: np.interp(x, m, r), r_below_on_ed, r_mid,
                 sigma=sig_black, color="black", n=300, seed=2)

    ax.set_xlabel(r"planet mass [$M_\oplus$]")
    ax.set_ylabel(r"planet radius [$R_\oplus$]")
    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 2.35)
    ax.grid(True, ls="-", lw=0.6, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9, loc="lower right", fontsize=9)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

