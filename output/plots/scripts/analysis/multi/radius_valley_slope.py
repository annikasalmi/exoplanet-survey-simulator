"""
29_radius_valley_slope.py — does the rocky/sub-Neptune boundary (the radius valley) move with
insolation the way ATMOSPHERE STRIPPING predicts? A simple, vanilla test of whether the cold
red-window deficit of large rocky planets is PHYSICAL.

Method (deliberately plain — histograms + one straight-line fit):
  1. Take transiting planets with good radii (sigma_R/R < 10%), R in 1-4 R_earth.
  2. Slice them into insolation bins. In each slice, histogram the radius and find the VALLEY
     (the dip near ~1.8 R_earth that separates rocky super-Earths below from sub-Neptunes above).
  3. Fit a straight line:  log10(R_valley) = beta * log10(insolation) + const.  The SIGN of beta
     is the whole result.

Physics:
  * Thermally-driven mass loss (photoevaporation / core-powered): the valley sits at SMALLER
    radius at LOWER insolation -> beta_insol POSITIVE (~+0.07; equivalently d logR/d logP ~ -0.09,
    Van Eylen 2018). Then at low insolation the rocky/puffy boundary drops BELOW 1.7 R_earth, so
    large (R>1.7) rocky planets in the cold red window are PHYSICALLY expected to be rare (no
    stripping in the cold -> planets keep H/He -> puffy, not large-rocky).
  * Gas-poor / primordial formation: opposite sign (beta_insol NEGATIVE; Cloutier & Menou 2020
    found this for M dwarfs) -> the cold deficit would NOT be a stripping effect.

We also split FGK vs M dwarfs, because the M-dwarf valley slope is reported to flip sign.

Run:
    python scripts/29_radius_valley_slope.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.paths import LIFESIM_OUTER_DIR, SILICON_CURVE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "29_radius_valley_slope")

R_LO, R_HI = 1.0, 4.0          # radius range that brackets the valley
VALLEY_LO, VALLEY_HI = 1.5, 2.2  # search the dip here
PREC_CUT = 0.10                # keep radii measured to <10%
MDWARF_TEFF = 3900.0
N_BOOT = 300                   # simple resampling for the slope error bar
RED_WINDOW_R = 1.7
RED_WINDOW_I = 10.0

# literature reference slopes (insolation), converted from the period slopes
PHOTOEVAP_BETA_I = +0.07       # thermal mass loss (Van Eylen 2018: dlogR/dlogP=-0.09)
GASPOOR_BETA_I = -0.08         # gas-poor / primordial (dlogR/dlogP=+0.11)

# silicate (pure-rock) curve from the rest of the project -> the rocky/puffy boundary
SILICATE_CURVE = Path(SILICON_CURVE)
SIL_MASS_LO, SIL_MASS_HI = 3.0, 10.0   # super-Earth mass range for the rock-radius band


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def draw_silicate_band(ax, m_sil, r_sil):
    """Shade the silicate (pure-rock) max-radius for M = 3-10 M_earth. The empirical valley
    should land in this band if the radius gap really is the rock/volatile boundary."""
    lo = float(np.interp(SIL_MASS_LO, m_sil, r_sil))
    hi = float(np.interp(SIL_MASS_HI, m_sil, r_sil))
    ax.axhspan(lo, hi, color="tab:green", alpha=0.13, zorder=0)
    return lo, hi


def radius_valley(radii):
    """Plain histogram dip finder: smoothed minimum of the radius histogram in the valley window."""
    lr = np.log10(np.asarray(radii, float))
    edges = np.linspace(np.log10(R_LO), np.log10(R_HI), 15)
    cent = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    h, _ = np.histogram(lr, bins=edges)
    hs = np.convolve(h, np.ones(3) / 3.0, mode="same")     # light 3-bin smoothing
    win = (cent >= VALLEY_LO) & (cent <= VALLEY_HI)
    idx = np.flatnonzero(win)
    return cent[idx[np.argmin(hs[idx])]]


def fit_slope(insol, radii, n_bins):
    """Quantile insolation bins -> valley per bin -> straight-line fit. Returns slope, centers, valleys."""
    qs = np.quantile(insol, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    cents, valls = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        cell = (insol >= lo) & (insol < hi)
        if cell.sum() < 40:
            continue
        cents.append(np.median(insol[cell]))
        valls.append(radius_valley(radii[cell]))
    cents, valls = np.array(cents), np.array(valls)
    if len(cents) < 2:
        return np.nan, cents, valls
    beta = np.polyfit(np.log10(cents), np.log10(valls), 1)[0]
    return beta, cents, valls


def boot_slope(insol, radii, n_bins, rng):
    betas = []
    n = len(insol)
    for _ in range(N_BOOT):
        b = rng.integers(0, n, n)
        beta, _, _ = fit_slope(insol[b], radii[b], n_bins)
        if np.isfinite(beta):
            betas.append(beta)
    return (np.nan, np.nan) if not betas else (float(np.mean(betas)), float(np.std(betas)))


def verdict(beta):
    if not np.isfinite(beta):
        return "indeterminate"
    if beta > 0.02:
        return "POSITIVE -> thermal mass loss (cold red window physically rocky-poor)"
    if beta < -0.02:
        return "NEGATIVE -> gas-poor/primordial (cold deficit NOT a stripping effect)"
    return "~flat -> no clear insolation dependence"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    m_sil, r_sil = load_silicate()
    nd = pd.read_csv(NASA_FILE)
    r = pd.to_numeric(nd["pl_rade"], errors="coerce")
    ins = pd.to_numeric(nd["pl_insol"], errors="coerce")
    te = pd.to_numeric(nd["st_teff"], errors="coerce")
    re1 = pd.to_numeric(nd["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(nd["pl_radeerr2"], errors="coerce").abs()
    fr = 0.5 * (re1 + re2) / r
    keep = r.between(R_LO, R_HI) & ins.between(0.3, 3000) & te.notna() & (fr < PREC_CUT)
    r, ins, te = r[keep].to_numpy(), ins[keep].to_numpy(), te[keep].to_numpy()
    print(f"--> {len(r)} transiting planets with radius<{PREC_CUT:.0%} error, R in {R_LO}-{R_HI}")

    samples = {
        "All": np.ones(len(r), bool),
        "FGK (Teff>=3900)": te >= MDWARF_TEFF,
        "M dwarf (Teff<3900)": te < MDWARF_TEFF,
    }
    nbins = {"All": 4, "FGK (Teff>=3900)": 4, "M dwarf (Teff<3900)": 3}

    print(f"\n  reference slopes (insolation):  thermal mass loss beta~{PHOTOEVAP_BETA_I:+.2f}   "
          f"gas-poor beta~{GASPOOR_BETA_I:+.2f}   (Van Eylen 2018 period slope -0.09)")
    print(f"  {'sample':<22}{'N':>6}{'beta_insol':>16}{'beta_period(eq)':>16}   interpretation")
    print("  " + "-" * 100)
    res = {}
    for name, mask in samples.items():
        I, R = ins[mask], r[mask]
        bI, cI, vI = fit_slope(I, R, nbins[name])
        muI, sdI = boot_slope(I, R, nbins[name], rng)
        muP, sdP = -4.0 / 3.0 * muI, 4.0 / 3.0 * sdI   # period-equivalent (I ∝ P^-4/3)
        res[name] = dict(I=I, R=R, cI=cI, vI=vI, bI=bI, muI=muI, sdI=sdI, muP=muP, sdP=sdP, n=int(mask.sum()))
        print(f"  {name:<22}{mask.sum():>6}{f'{muI:+.3f}±{sdI:.3f}':>16}{f'{muP:+.3f}±{sdP:.3f}':>16}   {verdict(muI)}")

    print(f"\n  Red window = insolation<{RED_WINDOW_I:g}, radius>{RED_WINDOW_R:g}.")
    for name in ["FGK (Teff>=3900)", "M dwarf (Teff<3900)"]:
        d = res[name]
        # extrapolate the valley line down to the red window insolation
        if np.isfinite(d["bI"]) and len(d["cI"]) >= 2:
            inter = np.polyfit(np.log10(d["cI"]), np.log10(d["vI"]), 1)[1]
            rv_cold = 10 ** (d["bI"] * np.log10(RED_WINDOW_I) + inter)
            tag = "BELOW 1.7 -> large cold rocky physically rare (supports stripping)" if rv_cold < RED_WINDOW_R \
                else "ABOVE 1.7 -> large cold rocky NOT excluded by the valley"
            print(f"    {name:<22}: extrapolated valley at I={RED_WINDOW_I:g} is R={rv_cold:.2f}  -> {tag}")

    sb_lo = float(np.interp(SIL_MASS_LO, m_sil, r_sil)); sb_hi = float(np.interp(SIL_MASS_HI, m_sil, r_sil))
    print(f"\n  Silicate (pure-rock) limit: max rocky radius {sb_lo:.2f}-{sb_hi:.2f} R⊕ over M={SIL_MASS_LO:g}-{SIL_MASS_HI:g} M⊕")
    for name in samples:
        vv = res[name]["vI"]
        if len(vv):
            rv = float(np.median(vv)); mcross = float(np.interp(rv, r_sil, m_sil))
            inside = sb_lo <= rv <= sb_hi
            print(f"    {name:<22}: empirical valley R≈{rv:.2f} = silicate rock radius at M={mcross:.1f} M⊕"
                  f"  {'(sits ON the rock limit ✓)' if inside else '(off the band)'}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))

    # (a) all-sample scatter + valley + fit + model slopes + red window
    ax = axes[0, 0]
    d = res["All"]
    ax.scatter(d["I"], d["R"], s=7, c="0.6", alpha=0.5, lw=0)
    ax.plot(d["cI"], d["vI"], "o", color="k", ms=9, label="measured valley (per bin)")
    if np.isfinite(d["bI"]):
        inter = np.polyfit(np.log10(d["cI"]), np.log10(d["vI"]), 1)[1]
        xx = np.logspace(np.log10(0.3), np.log10(3000), 50)
        ax.plot(xx, 10 ** (d["bI"] * np.log10(xx) + inter), "k-", lw=2,
                label=f"fit β={d['muI']:+.3f}±{d['sdI']:.3f}")
        anchor = 10 ** (np.log10(d["vI"]).mean())
        ax.plot(xx, anchor * (xx / np.median(d["cI"])) ** PHOTOEVAP_BETA_I, "g--", lw=1.5,
                label=f"thermal mass loss β={PHOTOEVAP_BETA_I:+.2f}")
        ax.plot(xx, anchor * (xx / np.median(d["cI"])) ** GASPOOR_BETA_I, "r--", lw=1.5,
                label=f"gas-poor β={GASPOOR_BETA_I:+.2f}")
    sb_lo, sb_hi = draw_silicate_band(ax, m_sil, r_sil)
    ax.axvspan(0.3, RED_WINDOW_I, color="tab:blue", alpha=0.10)
    ax.axhline(RED_WINDOW_R, color="tab:blue", ls=":", lw=1)
    ax.text(0.4, 3.6, "cold\nred window", color="tab:blue", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"insolation flux [$I_\oplus$]"); ax.set_ylabel(r"planet radius [$R_\oplus$]")
    ax.set_yticks([1, 1.5, 2, 3, 4]); ax.set_yticklabels(["1", "1.5", "2", "3", "4"])
    ax.set_title("(a) Radius valley vs insolation — ALL stars\nsign of the slope = the result", fontsize=11)
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="tab:green", alpha=0.3))
    l.append(r"silicate rock limit (M=3–10 $M_\oplus$)")
    ax.legend(h, l, fontsize=8, loc="lower right")

    # (b) radius histograms per insolation bin (see the valley move)
    ax = axes[0, 1]
    I, R = res["All"]["I"], res["All"]["R"]
    qs = np.quantile(I, np.linspace(0, 1, 5))
    cmap = plt.cm.coolwarm(np.linspace(0, 1, 4))
    edges = np.linspace(np.log10(R_LO), np.log10(R_HI), 15)
    cent = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    for k in range(4):
        cell = (I >= qs[k]) & (I < qs[k + 1] + (1e-9 if k == 3 else 0))
        h, _ = np.histogram(np.log10(R[cell]), bins=edges, density=True)
        ax.step(cent, h, where="mid", color=cmap[k], lw=1.8,
                label=f"I={qs[k]:.0f}-{qs[k+1]:.0f} (N={cell.sum()})")
        vv = radius_valley(R[cell])
        ax.axvline(vv, color=cmap[k], ls=":", lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"planet radius [$R_\oplus$]"); ax.set_ylabel("normalized count")
    ax.set_xticks([1, 1.5, 2, 3, 4]); ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
    ax.set_title("(b) The valley itself, per insolation slice\n(dotted = valley; watch it shift)", fontsize=11)
    ax.legend(fontsize=8)

    # (c, d) FGK and M dwarf
    for ax, name in [(axes[1, 0], "FGK (Teff>=3900)"), (axes[1, 1], "M dwarf (Teff<3900)")]:
        d = res[name]
        ax.scatter(d["I"], d["R"], s=7, c="0.6", alpha=0.5, lw=0)
        ax.plot(d["cI"], d["vI"], "o", color="k", ms=9)
        if np.isfinite(d["bI"]) and len(d["cI"]) >= 2:
            inter = np.polyfit(np.log10(d["cI"]), np.log10(d["vI"]), 1)[1]
            xx = np.logspace(np.log10(0.3), np.log10(3000), 50)
            ax.plot(xx, 10 ** (d["bI"] * np.log10(xx) + inter), "k-", lw=2,
                    label=f"β={d['muI']:+.3f}±{d['sdI']:.3f}")
            ax.plot(xx, 10 ** (np.log10(d["vI"]).mean()) * (xx / np.median(d["cI"])) ** PHOTOEVAP_BETA_I, "g--", lw=1.3)
            ax.plot(xx, 10 ** (np.log10(d["vI"]).mean()) * (xx / np.median(d["cI"])) ** GASPOOR_BETA_I, "r--", lw=1.3)
        draw_silicate_band(ax, m_sil, r_sil)
        ax.axvspan(0.3, RED_WINDOW_I, color="tab:blue", alpha=0.10)
        ax.axhline(RED_WINDOW_R, color="tab:blue", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"insolation flux [$I_\oplus$]"); ax.set_ylabel(r"planet radius [$R_\oplus$]")
        ax.set_yticks([1, 1.5, 2, 3, 4]); ax.set_yticklabels(["1", "1.5", "2", "3", "4"])
        ax.set_title(f"({'c' if 'FGK' in name else 'd'}) {name}  (N={d['n']})  "
                     f"green band=silicate rock limit; lines: green=mass-loss, red=gas-poor", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Does the radius valley move with insolation? — a vanilla test of physical vs bias in the cold red window\n"
                 "β>0 (valley smaller at low insolation) ⇒ atmosphere stripping ⇒ cold deficit PHYSICAL.   "
                 "Green band = silicate pure-rock limit: the empirical valley sits on it ⇒ the gap IS the rock/volatile boundary.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(OUT_DIR, "radius_valley_slope.png")
    fig.savefig(out_png, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()
