"""
occurrence_correction.py — is the cold "red window" (low insolation, large
radius) empty of ROCKY planets because of PHYSICS, or because telescopes can't see them?

Method (inverse-detection-efficiency, the standard occurrence-rate correction):
  1. COMPLETENESS eta(I, R): inject rocky planets (mass set ~1.3x the silicate-boundary mass
     so they sit BELOW the curve = rocky) of each radius onto real P-Pop orbits/stars across
     all insolations, run transit + RV detection, and measure the detected fraction per cell.
  2. NASA rocky counts N(I, R): real measured-mass planets classified rocky vs the silicate
     curve, binned in the same (insolation, radius) grid.
  3. CORRECTED occurrence  ~  N / eta   (Poisson error sqrt(N)/eta).
     Where eta is tiny the correction amplifies noise enormously -> those cells are flagged
     UNCONSTRAINED (we simply cannot know). If, where eta is trustworthy, the cold large-rocky
     count is STILL low after correction, the deficit is physical (e.g. photoevaporation);
     if the correction fills it in, it was telescope bias.

Caveats (printed): eta uses a toy transit+RV model as a proxy for NASA's heterogeneous
selection, P-Pop's (M-dwarf-heavy) star+orbit mix, and one representative rocky mass per
radius. It shows the SHAPE of the bias, not an absolute planets-per-star rate.

Run:
    python scripts/occurrence_correction.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.paths import SILICON_CURVE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = Path(SILICON_CURVE)
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "occurrence_correction")

# ============================ KNOBS ============================
N_HOST = 40000           # P-Pop hosts used to measure completeness per radius
ROCKY_MASS_FACTOR = 1.3  # injected mass = factor x silicate-boundary mass (>1 => rocky)
ETA_MIN = 0.02           # below this completeness, a corrected cell is "unconstrained"
COLD_FLUX = 10.0         # red window: insolation below this
BIG_RADIUS = 1.4         # red window: radius above this
# ==============================================================

RNG_SEED = 0
RV_MAG_TARGET = 12.0
RHO_EARTH = 5.513

RADIUS_EDGES = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
RADIUS_CENT = 0.5 * (RADIUS_EDGES[:-1] + RADIUS_EDGES[1:])
FLUX_EDGES = np.logspace(np.log10(0.3), np.log10(3000), 10)
FLUX_CENT = np.sqrt(FLUX_EDGES[:-1] * FLUX_EDGES[1:])


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def rocky_mass(R, m_sil, r_sil):
    """Mass making a planet of radius R rocky (below the silicate curve)."""
    m_boundary = np.interp(R, r_sil, m_sil)   # mass where silicate radius == R
    return float(np.clip(m_boundary * ROCKY_MASS_FACTOR, 0.1, 12.0))


def load_hosts():
    df = pd.read_csv(PPOP_CATALOG, low_memory=False)
    df["flux_p"] = pd.to_numeric(df["flux_p"], errors="coerce")
    df = df[df["flux_p"].between(FLUX_EDGES[0], FLUX_EDGES[-1])].copy()
    if len(df) > N_HOST:
        df = df.sample(N_HOST, random_state=1)
    return df.reset_index(drop=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    hosts = load_hosts()
    print(f"--> {len(hosts)} P-Pop hosts spanning insolation {FLUX_EDGES[0]:g}-{FLUX_EDGES[-1]:g}")

    # ---------- (1) completeness eta(R, I) ----------
    nR, nF = len(RADIUS_CENT), len(FLUX_CENT)
    eta = np.full((nR, nF), np.nan)
    nhost_cell = np.zeros((nR, nF), int)
    F = hosts["flux_p"].to_numpy(float)
    fidx = np.digitize(F, FLUX_EDGES) - 1
    print("--> measuring completeness per radius (injecting rocky planets, transit+RV)...")
    for ri, Rc in enumerate(RADIUS_CENT):
        inj = hosts.copy()
        inj["radius_p"] = Rc
        inj["mass_p"] = rocky_mass(Rc, m_sil, r_sil)
        det = (run_kepler(inj)["detected"].to_numpy(bool)
               & run_rv_best(inj, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool))
        for fi in range(nF):
            cell = fidx == fi
            n = int(cell.sum())
            nhost_cell[ri, fi] = n
            if n >= 30:
                eta[ri, fi] = det[cell].mean()

    # ---------- (2) NASA rocky counts ----------
    nd = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(nd["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(nd["pl_rade"], errors="coerce")
    fl = pd.to_numeric(nd["pl_insol"], errors="coerce")
    prov = nd.get("pl_bmassprov", pd.Series("", index=nd.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keep = meas & r.between(RADIUS_EDGES[0], RADIUS_EDGES[-1]) & fl.between(FLUX_EDGES[0], FLUX_EDGES[-1]) & m.between(0.1, 12)
    m, r, fl = m[keep].to_numpy(), r[keep].to_numpy(), fl[keep].to_numpy()
    rocky = r < np.interp(m, m_sil, r_sil)
    mR, rR, flR = m[rocky], r[rocky], fl[rocky]
    Ncount, _, _ = np.histogram2d(rR, flR, bins=[RADIUS_EDGES, FLUX_EDGES])
    print(f"--> NASA: {keep.sum()} measured-mass planets in grid; {rocky.sum()} are rocky")

    # ---------- (3) corrected occurrence ----------
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(eta >= ETA_MIN, Ncount / eta, np.nan)
        corr_err = np.where(eta >= ETA_MIN, np.sqrt(Ncount) / eta, np.nan)
    unreliable = ~(eta >= ETA_MIN)

    # red-window summary
    big = RADIUS_CENT > BIG_RADIUS
    cold = FLUX_CENT < COLD_FLUX
    rw = np.outer(big, cold)
    n_rw = int(Ncount[rw].sum())
    eta_rw = np.nanmedian(eta[rw])
    print("\n  ===== RED WINDOW (insol<%.0f, radius>%.1f) =====" % (COLD_FLUX, BIG_RADIUS))
    print(f"  NASA rocky planets observed there : {n_rw}")
    print(f"  median completeness eta there     : {eta_rw:.3f}  ({'TRUSTWORTHY' if eta_rw>=ETA_MIN else 'BELOW ETA_MIN -> UNCONSTRAINED'})")
    if np.isfinite(eta_rw) and eta_rw > 0:
        print(f"  naive corrected count (N/eta)     : {n_rw/eta_rw:.0f} +/- {np.sqrt(max(n_rw,1))/eta_rw:.0f}")
    # hot large-rocky control
    hot = FLUX_CENT > 100
    rh = np.outer(big, hot)
    print(f"  [control] HOT large-rocky (insol>100): N={int(Ncount[rh].sum())}, median eta={np.nanmedian(eta[rh]):.3f}")

    # large-rocky corrected vs insolation (sum over big-radius bins)
    Nbig = Ncount[big, :].sum(0)
    eta_big = np.nanmean(np.where(big[:, None], eta, np.nan), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_big = np.where(eta_big >= ETA_MIN, Nbig / eta_big, np.nan)
        corr_big_err = np.where(eta_big >= ETA_MIN, np.sqrt(Nbig) / eta_big, np.nan)

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    X, Y = np.meshgrid(FLUX_EDGES, RADIUS_EDGES)

    def draw_redwindow(ax):
        ax.add_patch(plt.Rectangle((FLUX_EDGES[0], BIG_RADIUS), COLD_FLUX - FLUX_EDGES[0],
                                   RADIUS_EDGES[-1] - BIG_RADIUS, fill=False, ec="red", lw=2, ls="--"))
        ax.set_xscale("log"); ax.set_xlabel(r"insolation flux [$I_\oplus$]"); ax.set_ylabel(r"planet radius [$R_\oplus$]")

    # (a) completeness
    ax = axes[0, 0]
    pc = ax.pcolormesh(X, Y, eta, cmap="viridis", vmin=0, vmax=np.nanmax(eta))
    fig.colorbar(pc, ax=ax, label="transit+RV completeness η of rocky planets")
    draw_redwindow(ax)
    ax.set_title("(a) Completeness η(insolation, radius)\nred window = where we are nearly blind", fontsize=11)

    # (b) NASA rocky counts
    ax = axes[0, 1]
    pc = ax.pcolormesh(X, Y, np.where(Ncount > 0, Ncount, np.nan), cmap="magma")
    fig.colorbar(pc, ax=ax, label="NASA rocky planets detected (count)")
    ax.scatter(flR, rR, s=14, c="cyan", ec="k", lw=0.3, alpha=0.7, zorder=5)
    draw_redwindow(ax)
    ax.set_title("(b) Raw NASA rocky detections\n(points = individual planets)", fontsize=11)

    # (c) corrected occurrence
    ax = axes[1, 0]
    cc = np.where(corr > 0, corr, np.nan)
    pc = ax.pcolormesh(X, Y, cc, cmap="cividis",
                       norm=LogNorm(vmin=max(np.nanmin(cc), 1), vmax=np.nanmax(cc)) if np.isfinite(np.nanmax(cc)) else None)
    fig.colorbar(pc, ax=ax, label="completeness-corrected count  N/η")
    # hatch unconstrained cells
    for ri in range(nR):
        for fi in range(nF):
            if unreliable[ri, fi]:
                ax.add_patch(plt.Rectangle((FLUX_EDGES[fi], RADIUS_EDGES[ri]),
                                           FLUX_EDGES[fi + 1] - FLUX_EDGES[fi],
                                           RADIUS_EDGES[ri + 1] - RADIUS_EDGES[ri],
                                           hatch="xx", fill=False, ec="0.5", lw=0))
    draw_redwindow(ax)
    ax.set_title(f"(c) Corrected occurrence N/η  (hatched = η<{ETA_MIN:g}, UNCONSTRAINED)", fontsize=11)

    # (d) large-rocky vs insolation
    ax = axes[1, 1]
    ax.plot(FLUX_CENT, Nbig, "o-", color="gray", label="raw NASA count (large rocky)")
    ok = np.isfinite(corr_big)
    ax.errorbar(FLUX_CENT[ok], corr_big[ok], yerr=corr_big_err[ok], fmt="s-", color="tab:blue",
                capsize=3, label="completeness-corrected (large rocky)")
    bad = ~ok
    for fi in np.where(bad)[0]:
        ax.axvspan(FLUX_EDGES[fi], FLUX_EDGES[fi + 1], color="0.85", alpha=0.6)
    ax.axvline(COLD_FLUX, color="red", ls="--", lw=1.2)
    ax.text(COLD_FLUX * 0.9, ax.get_ylim()[1] if False else 1, " red window →", color="red",
            fontsize=9, ha="right", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"insolation flux [$I_\oplus$]")
    ax.set_ylabel("large-rocky count (R>%.1f)" % BIG_RADIUS)
    ax.set_title("(d) Large-rocky abundance vs insolation\ngrey band = η too low to correct (unknowable)", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)

    fig.suptitle("Is the cold 'red window' deficit of large rocky planets PHYSICAL or TELESCOPE BIAS?\n"
                 "completeness-corrected (inverse-detection-efficiency) occurrence of rocky planets",
                 fontsize=13)
    fig.text(0.5, 0.005,
             "If, where η is trustworthy (un-hatched), the corrected large-rocky count stays LOW toward low insolation → physical "
             "(atmospheres not stripped in the cold). If the correction inflates it, or the cold cells are hatched/grey (η→0) → "
             "the deficit is bias / unknowable with current transit+RV data. η is a model proxy for NASA's true selection.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_png = os.path.join(OUT_DIR, "completeness_corrected_occurrence.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()
