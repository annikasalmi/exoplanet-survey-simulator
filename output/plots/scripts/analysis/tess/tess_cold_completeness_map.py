"""51_tess_cold_completeness_map.py — guidance product: where can TESS still find cold planets?

Question answered: around M dwarfs, how complete is the TESS detection pipeline as a function of
planet radius and insolation, and WHY does it fall off in the cold corner? The actionable answer
for future astronomers is the number of transits: cold (long-period) planets fall out of TESS's
baseline, not out of its photometric sensitivity.

Completeness is the SAME definition script 12 uses:
    eta(R, I) = N_detected / N_(transiting & observed)
computed on the vendored TESS detection model (run/tess/data/Gaia_C_F_K_combined_cdpp_v1, the
CDPP/SNR TESS pipeline model), restricted to M-dwarf hosts, on a grid refined around the cold
corner. The four known cold (I<10) census planets are overlaid as reference points. The geometric
transit probability is NOT corrected (as in script 12) — this isolates the pipeline lever the
observer can actually influence (baseline, target brightness, number of transits).

This script is THIN: it reads the TESS model catalog directly (few columns) with the same boolean
parsing as script 12; it does not reimplement any detection physics.

Outputs (output/plots/51_tess_cold_completeness_map/):
    tess_cold_completeness.png, completeness_grid.csv

Run (repo root, PYTHONPATH set):
    python "scripts/statistical_analysis/51_tess_cold_completeness_map.py" [--n-catalogs 400]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PPOP_DATA_DIR = ROOT / "run" / "tess" / "data" / "Gaia_C_F_K_combined_cdpp_v1"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = ROOT / "output/plots" / "51_tess_cold_completeness_map"

# grid refined around the cold corner (radius floor 1.35, cold cut I<10 / <50)
R_EDGES = np.array([0.5, 1.0, 1.35, 1.7, 2.2, 3.0, 4.0])
I_EDGES = np.array([0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
MIN_CELL_N = 8

COLD_PLANETS = ["LHS 1140 b", "TOI-1452 b", "LHS 1903 e", "TOI-198 b"]
COLD_INSOL = 10.0
CORNER_RADIUS = 1.35

USE_COLS = {"radius_p", "flux_p", "stype", "tess_transiting_geometric",
            "tess_observed", "tess_detected", "tess_n_transits"}


def _tobool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def load_mdwarf_catalog(n_catalogs):
    files = sorted(glob.glob(str(PPOP_DATA_DIR / "tess_catalog_*.csv")))[:n_catalogs]
    if not files:
        raise FileNotFoundError(f"No tess_catalog_*.csv in {PPOP_DATA_DIR}")
    frames = [pd.read_csv(p, usecols=lambda c: c in USE_COLS) for p in files]
    df = pd.concat(frames, ignore_index=True)
    for c in ["tess_transiting_geometric", "tess_observed", "tess_detected"]:
        df[c] = _tobool(df[c])
    for c in ["radius_p", "flux_p", "tess_n_transits"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    m = df["stype"].astype(str).str.strip().str.upper().str.startswith("M")
    df = df[m].dropna(subset=["radius_p", "flux_p"]).reset_index(drop=True)
    print(f"  Loaded {len(files)} TESS catalogs; M-dwarf planets: {len(df):,}")
    return df


def completeness_grid(df):
    base = df[df["tess_transiting_geometric"] & df["tess_observed"]]
    r, f = base["radius_p"].to_numpy(), base["flux_p"].to_numpy()
    det = base["tess_detected"].to_numpy(bool)
    ntr = base["tess_n_transits"].to_numpy()
    nr, ni = len(R_EDGES) - 1, len(I_EDGES) - 1
    eta = np.full((nr, ni), np.nan)
    ncell = np.zeros((nr, ni), int)
    med_tr = np.full((nr, ni), np.nan)
    for i in range(nr):
        for j in range(ni):
            cell = ((r >= R_EDGES[i]) & (r < R_EDGES[i + 1]) &
                    (f >= I_EDGES[j]) & (f < I_EDGES[j + 1]))
            n = int(cell.sum())
            ncell[i, j] = n
            if n >= MIN_CELL_N:
                eta[i, j] = det[cell].mean()
                med_tr[i, j] = np.nanmedian(ntr[cell])
    return eta, ncell, med_tr


def load_overlay():
    df = pd.read_csv(NASA_FILE, comment="#")
    sub = df[df["pl_name"].isin(COLD_PLANETS)][["pl_name", "pl_rade", "pl_insol"]].copy()
    sub["pl_rade"] = pd.to_numeric(sub["pl_rade"], errors="coerce")
    sub["pl_insol"] = pd.to_numeric(sub["pl_insol"], errors="coerce")
    sub["is_toi"] = sub["pl_name"].str.startswith("TOI")
    return sub


def _draw_grid(ax, grid, cmap, norm, fmt, fmt_color):
    ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm,
              extent=[0, len(I_EDGES) - 1, 0, len(R_EDGES) - 1])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j + 0.5, i + 0.5, fmt(grid[i, j]), ha="center", va="center",
                        fontsize=8, color=fmt_color(grid[i, j]))
    ax.set_xticks(np.arange(len(I_EDGES)))
    ax.set_xticklabels([f"{v:g}" for v in I_EDGES], fontsize=8)
    ax.set_yticks(np.arange(len(R_EDGES)))
    ax.set_yticklabels([f"{v:g}" for v in R_EDGES], fontsize=8)
    ax.set_xlabel(r"Insolation  $I$  [$I_\oplus$]")
    ax.set_ylabel(r"Radius  [$R_\oplus$]")
    # cold corner outline: I<10 and R>=1.35 (edges land exactly on grid lines)
    j_cold = int(np.searchsorted(I_EDGES, COLD_INSOL))       # x-position of I=10
    i_floor = int(np.searchsorted(R_EDGES, CORNER_RADIUS))   # y-position of R=1.35
    ax.add_patch(Rectangle((0, i_floor), j_cold, (len(R_EDGES) - 1) - i_floor,
                           fill=False, edgecolor="crimson", lw=2.2, zorder=5))
    ax.text(j_cold / 2, len(R_EDGES) - 1.28, "cold corner\n(I<10, R>=1.35)",
            color="crimson", ha="center", va="center", fontsize=8.5, zorder=6)


# manual label offsets (points) to avoid collisions where planets sit close together
LABEL_OFFSET = {"LHS 1140 b": (7, 7), "TOI-1452 b": (7, -14),
                "LHS 1903 e": (7, 9), "TOI-198 b": (7, 7)}


def _overlay_planets(ax, overlay, label=True):
    def cell_xy(r, f):
        x = np.interp(np.log10(f), np.log10(I_EDGES), np.arange(len(I_EDGES)))
        y = np.interp(r, R_EDGES, np.arange(len(R_EDGES)))
        return x, y
    for _, p in overlay.iterrows():
        x, y = cell_xy(p["pl_rade"], p["pl_insol"])
        mk = "^" if p["is_toi"] else "o"
        ax.scatter(x, y, marker=mk, s=90, facecolor="white", edgecolor="k",
                   lw=1.4, zorder=7)
        if label:
            ax.annotate(p["pl_name"], (x, y), textcoords="offset points",
                        xytext=LABEL_OFFSET.get(p["pl_name"], (6, 6)),
                        fontsize=7.5, color="k", zorder=8)


def make_figure(eta, med_tr, ncell, overlay):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.6))

    # Panel A: completeness eta (sequential single hue)
    cmapA = plt.get_cmap("viridis")
    normA = BoundaryNorm(np.linspace(0, 1, 11), cmapA.N)
    _draw_grid(axA, eta, cmapA, normA, lambda v: f"{v:.2f}",
               lambda v: "white" if v < 0.6 else "black")
    _overlay_planets(axA, overlay)
    sm = plt.cm.ScalarMappable(cmap=cmapA, norm=normA)
    fig.colorbar(sm, ax=axA, label=r"TESS detection completeness  $\eta = N_{\rm det}/N_{\rm trans}$")
    axA.set_title("A. TESS pipeline completeness around M dwarfs\n"
                  "collapses in the cold corner (not from small radius)")

    # Panel B: median number of transits (the reason) — log-ish sequential
    cmapB = plt.get_cmap("magma")
    finite = med_tr[np.isfinite(med_tr)]
    vmax = np.nanpercentile(finite, 95) if finite.size else 100
    normB = BoundaryNorm(np.linspace(0, max(vmax, 10), 11), cmapB.N)
    _draw_grid(axB, med_tr, cmapB, normB, lambda v: f"{v:.0f}",
               lambda v: "white" if v < 0.5 * max(vmax, 10) else "black")
    _overlay_planets(axB, overlay, label=False)
    smB = plt.cm.ScalarMappable(cmap=cmapB, norm=normB)
    fig.colorbar(smB, ax=axB, label="median number of TESS transits per planet")
    axB.set_title("B. The lever: number of transits in TESS's baseline\n"
                  "cold planets transit too rarely -> low SNR -> not detected")

    fig.suptitle("Where can TESS still find cold super-Earths? Completeness is limited by "
                 "transit count, so the fix is baseline (CVZ / re-observation), not aperture",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tess_cold_completeness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-catalogs", type=int, default=400)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_mdwarf_catalog(args.n_catalogs)
    eta, ncell, med_tr = completeness_grid(df)
    overlay = load_overlay()

    print("\n  Completeness eta(R,I) — rows = radius bins (low->high), cols = insolation bins:")
    for i in range(eta.shape[0] - 1, -1, -1):
        row = "  ".join("  nan" if not np.isfinite(v) else f"{v:5.2f}" for v in eta[i])
        print(f"    R[{R_EDGES[i]:.2f}-{R_EDGES[i+1]:.2f}]  {row}")
    # summary: cold corner (I<10, R>=1.35) vs hot (I>50)
    def _agg(imin, imax):
        base = df[df.tess_transiting_geometric & df.tess_observed]
        c = base[(base.flux_p >= imin) & (base.flux_p < imax) &
                 (base.radius_p >= CORNER_RADIUS) & (base.radius_p < 2.2)]
        return len(c), (c.tess_detected.mean() if len(c) else np.nan), \
            (np.nanmedian(c.tess_n_transits) if len(c) else np.nan)
    for lab, lo, hi in [("cold corner I<10", 0.3, 10.0), ("mid 10<=I<50", 10.0, 50.0),
                        ("hot I>50", 50.0, 1e5)]:
        n, e, tr = _agg(lo, hi)
        print(f"  {lab:<18} N={n:>5}  eta={e:.3f}  median transits={tr:.0f}")

    rows = []
    for i in range(len(R_EDGES) - 1):
        for j in range(len(I_EDGES) - 1):
            rows.append(dict(r_lo=R_EDGES[i], r_hi=R_EDGES[i + 1],
                             i_lo=I_EDGES[j], i_hi=I_EDGES[j + 1],
                             eta=eta[i, j], n_cell=ncell[i, j], median_transits=med_tr[i, j]))
    pd.DataFrame(rows).to_csv(OUT_DIR / "completeness_grid.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'completeness_grid.csv'}")
    make_figure(eta, med_tr, ncell, overlay)
    print("\n--> done.")


if __name__ == "__main__":
    main()
