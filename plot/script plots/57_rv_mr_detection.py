"""
57_rv_mr_detection.py

RV (HARPS / HARPS-N / NIRPS) mass-measurement completeness, the RV sibling of the
Kepler/TESS transit detected-fraction maps (scripts 45/46/44/51).

Produces 2x4 detectability maps (rows = sin i treatment, cols = F G K M) on THREE
planes so you can read the RV floor against different axes:

    1. mass-radius        (the M-R plane; radius carries no RV info, mass does)
    2. insolation-radius  (matches the transit maps directly -> red-window comparison)
    3. insolation-mass    (RV-native: mass = signal, insolation ~ period -> K ∝ P^-1/3)

Color = P-Pop RV detection probability  P(K/σ_K ≥ threshold | RV-target star).
Overlay = real NASA planets with a *measured* mass, split by how the mass was obtained:
    RV (red circles)   TTV (blue triangles)   other/unclear (grey squares).
M-R-"Calculated" masses are excluded (they are derived from radius -> circular).

Instruments (--instrument):
    HARPS  optical V band (default; faint, jitter-limited on M dwarfs)
    NIRPS  near-IR J band  (M dwarfs bright + lower jitter -> rescues the M column)
    best   per-planet max(HARPS, NIRPS)  (you would pick the right tool per target)

Why a brightness cut on the P-Pop background: P-Pop is volume-limited (Gaia ~60 pc);
a real RV survey targets nearby bright stars, so the background is conditioned on a
band-magnitude cut (--mag-target), the RV analog of the transit "P(detect | transiting)".

Run from repo root:
    python scripts/57_rv_mr_detection.py
    python scripts/57_rv_mr_detection.py --instrument best --n-catalogs 100
    python scripts/57_rv_mr_detection.py --instrument NIRPS --mag-target 11
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
PPOP_DIR = ROOT / "run" / "tess" / "data" / "Gaia_C_F_K_combined_cdpp_v1"
PPOP_PATTERN = "tess_catalog_*.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = ROOT / "my_outputs" / "57_rv_mr_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("rv_data", ROOT / "lifesim" / "core" / "rv_data.py")
_rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rv)
RVData = _rv.RVData

# Box and grids
R_MAX, M_MAX = 2.2, 12.0
MASS_EDGES = np.array([0.3, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
RADIUS_EDGES = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2])
INSOL_EDGES = np.logspace(np.log10(0.2), np.log10(1e4), 9)
MIN_CELL_N = 8
STYPES = ["F", "G", "K", "M"]

# (key, xvar, yvar, xedges, yedges, xlabel, ylabel, xlog)
PLANES = [
    ("mr", "mass_p", "radius_p", MASS_EDGES, RADIUS_EDGES,
     r"Planet mass [M$_\oplus$]", r"Planet radius [R$_\oplus$]", True),
    ("ir", "flux_p", "radius_p", INSOL_EDGES, RADIUS_EDGES,
     r"Insolation flux [$I_\oplus$]", r"Planet radius [R$_\oplus$]", True),
    ("im", "flux_p", "mass_p", INSOL_EDGES, MASS_EDGES,
     r"Insolation flux [$I_\oplus$]", r"Planet mass [M$_\oplus$]", True),
]

PPOP_COLS = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p",
             "radius_s", "mass_s", "teff_s", "l_sun", "distance_s", "flux_p", "stype"]

# Heuristic: authors whose small-planet masses come from transit-timing variations (TTV),
# not RV. Used only to colour the overlay so RV-empty cells with TTV points read correctly.
TTV_AUTHORS = ["hadden", "agol", "jontof", "leleu", "ofir", "lithwick",
               "mills", "holczer", "nesvorny", "sun ", "gillon"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
})


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_ppop(n_catalogs: int) -> pd.DataFrame:
    files = sorted(PPOP_DIR.glob(PPOP_PATTERN))[:n_catalogs]
    if not files:
        raise FileNotFoundError(f"No {PPOP_PATTERN} in {PPOP_DIR}")
    frames = []
    for p in files:
        try:
            frames.append(pd.read_csv(p, usecols=lambda c: c in PPOP_COLS, low_memory=False))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    for c in ["radius_p", "mass_p", "p_orb", "flux_p"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  Loaded {len(df):,} P-Pop planets from {len(files)} catalogs")
    return df


def run_one_instrument(ppop, instrument, apply_sini, snr_threshold, n_obs, mag_target):
    rv = RVData(ppop, source="ppop", instrument=instrument, apply_sini=apply_sini,
                snr_threshold=snr_threshold, n_obs=n_obs,
                detection_model="threshold", validate_for_detection=False)
    cat = rv.determine_detectable()
    cat["rv_mag"] = pd.to_numeric(cat["rv_mag"], errors="coerce")
    cat["is_rv_target"] = cat["rv_mag"] <= mag_target
    return cat


def run_rv(ppop, instrument, apply_sini, snr_threshold, n_obs, mag_target):
    """Single instrument, or per-planet best of HARPS+NIRPS."""
    if instrument.lower() != "best":
        return run_one_instrument(ppop, instrument, apply_sini, snr_threshold, n_obs, mag_target)
    h = run_one_instrument(ppop, "HARPS", apply_sini, snr_threshold, n_obs, mag_target)
    n = run_one_instrument(ppop, "NIRPS", apply_sini, snr_threshold, n_obs, mag_target)
    best = h.copy()
    best["rv_p_detect"] = np.maximum(pd.to_numeric(h["rv_p_detect"], errors="coerce"),
                                     pd.to_numeric(n["rv_p_detect"], errors="coerce"))
    # an RV target if bright enough in EITHER band; p_detect already zeroed where not bright
    best["is_rv_target"] = h["is_rv_target"].astype(bool) | n["is_rv_target"].astype(bool)
    return best


def load_nasa() -> pd.DataFrame:
    import re
    raw = pd.read_csv(NASA_FILE)
    ren = {"pl_rade": "radius_p", "pl_radeerr1": "rerr1", "pl_radeerr2": "rerr2",
           "pl_bmasse": "mass_p", "pl_bmasseerr1": "merr1", "pl_bmasseerr2": "merr2",
           "pl_bmassprov": "prov", "pl_bmasse_reflink": "reflink",
           "pl_insol": "flux_p", "st_teff": "teff_s", "pl_name": "name"}
    df = raw.rename(columns={k: v for k, v in ren.items() if k in raw.columns})
    for c in ["radius_p", "mass_p", "rerr1", "rerr2", "merr1", "merr2", "teff_s", "flux_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # measured masses only (drop M-R "Calculated")
    df = df[df["prov"].isin(["Mass", "Msini", "Msin(i)/sin(i)"])].copy()
    df = df.dropna(subset=["radius_p", "mass_p"])
    df = df[(df["radius_p"] > 0) & (df["mass_p"] > 0)
            & (df["radius_p"] <= R_MAX) & (df["mass_p"] <= M_MAX)].copy()
    df["stype"] = df["teff_s"].apply(_stype)

    def author(s):
        s = str(s); m = re.search(r">([^<]+)</a>", s); return (m.group(1) if m else s).lower()
    auth = df.get("reflink", pd.Series("", index=df.index)).apply(author)
    is_ttv = auth.str.contains("|".join(TTV_AUTHORS), na=False)
    df["mass_method"] = np.where(is_ttv, "TTV", "RV")
    print(f"  Loaded {len(df):,} NASA measured-mass planets in box "
          f"(RV={int((df.mass_method=='RV').sum())}, TTV={int((df.mass_method=='TTV').sum())})")
    return df


def _stype(teff) -> str:
    if pd.isna(teff): return "Unknown"
    teff = float(teff)
    if teff >= 7500: return "A"
    if teff >= 6000: return "F"
    if teff >= 5200: return "G"
    if teff >= 3700: return "K"
    if teff > 0: return "M"
    return "Unknown"


# --------------------------------------------------------------------------- #
# Grid + plot
# --------------------------------------------------------------------------- #

def detection_grid(cat, xvar, yvar, xedges, yedges):
    c = cat[cat["is_rv_target"]].copy()
    x = pd.to_numeric(c[xvar], errors="coerce").to_numpy()
    y = pd.to_numeric(c[yvar], errors="coerce").to_numpy()
    p = pd.to_numeric(c["rv_p_detect"], errors="coerce").to_numpy(float)
    nr, nc = len(yedges) - 1, len(xedges) - 1
    frac = np.full((nr, nc), np.nan); count = np.zeros((nr, nc))
    for i in range(nr):
        for j in range(nc):
            cell = ((y >= yedges[i]) & (y < yedges[i + 1])
                    & (x >= xedges[j]) & (x < xedges[j + 1]) & np.isfinite(p))
            count[i, j] = cell.sum()
            if cell.sum() >= MIN_CELL_N:
                frac[i, j] = p[cell].mean()
    return frac, count


def _panel(ax, frac, count, xedges, yedges, show_cell_n):
    cmap = plt.cm.viridis.copy(); cmap.set_bad("0.88")
    mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(frac),
                         cmap=cmap, vmin=0, vmax=1, shading="flat")
    if show_cell_n:
        for i in range(frac.shape[0]):
            for j in range(frac.shape[1]):
                if np.isnan(frac[i, j]):
                    continue
                xc = np.sqrt(xedges[j] * xedges[j + 1])
                yc = 0.5 * (yedges[i] + yedges[i + 1])
                ax.text(xc, yc, f"{frac[i,j]:.0%}", ha="center", va="center",
                        fontsize=6, color="black" if frac[i, j] > 0.55 else "white")
    return mesh


_NASA_STYLE = {"RV": dict(marker="o", color="#d62728"),
               "TTV": dict(marker="^", color="#1f77b4"),
               "other": dict(marker="s", color="0.4")}


def make_figure(plane, grids, frac_summary, nasa, args):
    key, xvar, yvar, xedges, yedges, xlab, ylab, xlog = plane
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, sharey=True)
    row_keys = [False, True]; row_lab = ["true mass (no sin i)", "M sin i (realistic)"]
    mesh = None
    for ri, sini in enumerate(row_keys):
        for ci, st in enumerate(STYPES):
            ax = axes[ri, ci]
            frac, count = grids[(sini, st)]
            mesh = _panel(ax, frac, count, xedges, yedges, show_cell_n=(key == "mr"))
            sub = nasa[nasa["stype"] == st]
            for method, style in _NASA_STYLE.items():
                ss = sub[sub["mass_method"] == method]
                if len(ss):
                    ax.scatter(ss[xvar], ss[yvar], s=20, edgecolors="white", linewidths=0.4,
                               zorder=5, alpha=0.85, label=f"{method} (N={len(ss)})", **style)
            if ri == 0 and ci == 3:
                ax.legend(loc="lower right", fontsize=6.5)
            if xlog:
                ax.set_xscale("log")
            ax.set_xlim(xedges[0], xedges[-1]); ax.set_ylim(yedges[0], yedges[-1])
            dfrac = frac_summary[(sini, st)]
            ax.set_title((f"{st} stars\n" if ri == 0 else "") + f"box detect = {dfrac:.1%}")
            if ci == 0:
                ax.set_ylabel(f"{row_lab[ri]}\n{ylab}")
            if ri == 1:
                ax.set_xlabel(xlab)
            ax.grid(alpha=0.15)
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("RV detection probability  P(K/σ_K ≥ threshold | RV-target star)")
    fig.suptitle(
        f"{args.instrument} RV mass-measurement completeness — {xlab} vs {ylab}\n"
        f"Top: true mass · Bottom: M sin i  |  K/σ_K (σ_K=√(2/N)σ_RV) ≥ {args.snr_threshold}, "
        f"N={args.n_obs}, band-mag≤{args.mag_target}  |  overlay: RV=red○ TTV=blue△",
        fontsize=11, y=1.0)
    out = OUT_DIR / f"rv_detection_{key}_{args.instrument.lower()}_2x4.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-catalogs", type=int, default=60)
    ap.add_argument("--instrument", default="HARPS", help="HARPS | NIRPS | best")
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--mag-target", type=float, default=12.0,
                    help="RV-target band-magnitude cut on the P-Pop background.")
    args = ap.parse_args()

    print(f"Output dir: {OUT_DIR}  |  instrument: {args.instrument}")
    ppop = load_ppop(args.n_catalogs)
    nasa = load_nasa()

    cats = {sini: run_rv(ppop, args.instrument, sini, args.snr_threshold, args.n_obs, args.mag_target)
            for sini in [False, True]}

    summary_rows = []
    for plane in PLANES:
        key, xvar, yvar, xedges, yedges, *_ = plane
        grids, frac_summary = {}, {}
        for sini in [False, True]:
            cat = cats[sini]
            for st in STYPES:
                sub = cat[cat["stype"] == st]
                grids[(sini, st)] = detection_grid(sub, xvar, yvar, xedges, yedges)
                box = sub[sub["is_rv_target"]
                          & (pd.to_numeric(sub["radius_p"], errors="coerce") <= R_MAX)
                          & (pd.to_numeric(sub["mass_p"], errors="coerce") <= M_MAX)]
                frac_summary[(sini, st)] = float(box["rv_p_detect"].mean()) if len(box) else float("nan")
        make_figure(plane, grids, frac_summary, nasa, args)
        if key == "mr":
            for st in STYPES:
                summary_rows.append((st, frac_summary[(False, st)], frac_summary[(True, st)]))

    write_summary(summary_rows, nasa, args)
    print("\nDone.")


def write_summary(rows, nasa, args):
    lines = [
        "=" * 70,
        f"RV MASS-MEASUREMENT COMPLETENESS  (instrument={args.instrument})",
        "=" * 70,
        f"K/σ_K (σ_K=√(2/N)σ_RV) ≥ {args.snr_threshold} | N_obs={args.n_obs} | band-mag cut ≤ {args.mag_target}",
        f"Box: R ≤ {R_MAX} Re, M ≤ {M_MAX} Me   (M-R-plane box-averaged P_detect)",
        "",
        f"{'stype':<8}{'true mass':>14}{'M sin i':>14}",
    ]
    for st, a, b in rows:
        lines.append(f"{st:<8}{a:>13.1%}{b:>13.1%}")
    lines += ["", "Real NASA measured-mass planets in box (by stype, by method):"]
    for st in STYPES:
        s = nasa[nasa["stype"] == st]
        lines.append(f"  {st}: total {len(s):>3}  | RV {int((s.mass_method=='RV').sum()):>3}  "
                     f"TTV {int((s.mass_method=='TTV').sum()):>3}")
    lines += [
        "",
        "Planes written: mass-radius, insolation-radius, insolation-mass (each 2x4, sin i rows).",
        "RV-empty low-mass cells with only TTV overlay points = masses from timing, not RV;",
        "those are the planets RV cannot confirm -> the transit-good/NASA-absent regions.",
        "=" * 70,
    ]
    text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
