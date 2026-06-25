"""
37_flat_detfrac_vs_radius_and_mass.py — the plot that makes the detector physics
unambiguous, on the fully-FLAT synthetic catalogue.

The M-R scatter (script 34) is misleading: it shows the DETECTED subset, whose
density is parent_sampling x detection_probability. Because mass is drawn
log-uniform (a dense low-mass stripe on a linear axis) and transit detection is
high & gentle, the transit scatter just traces the input sampling and *looks*
like "detection depends on mass, not radius" — the opposite of the truth.

This script instead plots the DETECTION FRACTION (the real efficiency) against
radius and against mass, one curve per detector, each among that instrument's
observable set:
    Kepler / TESS : among geometrically transiting planets
    RV (best H+N) : among RV-target hosts (band-mag <= 12 in either band)

Expected, textbook result:
    Transit  -> detection RISES with radius (depth ~ R^2), FLAT in mass
    RV       -> detection RISES with mass   (K ~ M_p),      FLAT in radius

Run:
    python scripts/37_flat_detfrac_vs_radius_and_mass.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run.ppop.uniform_generator import get_or_build_catalog, UNIFORM_OUT_DIR
from run.ppop.flat_detect import run_kepler, run_tess, run_rv_best

N_PLANETS = 300000
RV_MAG_TARGET = 12.0
CACHE_CSV = os.path.join(UNIFORM_OUT_DIR, "flat_catalog.csv")
OUT_DIR = os.path.join(ROOT, "my_outputs", "37_flat_detfrac_vs_radius_and_mass")

RADIUS_BINS = np.linspace(0.5, 2.2, 13)
MASS_BINS = np.logspace(np.log10(0.1), np.log10(12.0), 13)
MIN_BIN_COUNT = 30

DETECTOR_STYLE = {
    "Kepler": dict(color="tab:green", marker="o"),
    "TESS": dict(color="tab:blue", marker="s"),
    "RV": dict(color="tab:red", marker="^"),
}


def _as_bool(s) -> np.ndarray:
    return pd.Series(s).fillna(False).astype(bool).to_numpy()


def observable_mask(df: pd.DataFrame, key: str) -> np.ndarray:
    if key == "Kepler":
        return _as_bool(df.get("transiting_geometric", True))
    if key == "TESS":
        trans = _as_bool(df.get("tess_transiting_geometric", True))
        obs = _as_bool(df["tess_observed"]) if "tess_observed" in df.columns else np.ones(len(df), bool)
        return trans & obs
    if key == "RV":
        return _as_bool(df.get("rv_is_target", True))
    return np.ones(len(df), bool)


def fraction_vs(df: pd.DataFrame, denom: np.ndarray, var: str, edges: np.ndarray):
    """Detection fraction (+ binomial error) of the observable set, binned by `var`."""
    x = pd.to_numeric(df[var], errors="coerce").to_numpy()
    det = df["detected"].astype(bool).to_numpy()
    obs = denom & np.isfinite(x)
    centers, frac, err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cell = obs & (x >= lo) & (x < hi)
        n = int(cell.sum())
        c = np.sqrt(lo * hi) if edges is MASS_BINS else 0.5 * (lo + hi)
        if n < MIN_BIN_COUNT:
            centers.append(c); frac.append(np.nan); err.append(np.nan)
            continue
        p = det[cell].mean()
        centers.append(c); frac.append(p); err.append(np.sqrt(max(p * (1 - p), 0) / n))
    return np.array(centers), np.array(frac), np.array(err)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog = get_or_build_catalog(CACHE_CSV, rebuild=False, n_planets=N_PLANETS, seed=0)
    print(f"--> Running detectors on {len(catalog)} flat planets...")
    results = {
        "Kepler": run_kepler(catalog),
        "TESS": run_tess(catalog),
        "RV": run_rv_best(catalog, mag_target=RV_MAG_TARGET),
    }
    labels = {"Kepler": "Kepler", "TESS": "TESS",
              "RV": f"RV best(HARPS,NIRPS), mag≤{RV_MAG_TARGET:g}"}

    fig, (axr, axm) = plt.subplots(1, 2, figsize=(15, 6))

    for key in ["Kepler", "TESS", "RV"]:
        df = results[key]
        denom = observable_mask(df, key)
        st = DETECTOR_STYLE[key]

        xr, fr, er = fraction_vs(df, denom, "radius_p", RADIUS_BINS)
        axr.errorbar(xr, 100 * fr, yerr=100 * er, label=labels[key], capsize=2,
                     lw=2, **st)

        xm, fm, em = fraction_vs(df, denom, "mass_p", MASS_BINS)
        axm.errorbar(xm, 100 * fm, yerr=100 * em, label=labels[key], capsize=2,
                     lw=2, **st)

    axr.set_xlabel(r"Planet radius [$R_\oplus$]")
    axr.set_ylabel("Detection fraction of observable planets [%]")
    axr.set_title("vs RADIUS\n(transit depth $\\propto R^2$ -> transit rises; RV flat)")
    axr.set_xlim(0.5, 2.2)
    axr.set_ylim(-2, 102)
    axr.grid(alpha=0.25)
    axr.legend(loc="center right", fontsize=9)

    axm.set_xscale("log")
    axm.set_xlabel(r"Planet mass [$M_\oplus$]")
    axm.set_ylabel("Detection fraction of observable planets [%]")
    axm.set_title("vs MASS\n(RV semi-amplitude $K \\propto M_p$ -> RV rises; transit flat)")
    axm.set_xlim(0.1, 12)
    axm.set_ylim(-2, 102)
    axm.grid(alpha=0.25, which="both")
    axm.legend(loc="center right", fontsize=9)

    fig.suptitle("Detection efficiency vs radius and vs mass — FLAT control population\n"
                 "Transit detectors care about RADIUS (not mass); RV cares about MASS (not radius)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "flat_detfrac_vs_radius_and_mass.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


if __name__ == "__main__":
    main()
