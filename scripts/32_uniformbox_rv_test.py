"""
32_uniformbox_rv_test.py — push the UNIFORM-BOX control planets through RVData.

Same flat radius/period control population as the Kepler / TESS tests (shared
cache), run through the radial-velocity toy detector to map which planet MASSES
an RV campaign would and would not recover. (Planet mass comes from the
Chen2017 mass-from-radius relation that P-Pop applies to the uniform radii.)

INSTRUMENT defaults to HARPS (optical V band). Switch to "NIRPS" (NIR J band)
to see how a near-infrared spectrograph does much better on faint M dwarfs.

Run:
    python scripts/32_uniformbox_rv_test.py
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
from run.ppop.flat_detect import run_rv

N_PLANETS = 300000
INSTRUMENT = "HARPS"    # or "NIRPS" for the NIR M-dwarf-friendly spectrograph
CACHE_CSV = os.path.join(UNIFORM_OUT_DIR, "flat_catalog.csv")


def radius_bin(df: pd.DataFrame) -> pd.Series:
    return pd.cut(
        pd.to_numeric(df["radius_p"], errors="coerce"),
        bins=[0, 1.5, 3.0, 6.0, np.inf],
        labels=["<1.5", "1.5-3.0", "3.0-6.0", ">6.0"],
        include_lowest=True,
    )


def caveman_summary(df: pd.DataFrame, instrument: str) -> None:
    n = len(df)
    det = df["detected"].astype(bool)
    print(f"\n=============== RV ({instrument}) on UNIFORM-BOX planets ===============")
    print(f"Total fake planets thrown at RV     : {n}")
    print(f"RV SEES (mass recovered)            : {det.sum()}  ({100*det.mean():.1f}%)")
    print(f"RV MISSES                            : {(~det).sum()}  ({100*(~det).mean():.1f}%)")

    print("\nMass-recovered fraction by host-star type:")
    by_stype = df.groupby(df["stype"].astype(str).str[0])["detected"].mean()
    for st, frac in by_stype.items():
        print(f"   {st}: {100*frac:5.1f}% seen")

    print("\nMass-recovered fraction by planet size:")
    df = df.assign(_rb=radius_bin(df))
    for rb, frac in df.groupby("_rb", observed=True)["detected"].mean().items():
        print(f"   {str(rb):>8} Rearth: {100*frac:5.1f}% seen")

    reason_col = "rv_reason_category" if "rv_reason_category" in df.columns else "reason_category"
    if reason_col in df.columns:
        print(f"\nWHY masses were missed/seen ({reason_col}):")
        print(df[reason_col].value_counts(dropna=False).to_string())
    print("=====================================================================\n")


def make_plot(df: pd.DataFrame, out_png: str, instrument: str) -> None:
    det = df["detected"].astype(bool)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[~det, "p_orb"], df.loc[~det, "mass_p"],
               s=3, c="lightgray", label="missed", rasterized=True)
    ax.scatter(df.loc[det, "p_orb"], df.loc[det, "mass_p"],
               s=3, c="tab:red", label="mass recovered", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period [d]")
    ax.set_ylabel("Planet mass [$M_\\oplus$]")
    ax.set_title(f"RV ({instrument}) on uniform-box planets")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"--> Saved plot: {out_png}")


def main():
    catalog = get_or_build_catalog(CACHE_CSV, rebuild=False, n_planets=N_PLANETS, seed=0)

    print(f"--> Running RVData ({INSTRUMENT}) on {len(catalog)} flat planets...")
    df = run_rv(catalog, instrument=INSTRUMENT)

    caveman_summary(df, INSTRUMENT)

    out_csv = os.path.join(UNIFORM_OUT_DIR, "uniform_rv_detected.csv")
    df.to_csv(out_csv, index=False)
    print(f"--> Saved RV results: {out_csv}")

    make_plot(df, os.path.join(UNIFORM_OUT_DIR, "uniform_rv_detection_map.png"), INSTRUMENT)


if __name__ == "__main__":
    main()
