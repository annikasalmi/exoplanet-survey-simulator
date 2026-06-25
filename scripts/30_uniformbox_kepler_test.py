"""
30_uniformbox_kepler_test.py — push the UNIFORM-BOX control planets through KeplerData.

This generates (once, cached) a flat radius/period planet population on real
Gaia-60pc stars using PPop/PlanetDistributions/UniformBox.py, then runs the
Kepler toy detector on it and reports what Kepler would and would not catch.

Because the input is FLAT in (radius, period) instead of clumped where real
planets live, the detected/undetected split is a clean map of Kepler's
sensitivity edge rather than a reflection of the occurrence-rate prior.

Run:
    python scripts/30_uniformbox_kepler_test.py
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
from run.ppop.flat_detect import run_kepler

N_PLANETS = 300000
CACHE_CSV = os.path.join(UNIFORM_OUT_DIR, "flat_catalog.csv")


def radius_bin(df: pd.DataFrame) -> pd.Series:
    return pd.cut(
        pd.to_numeric(df["radius_p"], errors="coerce"),
        bins=[0, 1.5, 3.0, 6.0, np.inf],
        labels=["<1.5", "1.5-3.0", "3.0-6.0", ">6.0"],
        include_lowest=True,
    )


def caveman_summary(df: pd.DataFrame) -> None:
    n = len(df)
    det = df["detected"].astype(bool)
    print("\n=================== KEPLER on UNIFORM-BOX planets ===================")
    print(f"Total fake planets thrown at Kepler : {n}")
    print(f"Kepler SEES (detected)              : {det.sum()}  ({100*det.mean():.1f}%)")
    print(f"Kepler MISSES                        : {(~det).sum()}  ({100*(~det).mean():.1f}%)")

    print("\nDetected fraction by host-star type:")
    by_stype = df.groupby(df["stype"].astype(str).str[0])["detected"].mean()
    for st, frac in by_stype.items():
        print(f"   {st}: {100*frac:5.1f}% seen")

    print("\nDetected fraction by planet size:")
    df = df.assign(_rb=radius_bin(df))
    for rb, frac in df.groupby("_rb", observed=True)["detected"].mean().items():
        print(f"   {str(rb):>8} Rearth: {100*frac:5.1f}% seen")

    print("\nWHY planets were missed/seen (reason_category):")
    print(df["reason_category"].value_counts(dropna=False).to_string())
    print("=====================================================================\n")


def make_plot(df: pd.DataFrame, out_png: str) -> None:
    det = df["detected"].astype(bool)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[~det, "p_orb"], df.loc[~det, "radius_p"],
               s=3, c="lightgray", label="missed", rasterized=True)
    ax.scatter(df.loc[det, "p_orb"], df.loc[det, "radius_p"],
               s=3, c="tab:green", label="detected", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period [d]")
    ax.set_ylabel("Planet radius [$R_\\oplus$]")
    ax.set_title("Kepler on uniform-box planets")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"--> Saved plot: {out_png}")


def main():
    catalog = get_or_build_catalog(CACHE_CSV, rebuild=False, n_planets=N_PLANETS, seed=0)

    print(f"--> Running KeplerData on {len(catalog)} flat planets...")
    df = run_kepler(catalog)

    caveman_summary(df)

    out_csv = os.path.join(UNIFORM_OUT_DIR, "uniform_kepler_detected.csv")
    df.to_csv(out_csv, index=False)
    print(f"--> Saved Kepler results: {out_csv}")

    make_plot(df, os.path.join(UNIFORM_OUT_DIR, "uniform_kepler_detection_map.png"))


if __name__ == "__main__":
    main()
