"""
02_kepler_koi_dr25_check.py

Report-ready KOI/DR25 validation plots for KeplerData.

Purpose
-------
Check whether lifesim.core.kepler_data.KeplerData behaves like a physically
reasonable Kepler selection-function model using NASA's KOI cumulative table.

This is NOT a false-positive classifier.
A false positive can still be a strong transit-like signal, so it may pass the
toy detector. The goal is signal detectability, not planet validation.

Outputs
-------
ROOT / "my_outputs" / "02_kepler_koi_dr25_check"

Main products:
    koi_dr25_detector_check_results.csv
    koi_dr25_summary_by_disposition.csv
    koi_dr25_summary_by_mes_bin.csv
    koi_dr25_mes_ratio_by_mes_bin.csv
    figure_data_dictionary.txt

    A_mes_calibration_report.png
    B_recovery_curve_report.png
    C1_selection_radius_period.png
    C2_selection_radius_host_radius.png
    C3_selection_radius_kepmag.png
    D_signal_detectability_by_disposition.png
    E_mes_ratio_by_koi_mes.png

How to run
----------
From repo root:
    python scripts/02_kepler_koi_dr25_check.py

Useful options:
    python scripts/02_kepler_koi_dr25_check.py --redownload
    python scripts/02_kepler_koi_dr25_check.py --no-dr25-only
    python scripts/02_kepler_koi_dr25_check.py --transit-mode inclination
    python scripts/02_kepler_koi_dr25_check.py --no-use-observed-depth

Recommended default:
    --transit-mode observed

Why? KOIs are already transit-like events. The default checks whether your
signal/MES/noise part behaves correctly, without punishing rows for missing or
weird fitted inclination.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths and imports
# ============================================================


def find_project_root(start: Path) -> Path:
    """Walk upward until we find the project root."""
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "run" / "kepler").exists() or (p / "lifesim" / "core").exists():
            return p
    return Path.cwd()


ROOT = find_project_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from lifesim.core.kepler_data import KeplerData
except Exception as exc:
    raise ImportError(
        "Could not import lifesim.core.kepler_data.KeplerData.\n"
        "Run this script from your mdwarf-habitability repo root, or place it in scripts/.\n"
        f"Detected ROOT = {ROOT}\n"
        f"Original error: {type(exc).__name__}: {exc}"
    ) from exc


OUT_DIR = ROOT / "my_outputs" / "02_kepler_koi_dr25_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = OUT_DIR / "koi_cumulative_dr25_cached.csv"
OLD_CACHE_PATH = ROOT / "my_outputs" / "24_koi_dr25_detector_check" / "koi_cumulative_dr25_cached.csv"
RESULTS_PATH = OUT_DIR / "koi_dr25_detector_check_results.csv"

MES_THRESHOLD = 7.1
MIN_CELL_N = 20


# ============================================================
# Plot style
# ============================================================


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 260,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ============================================================
# NASA KOI download
# ============================================================


KOI_COLUMNS = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_tce_plnt_num",
    "koi_tce_delivname",

    "koi_period",
    "koi_prad",
    "koi_sma",
    "koi_insol",
    "koi_depth",
    "koi_duration",
    "koi_incl",

    "koi_steff",
    "koi_srad",
    "koi_smass",
    "koi_kepmag",

    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_model_snr",
    "koi_num_transits",
]


def nasa_tap_csv_url(query: str) -> str:
    base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    return f"{base}?query={quote(query)}&format=csv"


def load_or_download_koi(redownload: bool = False) -> pd.DataFrame:
    """Load cached KOIs, copy old cache if available, or download from NASA."""
    if CACHE_PATH.exists() and not redownload:
        print(f"Loading cached KOI table: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    if OLD_CACHE_PATH.exists() and not redownload:
        print(f"Copying previous KOI cache: {OLD_CACHE_PATH}")
        df = pd.read_csv(OLD_CACHE_PATH)
        df.to_csv(CACHE_PATH, index=False)
        return df

    query = f"""
    SELECT
        {", ".join(KOI_COLUMNS)}
    FROM cumulative
    WHERE koi_period IS NOT NULL
      AND koi_prad IS NOT NULL
      AND koi_srad IS NOT NULL
      AND koi_depth IS NOT NULL
      AND koi_duration IS NOT NULL
      AND koi_kepmag IS NOT NULL
    """
    print("Downloading NASA KOI cumulative table...")
    df = pd.read_csv(nasa_tap_csv_url(query))
    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved cache: {CACHE_PATH}")
    return df


# ============================================================
# Cleaning and detector run
# ============================================================


def force_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_dr25_koi(df: pd.DataFrame, transit_mode: str, dr25_only: bool) -> pd.DataFrame:
    """Prepare NASA KOI rows for KeplerData(source='koi')."""
    df = df.copy()

    numeric_cols = [
        "kepid", "koi_score", "koi_period", "koi_prad", "koi_sma", "koi_insol",
        "koi_depth", "koi_duration", "koi_incl", "koi_steff", "koi_srad",
        "koi_smass", "koi_kepmag", "koi_max_sngle_ev", "koi_max_mult_ev",
        "koi_model_snr", "koi_num_transits",
    ]
    df = force_numeric(df, numeric_cols)

    if dr25_only and "koi_tce_delivname" in df.columns:
        before = len(df)
        is_dr25 = df["koi_tce_delivname"].astype(str).str.contains(
            "Q1_Q17_DR25", case=False, na=False
        )
        if is_dr25.any():
            df = df[is_dr25].copy()
            print(f"DR25 filter using koi_tce_delivname: {before:,} -> {len(df):,}")
        else:
            print("WARNING: no Q1_Q17_DR25 labels found; keeping all rows.")

    before = len(df)
    required = ["koi_period", "koi_prad", "koi_srad", "koi_depth", "koi_duration"]
    df = df.dropna(subset=required).copy()
    df = df[(df["koi_period"] > 0) & (df["koi_prad"] > 0) & (df["koi_srad"] > 0)].copy()
    print(f"Valid KOI rows for detector: {before:,} -> {len(df):,}")

    if transit_mode == "observed":
        # KOIs are already transit-like events. This validates signal recovery.
        df["tran_flag"] = 1
    elif transit_mode == "inclination":
        # KeplerData will use koi_incl geometry.
        pass
    else:
        raise ValueError("transit_mode must be 'observed' or 'inclination'")

    return df


def run_kepler_detector(
    koi_df: pd.DataFrame,
    use_observed_depth: bool,
    fallback_cdpp_ppm: float,
) -> pd.DataFrame:
    """Run KeplerData and return a report-friendly result table."""
    kd = KeplerData(
        koi_df,
        source="koi",
        validate_for_detection=True,
        min_transits=3,
        mes_threshold=MES_THRESHOLD,
        kepler_mag_limit=16.0,
        fallback_cdpp_ppm=fallback_cdpp_ppm,
        use_kepmag_cdpp_fallback=True,
        use_observed_transit_flag_for_nasa=True,
        use_observed_transit_depth_for_nasa=use_observed_depth,
        assume_bright_if_kepmag_missing_for_nasa=False,
        estimate_missing_semimajor_axis=True,
    )
    out = kd.determine_detectable()

    # KeplerData(source='koi') renames kepoi_name -> planet_name.
    if "kepoi_name" not in out.columns:
        if "planet_name" in out.columns:
            out["kepoi_name"] = out["planet_name"]
        else:
            out["kepoi_name"] = np.arange(len(out)).astype(str)

    out["koi_mes_pass"] = pd.to_numeric(out["koi_max_mult_ev"], errors="coerce") >= MES_THRESHOLD
    out["toy_over_koi_mes"] = (
        pd.to_numeric(out["kepler_mes"], errors="coerce") /
        pd.to_numeric(out["koi_max_mult_ev"], errors="coerce").replace(0, np.nan)
    )
    out["planet_like_disposition"] = out["koi_disposition"].astype(str).isin(["CONFIRMED", "CANDIDATE"])
    return add_bins(out)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add bins used by report tables and plots."""
    df = df.copy()
    df["period_bin"] = pd.cut(
        pd.to_numeric(df["p_orb"], errors="coerce"),
        bins=[0, 3, 10, 30, 100, 300, np.inf],
        labels=["<3", "3-10", "10-30", "30-100", "100-300", ">300"],
        include_lowest=True,
    )
    df["radius_bin"] = pd.cut(
        pd.to_numeric(df["radius_p"], errors="coerce"),
        bins=[0, 1.0, 1.5, 3.0, 6.0, np.inf],
        labels=["<1", "1-1.5", "1.5-3", "3-6", ">6"],
        include_lowest=True,
    )
    df["stellar_radius_bin"] = pd.cut(
        pd.to_numeric(df["radius_s"], errors="coerce"),
        bins=[0, 0.7, 1.0, 1.5, 2.0, np.inf],
        labels=["<0.7", "0.7-1.0", "1.0-1.5", "1.5-2.0", ">2.0"],
        include_lowest=True,
    )
    df["kepmag_bin"] = pd.cut(
        pd.to_numeric(df["kepmag"], errors="coerce"),
        bins=[0, 10, 12, 14, 15, 16, np.inf],
        labels=["<10", "10-12", "12-14", "14-15", "15-16", ">16"],
        include_lowest=True,
    )
    df["koi_mes_bin"] = pd.cut(
        pd.to_numeric(df["koi_max_mult_ev"], errors="coerce"),
        bins=[0, 7.1, 10, 20, 50, 100, 300, np.inf],
        labels=["<7.1", "7.1-10", "10-20", "20-50", "50-100", "100-300", ">300"],
        include_lowest=True,
    )
    return df


# ============================================================
# Tables
# ============================================================


def save_tables(out: pd.DataFrame) -> None:
    """Save compact summary tables for the report and debugging."""
    summary_disp = (
        out.groupby("koi_disposition", dropna=False)
        .agg(
            n=("kepoi_name", "size"),
            toy_pass_fraction=("detected", "mean"),
            official_mes_pass_fraction=("koi_mes_pass", "mean"),
            median_toy_mes=("kepler_mes", "median"),
            median_koi_mes=("koi_max_mult_ev", "median"),
            median_toy_over_koi_mes=("toy_over_koi_mes", "median"),
        )
        .reset_index()
    )
    summary_disp.to_csv(OUT_DIR / "koi_dr25_summary_by_disposition.csv", index=False)

    summary_mes = (
        out.groupby("koi_mes_bin", observed=True, dropna=False)
        .agg(
            n=("kepoi_name", "size"),
            toy_pass_fraction=("detected", "mean"),
            median_toy_mes=("kepler_mes", "median"),
            median_koi_mes=("koi_max_mult_ev", "median"),
            median_toy_over_koi_mes=("toy_over_koi_mes", "median"),
        )
        .reset_index()
    )
    summary_mes.to_csv(OUT_DIR / "koi_dr25_summary_by_mes_bin.csv", index=False)
    summary_mes.to_csv(OUT_DIR / "koi_dr25_mes_ratio_by_mes_bin.csv", index=False)

    reason = (
        out.groupby(["koi_disposition", "reason_category"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    reason.to_csv(OUT_DIR / "koi_dr25_reason_by_disposition.csv", index=False)

    near = out[
        pd.to_numeric(out["koi_max_mult_ev"], errors="coerce").between(7.1, 20)
    ].copy()
    near_cols = [
        "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition",
        "p_orb", "radius_p", "radius_s", "kepmag",
        "koi_depth", "koi_duration", "koi_num_transits",
        "n_transits_keplerish", "koi_max_mult_ev", "koi_model_snr",
        "kepler_mes", "toy_over_koi_mes", "detected", "reason_category",
        "kepler_cdpp_ppm", "transit_depth_ppm", "transit_duration_hr_est",
    ]
    near_cols = [c for c in near_cols if c in near.columns]
    near.sort_values("toy_over_koi_mes", ascending=True)[near_cols].head(200).to_csv(
        OUT_DIR / "near_threshold_low_toy_mes_examples.csv", index=False
    )

    print("\n=== Summary by disposition ===")
    print(summary_disp.to_string(index=False))
    print("\n=== Summary by official KOI MES bin ===")
    print(summary_mes.to_string(index=False))


def write_data_dictionary() -> None:
    text = f"""
Figure/data dictionary for 02_kepler_koi_dr25_check
========================================================

Input sample
------------
- NASA KOI cumulative table, optionally filtered to Q1_Q17_DR25 using koi_tce_delivname.
- Default transit mode is observed: tran_flag=1 is assigned because KOIs are already transit-like events.
- KeplerData(source='koi') standardizes KOI columns:
    koi_period  -> p_orb
    koi_prad    -> radius_p
    koi_sma     -> semimajor_p
    koi_depth   -> observed_transit_depth_ppm
    koi_duration-> observed_transit_duration_hr
    koi_srad    -> radius_s
    koi_smass   -> mass_s
    koi_kepmag  -> kepmag

Main detector columns
---------------------
- Toy Kepler MES:
    column = kepler_mes
    created by KeplerData.determine_detectable(), through calc_kepler_mes().
    Formula in KeplerData is approximately:
        MES = transit_depth_ppm / kepler_cdpp_ppm * sqrt(n_transits_keplerish) * correction

- Official NASA KOI MES:
    column = koi_max_mult_ev
    downloaded directly from NASA KOI cumulative table.

- Toy pass/fail:
    column = detected
    created by KeplerData.determine_detectable().
    Logic:
        detected = transiting_geometric & kepler_star_bright_enough & kepler_depth_pass

- Miss reason:
    column = reason_category
    created by KeplerData.add_miss_reason_category().
    Values include:
        detected
        host_star_too_faint
        too_few_transits
        too_shallow_or_low_mes
        not_transiting

Figures
-------
A_mes_calibration_report.png
    Data: koi_max_mult_ev vs kepler_mes.
    Purpose: test whether toy MES tracks official Kepler KOI MES.
    Helper functions: plot_mes_calibration(), binned_median_by_mes().

B_recovery_curve_report.png
    Data: detected fraction in bins of koi_max_mult_ev.
    Purpose: show recovery rises with official Kepler signal strength.
    Helper functions: plot_recovery_curve().

C1_selection_radius_period.png
    Data: detected fraction grouped by radius_bin and period_bin.
    Purpose: small planets and long-period planets should be harder.
    Helper functions: plot_selection_heatmap().

C2_selection_radius_host_radius.png
    Data: detected fraction grouped by radius_bin and stellar_radius_bin.
    Purpose: larger host stars dilute transit depth and make small planets harder.
    Helper functions: plot_selection_heatmap().

C3_selection_radius_kepmag.png
    Data: detected fraction grouped by radius_bin and kepmag_bin.
    Purpose: faint/noisy stars and the Kp=16 toy cutoff reduce recovery.
    Helper functions: plot_selection_heatmap().

D_signal_detectability_by_disposition.png
    Data: reason_category fractions grouped by koi_disposition.
    Purpose: show the detector sees transit-like signals, not planet validity.
    Important: false positives can pass if they are strong signals.

E_mes_ratio_by_koi_mes.png
    Data: median toy_over_koi_mes in bins of koi_max_mult_ev.
    Purpose: show whether toy MES is conservative or optimistic in each signal regime.

Settings
--------
MES threshold = {MES_THRESHOLD}
Minimum heatmap cell count shown as robust = {MIN_CELL_N}
Output directory = {OUT_DIR}
"""
    (OUT_DIR / "figure_data_dictionary.txt").write_text(text.strip() + "\n", encoding="utf-8")


# ============================================================
# Plot helpers
# ============================================================


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def finite_positive(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for col in cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    mask = np.ones(len(d), dtype=bool)
    for col in cols:
        mask &= np.isfinite(d[col]) & (d[col] > 0)
    return d[mask].copy()


def binned_median_by_mes(out: pd.DataFrame) -> pd.DataFrame:
    """Median toy and official MES in official KOI MES bins."""
    d = out.dropna(subset=["koi_mes_bin"]).copy()
    return (
        d.groupby("koi_mes_bin", observed=True)
        .agg(
            n=("kepoi_name", "size"),
            median_koi_mes=("koi_max_mult_ev", "median"),
            median_toy_mes=("kepler_mes", "median"),
            q25_toy_mes=("kepler_mes", lambda x: np.nanpercentile(x, 25)),
            q75_toy_mes=("kepler_mes", lambda x: np.nanpercentile(x, 75)),
        )
        .reset_index()
    )


def plot_mes_calibration(out: pd.DataFrame) -> None:
    """A. Toy MES vs official KOI MES, with binned medians."""
    d = finite_positive(out, ["koi_max_mult_ev", "kepler_mes"])

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    # Light background cloud.
    for label, g in d.groupby("koi_disposition", dropna=False):
        ax.scatter(
            g["koi_max_mult_ev"], g["kepler_mes"],
            s=7, alpha=0.22, label=str(label)
        )

    med = binned_median_by_mes(d)
    med = med[np.isfinite(med["median_koi_mes"]) & np.isfinite(med["median_toy_mes"])]
    if len(med):
        yerr = np.vstack([
            med["median_toy_mes"] - med["q25_toy_mes"],
            med["q75_toy_mes"] - med["median_toy_mes"],
        ])
        ax.errorbar(
            med["median_koi_mes"], med["median_toy_mes"],
            yerr=yerr, fmt="o-", color="black", linewidth=1.5,
            markersize=4, capsize=2, label="Binned median"
        )

    lo = max(0.5, min(d["koi_max_mult_ev"].min(), d["kepler_mes"].min()))
    hi = max(d["koi_max_mult_ev"].max(), d["kepler_mes"].max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)

    ax.plot(grid, grid, linestyle="--", linewidth=1.0, color="black", alpha=0.65, label="1:1")
    ax.axvline(MES_THRESHOLD, linestyle=":", linewidth=1.1, color="black")
    ax.axhline(MES_THRESHOLD, linestyle=":", linewidth=1.1, color="black")
    ax.axvspan(MES_THRESHOLD, 20, color="0.85", alpha=0.35, label="Marginal KOI MES")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Official NASA KOI MES: koi_max_mult_ev")
    ax.set_ylabel("Toy Kepler MES: kepler_mes")
    ax.set_title("A. Toy MES tracks official Kepler DR25 MES")
    ax.text(
        0.02, 0.02,
        "Dashed line = perfect agreement\nDotted lines = MES 7.1 threshold",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(OUT_DIR / "A_mes_calibration_report.png", bbox_inches="tight")
    plt.close(fig)


def plot_recovery_curve(out: pd.DataFrame) -> None:
    """B. Toy pass fraction as a function of official KOI MES."""
    s = (
        out.dropna(subset=["koi_mes_bin"])
        .groupby("koi_mes_bin", observed=True)
        .agg(n=("kepoi_name", "size"), pass_fraction=("detected", "mean"))
        .reset_index()
    )
    s["x"] = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(s["x"], s["pass_fraction"], marker="o", linewidth=1.8)
    for _, row in s.iterrows():
        ax.text(
            row["x"], row["pass_fraction"] + 0.035,
            f"{row['pass_fraction']:.0%}\nN={int(row['n'])}",
            ha="center", va="bottom", fontsize=8
        )
    ax.set_ylim(0, 1.10)
    ax.set_xticks(s["x"])
    ax.set_xticklabels(s["koi_mes_bin"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Toy Kepler pass fraction")
    ax.set_xlabel("Official NASA KOI MES bin")
    ax.set_title("B. Recovery rises with official Kepler signal strength")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "B_recovery_curve_report.png", bbox_inches="tight")
    plt.close(fig)


def heatmap_table(df: pd.DataFrame, row: str, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frac = df.groupby([row, col], observed=True)["detected"].mean().unstack(col)
    count = df.groupby([row, col], observed=True)["detected"].size().unstack(col)
    return frac, count


def plot_selection_heatmap(
    out: pd.DataFrame,
    row: str,
    col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Report-ready heatmap. Cells with N < MIN_CELL_N are greyed out."""
    frac, count = heatmap_table(out, row, col)
    display = frac.where(count >= MIN_CELL_N)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("0.90")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(display.to_numpy(dtype=float), vmin=0, vmax=1, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(frac.columns)))
    ax.set_yticks(np.arange(len(frac.index)))
    ax.set_xticklabels([str(x) for x in frac.columns], rotation=30, ha="right")
    ax.set_yticklabels([str(y) for y in frac.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            n = count.iloc[i, j]
            f = frac.iloc[i, j]
            if pd.isna(n) or n == 0:
                continue
            if n < MIN_CELL_N:
                label = f"N={int(n)}\nlow N"
                color = "0.45"
            else:
                label = f"{f:.0%}\nN={int(n)}"
                color = "black" if f > 0.55 else "white"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Toy Kepler pass fraction")
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_all_selection_heatmaps(out: pd.DataFrame) -> None:
    plot_selection_heatmap(
        out,
        row="radius_bin",
        col="period_bin",
        xlabel="Orbital period bin [days]",
        ylabel="Planet radius bin [R_earth]",
        title="C1. Small, long-period KOIs are hardest",
        filename="C1_selection_radius_period.png",
    )
    plot_selection_heatmap(
        out,
        row="radius_bin",
        col="stellar_radius_bin",
        xlabel="Host-star radius bin [R_sun]",
        ylabel="Planet radius bin [R_earth]",
        title="C2. Large host stars dilute transit depth",
        filename="C2_selection_radius_host_radius.png",
    )
    plot_selection_heatmap(
        out,
        row="radius_bin",
        col="kepmag_bin",
        xlabel="Kepler magnitude bin; smaller = brighter",
        ylabel="Planet radius bin [R_earth]",
        title="C3. Faint/noisy hosts and Kp > 16 reduce recovery",
        filename="C3_selection_radius_kepmag.png",
    )


def plot_reason_by_disposition(out: pd.DataFrame) -> None:
    """D. Detection reason grouped by disposition."""
    order = ["detected", "too_shallow_or_low_mes", "too_few_transits", "host_star_too_faint", "not_transiting", "other_missed"]
    pivot = (
        out.groupby(["koi_disposition", "reason_category"], dropna=False)
        .size()
        .unstack("reason_category", fill_value=0)
    )
    for col in order:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[order]
    frac = pivot.div(pivot.sum(axis=1), axis=0)

    labels = [f"{idx}\nN={int(pivot.loc[idx].sum())}" for idx in frac.index]
    ax = frac.plot(kind="bar", stacked=True, figsize=(7.8, 4.8), width=0.72)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of KOIs")
    ax.set_xlabel("KOI disposition")
    ax.set_title("D. Signal detectability by KOI disposition\nFalse positives can pass: detection ≠ planet validation")
    ax.legend(title="Toy reason", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.savefig(OUT_DIR / "D_signal_detectability_by_disposition.png", bbox_inches="tight")
    plt.close()


def plot_mes_ratio(out: pd.DataFrame) -> None:
    """E. Median toy/official MES ratio by official MES bin."""
    d = out.copy()
    d["toy_over_koi_mes"] = pd.to_numeric(d["toy_over_koi_mes"], errors="coerce")
    d = d[np.isfinite(d["toy_over_koi_mes"]) & (d["toy_over_koi_mes"] > 0)]

    s = (
        d.dropna(subset=["koi_mes_bin"])
        .groupby("koi_mes_bin", observed=True)
        .agg(
            n=("kepoi_name", "size"),
            median_ratio=("toy_over_koi_mes", "median"),
            q25=("toy_over_koi_mes", lambda x: np.nanpercentile(x, 25)),
            q75=("toy_over_koi_mes", lambda x: np.nanpercentile(x, 75)),
        )
        .reset_index()
    )
    s["x"] = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    yerr = np.vstack([s["median_ratio"] - s["q25"], s["q75"] - s["median_ratio"]])
    ax.errorbar(s["x"], s["median_ratio"], yerr=yerr, fmt="o-", capsize=3, linewidth=1.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(s["x"])
    ax.set_xticklabels(s["koi_mes_bin"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Toy MES / official KOI MES")
    ax.set_xlabel("Official NASA KOI MES bin")
    ax.set_title("E. MES calibration ratio\nBelow 1 = toy detector is conservative")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / "E_mes_ratio_by_koi_mes.png", bbox_inches="tight")
    plt.close(fig)


def make_report_plots(out: pd.DataFrame) -> None:
    plot_mes_calibration(out)
    plot_recovery_curve(out)
    plot_all_selection_heatmaps(out)
    plot_reason_by_disposition(out)
    plot_mes_ratio(out)


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redownload", action="store_true", help="Ignore caches and redownload NASA KOI table.")
    parser.add_argument("--no-dr25-only", action="store_true", help="Do not filter to Q1_Q17_DR25 KOIs.")
    parser.add_argument("--transit-mode", choices=["observed", "inclination"], default="observed")
    parser.add_argument("--no-use-observed-depth", action="store_true", help="Use Rp/Rstar depth instead of KOI depth.")
    parser.add_argument("--fallback-cdpp-ppm", type=float, default=100.0)
    args = parser.parse_args()

    print(f"Project root: {ROOT}")
    print(f"Output dir:   {OUT_DIR}")

    raw = load_or_download_koi(redownload=args.redownload)
    print(f"Raw downloaded/cached rows: {len(raw):,}")

    if "koi_tce_delivname" in raw.columns:
        print("TCE delivery names:")
        print(raw["koi_tce_delivname"].value_counts(dropna=False).head(10).to_string())

    koi = prepare_dr25_koi(
        raw,
        transit_mode=args.transit_mode,
        dr25_only=not args.no_dr25_only,
    )

    out = run_kepler_detector(
        koi,
        use_observed_depth=not args.no_use_observed_depth,
        fallback_cdpp_ppm=args.fallback_cdpp_ppm,
    )

    out.to_csv(RESULTS_PATH, index=False)
    print(f"Saved detector results: {RESULTS_PATH}")

    save_tables(out)
    write_data_dictionary()
    make_report_plots(out)

    print("\nSaved report-ready plots:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")

    print("\nInterpretation:")
    print("  A: toy MES should correlate with official KOI MES.")
    print("  B: toy recovery should rise with official KOI MES.")
    print("  C1-C3: small/long-period/large-star/faint-star bins should be harder.")
    print("  D: false positives can pass because signal detection is not planet validation.")
    print("  E: toy/official MES below 1 means the toy detector is conservative.")


if __name__ == "__main__":
    main()
