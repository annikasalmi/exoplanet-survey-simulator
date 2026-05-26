# scripts/16_kepler_sciences_simple.py
#
# Simplified version of 16_kepler_sciences_comparison_panels.py
#
# Purpose:
#   Make the four main science figures:
#   A. Kepler detection funnel
#   B. Host-star type x planet-radius comparison
#   C. Detection-efficiency heatmap comparisons
#   E. Minimum detectable radius threshold comparisons
#
# Philosophy:
#   Less "clever helper code", more readable science logic.

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "run" / "kepler" / "data" / "Gaia_C_F_K_combined"
OUT_DIR = ROOT / "my_outputs" / "16_w3_7_kepler_simple_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATTERN = "kepler_catalog_*.csv"

MES_THRESHOLD = 7.1
RSUN_TO_REARTH = 109.076

# These are diagnostic labels, not official Kepler rules.
LONG_PERIOD_DAYS = 60.817
LOW_REPEAT_NTRANSITS = 24.0

STAR_ORDER = ["F", "G", "K", "M"]

RADIUS_BINS = [0, 1.0, 1.5, 3.0, 6.0, np.inf]
RADIUS_LABELS = ["<1.0", "1.0-1.5", "1.5-3.0", "3.0-6.0", ">6.0"]

PERIOD_BINS = np.logspace(np.log10(0.5), np.log10(500), 28)
PLANET_RADIUS_BINS = np.logspace(np.log10(0.3), np.log10(15), 24)

MIN_BIN_COUNT = 5


# ============================================================
# Data loading and cleaning
# ============================================================

def load_catalogs():
    paths = sorted(glob.glob(str(DATA_DIR / CSV_PATTERN)))
    if not paths:
        raise FileNotFoundError(f"No files found at {DATA_DIR / CSV_PATTERN}")

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = Path(path).name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(paths)} file(s)")
    print(f"Total rows: {len(df):,}")
    return df


def as_bool(series):
    """Safe bool conversion: avoids the Python problem where bool('False') is True."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def clean_star_type(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().upper()
    return s[0] if s else "Unknown"


def prepare_data(df):
    required = [
        "radius_p", "p_orb", "radius_s", "stype",
        "transiting_geometric", "detected",
        "kepler_mes", "n_transits_keplerish", "transit_depth_ppm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Clean categories.
    df["stype_clean"] = df["stype"].apply(clean_star_type)
    df["radius_bin"] = pd.cut(
        df["radius_p"],
        bins=RADIUS_BINS,
        labels=RADIUS_LABELS,
        include_lowest=True,
    )

    # Safe boolean columns.
    df["detected"] = as_bool(df["detected"])
    df["transiting_geometric"] = as_bool(df["transiting_geometric"])

    if "bright_enough_kepler" in df.columns:
        df["star_bright_enough"] = as_bool(df["bright_enough_kepler"])
    elif "kepler_star_bright_enough" in df.columns:
        df["star_bright_enough"] = as_bool(df["kepler_star_bright_enough"])
    else:
        df["star_bright_enough"] = True

    if "kepler_enough_transits" in df.columns:
        df["enough_transits"] = as_bool(df["kepler_enough_transits"])
    else:
        df["enough_transits"] = df["n_transits_keplerish"] >= 3

    if "kepler_depth_good" in df.columns:
        df["depth_good"] = as_bool(df["kepler_depth_good"])
    elif "kepler_depth_pass" in df.columns:
        df["depth_good"] = as_bool(df["kepler_depth_pass"])
    else:
        df["depth_good"] = df["kepler_mes"] >= MES_THRESHOLD

    return df


def add_reason_category(df):
    df = df.copy()

    reason = np.full(len(df), "other_missed", dtype=object)

    reason[df["detected"].to_numpy()] = "detected"
    reason[(~df["transiting_geometric"]).to_numpy()] = "not_transiting"

    transiting_missed = df["transiting_geometric"] & ~df["detected"]

    faint = transiting_missed & ~df["star_bright_enough"]

    long_low = (
        transiting_missed
        & df["star_bright_enough"]
        & (
            (df["p_orb"] >= LONG_PERIOD_DAYS)
            | (df["n_transits_keplerish"] <= LOW_REPEAT_NTRANSITS)
            | (~df["enough_transits"])
        )
    )

    shallow = (
        transiting_missed
        & df["star_bright_enough"]
        & ~long_low
        & (
            (~df["depth_good"])
            | (df["kepler_mes"] < MES_THRESHOLD)
        )
    )

    reason[faint.to_numpy()] = "host_star_too_faint"
    reason[long_low.to_numpy()] = "long_period_low_repeat"
    reason[shallow.to_numpy()] = "too_shallow_or_low_mes"

    df["reason_category"] = reason
    return df


# ============================================================
# Small plotting helpers
# ============================================================

def label_bars(ax, bars, denominator):
    ymax = max([b.get_height() for b in bars]) if len(bars) else 1
    for bar in bars:
        value = bar.get_height()
        pct = 100 * value / denominator if denominator > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015 * ymax,
            f"{int(value):,}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def detection_grid(df, xcol, ycol, xbins, ybins):
    d = df.copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]

    total, _, _ = np.histogram2d(d[xcol], d[ycol], bins=[xbins, ybins])
    detected, _, _ = np.histogram2d(
        d.loc[d["detected"], xcol],
        d.loc[d["detected"], ycol],
        bins=[xbins, ybins],
    )

    frac = np.divide(
        detected,
        total,
        out=np.full_like(detected, np.nan, dtype=float),
        where=total > 0,
    )
    frac[total < MIN_BIN_COUNT] = np.nan
    return frac.T


def linear_bins(series, nbins):
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    if x.min() == x.max():
        return np.linspace(x.min() - 0.5, x.max() + 0.5, nbins + 1)
    return np.linspace(x.min(), x.max(), nbins + 1)


def log_bins(series, nbins):
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x) & (x > 0)]
    return np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)


# ============================================================
# A. Funnel
# ============================================================

def plot_A_funnel(df):
    stages = [
        ("All planets", pd.Series(True, index=df.index)),
        ("Geometrically\ntransiting", df["transiting_geometric"]),
        ("Bright enough\nhost", df["transiting_geometric"] & df["star_bright_enough"]),
        ("Enough\nrepeat transits", df["transiting_geometric"] & df["star_bright_enough"] & df["enough_transits"]),
        ("MES >= 7.1", df["transiting_geometric"] & df["star_bright_enough"] & df["enough_transits"] & (df["kepler_mes"] >= MES_THRESHOLD)),
        ("Detected", df["detected"]),
    ]

    labels = [name for name, _ in stages]
    counts = np.array([int(mask.sum()) for _, mask in stages])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # A1: all planets.
    bars = axes[0].bar(labels, counts)
    axes[0].set_title("A1. Absolute funnel: all simulated planets")
    axes[0].set_ylabel("Number of planets")
    axes[0].tick_params(axis="x", rotation=25)
    label_bars(axes[0], bars, denominator=counts[0])

    # A2: remove random geometry and zoom into physical detectability.
    labels2 = labels[1:]
    counts2 = counts[1:]
    bars = axes[1].bar(labels2, counts2)
    axes[1].set_title("A2. Conditional funnel: only transiting planets")
    axes[1].set_ylabel("Number of transiting planets")
    axes[1].tick_params(axis="x", rotation=25)
    label_bars(axes[1], bars, denominator=counts2[0])

    fig.suptitle("A. Kepler detection funnel: geometry first, then physical detectability")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A_detection_funnel_simple.png", dpi=250)
    plt.close(fig)


# ============================================================
# B. Star type x radius bin
# ============================================================

def star_radius_tables(df):
    trans = df[df["transiting_geometric"] & df["stype_clean"].isin(STAR_ORDER)]

    total = pd.DataFrame(0.0, index=STAR_ORDER, columns=RADIUS_LABELS)
    detected = pd.DataFrame(0.0, index=STAR_ORDER, columns=RADIUS_LABELS)
    frac = pd.DataFrame(np.nan, index=STAR_ORDER, columns=RADIUS_LABELS)
    mes = pd.DataFrame(np.nan, index=STAR_ORDER, columns=RADIUS_LABELS)

    for stype in STAR_ORDER:
        for rbin in RADIUS_LABELS:
            sub = trans[(trans["stype_clean"] == stype) & (trans["radius_bin"] == rbin)]
            total.loc[stype, rbin] = len(sub)
            detected.loc[stype, rbin] = sub["detected"].sum()
            if len(sub) > 0:
                frac.loc[stype, rbin] = sub["detected"].mean()
                mes.loc[stype, rbin] = sub["kepler_mes"].median()

    return total, detected, frac, mes


def plot_B_star_radius(df):
    total, detected, frac, mes = star_radius_tables(df)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # B1: counts.
    ax = axes[0]
    x0 = np.arange(len(STAR_ORDER))
    width = 0.14

    for i, rbin in enumerate(RADIUS_LABELS):
        x = x0 + (i - 2) * width
        ax.bar(x, total[rbin], width=width, alpha=0.25, edgecolor="black",
               label="transiting total" if i == 0 else None)
        ax.bar(x, detected[rbin], width=width, alpha=0.85, label=rbin)

        ymax = max(total.max().max(), 1)
        for xi, t, f in zip(x, total[rbin], frac[rbin]):
            if t > 0 and np.isfinite(f):
                ax.text(xi, t + 0.02 * ymax, f"{100*f:.0f}%",
                        ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x0)
    ax.set_xticklabels(STAR_ORDER)
    ax.set_xlabel("Host star type")
    ax.set_ylabel("Number of geometrically transiting planets")
    ax.set_title("B1. Counts by star type and planet radius")
    ax.legend(title="Radius bin", fontsize=8)

    # B2: detection fraction.
    im = axes[1].imshow(frac.to_numpy(float), vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("B2. Detection fraction")
    axes[1].set_xticks(np.arange(len(RADIUS_LABELS)))
    axes[1].set_xticklabels(RADIUS_LABELS, rotation=30, ha="right")
    axes[1].set_yticks(np.arange(len(STAR_ORDER)))
    axes[1].set_yticklabels(STAR_ORDER)
    axes[1].set_xlabel("Planet radius bin [R_earth]")
    axes[1].set_ylabel("Host star type")

    for i in range(len(STAR_ORDER)):
        for j in range(len(RADIUS_LABELS)):
            if np.isfinite(frac.iloc[i, j]):
                axes[1].text(j, i, f"{100*frac.iloc[i,j]:.0f}%\nN={int(total.iloc[i,j])}",
                             ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], label="Detected fraction")

    # B3: median MES.
    log_mes = np.log10(mes.astype(float).where(mes.astype(float) > 0))
    im = axes[2].imshow(log_mes.to_numpy(float), aspect="auto")
    axes[2].set_title("B3. Median signal strength: log10(MES)")
    axes[2].set_xticks(np.arange(len(RADIUS_LABELS)))
    axes[2].set_xticklabels(RADIUS_LABELS, rotation=30, ha="right")
    axes[2].set_yticks(np.arange(len(STAR_ORDER)))
    axes[2].set_yticklabels(STAR_ORDER)
    axes[2].set_xlabel("Planet radius bin [R_earth]")
    axes[2].set_ylabel("Host star type")

    for i in range(len(STAR_ORDER)):
        for j in range(len(RADIUS_LABELS)):
            if np.isfinite(mes.iloc[i, j]):
                axes[2].text(j, i, f"{mes.iloc[i,j]:.1f}",
                             ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[2], label="log10 median MES")

    fig.suptitle("B. Host type and planet radius: counts, recovery, and signal strength")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "B_star_type_radius_simple.png", dpi=250)
    plt.close(fig)


# ============================================================
# C. Detection-efficiency heatmaps
# ============================================================

def plot_C_heatmaps(df):
    trans = df[
        df["transiting_geometric"]
        & (df["p_orb"] > 0)
        & (df["radius_p"] > 0)
        & (df["radius_s"] > 0)
    ].copy()

    # Third comparison: use CDPP if it varies; otherwise use Kepler magnitude.
    if "kepler_cdpp_ppm" in trans.columns and trans["kepler_cdpp_ppm"].nunique(dropna=True) > 5:
        third = ("kepler_cdpp_ppm", "Kepler CDPP [ppm]", log_bins(trans["kepler_cdpp_ppm"], 26), True, "C3. Radius vs photometric noise")
    elif "kepler_mag_used" in trans.columns:
        third = ("kepler_mag_used", "Approx. Kepler magnitude", linear_bins(trans["kepler_mag_used"], 26), False, "C3. Radius vs host brightness")
    else:
        third = ("transit_depth_ppm", "Transit depth [ppm]", log_bins(trans["transit_depth_ppm"], 26), True, "C3. Radius vs transit depth")

    panels = [
        ("p_orb", "Orbital period [days]", PERIOD_BINS, True, "C1. Radius vs period"),
        ("radius_s", "Host-star radius [R_sun]", linear_bins(trans["radius_s"], 26), False, "C2. Radius vs host-star radius"),
        third,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    mesh = None

    for ax, (xcol, xlabel, xbins, xlog, title) in zip(axes, panels):
        frac = detection_grid(trans, xcol, "radius_p", xbins, PLANET_RADIUS_BINS)
        mesh = ax.pcolormesh(xbins, PLANET_RADIUS_BINS, frac, shading="auto", vmin=0, vmax=1)

        if xlog:
            ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Planet radius [R_earth]")
        ax.set_title(title)

    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Detected fraction")
    fig.suptitle("C. Kepler detection efficiency: which parameters control recovery?")
    fig.savefig(OUT_DIR / "C_detection_efficiency_simple.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# E. Minimum detectable radius curves
# ============================================================

def rp_min_curve(sub, period_grid):
    sub = sub[np.isfinite(sub["radius_s"]) & (sub["radius_s"] > 0)]
    if len(sub) == 0:
        return np.full_like(period_grid, np.nan)

    rstar = np.nanmedian(sub["radius_s"])
    cdpp = np.nanmedian(sub["kepler_cdpp_ppm"]) if "kepler_cdpp_ppm" in sub.columns else 100.0
    baseline = np.nanmedian(sub["kepler_dataspan_used_days"]) if "kepler_dataspan_used_days" in sub.columns else 1461.0
    duty = np.nanmedian(sub["kepler_dutycycle_used"]) if "kepler_dutycycle_used" in sub.columns else 1.0

    n_transits = baseline * duty / period_grid
    depth_needed = MES_THRESHOLD * cdpp / np.sqrt(np.maximum(n_transits, 1))

    rp_min = rstar * RSUN_TO_REARTH * np.sqrt(depth_needed / 1e6)
    rp_min[n_transits < 3] = np.nan
    return rp_min


def add_tercile(df, source_col, new_col, labels):
    # rank(method="first") prevents qcut from crashing when many values are tied.
    codes = pd.qcut(df[source_col].rank(method="first"), 3, labels=False, duplicates="drop")
    df[new_col] = codes.map({0: labels[0], 1: labels[1], 2: labels[2]})
    return df


def draw_threshold_panel(ax, trans, groups, title):
    missed = trans[~trans["detected"]]
    detected = trans[trans["detected"]]

    ax.scatter(missed["p_orb"], missed["radius_p"], s=4, alpha=0.10,
               label=f"Transiting missed ({len(missed):,})")
    ax.scatter(detected["p_orb"], detected["radius_p"], s=4, alpha=0.14,
               label=f"Detected ({len(detected):,})")

    period_grid = np.logspace(np.log10(0.5), np.log10(500), 300)

    for label, sub in groups:
        if len(sub) < 10:
            continue
        curve = rp_min_curve(sub, period_grid)
        if np.isfinite(curve).any():
            ax.plot(period_grid, curve, linewidth=2, label=f"{label} threshold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period [days]")
    ax.set_ylabel("Planet radius [R_earth]")
    ax.set_title(title)
    ax.legend(fontsize=7)


def plot_E_thresholds(df):
    trans = df[
        df["transiting_geometric"]
        & (df["p_orb"] > 0)
        & (df["radius_p"] > 0)
        & (df["radius_s"] > 0)
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # E1: host spectral type.
    groups = [(s, trans[trans["stype_clean"] == s]) for s in STAR_ORDER]
    draw_threshold_panel(axes[0], trans, groups, "E1. Threshold by host-star type")

    # E2: stellar radius tercile.
    trans = add_tercile(
        trans, "radius_s", "rstar_bin",
        ["small R_star", "medium R_star", "large R_star"]
    )
    groups = [(label, trans[trans["rstar_bin"] == label])
              for label in ["small R_star", "medium R_star", "large R_star"]]
    draw_threshold_panel(axes[1], trans, groups, "E2. Threshold by stellar radius")

    # # E3: brightness if available; otherwise noise.
    # if "kepler_mag_used" in trans.columns and trans["kepler_mag_used"].nunique(dropna=True) > 5:
    #     trans = add_tercile(
    #         trans, "kepler_mag_used", "brightness_bin",
    #         ["bright hosts", "medium hosts", "faint hosts"]
    #     )
    #     groups = [(label, trans[trans["brightness_bin"] == label])
    #               for label in ["bright hosts", "medium hosts", "faint hosts"]]
    #     title = "E3. Threshold by host brightness"
    # elif "kepler_cdpp_ppm" in trans.columns:
    #     trans = add_tercile(
    #         trans, "kepler_cdpp_ppm", "noise_bin",
    #         ["low noise", "medium noise", "high noise"]
    #     )
    #     groups = [(label, trans[trans["noise_bin"] == label])
    #               for label in ["low noise", "medium noise", "high noise"]]
    #     title = "E3. Threshold by photometric noise"
    # else:
    #     groups = []
    #     title = "E3. Brightness/noise unavailable"

    # draw_threshold_panel(axes[2], trans, groups, title)

    fig.suptitle("E. Approximate minimum detectable radius: above curve = easier for Kepler")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E_min_detectable_radius_simple.png", dpi=250)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    df = load_catalogs()
    df = prepare_data(df)
    df = add_reason_category(df)

    print("\nReason counts:")
    print(df["reason_category"].value_counts())

    print("\nCore summary:")
    print(f"All planets: {len(df):,}")
    print(f"Geometrically transiting: {int(df['transiting_geometric'].sum()):,}")
    print(f"Detected: {int(df['detected'].sum()):,}")
    print(f"Transiting but missed: {int((df['transiting_geometric'] & ~df['detected']).sum()):,}")

    plot_A_funnel(df)
    plot_B_star_radius(df)
    plot_C_heatmaps(df)
    plot_E_thresholds(df)

    print(f"\nSaved simplified figures to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
