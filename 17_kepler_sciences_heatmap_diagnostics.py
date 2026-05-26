# scripts/17_kepler_sciences_heatmap_diagnostics.py
#
# Cleaner Kepler science plots for P-Pop / Gaia C_F_K_combined runs.
# Main change from the messy scatter-D figure:
#   D is now heatmap-based, like B/C/E, instead of point clouds.

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists():
            return p
    return start_path.parents[1]


ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Change this if you are plotting the original non-experiment folder.
STAR_CATALOG_FOLDER = "Gaia_C_F_K_combined"
DATA_DIR = ROOT / "run" / "kepler" / "data" / STAR_CATALOG_FOLDER
OUT_DIR = ROOT / "my_outputs" / f"17_kepler_heatmap_diagnostics_{STAR_CATALOG_FOLDER}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MES_THRESHOLD = 7.1
MIN_BIN_COUNT = 8
RSUN_TO_REARTH = 109.076
RSTAR_XMAX_TARGET = 2.7  # show larger-host search space if present; does not fake data

PERIOD_BINS = np.logspace(np.log10(0.5), np.log10(500), 30)
PLANET_RADIUS_BINS = np.logspace(np.log10(0.3), np.log10(20), 28)
STAR_ORDER = ["F", "G", "K", "M"]
RADIUS_BINS = [0, 1.0, 1.5, 3.0, 6.0, np.inf]
RADIUS_LABELS = ["<1.0", "1.0-1.5", "1.5-3.0", "3.0-6.0", ">6.0"]

REASON_ORDER = [
    "not_transiting",
    "host_star_too_faint",
    "too_few_transits",
    "too_shallow_or_low_mes",
    "other_missed",
    "detected",
]


# ============================================================
# Load and prepare
# ============================================================

def load_catalog() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("kepler_catalog_*.csv"))
    if not files:
        raise FileNotFoundError(f"No kepler_catalog_*.csv files found in {DATA_DIR}")

    frames = []
    for p in files:
        df = pd.read_csv(p)
        if "run" not in df.columns:
            m = re.search(r"kepler_catalog_(\d+)\.csv", p.name)
            df["run"] = int(m.group(1)) if m else len(frames)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(df):,} rows from {DATA_DIR}")
    return df


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def clean_star_type(x) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().upper()
    return s[0] if s else "Unknown"


def ensure_reason_category(df: pd.DataFrame) -> pd.DataFrame:
    if "reason_category" in df.columns:
        return df

    reason = np.full(len(df), "other_missed", dtype=object)
    transiting = df["transiting_geometric"]
    bright = df["star_bright_enough"]
    enough = df["enough_transits"]
    detected = df["detected"]
    mes = pd.to_numeric(df["kepler_mes"], errors="coerce").fillna(0)

    reason[(~transiting).to_numpy()] = "not_transiting"
    reason[(transiting & ~bright).to_numpy()] = "host_star_too_faint"
    reason[(transiting & bright & ~enough).to_numpy()] = "too_few_transits"
    reason[(transiting & bright & enough & (mes < MES_THRESHOLD)).to_numpy()] = "too_shallow_or_low_mes"
    reason[detected.to_numpy()] = "detected"

    df["reason_category"] = reason
    df["miss_reason"] = reason
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "bright_enough_kepler" not in df.columns and "kepler_star_bright_enough" in df.columns:
        df["bright_enough_kepler"] = df["kepler_star_bright_enough"]
    if "kepler_enough_transits" not in df.columns and "n_transits_keplerish" in df.columns:
        df["kepler_enough_transits"] = df["n_transits_keplerish"] >= 3
    if "detected" not in df.columns and "detected_best" in df.columns:
        df["detected"] = df["detected_best"]
    if "stype" not in df.columns:
        df["stype"] = "Unknown"

    required = [
        "radius_p", "p_orb", "radius_s", "transiting_geometric", "detected",
        "bright_enough_kepler", "kepler_enough_transits", "kepler_mes",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [
        "radius_p", "p_orb", "radius_s", "kepler_mes", "kepler_mag_used",
        "kepler_cdpp_ppm", "transit_depth_ppm", "n_transits_keplerish",
        "kepler_dataspan_used_days", "kepler_dutycycle_used",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["detected"] = as_bool(df["detected"])
    df["transiting_geometric"] = as_bool(df["transiting_geometric"])
    df["star_bright_enough"] = as_bool(df["bright_enough_kepler"])
    df["enough_transits"] = as_bool(df["kepler_enough_transits"])
    df["stype_clean"] = df["stype"].apply(clean_star_type)

    df = ensure_reason_category(df)
    df = df.dropna(subset=["radius_p", "p_orb", "radius_s", "kepler_mes"]).copy()
    return df


# ============================================================
# Heatmap helpers
# ============================================================

def finite_bins(series, nbins=28, log=False, lower=None, upper=None):
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    if log:
        x = x[x > 0]
    if len(x) == 0:
        return np.logspace(0, 1, nbins + 1) if log else np.linspace(0, 1, nbins + 1)

    lo = float(x.min()) if lower is None else float(lower)
    hi = float(x.max()) if upper is None else max(float(upper), float(x.max()))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.logspace(np.log10(lo), np.log10(hi), nbins + 1) if log else np.linspace(lo, hi, nbins + 1)


def fraction_grid(df, xcol, ycol, xbins, ybins, numerator_mask, denominator_mask=None):
    d = df.copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    valid = np.isfinite(d[xcol]) & np.isfinite(d[ycol])
    d = d[valid]
    numerator_mask = pd.Series(numerator_mask, index=df.index).loc[d.index]
    denominator_mask = pd.Series(True, index=df.index).loc[d.index] if denominator_mask is None else pd.Series(denominator_mask, index=df.index).loc[d.index]

    total, _, _ = np.histogram2d(d.loc[denominator_mask, xcol], d.loc[denominator_mask, ycol], bins=[xbins, ybins])
    numerator, _, _ = np.histogram2d(d.loc[numerator_mask & denominator_mask, xcol], d.loc[numerator_mask & denominator_mask, ycol], bins=[xbins, ybins])
    frac = np.divide(numerator, total, out=np.full_like(numerator, np.nan, dtype=float), where=total > 0)
    frac[total < MIN_BIN_COUNT] = np.nan
    return frac.T


def plot_grid(ax, df, xcol, ycol, xbins, ybins, numerator_mask, denominator_mask, title, xlabel, ylabel, xlog=False, ylog=True):
    frac = fraction_grid(df, xcol, ycol, xbins, ybins, numerator_mask, denominator_mask)
    mesh = ax.pcolormesh(xbins, ybins, frac, shading="auto", vmin=0, vmax=1)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return mesh


def label_bars(ax, bars, denominator):
    ymax = max([b.get_height() for b in bars]) if len(bars) else 1
    for bar in bars:
        value = bar.get_height()
        pct = 100 * value / denominator if denominator > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.015*ymax,
                f"{int(value):,}\n{pct:.1f}%", ha="center", va="bottom", fontsize=8)


# ============================================================
# A. Funnel with reasons
# ============================================================

def plot_A_funnel_reason(df):
    stages = [
        ("All\nplanets", pd.Series(True, index=df.index)),
        ("Geometrically\ntransiting", df["transiting_geometric"]),
        ("Bright enough\nhost", df["transiting_geometric"] & df["star_bright_enough"]),
        ("Enough repeat\ntransits", df["transiting_geometric"] & df["star_bright_enough"] & df["enough_transits"]),
        ("MES >= 7.1", df["transiting_geometric"] & df["star_bright_enough"] & df["enough_transits"] & (df["kepler_mes"] >= MES_THRESHOLD)),
        ("Detected", df["detected"]),
    ]
    labels = [s[0] for s in stages]
    counts = np.array([int(s[1].sum()) for s in stages])

    fig, axes = plt.subplots(1, 3, figsize=(23, 6))
    bars = axes[0].bar(labels, counts)
    axes[0].set_title("A1. Absolute funnel")
    axes[0].set_ylabel("Number of planets")
    axes[0].tick_params(axis="x", rotation=25)
    label_bars(axes[0], bars, max(counts[0], 1))

    labels2, counts2 = labels[1:], counts[1:]
    bars = axes[1].bar(labels2, counts2)
    axes[1].set_title("A2. Conditional funnel: transiting stage onward")
    axes[1].set_ylabel("Number of planets")
    axes[1].tick_params(axis="x", rotation=25)
    label_bars(axes[1], bars, max(counts2[0], 1))

    reason_counts = df["reason_category"].value_counts().reindex(REASON_ORDER).dropna()
    bars = axes[2].bar(reason_counts.index, reason_counts.values)
    axes[2].set_title("A3. Final reason category")
    axes[2].set_ylabel("Number of planets")
    axes[2].tick_params(axis="x", rotation=30)
    label_bars(axes[2], bars, max(len(df), 1))

    fig.suptitle("A. Kepler detection funnel plus why planets were missed")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A_detection_funnel_with_reasons.png", dpi=250)
    plt.close(fig)


# ============================================================
# C. Detection heatmaps, with host-radius axis extended to 2.7 if possible
# ============================================================

def plot_C_detection_heatmaps(df):
    trans = df[df["transiting_geometric"] & (df["p_orb"] > 0) & (df["radius_p"] > 0) & (df["radius_s"] > 0)].copy()
    rstar_bins = finite_bins(trans["radius_s"], 28, lower=0.45, upper=RSTAR_XMAX_TARGET)

    if "kepler_cdpp_ppm" in trans.columns and trans["kepler_cdpp_ppm"].nunique(dropna=True) > 5:
        third_col, third_label, third_log = "kepler_cdpp_ppm", "CDPP noise [ppm]", True
    else:
        third_col, third_label, third_log = "kepler_mag_used", "Approx. Kepler magnitude", False

    panels = [
        ("p_orb", "Orbital period [days]", PERIOD_BINS, True, "C1. Radius vs period"),
        ("radius_s", "Host-star radius [R_sun]", rstar_bins, False, "C2. Radius vs host-star radius"),
        (third_col, third_label, finite_bins(trans[third_col], 28, log=third_log), third_log, "C3. Radius vs noise/brightness"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    mesh = None
    for ax, (xcol, xlabel, xbins, xlog, title) in zip(axes, panels):
        mesh = plot_grid(
            ax, trans, xcol, "radius_p", xbins, PLANET_RADIUS_BINS,
            numerator_mask=trans["detected"], denominator_mask=pd.Series(True, index=trans.index),
            title=title, xlabel=xlabel, ylabel="Planet radius [R_earth]", xlog=xlog, ylog=True,
        )
        if xcol == "radius_s":
            ax.set_xlim(rstar_bins[0], rstar_bins[-1])

    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Detected fraction")
    fig.suptitle("C. Kepler detection efficiency: which parameters control recovery?")
    fig.savefig(OUT_DIR / "C_detection_efficiency_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# D. Cleaner missed-reason heatmaps, replacing scatter
# ============================================================

def plot_D_reason_heatmaps(df):
    trans = df[df["transiting_geometric"] & (df["p_orb"] > 0) & (df["radius_p"] > 0) & (df["radius_s"] > 0)].copy()
    rstar_bins = finite_bins(trans["radius_s"], 28, lower=0.45, upper=RSTAR_XMAX_TARGET)

    if "kepler_cdpp_ppm" in trans.columns and trans["kepler_cdpp_ppm"].nunique(dropna=True) > 5:
        third_col, third_label, third_log = "kepler_cdpp_ppm", "CDPP noise [ppm]", True
    else:
        third_col, third_label, third_log = "kepler_mag_used", "Approx. Kepler magnitude", False

    panels = [
        ("p_orb", "Orbital period [days]", PERIOD_BINS, True, "period"),
        ("radius_s", "Host-star radius [R_sun]", rstar_bins, False, "host radius"),
        (third_col, third_label, finite_bins(trans[third_col], 28, log=third_log), third_log, "noise/brightness"),
    ]

    low_mes = trans["reason_category"].astype(str).eq("too_shallow_or_low_mes")
    detected = trans["detected"]
    denom = pd.Series(True, index=trans.index)

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    for j, (xcol, xlabel, xbins, xlog, short) in enumerate(panels):
        mesh1 = plot_grid(
            axes[0, j], trans, xcol, "radius_p", xbins, PLANET_RADIUS_BINS,
            numerator_mask=low_mes, denominator_mask=denom,
            title=f"D{j+1}. Low-MES missed fraction vs {short}",
            xlabel=xlabel, ylabel="Planet radius [R_earth]", xlog=xlog, ylog=True,
        )
        mesh2 = plot_grid(
            axes[1, j], trans, xcol, "radius_p", xbins, PLANET_RADIUS_BINS,
            numerator_mask=detected, denominator_mask=denom,
            title=f"D{j+4}. Detected fraction vs {short}",
            xlabel=xlabel, ylabel="Planet radius [R_earth]", xlog=xlog, ylog=True,
        )
        if xcol == "radius_s":
            axes[0, j].set_xlim(rstar_bins[0], rstar_bins[-1])
            axes[1, j].set_xlim(rstar_bins[0], rstar_bins[-1])

    fig.colorbar(mesh1, ax=axes[0, :].ravel().tolist(), label="Fraction of transiting planets missed by low MES")
    fig.colorbar(mesh2, ax=axes[1, :].ravel().tolist(), label="Detected fraction among transiting planets")
    fig.suptitle("D. Why missed? Heatmaps are binned by reason instead of messy point clouds")
    fig.savefig(OUT_DIR / "D_miss_reason_heatmaps.png", dpi=250, bbox_inches="tight")
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
    codes = pd.qcut(df[source_col].rank(method="first"), 3, labels=False, duplicates="drop")
    df[new_col] = codes.map({0: labels[0], 1: labels[1], 2: labels[2]})
    return df


def draw_threshold_panel(ax, trans, groups, title):
    missed = trans[trans["transiting_geometric"] & ~trans["detected"]]
    detected = trans[trans["detected"]]
    ax.scatter(missed["p_orb"], missed["radius_p"], s=8, alpha=0.25, label=f"Transiting missed ({len(missed):,})")
    ax.scatter(detected["p_orb"], detected["radius_p"], s=8, alpha=0.18, label=f"Detected ({len(detected):,})")
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
    trans = df[df["transiting_geometric"] & (df["p_orb"] > 0) & (df["radius_p"] > 0) & (df["radius_s"] > 0)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    star_types = [s for s in STAR_ORDER if (trans["stype_clean"] == s).sum() >= 10]
    draw_threshold_panel(axes[0], trans, [(s, trans[trans["stype_clean"] == s]) for s in star_types], "E1. Threshold by host-star type")

    trans = add_tercile(trans, "radius_s", "rstar_bin", ["small R_star", "medium R_star", "large R_star"])
    draw_threshold_panel(axes[1], trans, [(x, trans[trans["rstar_bin"] == x]) for x in ["small R_star", "medium R_star", "large R_star"]], "E2. Threshold by stellar radius")

    if "kepler_cdpp_ppm" in trans.columns and trans["kepler_cdpp_ppm"].nunique(dropna=True) > 5:
        trans = add_tercile(trans, "kepler_cdpp_ppm", "noise_bin", ["low noise", "medium noise", "high noise"])
        groups = [(x, trans[trans["noise_bin"] == x]) for x in ["low noise", "medium noise", "high noise"]]
        title = "E3. Threshold by CDPP noise"
    else:
        trans = add_tercile(trans, "kepler_mag_used", "brightness_bin", ["bright hosts", "medium hosts", "faint hosts"])
        groups = [(x, trans[trans["brightness_bin"] == x]) for x in ["bright hosts", "medium hosts", "faint hosts"]]
        title = "E3. Threshold by host brightness"
    draw_threshold_panel(axes[2], trans, groups, title)

    fig.suptitle("E. Approximate minimum detectable radius: above curve is easier for Kepler")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E_min_detectable_radius_cdpp.png", dpi=250)
    plt.close(fig)


# ============================================================
# Diagnostics and main
# ============================================================

def save_tables(df):
    df["reason_category"].value_counts().rename_axis("reason_category").reset_index(name="count").to_csv(
        OUT_DIR / "reason_category_counts.csv", index=False
    )
    summary = {
        "rows": len(df),
        "transiting": int(df["transiting_geometric"].sum()),
        "detected": int(df["detected"].sum()),
        "max_radius_s": float(pd.to_numeric(df["radius_s"], errors="coerce").max()),
        "n_radius_s_gt_2p2": int((pd.to_numeric(df["radius_s"], errors="coerce") > 2.2).sum()),
        "n_radius_s_gt_2p5": int((pd.to_numeric(df["radius_s"], errors="coerce") > 2.5).sum()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "catalog_summary.csv", index=False)


def print_large_star_diagnostic(df):
    r = pd.to_numeric(df["radius_s"], errors="coerce")
    print("\nHost-star radius diagnostic:")
    print(f"  max radius_s = {r.max():.3f} R_sun")
    print(f"  rows with radius_s > 2.2 = {(r > 2.2).sum():,}")
    print(f"  rows with radius_s > 2.5 = {(r > 2.5).sum():,}")
    if r.max() < 2.5:
        print("  Note: no actual generated Gaia/P-Pop stars above 2.5 R_sun in this run.")
        print("  The C2 axis now extends to 2.7, but empty bins mean the generator did not produce such stars.")


def main():
    df = prepare_data(load_catalog())
    print_large_star_diagnostic(df)
    save_tables(df)

    print("\nReason counts:")
    print(df["reason_category"].value_counts())

    plot_A_funnel_reason(df)
    plot_C_detection_heatmaps(df)
    plot_D_reason_heatmaps(df)
    plot_E_thresholds(df)

    print(f"\nSaved figures/tables to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
