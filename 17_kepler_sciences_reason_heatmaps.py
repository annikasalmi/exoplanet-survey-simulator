# scripts/17_kepler_sciences_reason_heatmaps.py
#
# Cleaner D panels: binned heatmaps instead of messy point clouds.
# Also fixes C2/E2 host-radius plotting range so it can show up to 2.7 R_sun.

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists():
            return p
    return start_path.parents[1]


ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifesim.core.kepler_data import KeplerData  # noqa: E402

INPUT_MODE = "ppop"  # "ppop" or "nasa"
STAR_CATALOG_FOLDER = "Gaia_C_F_K_combined"
RERUN_DETECTOR_ON_LOAD = False

PPOP_DIR = ROOT / "run" / "kepler" / "data" / STAR_CATALOG_FOLDER
PPOP_GLOB = "kepler_catalog_*.csv"

NASA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
NASA_RAW_CSV = NASA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation.csv"
NASA_MODEL_CSV = NASA_DIR / "kepler_catalog_nasa_pscomppars.csv"

OUT_DIR = ROOT / "my_outputs" / f"17_kepler_reason_heatmaps_{INPUT_MODE}_{STAR_CATALOG_FOLDER}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MES_THRESHOLD = 7.1
RSUN_TO_REARTH = 109.076
MIN_BIN_COUNT = 8
HOST_RADIUS_XMAX = 2.7

PERIOD_BINS = np.logspace(np.log10(0.5), np.log10(500), 30)
PLANET_RADIUS_BINS = np.logspace(np.log10(0.3), np.log10(20), 28)
HOST_RADIUS_BINS = np.linspace(0.4, HOST_RADIUS_XMAX, 32)

REASON_ORDER = [
    "detected",
    "too_shallow_or_low_mes",
    "too_few_transits",
    "host_star_too_faint",
    "not_transiting",
    "other_missed",
]
REASON_LABELS = {
    "detected": "detected",
    "too_shallow_or_low_mes": "low MES",
    "too_few_transits": "too few transits",
    "host_star_too_faint": "host too faint",
    "not_transiting": "not transiting",
    "other_missed": "other missed",
}


def load_ppop_catalog() -> pd.DataFrame:
    files = sorted(PPOP_DIR.glob(PPOP_GLOB))
    if not files:
        raise FileNotFoundError(f"No P-Pop Kepler CSVs found in {PPOP_DIR} with pattern {PPOP_GLOB}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "run" not in df.columns:
            m = re.search(r"kepler_catalog_(\d+)\.csv$", path.name)
            df["run"] = int(m.group(1)) if m else len(frames)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(df):,} rows from {PPOP_DIR}")
    return df


def load_nasa_catalog() -> pd.DataFrame:
    if RERUN_DETECTOR_ON_LOAD or not NASA_MODEL_CSV.exists():
        raw = pd.read_csv(NASA_RAW_CSV)
        df = KeplerData(raw, source="pscomppars").determine_detectable()
        df.to_csv(NASA_MODEL_CSV, index=False)
    else:
        df = pd.read_csv(NASA_MODEL_CSV)
    return df


def load_catalog() -> pd.DataFrame:
    df = load_ppop_catalog() if INPUT_MODE == "ppop" else load_nasa_catalog()
    if RERUN_DETECTOR_ON_LOAD:
        source = "ppop" if INPUT_MODE == "ppop" else "pscomppars"
        df = KeplerData(df, source=source).determine_detectable()
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
    if "kepler_depth_good" not in df.columns and "kepler_depth_pass" in df.columns:
        df["kepler_depth_good"] = df["kepler_depth_pass"]
    if "detected" not in df.columns and "detected_best" in df.columns:
        df["detected"] = df["detected_best"]
    if "kepler_enough_transits" not in df.columns:
        df["kepler_enough_transits"] = pd.to_numeric(df["n_transits_keplerish"], errors="coerce") >= 3
    if "stype" not in df.columns:
        df["stype"] = "Unknown"

    required = ["radius_p", "p_orb", "radius_s", "transiting_geometric", "detected", "bright_enough_kepler", "kepler_enough_transits", "kepler_mes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["radius_p", "p_orb", "radius_s", "kepler_mes", "kepler_mag_used", "kepler_cdpp_ppm", "transit_depth_ppm", "n_transits_keplerish"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["detected"] = as_bool(df["detected"])
    df["transiting_geometric"] = as_bool(df["transiting_geometric"])
    df["star_bright_enough"] = as_bool(df["bright_enough_kepler"])
    df["enough_transits"] = as_bool(df["kepler_enough_transits"])
    df["stype_clean"] = df["stype"].apply(clean_star_type)
    df = ensure_reason_category(df)
    return df.dropna(subset=["radius_p", "p_orb", "radius_s", "kepler_mes"]).copy()


def label_bars(ax, bars, denominator):
    ymax = max([b.get_height() for b in bars]) if len(bars) else 1
    for bar in bars:
        value = bar.get_height()
        pct = 100 * value / denominator if denominator > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.015*ymax, f"{int(value):,}\n{pct:.1f}%", ha="center", va="bottom", fontsize=8)


def detection_grid(df, xcol, ycol, xbins, ybins):
    d = df.copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    total, _, _ = np.histogram2d(d[xcol], d[ycol], bins=[xbins, ybins])
    detected, _, _ = np.histogram2d(d.loc[d["detected"], xcol], d.loc[d["detected"], ycol], bins=[xbins, ybins])
    frac = np.divide(detected, total, out=np.full_like(detected, np.nan, dtype=float), where=total > 0)
    frac[total < MIN_BIN_COUNT] = np.nan
    return frac.T


def finite_bins(series, nbins=28, log=False):
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    if log:
        x = x[x > 0]
    if len(x) == 0:
        return np.logspace(0, 1, nbins + 1) if log else np.linspace(0, 1, nbins + 1)
    if x.min() == x.max():
        return np.linspace(x.min() - 0.5, x.max() + 0.5, nbins + 1)
    return np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1) if log else np.linspace(x.min(), x.max(), nbins + 1)


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
    bars = axes[1].bar(labels[1:], counts[1:])
    axes[1].set_title("A2. Conditional funnel: transiting stage onward")
    axes[1].set_ylabel("Number of planets")
    axes[1].tick_params(axis="x", rotation=25)
    label_bars(axes[1], bars, max(counts[1], 1))
    order = ["not_transiting", "host_star_too_faint", "too_few_transits", "too_shallow_or_low_mes", "detected", "other_missed"]
    reason_counts = df["reason_category"].value_counts().reindex(order).dropna()
    bars = axes[2].bar(reason_counts.index, reason_counts.values)
    axes[2].set_title("A3. Final reason category")
    axes[2].set_ylabel("Number of planets")
    axes[2].tick_params(axis="x", rotation=30)
    label_bars(axes[2], bars, max(len(df), 1))
    fig.suptitle("A. Kepler detection funnel plus why planets were missed")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A_detection_funnel_with_reasons.png", dpi=250)
    plt.close(fig)


def plot_C_detection_heatmaps(df):
    trans = df[df["transiting_geometric"] & (df["p_orb"] > 0) & (df["radius_p"] > 0) & (df["radius_s"] > 0)].copy()
    third_col = "kepler_cdpp_ppm" if "kepler_cdpp_ppm" in trans.columns and trans["kepler_cdpp_ppm"].nunique(dropna=True) > 5 else "kepler_mag_used"
    third_label = "CDPP noise [ppm]" if third_col == "kepler_cdpp_ppm" else "Approx. Kepler magnitude"
    third_log = third_col == "kepler_cdpp_ppm"
    panels = [
        ("p_orb", "Orbital period [days]", PERIOD_BINS, True, "C1. Radius vs period"),
        ("radius_s", "Host-star radius [R_sun]", HOST_RADIUS_BINS, False, "C2. Radius vs host-star radius"),
        (third_col, third_label, finite_bins(trans[third_col], 28, log=third_log), third_log, "C3. Radius vs noise/brightness"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    mesh = None
    for ax, (xcol, xlabel, xbins, xlog, title) in zip(axes, panels):
        frac = detection_grid(trans, xcol, "radius_p", xbins, PLANET_RADIUS_BINS)
        mesh = ax.pcolormesh(xbins, PLANET_RADIUS_BINS, frac, shading="auto", vmin=0, vmax=1)
        if xlog:
            ax.set_xscale("log")
        ax.set_yscale("log")
        if xcol == "radius_s":
            ax.set_xlim(HOST_RADIUS_BINS[0], HOST_RADIUS_BINS[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Planet radius [R_earth]")
        ax.set_title(title)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Detected fraction")
    fig.suptitle("C. Kepler detection efficiency: which parameters control recovery?")
    fig.savefig(OUT_DIR / "C_detection_efficiency_cdpp.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def dominant_reason_grid(df, xcol, ycol, xbins, ybins, transiting_only=True):
    d = df.copy()
    if transiting_only:
        d = d[d["transiting_geometric"]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    out = np.full((len(ybins)-1, len(xbins)-1), np.nan)
    for ix in range(len(xbins)-1):
        xmask = (d[xcol] >= xbins[ix]) & (d[xcol] < xbins[ix+1])
        for iy in range(len(ybins)-1):
            cell = d[xmask & (d[ycol] >= ybins[iy]) & (d[ycol] < ybins[iy+1])]
            if len(cell) < MIN_BIN_COUNT:
                continue
            counts = cell["reason_category"].value_counts()
            for code, reason in enumerate(REASON_ORDER):
                if reason in counts.index and counts.idxmax() == reason:
                    out[iy, ix] = code
                    break
    return out


def reason_fraction_grid(df, xcol, ycol, xbins, ybins, reason="too_shallow_or_low_mes"):
    d = df[df["transiting_geometric"]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    total, _, _ = np.histogram2d(d[xcol], d[ycol], bins=[xbins, ybins])
    selected = d[d["reason_category"] == reason]
    count, _, _ = np.histogram2d(selected[xcol], selected[ycol], bins=[xbins, ybins])
    frac = np.divide(count, total, out=np.full_like(count, np.nan, dtype=float), where=total > 0)
    frac[total < MIN_BIN_COUNT] = np.nan
    return frac.T


def plot_D_reason_heatmaps(df):
    trans = df[df["transiting_geometric"] & (df["radius_p"] > 0)].copy()
    third_col = "kepler_cdpp_ppm" if "kepler_cdpp_ppm" in trans.columns and trans["kepler_cdpp_ppm"].nunique(dropna=True) > 5 else "kepler_mag_used"
    third_label = "CDPP noise [ppm]" if third_col == "kepler_cdpp_ppm" else "Approx. Kepler magnitude"
    third_log = third_col == "kepler_cdpp_ppm"
    panels = [
        ("p_orb", "Orbital period [days]", PERIOD_BINS, True, "D1. Radius vs period"),
        ("radius_s", "Host-star radius [R_sun]", HOST_RADIUS_BINS, False, "D2. Radius vs host radius"),
        (third_col, third_label, finite_bins(trans[third_col], 28, log=third_log), third_log, "D3. Radius vs noise/brightness"),
    ]
    cmap = ListedColormap(["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#9ecae1", "#7f7f7f"])
    norm = BoundaryNorm(np.arange(-0.5, len(REASON_ORDER)+0.5, 1), cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(23, 11))
    last_mesh = None
    for j, (xcol, xlabel, xbins, xlog, title) in enumerate(panels):
        dom = dominant_reason_grid(trans, xcol, "radius_p", xbins, PLANET_RADIUS_BINS, transiting_only=True)
        axes[0, j].pcolormesh(xbins, PLANET_RADIUS_BINS, dom, cmap=cmap, norm=norm, shading="auto")
        lowmes = reason_fraction_grid(trans, xcol, "radius_p", xbins, PLANET_RADIUS_BINS, reason="too_shallow_or_low_mes")
        last_mesh = axes[1, j].pcolormesh(xbins, PLANET_RADIUS_BINS, lowmes, shading="auto", vmin=0, vmax=1)
        for ax in [axes[0, j], axes[1, j]]:
            if xlog:
                ax.set_xscale("log")
            ax.set_yscale("log")
            if xcol == "radius_s":
                ax.set_xlim(HOST_RADIUS_BINS[0], HOST_RADIUS_BINS[-1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Planet radius [R_earth]")
        axes[0, j].set_title(title + "\nDominant outcome, transiting planets only")
        axes[1, j].set_title(title + "\nFraction missed by low MES")
    legend_handles = [Patch(facecolor=cmap(i), label=REASON_LABELS[r]) for i, r in enumerate(REASON_ORDER)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9)
    fig.colorbar(last_mesh, ax=axes[1, :].ravel().tolist(), label="Low-MES missed fraction")
    fig.suptitle("D. Missed-reason heatmaps: cleaner binned view, not point clouds")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(OUT_DIR / "D_miss_reason_heatmaps.png", dpi=250)
    plt.close(fig)


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
    star_types = [s for s in ["F", "G", "K", "M"] if (trans["stype_clean"] == s).sum() >= 10]
    draw_threshold_panel(axes[0], trans, [(s, trans[trans["stype_clean"] == s]) for s in star_types], "E1. Threshold by host-star type")
    trans = add_tercile(trans, "radius_s", "rstar_bin", ["small R_star", "medium R_star", "large R_star"])
    draw_threshold_panel(axes[1], trans, [(x, trans[trans["rstar_bin"] == x]) for x in ["small R_star", "medium R_star", "large R_star"]], "E2. Threshold by stellar radius")
    axes[1].set_xlim(0.5, HOST_RADIUS_XMAX)
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


def save_tables(df):
    df["reason_category"].value_counts().rename_axis("reason_category").reset_index(name="count").to_csv(OUT_DIR / "reason_category_counts.csv", index=False)
    pd.DataFrame({
        "summary": ["rows", "max_radius_s", "transiting", "detected"],
        "value": [len(df), df["radius_s"].max(), int(df["transiting_geometric"].sum()), int(df["detected"].sum())],
    }).to_csv(OUT_DIR / "quick_summary.csv", index=False)


def main():
    df = prepare_data(load_catalog())
    save_tables(df)
    print("Reason counts:")
    print(df["reason_category"].value_counts())
    print(f"Max host radius in loaded data: {df['radius_s'].max():.3f} R_sun")
    plot_A_funnel_reason(df)
    plot_C_detection_heatmaps(df)
    plot_D_reason_heatmaps(df)
    plot_E_thresholds(df)
    print(f"Saved figures to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
