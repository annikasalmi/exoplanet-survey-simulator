# scripts/16_kepler_sciences_nasa_pscomppars.py
#
# Purpose:
#   Run/read the Kepler toy model applied to the official NASA PSCompPars exoplanets,
#   then make science diagnostic figures similar to your original 16 script.
#
# Input raw NASA file:
#   run/kepler/data/NASA/NASA_PSCompPars_transiting_confirmed_RM_insolation.csv
#
# Model output file:
#   run/kepler/data/NASA/kepler_catalog_nasa_pscomppars.csv
#
# Caveman:
#   Original 16 = graphs P-Pop fake planets.
#   This 16 = graphs NASA official planets after the same KeplerData detector.

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths and settings
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

from lifesim.core.kepler_data import KeplerData  # noqa: E402


NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
NASA_RAW_CSV = NASA_DATA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
NASA_MODEL_CSV = NASA_DATA_DIR / "kepler_catalog_nasa_pscomppars.csv"

OUT_DIR = ROOT / "my_outputs" / "17_w3_nasa_pscomppars_kepler_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_MODEL_IF_NEEDED = True
FORCE_RERUN_MODEL = False

# Optional: keep all official NASA exoplanets from your file.
# Set to "Kepler" if you only want NASA rows discovered by Kepler.
FILTER_DISC_FACILITY = None
# FILTER_DISC_FACILITY = "Kepler"

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
# Model runner for NASA PSCompPars
# ============================================================

def run_model_on_nasa_pscomppars() -> pd.DataFrame:
    if not NASA_RAW_CSV.exists():
        raise FileNotFoundError(
            f"Could not find NASA raw CSV:\n{NASA_RAW_CSV}\n\n"
            "Put NASA_PSCompPars_transiting_confirmed_RM_insolation.csv in run/kepler/data/NASA/."
        )

    print("Running KeplerData model on NASA PSCompPars raw file:")
    print(NASA_RAW_CSV)

    raw = pd.read_csv(NASA_RAW_CSV)
    print(f"Raw NASA rows: {len(raw):,}")

    if FILTER_DISC_FACILITY is not None and "disc_facility" in raw.columns:
        before = len(raw)
        raw = raw[raw["disc_facility"].astype(str) == FILTER_DISC_FACILITY].copy()
        print(f"Filtered disc_facility == {FILTER_DISC_FACILITY!r}: {before:,} -> {len(raw):,}")

    model = KeplerData(
        raw,
        source="pscomppars",
        validate_for_detection=True,
        use_observed_transit_flag_for_nasa=True,
        use_observed_transit_depth_for_nasa=True,
        assume_bright_if_kepmag_missing_for_nasa=True,
    )
    df = model.determine_detectable()

    df["run"] = 0
    df["model_input_file"] = NASA_RAW_CSV.name
    df["model_population"] = "NASA_PSCompPars_transiting_confirmed_RM_insolation"

    df["radius_bin"] = pd.cut(
        df["radius_p"],
        bins=[0, 1.5, 3.0, 6.0, np.inf],
        labels=["<1.5", "1.5–3.0", "3.0–6.0", ">6.0"],
        include_lowest=True,
    )

    NASA_MODEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(NASA_MODEL_CSV, index=False)

    print("Saved model output:")
    print(NASA_MODEL_CSV)
    return df


# ============================================================
# Loading and cleaning
# ============================================================

def load_catalog():
    if FORCE_RERUN_MODEL or not NASA_MODEL_CSV.exists():
        if not RUN_MODEL_IF_NEEDED:
            raise FileNotFoundError(
                f"Model output not found:\n{NASA_MODEL_CSV}\n\n"
                "Run run_Kepler_nasa_pscomppars.py first, or set RUN_MODEL_IF_NEEDED = True."
            )
        df = run_model_on_nasa_pscomppars()
    else:
        print("Loading existing Kepler-on-NASA model output:")
        print(NASA_MODEL_CSV)
        df = pd.read_csv(NASA_MODEL_CSV)

    print(f"Loaded rows: {len(df):,}")
    return df


def as_bool(series):
    """Safe bool conversion: avoids bool('False') == True."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def clean_star_type(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().upper()
    return s[0] if s else "Unknown"


def infer_star_type_from_teff(teff):
    teff = pd.to_numeric(teff, errors="coerce")
    stype = np.full(len(teff), "Unknown", dtype=object)
    stype[(teff >= 7500)] = "A"
    stype[(teff >= 6000) & (teff < 7500)] = "F"
    stype[(teff >= 5200) & (teff < 6000)] = "G"
    stype[(teff >= 3700) & (teff < 5200)] = "K"
    stype[(teff > 0) & (teff < 3700)] = "M"
    return stype


def prepare_data(df):
    df = df.copy()

    # Backward/forward compatibility with different KeplerData versions.
    if "bright_enough_kepler" not in df.columns and "kepler_star_bright_enough" in df.columns:
        df["bright_enough_kepler"] = df["kepler_star_bright_enough"]

    if "kepler_depth_good" not in df.columns and "kepler_depth_pass" in df.columns:
        df["kepler_depth_good"] = df["kepler_depth_pass"]

    if "n_transits_keplerish" not in df.columns:
        df["n_transits_keplerish"] = np.floor((4 * 365.25) / pd.to_numeric(df["p_orb"], errors="coerce"))

    if "kepler_enough_transits" not in df.columns:
        df["kepler_enough_transits"] = df["n_transits_keplerish"] >= 3

    if "stype" not in df.columns:
        if "temp_s" in df.columns:
            df["stype"] = infer_star_type_from_teff(df["temp_s"])
        elif "st_teff" in df.columns:
            df["stype"] = infer_star_type_from_teff(df["st_teff"])
        else:
            df["stype"] = "Unknown"

    if "transiting_geometric" not in df.columns:
        if "tran_flag" in df.columns:
            df["transiting_geometric"] = pd.to_numeric(df["tran_flag"], errors="coerce").fillna(0).astype(int) == 1
        else:
            raise ValueError("Missing transiting_geometric and tran_flag.")

    if "detected" not in df.columns and "detected_best" in df.columns:
        df["detected"] = df["detected_best"]

    required = [
        "radius_p", "p_orb", "radius_s", "stype",
        "transiting_geometric", "detected",
        "kepler_mes", "n_transits_keplerish", "transit_depth_ppm",
        "bright_enough_kepler", "kepler_depth_good", "kepler_enough_transits",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after preparation: {missing}\nAvailable: {df.columns.tolist()}")

    numeric_cols = [
        "radius_p", "p_orb", "radius_s", "mass_p", "flux_p",
        "kepler_mes", "n_transits_keplerish", "transit_depth_ppm",
        "kepler_cdpp_ppm", "kepler_mag_used",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["radius_p", "p_orb", "radius_s", "kepler_mes"]).copy()

    df["stype_clean"] = df["stype"].apply(clean_star_type)
    df["radius_bin"] = pd.cut(
        df["radius_p"],
        bins=RADIUS_BINS,
        labels=RADIUS_LABELS,
        include_lowest=True,
    )

    df["detected"] = as_bool(df["detected"])
    df["transiting_geometric"] = as_bool(df["transiting_geometric"])
    df["star_bright_enough"] = as_bool(df["bright_enough_kepler"])
    df["enough_transits"] = as_bool(df["kepler_enough_transits"])
    df["depth_good"] = as_bool(df["kepler_depth_good"])

    return df


def add_reason_category(df):
    df = df.copy()

    reason = np.full(len(df), "other_missed", dtype=object)

    # Assign missed reasons first, then detected last.
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
    reason[df["detected"].to_numpy()] = "detected"

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
    if len(x) == 0:
        return np.linspace(0, 1, nbins + 1)
    if x.min() == x.max():
        return np.linspace(x.min() - 0.5, x.max() + 0.5, nbins + 1)
    return np.linspace(x.min(), x.max(), nbins + 1)


def log_bins(series, nbins):
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.logspace(0, 1, nbins + 1)
    return np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)


# ============================================================
# A. Funnel
# ============================================================

def plot_A_funnel(df):
    stages = [
        ("All NASA\nplanets", pd.Series(True, index=df.index)),
        ("Marked\ntransiting", df["transiting_geometric"]),
        ("Bright enough\nhost", df["transiting_geometric"] & df["star_bright_enough"]),
        ("Enough\nrepeat transits", df["transiting_geometric"] & df["star_bright_enough"] & df["enough_transits"]),
        ("MES >= 7.1", df["transiting_geometric"] & df["star_bright_enough"] & df["enough_transits"] & (df["kepler_mes"] >= MES_THRESHOLD)),
        ("Toy Kepler\npass", df["detected"]),
    ]

    labels = [name for name, _ in stages]
    counts = np.array([int(mask.sum()) for _, mask in stages])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    bars = axes[0].bar(labels, counts)
    axes[0].set_title("A1. Absolute funnel: NASA PSCompPars through toy Kepler")
    axes[0].set_ylabel("Number of planets")
    axes[0].tick_params(axis="x", rotation=25)
    label_bars(axes[0], bars, denominator=max(counts[0], 1))

    labels2 = labels[1:]
    counts2 = counts[1:]
    bars = axes[1].bar(labels2, counts2)
    axes[1].set_title("A2. Conditional funnel: known transiting planets")
    axes[1].set_ylabel("Number of NASA transiting planets")
    axes[1].tick_params(axis="x", rotation=25)
    label_bars(axes[1], bars, denominator=max(counts2[0], 1))

    fig.suptitle("A. Official NASA exoplanets passed through your Kepler toy detector")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "A_nasa_detection_funnel.png", dpi=250)
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
    ax.set_ylabel("Number of NASA transiting planets")
    ax.set_title("B1. Counts by star type and planet radius")
    ax.legend(title="Radius bin", fontsize=8)

    im = axes[1].imshow(frac.to_numpy(float), vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("B2. Toy-Kepler pass fraction")
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
    fig.colorbar(im, ax=axes[1], label="Toy Kepler pass fraction")

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

    fig.suptitle("B. NASA planets through toy Kepler: host type, radius, and signal strength")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "B_nasa_star_type_radius.png", dpi=250)
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

    if "kepler_mag_used" in trans.columns and trans["kepler_mag_used"].nunique(dropna=True) > 5:
        third = ("kepler_mag_used", "Kepler magnitude / proxy", linear_bins(trans["kepler_mag_used"], 26), False, "C3. Radius vs host brightness")
    elif "kepler_cdpp_ppm" in trans.columns and trans["kepler_cdpp_ppm"].nunique(dropna=True) > 5:
        third = ("kepler_cdpp_ppm", "Kepler CDPP [ppm]", log_bins(trans["kepler_cdpp_ppm"], 26), True, "C3. Radius vs photometric noise")
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

    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Toy Kepler pass fraction")
    fig.suptitle("C. Toy Kepler pass fraction across official NASA planets")
    fig.savefig(OUT_DIR / "C_nasa_detection_efficiency.png", dpi=250, bbox_inches="tight")
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
    missed = trans[~trans["detected"]]
    detected = trans[trans["detected"]]

    ax.scatter(missed["p_orb"], missed["radius_p"], s=8, alpha=0.25,
               label=f"Toy missed ({len(missed):,})")
    ax.scatter(detected["p_orb"], detected["radius_p"], s=8, alpha=0.18,
               label=f"Toy pass ({len(detected):,})")

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

    groups = [(s, trans[trans["stype_clean"] == s]) for s in STAR_ORDER]
    draw_threshold_panel(axes[0], trans, groups, "E1. Threshold by host-star type")

    trans = add_tercile(
        trans, "radius_s", "rstar_bin",
        ["small R_star", "medium R_star", "large R_star"]
    )
    groups = [(label, trans[trans["rstar_bin"] == label])
              for label in ["small R_star", "medium R_star", "large R_star"]]
    draw_threshold_panel(axes[1], trans, groups, "E2. Threshold by stellar radius")

    if "kepler_mag_used" in trans.columns and trans["kepler_mag_used"].nunique(dropna=True) > 5:
        trans = add_tercile(
            trans, "kepler_mag_used", "brightness_bin",
            ["bright hosts", "medium hosts", "faint hosts"]
        )
        groups = [(label, trans[trans["brightness_bin"] == label])
                  for label in ["bright hosts", "medium hosts", "faint hosts"]]
        title = "E3. Threshold by host brightness"
    elif "kepler_cdpp_ppm" in trans.columns:
        trans = add_tercile(
            trans, "kepler_cdpp_ppm", "noise_bin",
            ["low noise", "medium noise", "high noise"]
        )
        groups = [(label, trans[trans["noise_bin"] == label])
                  for label in ["low noise", "medium noise", "high noise"]]
        title = "E3. Threshold by photometric noise"
    else:
        groups = []
        title = "E3. Brightness/noise unavailable"

    draw_threshold_panel(axes[2], trans, groups, title)

    fig.suptitle("E. Approximate minimum detectable radius for official NASA planets")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E_nasa_min_detectable_radius.png", dpi=250)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def save_summary_tables(df):
    counts = df["reason_category"].value_counts().rename_axis("reason_category").reset_index(name="count")
    counts.to_csv(OUT_DIR / "nasa_reason_category_counts.csv", index=False)

    source_cols = [
        "kepler_mag_source", "transit_depth_source", "kepler_cdpp_source",
        "transiting_source", "discovery_facility", "dataset_source",
    ]
    summaries = []
    for col in source_cols:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).reset_index()
            vc.columns = ["value", "count"]
            vc.insert(0, "column", col)
            summaries.append(vc)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(OUT_DIR / "nasa_source_diagnostics.csv", index=False)


def main():
    df = load_catalog()
    df = prepare_data(df)
    df = add_reason_category(df)
    save_summary_tables(df)

    print("\nReason counts:")
    print(df["reason_category"].value_counts())

    print("\nCore summary:")
    print(f"All NASA planets: {len(df):,}")
    print(f"Marked transiting: {int(df['transiting_geometric'].sum()):,}")
    print(f"Toy Kepler pass/detected: {int(df['detected'].sum()):,}")
    print(f"Toy Kepler missed: {int((df['transiting_geometric'] & ~df['detected']).sum()):,}")

    print("\nImportant diagnostics:")
    for col in ["kepler_mag_source", "transit_depth_source", "kepler_cdpp_source"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).head(10))

    plot_A_funnel(df)
    plot_B_star_radius(df)
    plot_C_heatmaps(df)
    plot_E_thresholds(df)

    print(f"\nSaved NASA Kepler-model science figures to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
