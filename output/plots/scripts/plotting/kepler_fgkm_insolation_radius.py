# scripts/28_w4_kepler_fgkm_insolation_radius_detected_only.py
#
# FGKM split of Kepler v1 Insolation-Radius validation — detected fraction only.
#
# Outputs:
#   - One 1x4 figure: F, G, K, M host stars; color = detected fraction.
#   - One support-count figure.
#   - One CSV summary of NASA overlay bin values by host type.
#
# Scientific rule:
#   This validates the Kepler detector, so it uses Kepler-run P-Pop catalogs.
#   Do not mix TESS detected columns into a Kepler validation plot unless you
#   rerun KeplerData on the TESS-generated planet population first.

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists():
            return p
    return start_path.parents[2]


ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAR_CATALOG_FOLDER = "Gaia_C_F_K_combined"

PPOP_DATA_DIR = ROOT / "run" / "kepler" / "data" / STAR_CATALOG_FOLDER
NASA_MODEL_CSV = ROOT / "run" / "kepler" / "data" / "NASA" / "kepler_catalog_nasa_pscomppars.csv"

OUT_DIR = ROOT / "output/plots" / f"52_kepler_fgkm_insolation_radius_detected_only_{STAR_CATALOG_FOLDER}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAR_ORDER = ["F", "G", "K", "M"]

CASES = {
    "optimistic": {"mes_threshold": 6.5, "cdpp_scale": 0.8},
    "baseline": {"mes_threshold": 7.1, "cdpp_scale": 1.0},
    "conservative": {"mes_threshold": 7.7, "cdpp_scale": 1.2},
}
RUN_CASES = ["baseline"]

NASA_DISC_FACILITY_FILTER = "Kepler"
NASA_TRAN_FLAG_ONLY = True

RADIUS_LIMITS = (0.5, 4.0)
INSOLATION_LIMITS = (0.1, 1e5)

INSOLATION_BINS = np.logspace(np.log10(INSOLATION_LIMITS[0]), np.log10(INSOLATION_LIMITS[1]), 30)
PLANET_RADIUS_BINS = np.logspace(np.log10(RADIUS_LIMITS[0]), np.log10(RADIUS_LIMITS[1]), 20)

MIN_BIN_COUNT = 3
CMAP_DETECTED = "viridis"


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def clean_star_type(x) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().upper()
    return s[0] if s else "Unknown"


def infer_star_type_from_teff(teff: pd.Series) -> pd.Series:
    t = pd.to_numeric(teff, errors="coerce")
    out = pd.Series("Unknown", index=t.index, dtype=object)
    out[(t >= 7500)] = "A"
    out[(t >= 6000) & (t < 7500)] = "F"
    out[(t >= 5200) & (t < 6000)] = "G"
    out[(t >= 3700) & (t < 5200)] = "K"
    out[(t > 0) & (t < 3700)] = "M"
    return out


def add_stype_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "stype" in df.columns:
        df["stype_clean"] = df["stype"].apply(clean_star_type)
        return df
    teff_col = first_col(df, ["teff_s", "temp_s", "st_teff", "stellar_eff_temp", "koi_steff"])
    if teff_col is not None:
        df["stype_clean"] = infer_star_type_from_teff(df[teff_col])
    else:
        df["stype_clean"] = "Unknown"
    return df


def standardize_flux_radius(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    if "flux_p" not in df.columns:
        c = first_col(df, ["pl_insol", "insolation", "insolation_flux", "koi_insol"])
        if c is not None:
            rename[c] = "flux_p"
    if "radius_p" not in df.columns:
        c = first_col(df, ["pl_rade", "koi_prad", "planet_radius"])
        if c is not None:
            rename[c] = "radius_p"
    if rename:
        df = df.rename(columns=rename)

    for col in ["flux_p", "radius_p", "kepler_mes", "n_transits_keplerish"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def restrict_science_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["flux_p"] > INSOLATION_LIMITS[0])
        & (df["flux_p"] < INSOLATION_LIMITS[1])
        & (df["radius_p"] > RADIUS_LIMITS[0])
        & (df["radius_p"] < RADIUS_LIMITS[1])
        & df["stype_clean"].isin(STAR_ORDER)
    ].copy()


def load_ppop() -> pd.DataFrame:
    files = sorted(PPOP_DATA_DIR.glob("kepler_catalog_*.csv"))
    if not files:
        raise FileNotFoundError(f"No kepler_catalog_*.csv files found in {PPOP_DATA_DIR}")

    frames = []
    for p in files:
        d = pd.read_csv(p)
        if "run" not in d.columns:
            m = re.search(r"kepler_catalog_(\d+)\.csv", p.name)
            d["run"] = int(m.group(1)) if m else len(frames)
        d["source_file"] = p.name
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} P-Pop Kepler files: {len(df):,} rows")
    return prepare_ppop(df)


def prepare_ppop(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_flux_radius(df)
    df = add_stype_clean(df)

    if "detected" not in df.columns and "detected_best" in df.columns:
        df["detected"] = df["detected_best"]

    if "bright_enough_kepler" not in df.columns and "kepler_star_bright_enough" in df.columns:
        df["bright_enough_kepler"] = df["kepler_star_bright_enough"]
    if "bright_enough_kepler" not in df.columns:
        df["bright_enough_kepler"] = True

    if "kepler_enough_transits" not in df.columns:
        if "n_transits_keplerish" in df.columns:
            df["kepler_enough_transits"] = df["n_transits_keplerish"] >= 3
        else:
            df["kepler_enough_transits"] = True

    required = ["flux_p", "radius_p", "kepler_mes", "transiting_geometric", "bright_enough_kepler", "kepler_enough_transits"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"P-Pop missing required columns: {missing}")

    df["transiting_geometric"] = as_bool(df["transiting_geometric"])
    df["bright_enough_kepler"] = as_bool(df["bright_enough_kepler"])
    df["kepler_enough_transits"] = as_bool(df["kepler_enough_transits"])

    df = df.dropna(subset=["flux_p", "radius_p", "kepler_mes"]).copy()
    df = restrict_science_window(df)

    print("\nP-Pop after preparation:")
    print(f"  rows in science window = {len(df):,}")
    print(f"  transiting = {int(df['transiting_geometric'].sum()):,}")
    print("  host types:")
    print(df["stype_clean"].value_counts().reindex(STAR_ORDER).fillna(0).astype(int).to_string())
    return df


def load_nasa_overlay() -> pd.DataFrame:
    if not NASA_MODEL_CSV.exists():
        raise FileNotFoundError(f"NASA overlay file not found: {NASA_MODEL_CSV}")

    df = pd.read_csv(NASA_MODEL_CSV)
    raw_n = len(df)

    if NASA_DISC_FACILITY_FILTER and "discovery_facility" in df.columns:
        df = df[df["discovery_facility"].astype(str) == NASA_DISC_FACILITY_FILTER].copy()
    if NASA_TRAN_FLAG_ONLY and "tran_flag" in df.columns:
        df = df[pd.to_numeric(df["tran_flag"], errors="coerce").fillna(0).astype(int) == 1].copy()

    df = standardize_flux_radius(df)
    df = add_stype_clean(df)
    df = df.dropna(subset=["flux_p", "radius_p"]).copy()
    df = restrict_science_window(df)

    print("\nNASA overlay after preparation:")
    print(f"  raw rows = {raw_n:,}")
    print(f"  overlay rows in science window = {len(df):,}")
    print("  host types:")
    print(df["stype_clean"].value_counts().reindex(STAR_ORDER).fillna(0).astype(int).to_string())
    return df


def add_case_columns(df: pd.DataFrame, case_name: str) -> pd.DataFrame:
    case = CASES[case_name]
    df = df.copy()
    mes = pd.to_numeric(df["kepler_mes"], errors="coerce").fillna(0.0) / case["cdpp_scale"]
    trans = df["transiting_geometric"]
    bright = df["bright_enough_kepler"]
    enough = df["kepler_enough_transits"]

    df["detected_case"] = trans & bright & enough & (mes >= case["mes_threshold"])
    df["denominator_case"] = trans
    return df


def fraction_grid(df: pd.DataFrame, numerator: pd.Series, denominator: pd.Series):
    x = pd.to_numeric(df["flux_p"], errors="coerce")
    y = pd.to_numeric(df["radius_p"], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y)

    total, _, _ = np.histogram2d(
        x[valid & denominator],
        y[valid & denominator],
        bins=[INSOLATION_BINS, PLANET_RADIUS_BINS],
    )
    num, _, _ = np.histogram2d(
        x[valid & numerator & denominator],
        y[valid & numerator & denominator],
        bins=[INSOLATION_BINS, PLANET_RADIUS_BINS],
    )

    frac = np.divide(num, total, out=np.full_like(num, np.nan, dtype=float), where=total > 0)
    frac[total < MIN_BIN_COUNT] = np.nan
    return frac.T, total.T


def bin_centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


def add_contours(ax, grid: np.ndarray):
    if not np.isfinite(grid).any():
        return
    x = bin_centers(INSOLATION_BINS)
    y = bin_centers(PLANET_RADIUS_BINS)
    X, Y = np.meshgrid(x, y)
    levels = [0.2, 0.5, 0.8]
    try:
        cs = ax.contour(X, Y, grid, levels=levels, colors="white", linewidths=0.8, alpha=0.8)
        ax.clabel(cs, fmt="%.1f", fontsize=7)
    except Exception:
        pass


def overlay_nasa(ax, nasa_sub: pd.DataFrame, label=None):
    if len(nasa_sub) == 0:
        return
    ax.scatter(
        nasa_sub["flux_p"],
        nasa_sub["radius_p"],
        s=12,
        facecolors="none",
        edgecolors="0.35",
        linewidths=0.45,
        alpha=0.45,
        label=label,
        zorder=3,
    )


def lookup_grid_values(grid: np.ndarray, points: pd.DataFrame) -> np.ndarray:
    if len(points) == 0:
        return np.array([])
    xi = np.digitize(points["flux_p"], INSOLATION_BINS) - 1
    yi = np.digitize(points["radius_p"], PLANET_RADIUS_BINS) - 1
    good = (xi >= 0) & (xi < len(INSOLATION_BINS) - 1) & (yi >= 0) & (yi < len(PLANET_RADIUS_BINS) - 1)
    vals = np.full(len(points), np.nan)
    vals[good] = grid[yi[good], xi[good]]
    return vals[np.isfinite(vals)]


def setup_axis(ax, title: str):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(INSOLATION_LIMITS)
    ax.set_ylim(RADIUS_LIMITS)
    ax.set_title(title)
    ax.set_xlabel("Insolation flux [I⊕]")
    ax.set_ylabel("Planet radius [R⊕]")
    ax.grid(True, which="both", alpha=0.18)


def plot_fgkm_detected_only(ppop: pd.DataFrame, nasa: pd.DataFrame, case_name: str):
    ppop = add_case_columns(ppop, case_name)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.2), sharex=True, sharey=True, constrained_layout=True)

    mesh = None
    summary_rows = []

    for j, stype in enumerate(STAR_ORDER):
        p = ppop[ppop["stype_clean"] == stype].copy()
        n = nasa[nasa["stype_clean"] == stype].copy()

        det_grid, support = fraction_grid(p, p["detected_case"], p["denominator_case"])

        ax = axes[j]
        mesh = ax.pcolormesh(
            INSOLATION_BINS, PLANET_RADIUS_BINS, det_grid,
            shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED,
        )
        add_contours(ax, det_grid)
        overlay_nasa(ax, n, label=f"NASA/Kepler {stype}" if j == 0 else None)
        setup_axis(ax, f"{stype} stars: Detected fraction")

        vals = lookup_grid_values(det_grid, n)
        summary_rows.append({
            "case": case_name,
            "host_type": stype,
            "metric": "detected",
            "ppop_transiting_support": int(support.sum()),
            "nasa_overlay_points": int(len(n)),
            "nasa_supported_bins": int(len(vals)),
            "median_bin_value_at_nasa_points": float(np.nanmedian(vals)) if len(vals) else np.nan,
            "n_ge_0p5": int((vals >= 0.5).sum()) if len(vals) else 0,
            "n_le_0p2": int((vals <= 0.2).sum()) if len(vals) else 0,
        })

    if len(nasa):
        axes[0].legend(loc="upper right", fontsize=8)

    fig.colorbar(mesh, ax=axes, label="Detected fraction", shrink=0.92)
    fig.suptitle(
        f"Kepler v1 FGKM view — Insolation-Radius detected fraction ({case_name})\n"
        "White/blank bins = too few transiting P-Pop planets, not zero detection",
        fontsize=16,
    )

    out = OUT_DIR / f"fgkm_insolation_radius_detected_only_{case_name}.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)

    return pd.DataFrame(summary_rows), out


def plot_support_counts(ppop: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.8), sharex=True, sharey=True, constrained_layout=True)
    mesh = None

    for ax, stype in zip(axes, STAR_ORDER):
        p = ppop[(ppop["stype_clean"] == stype) & ppop["transiting_geometric"]].copy()
        total, _, _ = np.histogram2d(p["flux_p"], p["radius_p"], bins=[INSOLATION_BINS, PLANET_RADIUS_BINS])
        support = total.T
        support[support == 0] = np.nan
        mesh = ax.pcolormesh(INSOLATION_BINS, PLANET_RADIUS_BINS, support, shading="auto")
        setup_axis(ax, f"{stype} stars: transiting P-Pop support")

    fig.colorbar(mesh, ax=axes, label="Transiting P-Pop planets per bin", shrink=0.92)
    fig.suptitle("FGKM support map — shows where the detected-fraction maps are statistically supported", fontsize=15)

    out = OUT_DIR / "fgkm_insolation_radius_support_counts.png"
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def main():
    ppop = load_ppop()
    nasa = load_nasa_overlay()

    all_summaries = []
    for case_name in RUN_CASES:
        print(f"\nRunning case: {case_name}")
        summary, fig_path = plot_fgkm_detected_only(ppop, nasa, case_name)
        all_summaries.append(summary)
        print(f"  saved {fig_path}")

    support_path = plot_support_counts(ppop)
    print(f"  saved {support_path}")

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_csv = OUT_DIR / "fgkm_insolation_radius_detected_only_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
