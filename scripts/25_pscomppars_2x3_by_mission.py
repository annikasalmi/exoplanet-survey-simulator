"""
25_pscomppars_2x3_by_mission.py

One 1x3 picture.
Shows only NASA strict planets (quality + uncertainty cuts from plot 24). "Clean pile."
Three columns = insolation bins: I < 10, I < 50, I > 50.

Color = which MISSION / facility found the planet (Kepler, TESS, K2, ...).
Legend shows missions that have >MIN_PLANETS_FOR_OWN_LEGEND planets in the plot
window (mass 0-12, radius 0.5-2.2). Missions with <= that threshold collapse to "Other".
Legend counts reflect planets inside the window only.

"Quality cuts" means:
  - No upper-limit mass or radius values
  - No calculated masses
  - Two-sided error bars on mass and radius
  - Relative uncertainty <= 8% (radius) and <= 25% (mass)

NOTE: NASA PSCompPars planets are raw real planets from NASA archive.
They do NOT go through your kepler_Data detector. Only P-Pop is detector-made.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Settings (kept same spirit as script 24)
# ============================================================


def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists():
            return p
    return Path.cwd()


ROOT = find_project_root(Path(__file__).resolve())

NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"
NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = ROOT / "my_outputs" / "25_pscomppars_2x3_by_mission"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same cache file script 24 uses, so no re-download if you already ran 24.
NASA_PSCOMPPARS_FLAGS_CACHE = (
    NASA_DATA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)
FORCE_REDOWNLOAD_NASA = False
DOWNLOAD_NASA_IF_MISSING = True

# Strict cuts for the BOTTOM row (must match plot 24 to be comparable).
NASA_EXCLUDE_MASS_LIMITS = True
NASA_EXCLUDE_RADIUS_LIMITS = True
NASA_EXCLUDE_INSOLATION_LIMITS = False
NASA_EXCLUDE_CALCULATED_MASSES = True
NASA_REQUIRE_TWO_SIDED_MASS_ERRORS = True
NASA_REQUIRE_TWO_SIDED_RADIUS_ERRORS = True
MAX_RADIUS_REL_UNCERTAINTY = 0.08   # 8%
MAX_MASS_REL_UNCERTAINTY = 0.25     # 25%

# Plot camera.
NOTEBOOK_XLIM = (0.0, 12.0)
NOTEBOOK_YLIM = (0.5, 2.2)

POINT_SIZE = 20
POINT_ALPHA = 0.75
DRAW_ROCKY_CURVE = True
ALLOW_TOY_ROCKY_CURVE_IF_REF_MISSING = True

# Missions with <= this many planets visible in the plot window collapse into "Other".
MIN_PLANETS_FOR_OWN_LEGEND = 2
OTHER_LABEL = "Other"
OTHER_COLOR = "0.6"


# ============================================================
# NASA download / cache (same query as script 24)
# ============================================================


def build_nasa_pscomppars_query() -> str:
    return """
    SELECT
        pl_name, hostname, discoverymethod, disc_facility, disc_telescope, tran_flag,
        pl_orbper, pl_orbsmax, pl_orbincl, pl_trandep, pl_trandur,
        pl_insol, pl_insolerr1, pl_insolerr2, pl_insollim, pl_eqt,
        pl_rade, pl_radeerr1, pl_radeerr2, pl_radelim, pl_rade_reflink,
        pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_bmasselim, pl_bmassprov, pl_bmasse_reflink,
        st_rad, st_mass, st_teff, st_lum, sy_dist, sy_kepmag, sy_gaiamag
    FROM pscomppars
    WHERE tran_flag = 1
      AND pl_rade IS NOT NULL
      AND pl_bmasse IS NOT NULL
      AND pl_insol IS NOT NULL
    """


def nasa_tap_url(query: str) -> str:
    return "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + quote(query) + "&format=csv"


def load_nasa_raw() -> pd.DataFrame:
    if FORCE_REDOWNLOAD_NASA or not NASA_PSCOMPPARS_FLAGS_CACHE.exists():
        if not DOWNLOAD_NASA_IF_MISSING and not NASA_PSCOMPPARS_FLAGS_CACHE.exists():
            raise FileNotFoundError(f"NASA cache missing: {NASA_PSCOMPPARS_FLAGS_CACHE}")
        print("Downloading NASA PSCompPars...")
        df = pd.read_csv(nasa_tap_url(build_nasa_pscomppars_query()))
        NASA_PSCOMPPARS_FLAGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NASA_PSCOMPPARS_FLAGS_CACHE, index=False)
        print(f"Downloaded NASA rows: {len(df):,}")
        return df
    print("Loading cached NASA PSCompPars:", NASA_PSCOMPPARS_FLAGS_CACHE)
    return pd.read_csv(NASA_PSCOMPPARS_FLAGS_CACHE)


# ============================================================
# Standardize + quality flags
# ============================================================


def force_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def standardize_nasa_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "pl_name": "planet_name", "hostname": "host_name",
        "disc_facility": "discovery_facility",
        "pl_bmasse": "mass_p", "pl_bmasseerr1": "mass_err_plus", "pl_bmasseerr2": "mass_err_minus",
        "pl_bmasselim": "mass_limit_flag", "pl_bmassprov": "mass_provider",
        "pl_bmasse_reflink": "mass_reference",
        "pl_rade": "radius_p", "pl_radeerr1": "radius_err_plus", "pl_radeerr2": "radius_err_minus",
        "pl_radelim": "radius_limit_flag", "pl_rade_reflink": "radius_reference",
        "pl_insol": "flux_p", "pl_insolerr1": "flux_err_plus", "pl_insolerr2": "flux_err_minus",
        "pl_insollim": "insolation_limit_flag",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def as_limit_flag(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).ne(0)


def prepare_pscomppars() -> pd.DataFrame:
    """Return one standardized table. Raw planets, valid M/R/insolation only."""
    raw = load_nasa_raw()
    df = standardize_nasa_columns(raw)
    df = force_numeric(df, [
        "mass_p", "radius_p", "flux_p", "mass_err_plus", "mass_err_minus",
        "radius_err_plus", "radius_err_minus",
        "mass_limit_flag", "radius_limit_flag", "insolation_limit_flag",
    ])

    before = len(df)
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()
    df = df[(df["mass_p"] > 0) & (df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()
    print(f"PSCompPars valid M/R/insolation rows: {before:,} -> {len(df):,}")

    df["planet_label"] = df.get("planet_name", pd.Series("", index=df.index)).astype(str)
    df["discovery_facility"] = df.get("discovery_facility", pd.Series("Unknown", index=df.index)).astype(str)

    # relative uncertainty (max of the two sides) for the strict cut
    m_err = pd.concat([df["mass_err_plus"].abs(), df["mass_err_minus"].abs()], axis=1).max(axis=1)
    r_err = pd.concat([df["radius_err_plus"].abs(), df["radius_err_minus"].abs()], axis=1).max(axis=1)
    df["mass_rel_uncertainty"] = m_err / df["mass_p"].abs()
    df["radius_rel_uncertainty"] = r_err / df["radius_p"].abs()

    df["has_two_sided_mass_errorbar"] = df["mass_err_plus"].notna() & df["mass_err_minus"].notna()
    df["has_two_sided_radius_errorbar"] = df["radius_err_plus"].notna() & df["radius_err_minus"].notna()

    # quality flags
    df["mass_is_limit"] = as_limit_flag(df.get("mass_limit_flag", 0))
    df["radius_is_limit"] = as_limit_flag(df.get("radius_limit_flag", 0))
    df["insolation_is_limit"] = as_limit_flag(df.get("insolation_limit_flag", 0))
    mass_ref = df.get("mass_reference", pd.Series("", index=df.index)).astype(str)
    radius_ref = df.get("radius_reference", pd.Series("", index=df.index)).astype(str)
    df["mass_is_calculated"] = mass_ref.str.contains("CALCULATED_VALUE|Calculated Value", case=False, na=False)
    df["radius_is_calculated"] = radius_ref.str.contains("CALCULATED_VALUE|Calculated Value", case=False, na=False)
    return df


def apply_strict_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Bottom row sample. Same logic as plot 24."""
    mass_good = pd.Series(True, index=df.index)
    radius_good = pd.Series(True, index=df.index)
    insol_good = pd.Series(True, index=df.index)

    if NASA_EXCLUDE_MASS_LIMITS:
        mass_good &= ~df["mass_is_limit"]
    if NASA_EXCLUDE_CALCULATED_MASSES:
        mass_good &= ~df["mass_is_calculated"]
    if NASA_REQUIRE_TWO_SIDED_MASS_ERRORS:
        mass_good &= df["has_two_sided_mass_errorbar"]
    if NASA_EXCLUDE_RADIUS_LIMITS:
        radius_good &= ~df["radius_is_limit"]
    if NASA_REQUIRE_TWO_SIDED_RADIUS_ERRORS:
        radius_good &= df["has_two_sided_radius_errorbar"]
    if NASA_EXCLUDE_INSOLATION_LIMITS:
        insol_good &= ~df["insolation_is_limit"]

    quality_good = mass_good & radius_good & insol_good
    unc_good = (
        (df["mass_rel_uncertainty"].isna() | (df["mass_rel_uncertainty"] <= MAX_MASS_REL_UNCERTAINTY))
        & (df["radius_rel_uncertainty"].isna() | (df["radius_rel_uncertainty"] <= MAX_RADIUS_REL_UNCERTAINTY))
    )
    # require the error to actually exist for the strict sample (matches plot 24)
    unc_good &= df["mass_rel_uncertainty"].notna() & df["radius_rel_uncertainty"].notna()

    kept = df[quality_good & unc_good].copy()
    print(f"Strict NASA: {len(df):,} -> {len(kept):,} kept")
    return kept


# ============================================================
# Rocky curve
# ============================================================


def load_rocky_reference_curve(ref_path=REF_CURVE_PATH):
    ref_path = Path(ref_path)
    if not ref_path.exists():
        if not ALLOW_TOY_ROCKY_CURVE_IF_REF_MISSING:
            raise FileNotFoundError(f"Missing rocky reference: {ref_path}")
        print("WARNING: ref.ddat not found. Using toy rocky curve.")
        m_ref = np.linspace(0.05, 100.0, 2000)
        return m_ref, m_ref ** 0.27
    ref = np.loadtxt(ref_path, comments="#")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    m_ref, r_ref = ref[:, 0].astype(float), ref[:, 1].astype(float)
    good = np.isfinite(m_ref) & np.isfinite(r_ref) & (m_ref > 0) & (r_ref > 0)
    m_ref, r_ref = m_ref[good], r_ref[good]
    order = np.argsort(m_ref)
    return m_ref[order], r_ref[order]


def add_rocky_curve(ax, m_ref, r_ref):
    if not DRAW_ROCKY_CURVE:
        return
    m_min = max(NOTEBOOK_XLIM[0], float(np.nanmin(m_ref)))
    m_max = min(NOTEBOOK_XLIM[1], float(np.nanmax(m_ref)))
    if m_max <= m_min:
        return
    m_line = np.linspace(m_min, m_max, 500)
    r_line = np.interp(m_line, m_ref, r_ref, left=np.nan, right=np.nan)
    ax.plot(m_line, r_line, "--", linewidth=1.6, color="black", alpha=0.92, zorder=5)


# ============================================================
# Mission colors + insolation bins
# ============================================================


def make_insolation_panels(df: pd.DataFrame):
    return [
        (r"I < 10 $I_\oplus$", df[df["flux_p"] < 10].copy()),
        (r"I < 50 $I_\oplus$", df[df["flux_p"] < 50].copy()),
        (r"I > 50 $I_\oplus$", df[df["flux_p"] > 50].copy()),
    ]


def build_mission_colormap(df_in_window: pd.DataFrame):
    """Build color map from planets visible in the plot window.

    Missions with > MIN_PLANETS_FOR_OWN_LEGEND planets in the window get their
    own color. All others collapse into OTHER_LABEL.
    Sorted by in-window count descending for stable color assignment.
    """
    counts = df_in_window["discovery_facility"].value_counts()
    top = list(counts[counts > MIN_PLANETS_FOR_OWN_LEGEND].index)

    cmap = plt.get_cmap("tab10")
    color_map = {name: cmap(i % 10) for i, name in enumerate(top)}
    color_map[OTHER_LABEL] = OTHER_COLOR

    legend_order = top + [OTHER_LABEL]
    return color_map, legend_order


def assign_mission_key(df: pd.DataFrame, top_missions) -> pd.Series:
    return df["discovery_facility"].apply(lambda n: n if n in top_missions else OTHER_LABEL)


# ============================================================
# Plotting
# ============================================================


def setup_axis(ax):
    ax.set_xlim(*NOTEBOOK_XLIM)
    ax.set_ylim(*NOTEBOOK_YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=10)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=10)
    ax.grid(True, alpha=0.28, linestyle="--")


def scatter_by_mission(ax, df, color_map, legend_order, m_ref, r_ref, title, add_labels=False):
    setup_axis(ax)
    add_rocky_curve(ax, m_ref, r_ref)
    if len(df) == 0:
        ax.text(0.5, 0.5, "No planets", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{title}\nN = 0", fontsize=10)
        return
    # draw Other first (background), then named missions on top
    for key in legend_order:
        sub = df[df["mission_color_key"] == key]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["mass_p"], sub["radius_p"],
            s=POINT_SIZE, alpha=POINT_ALPHA,
            color=color_map[key], edgecolors="none",
            zorder=2 if key == OTHER_LABEL else 3,
        )
    if add_labels:
        in_win = (
            df["mass_p"].between(*NOTEBOOK_XLIM) &
            df["radius_p"].between(*NOTEBOOK_YLIM)
        )
        for _, row in df[in_win].iterrows():
            ax.annotate(
                row["planet_label"],
                xy=(row["mass_p"], row["radius_p"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=3, ha="left", va="bottom",
                zorder=6, clip_on=True,
            )
    ax.set_title(f"{title}\nN = {len(df)}", fontsize=10)


def make_figure(df_strict: pd.DataFrame):
    m_ref, r_ref = load_rocky_reference_curve()

    # Build color map from planets visible in the plot window
    in_window_mask = (
        df_strict["mass_p"].between(*NOTEBOOK_XLIM) &
        df_strict["radius_p"].between(*NOTEBOOK_YLIM)
    )
    df_in_window = df_strict[in_window_mask]

    color_map, legend_order = build_mission_colormap(df_in_window)
    top_missions = [k for k in legend_order if k != OTHER_LABEL]

    df_strict = df_strict.copy()
    df_strict["mission_color_key"] = assign_mission_key(df_strict, top_missions)
    df_in_window = df_in_window.copy()
    df_in_window["mission_color_key"] = assign_mission_key(df_in_window, top_missions)

    panels = make_insolation_panels(df_strict)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for j, (bin_title, df_sub) in enumerate(panels):
        scatter_by_mission(
            axes[j], df_sub, color_map, legend_order, m_ref, r_ref, bin_title,
            add_labels=((j == 1) or (j==0)),  # label all in-window planets in both the I<50 panel
        )

    # Legend counts reflect in-window planets only; skip entries with 0 in-window
    in_window_counts = df_in_window["mission_color_key"].value_counts()
    handles = []
    for key in legend_order:
        n = int(in_window_counts.get(key, 0))
        if n == 0:
            continue
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=color_map[key], markeredgecolor="none",
            markersize=8, label=f"{key} (n={n})",
        ))
    fig.legend(
        handles=handles,
        title=f"Discovery facility (n = in-window, >{MIN_PLANETS_FOR_OWN_LEGEND} shown)",
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, title_fontsize=10,
    )

    fig.suptitle(
        "NASA PSCompPars strict sample — mass-radius by insolation, colored by discovery facility\n"
        f"Quality + uncertainty cuts: mass ≤ {MAX_MASS_REL_UNCERTAINTY:.0%}, radius ≤ {MAX_RADIUS_REL_UNCERTAINTY:.0%}  |  "
        f"Legend: facilities with >{MIN_PLANETS_FOR_OWN_LEGEND} planets in window",
        fontsize=13,
    )

    png = OUT_DIR / "pscomppars_1x3_by_mission.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png)

    # per-mission count table
    counts_rows = []
    for bin_title, df_sub in panels:
        vc = df_sub["mission_color_key"].value_counts()
        for mission, n in vc.items():
            counts_rows.append({"bin": bin_title, "mission": mission, "count": int(n)})
    counts_path = OUT_DIR / "pscomppars_1x3_by_mission_counts.csv"
    pd.DataFrame(counts_rows).to_csv(counts_path, index=False)
    print("Saved:", counts_path)
    return png, counts_path


# ============================================================
# Main
# ============================================================


def main():
    print("Project root:", ROOT)
    print("Output dir:", OUT_DIR)

    df_all_raw = prepare_pscomppars()
    df_strict = apply_strict_filter(df_all_raw)

    df_strict.to_csv(OUT_DIR / "pscomppars_strict.csv", index=False)

    make_figure(df_strict)
    print("\nDone.")
    print("Caveman note: NASA planets here are raw real planets, never run through your detector.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSCRIPT FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
