"""
35_kepler-rocky-fgkm-detection.py

Defines a "rocky threshold" curve by vertically shifting the ref.ddat
pure-rock boundary upward until it passes exactly through LHS 1140 b in
mass-radius space.  Then:

  1. Filters NASA PSCompPars to planets on or below that threshold
     ("rocky" planets, same selection logic as script 24 quality filters).
  2. Overlays those rocky planets on the FGKM insolation-radius
     detected-fraction background from script 29 (same P-Pop grid,
     same FGKM star-type split).
  3. Saves a mass-radius diagnostic plot to verify the threshold placement.

Rocky threshold anchor (Cadieux et al. 2024, JWST era):
    LHS 1140 b:  M_p = 5.60 M_earth,  R_p = 1.730 R_earth
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Project root ──────────────────────────────────────────────────────────────

def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists():
            return p
    return start_path.parents[1]


ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow unicode (⊕, ≤, …) in console output on Windows cp1252 terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────

STAR_CATALOG_FOLDER = "Gaia_C_F_K_combined"
PPOP_DATA_DIR = ROOT / "run" / "kepler" / "data" / STAR_CATALOG_FOLDER
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the cache from script 24 if available; download to the same location if not.
NASA_FLAGS_CACHE = (
    NASA_DATA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)

OUT_DIR = ROOT / "my_outputs" / "35_kepler-rocky-fgkm-detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LHS 1140 b anchor (Cadieux et al. 2024) ──────────────────────────────────

LHS1140B_MASS_MEARTH   = 5.60
LHS1140B_RADIUS_REARTH = 1.730

# ── NASA quality settings (mirror script 24) ─────────────────────────────────

FORCE_REDOWNLOAD_NASA     = False
DOWNLOAD_NASA_IF_MISSING  = True

EXCLUDE_MASS_LIMITS       = True
EXCLUDE_RADIUS_LIMITS     = True
EXCLUDE_CALCULATED_MASSES = True
REQUIRE_TWO_SIDED_MASS    = True
REQUIRE_TWO_SIDED_RADIUS  = True

MAX_MASS_REL_UNCERTAINTY   = 0.25
MAX_RADIUS_REL_UNCERTAINTY = 0.08

# ── P-Pop / FGKM settings (mirror script 29) ─────────────────────────────────

STAR_ORDER = ["F", "G", "K", "M"]

CASES = {
    "optimistic":   {"mes_threshold": 6.5, "cdpp_scale": 0.8},
    "baseline":     {"mes_threshold": 7.1, "cdpp_scale": 1.0},
    "conservative": {"mes_threshold": 7.7, "cdpp_scale": 1.2},
}
RUN_CASES = ["baseline"]

RADIUS_LIMITS     = (0.5, 4.0)
INSOLATION_LIMITS = (0.1, 1e5)

INSOLATION_BINS    = np.logspace(np.log10(INSOLATION_LIMITS[0]), np.log10(INSOLATION_LIMITS[1]), 18)
PLANET_RADIUS_BINS = np.logspace(np.log10(RADIUS_LIMITS[0]),     np.log10(RADIUS_LIMITS[1]), 13)

MIN_BIN_COUNT = 2
CMAP_DETECTED = "viridis"

# ── General helpers ───────────────────────────────────────────────────────────


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


def infer_star_type_from_teff(teff: pd.Series) -> pd.Series:
    t = pd.to_numeric(teff, errors="coerce")
    out = pd.Series("Unknown", index=t.index, dtype=object)
    out[t >= 7500] = "A"
    out[(t >= 6000) & (t < 7500)] = "F"
    out[(t >= 5200) & (t < 6000)] = "G"
    out[(t >= 3700) & (t < 5200)] = "K"
    out[(t > 0) & (t < 3700)] = "M"
    return out


def add_stype_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "stype" in df.columns:
        raw = df["stype"].astype(str).str.strip().str.upper()
        df["stype_clean"] = raw.str[0].where(raw.str.len() > 0, "Unknown")
        return df
    teff_col = first_col(df, ["teff_s", "temp_s", "st_teff", "stellar_eff_temp", "koi_steff"])
    if teff_col:
        df["stype_clean"] = infer_star_type_from_teff(df[teff_col])
    else:
        df["stype_clean"] = "Unknown"
    return df


def standardize_flux_radius(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    if "flux_p" not in df.columns:
        c = first_col(df, ["pl_insol", "insolation", "insolation_flux", "koi_insol"])
        if c:
            rename[c] = "flux_p"
    if "radius_p" not in df.columns:
        c = first_col(df, ["pl_rade", "koi_prad", "planet_radius"])
        if c:
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


# ── Rocky threshold ───────────────────────────────────────────────────────────


def load_rocky_reference_curve():
    if not REF_CURVE_PATH.exists():
        print(f"WARNING: {REF_CURVE_PATH} not found — using toy power-law rocky curve.")
        m = np.linspace(0.05, 30.0, 600)
        r = m ** 0.27
        return m, r
    ref = np.loadtxt(REF_CURVE_PATH, comments="#")
    m_ref = ref[:, 0].astype(float)
    r_ref = ref[:, 1].astype(float)
    good = np.isfinite(m_ref) & np.isfinite(r_ref) & (m_ref > 0) & (r_ref > 0)
    m_ref, r_ref = m_ref[good], r_ref[good]
    order = np.argsort(m_ref)
    return m_ref[order], r_ref[order]


def compute_rocky_threshold_shift(m_ref: np.ndarray, r_ref: np.ndarray) -> float:
    """
    Vertical shift (in R_earth) to apply to the ref.ddat rocky curve so that
    the resulting rocky threshold passes exactly through LHS 1140 b.
    """
    r_at_lhs = float(np.interp(LHS1140B_MASS_MEARTH, m_ref, r_ref))
    shift = LHS1140B_RADIUS_REARTH - r_at_lhs
    print(
        f"Rocky curve at LHS 1140 b mass ({LHS1140B_MASS_MEARTH} M⊕): "
        f"{r_at_lhs:.4f} R⊕"
    )
    print(
        f"LHS 1140 b radius: {LHS1140B_RADIUS_REARTH} R⊕  →  "
        f"rocky threshold shift = {shift:+.4f} R⊕"
    )
    return shift


def rocky_threshold_at_mass(
    masses: np.ndarray,
    m_ref: np.ndarray,
    r_ref: np.ndarray,
    shift: float,
) -> np.ndarray:
    return np.interp(
        np.asarray(masses, dtype=float),
        m_ref,
        r_ref + shift,
        left=np.nan,
        right=np.nan,
    )


# ── NASA PSCompPars loading and rocky filtering ───────────────────────────────


def _build_nasa_query() -> str:
    return """
    SELECT pl_name, hostname, discoverymethod, disc_facility, tran_flag,
           pl_insol, pl_insolerr1, pl_insolerr2, pl_insollim,
           pl_rade, pl_radeerr1, pl_radeerr2, pl_radelim, pl_rade_reflink,
           pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_bmasselim,
           pl_bmassprov, pl_bmasse_reflink,
           st_teff, st_rad, st_mass, st_lum, sy_dist
    FROM pscomppars
    WHERE tran_flag = 1
      AND pl_rade IS NOT NULL
      AND pl_bmasse IS NOT NULL
      AND pl_insol IS NOT NULL
    """


def load_nasa_raw() -> pd.DataFrame:
    if not FORCE_REDOWNLOAD_NASA and NASA_FLAGS_CACHE.exists():
        print(f"Loading NASA PSCompPars from cache:\n  {NASA_FLAGS_CACHE}")
        return pd.read_csv(NASA_FLAGS_CACHE)
    if not DOWNLOAD_NASA_IF_MISSING:
        raise FileNotFoundError(
            f"NASA cache missing and DOWNLOAD_NASA_IF_MISSING=False:\n{NASA_FLAGS_CACHE}"
        )
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        + quote(_build_nasa_query())
        + "&format=csv"
    )
    print("Downloading NASA PSCompPars with limit flags...")
    df = pd.read_csv(url)
    NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(NASA_FLAGS_CACHE, index=False)
    print(f"Saved to: {NASA_FLAGS_CACHE}  ({len(df):,} rows)")
    return df


def load_and_filter_nasa(m_ref: np.ndarray, r_ref: np.ndarray, shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (all_quality_filtered, rocky_filtered) DataFrames.
    all_quality_filtered : same strict quality cuts as script 24
    rocky_filtered       : subset where radius_p <= rocky_threshold(mass_p)
    """
    raw = load_nasa_raw()

    rename = {
        "pl_name":           "planet_name",
        "hostname":          "host_name",
        "disc_facility":     "discovery_facility",
        "pl_bmasse":         "mass_p",
        "pl_bmasseerr1":     "mass_err_plus",
        "pl_bmasseerr2":     "mass_err_minus",
        "pl_bmasselim":      "mass_limit_flag",
        "pl_bmassprov":      "mass_provider",
        "pl_bmasse_reflink": "mass_reference",
        "pl_rade":           "radius_p",
        "pl_radeerr1":       "radius_err_plus",
        "pl_radeerr2":       "radius_err_minus",
        "pl_radelim":        "radius_limit_flag",
        "pl_rade_reflink":   "radius_reference",
        "pl_insol":          "flux_p",
        "pl_insollim":       "insolation_limit_flag",
        "st_teff":           "teff_s",
        "st_rad":            "radius_s",
        "st_mass":           "mass_s",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    for col in [
        "mass_p", "radius_p", "flux_p",
        "mass_err_plus", "mass_err_minus",
        "radius_err_plus", "radius_err_minus",
        "mass_limit_flag", "radius_limit_flag", "insolation_limit_flag",
        "teff_s",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()
    df = df[(df["mass_p"] > 0) & (df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()
    print(f"NASA rows with valid M / R / flux: {n0:,} → {len(df):,}")

    # ── same quality filters as script 24 ────────────────────────────────────
    if EXCLUDE_MASS_LIMITS:
        mask = df["mass_limit_flag"].fillna(0).ne(0)
        df = df[~mask].copy()
        print(f"  remove mass limit flags: → {len(df):,}")

    if EXCLUDE_CALCULATED_MASSES and "mass_reference" in df.columns:
        calc = df["mass_reference"].astype(str).str.contains(
            "CALCULATED_VALUE|Calculated Value", case=False, na=False
        )
        df = df[~calc].copy()
        print(f"  remove calculated masses: → {len(df):,}")

    if EXCLUDE_RADIUS_LIMITS:
        mask = df["radius_limit_flag"].fillna(0).ne(0)
        df = df[~mask].copy()
        print(f"  remove radius limit flags: → {len(df):,}")

    if REQUIRE_TWO_SIDED_MASS:
        has = df["mass_err_plus"].notna() & df["mass_err_minus"].notna()
        df = df[has].copy()
        print(f"  require two-sided mass errors: → {len(df):,}")

    if REQUIRE_TWO_SIDED_RADIUS:
        has = df["radius_err_plus"].notna() & df["radius_err_minus"].notna()
        df = df[has].copy()
        print(f"  require two-sided radius errors: → {len(df):,}")

    # Relative uncertainty cuts
    mass_err = pd.concat(
        [df["mass_err_plus"].abs(), df["mass_err_minus"].abs()], axis=1
    ).max(axis=1)
    mass_rel = mass_err / df["mass_p"].abs()
    df = df[~(mass_rel.notna() & (mass_rel > MAX_MASS_REL_UNCERTAINTY))].copy()
    print(f"  mass uncertainty ≤ {MAX_MASS_REL_UNCERTAINTY:.0%}: → {len(df):,}")

    rad_err = pd.concat(
        [df["radius_err_plus"].abs(), df["radius_err_minus"].abs()], axis=1
    ).max(axis=1)
    rad_rel = rad_err / df["radius_p"].abs()
    df = df[~(rad_rel.notna() & (rad_rel > MAX_RADIUS_REL_UNCERTAINTY))].copy()
    print(f"  radius uncertainty ≤ {MAX_RADIUS_REL_UNCERTAINTY:.0%}: → {len(df):,}")

    df = add_stype_clean(df)
    df["planet_label"] = (
        df.get("planet_name", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
    )
    df["mass_rel_uncertainty"] = mass_rel.reindex(df.index)
    df["radius_rel_uncertainty"] = rad_rel.reindex(df.index)

    print(f"NASA after all quality filters: {len(df):,} planets")

    # ── rocky threshold filter ────────────────────────────────────────────────
    threshold_r = rocky_threshold_at_mass(df["mass_p"].to_numpy(), m_ref, r_ref, shift)
    is_rocky = (df["radius_p"].to_numpy() <= threshold_r) & np.isfinite(threshold_r)
    df["rocky_threshold_radius"] = threshold_r
    df["below_rocky_threshold"] = is_rocky

    rocky = df[is_rocky].copy()
    print(
        f"Rocky threshold filter (radius ≤ threshold): {len(df):,} → {len(rocky):,} rocky planets"
    )
    return df, rocky


# ── P-Pop loading (same as script 29) ─────────────────────────────────────────


def load_ppop() -> pd.DataFrame:
    files = sorted(PPOP_DATA_DIR.glob("kepler_catalog_*.csv"))
    if not files:
        raise FileNotFoundError(f"No kepler_catalog_*.csv found in:\n  {PPOP_DATA_DIR}")

    frames = []
    for p in files:
        d = pd.read_csv(p)
        if "run" not in d.columns:
            m = re.search(r"kepler_catalog_(\d+)\.csv", p.name)
            d["run"] = int(m.group(1)) if m else len(frames)
        d["source_file"] = p.name
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} P-Pop catalog file(s): {len(df):,} rows")
    return _prepare_ppop(df)


def _prepare_ppop(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_flux_radius(df)
    df = add_stype_clean(df)

    if "detected" not in df.columns and "detected_best" in df.columns:
        df["detected"] = df["detected_best"]
    if "bright_enough_kepler" not in df.columns:
        kc = first_col(df, ["kepler_star_bright_enough"])
        df["bright_enough_kepler"] = df[kc] if kc else True
    if "kepler_enough_transits" not in df.columns:
        if "n_transits_keplerish" in df.columns:
            df["kepler_enough_transits"] = df["n_transits_keplerish"] >= 3
        else:
            df["kepler_enough_transits"] = True

    required = [
        "flux_p", "radius_p", "kepler_mes",
        "transiting_geometric", "bright_enough_kepler", "kepler_enough_transits",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"P-Pop missing required columns: {missing}")

    df["transiting_geometric"]  = as_bool(df["transiting_geometric"])
    df["bright_enough_kepler"]  = as_bool(df["bright_enough_kepler"])
    df["kepler_enough_transits"] = as_bool(df["kepler_enough_transits"])

    df = df.dropna(subset=["flux_p", "radius_p", "kepler_mes"]).copy()
    df = restrict_science_window(df)

    print(f"P-Pop in science window: {len(df):,} rows")
    print("  host types:", df["stype_clean"].value_counts().reindex(STAR_ORDER).fillna(0).astype(int).to_dict())
    return df


def _add_case_columns(df: pd.DataFrame, case_name: str) -> pd.DataFrame:
    case = CASES[case_name]
    df = df.copy()
    mes = pd.to_numeric(df["kepler_mes"], errors="coerce").fillna(0.0) / case["cdpp_scale"]
    df["detected_case"] = (
        df["transiting_geometric"]
        & df["bright_enough_kepler"]
        & df["kepler_enough_transits"]
        & (mes >= case["mes_threshold"])
    )
    df["denominator_case"] = df["transiting_geometric"]
    return df


# ── Grid / contour helpers (same as script 29) ────────────────────────────────


def fraction_grid(df: pd.DataFrame, numerator: pd.Series, denominator: pd.Series):
    x = pd.to_numeric(df["flux_p"], errors="coerce")
    y = pd.to_numeric(df["radius_p"], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y)

    total, _, _ = np.histogram2d(
        x[valid & denominator], y[valid & denominator],
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


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1] * edges[1:])


def _add_contours(ax, grid: np.ndarray):
    if not np.isfinite(grid).any():
        return
    X, Y = np.meshgrid(_bin_centers(INSOLATION_BINS), _bin_centers(PLANET_RADIUS_BINS))
    try:
        cs = ax.contour(X, Y, grid, levels=[0.2, 0.5, 0.8], colors="white", linewidths=0.8, alpha=0.8)
        ax.clabel(cs, fmt="%.1f", fontsize=7)
    except Exception:
        pass


def _setup_axis(ax, title: str):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(INSOLATION_LIMITS)
    ax.set_ylim(RADIUS_LIMITS)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Insolation flux [I⊕]")
    ax.set_ylabel("Planet radius [R⊕]")
    ax.grid(True, which="both", alpha=0.18)


# ── FGKM plot with rocky overlay ─────────────────────────────────────────────


def _overlay_rocky(ax, sub: pd.DataFrame, label: str | None = None):
    if len(sub) == 0:
        return

    has_rerr = "radius_err_plus" in sub.columns and "radius_err_minus" in sub.columns
    if has_rerr:
        yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
        yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
        ax.errorbar(
            sub["flux_p"], sub["radius_p"],
            yerr=[yerr_lo, yerr_hi],
            fmt="o", ms=3, color="crimson", alpha=0.70,
            elinewidth=0.7, capsize=1.5, ecolor="crimson",
            label=label, zorder=4,
        )
    else:
        ax.scatter(
            sub["flux_p"], sub["radius_p"],
            s=16, color="crimson", alpha=0.70, linewidths=0,
            label=label, zorder=4,
        )
    # Highlight LHS 1140 b if it landed in this sub-panel
    lhs_mask = sub["planet_label"].str.contains(
        r"LHS\s*1140\s*b", case=False, na=False, regex=True
    )
    for _, row in sub[lhs_mask].iterrows():
        ax.scatter(
            [row["flux_p"]], [row["radius_p"]],
            s=60, marker="*", color="gold", edgecolors="darkred",
            linewidths=0.8, zorder=6,
        )
        ax.annotate(
            "LHS 1140 b",
            xy=(row["flux_p"], row["radius_p"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7, color="darkred", fontweight="bold", zorder=7,
        )


def plot_rocky_fgkm(
    ppop: pd.DataFrame,
    rocky: pd.DataFrame,
    case_name: str,
    shift: float,
) -> tuple[pd.DataFrame, Path]:
    ppop = _add_case_columns(ppop, case_name)
    fig, axes = plt.subplots(
        1, 4, figsize=(24, 5.6), sharex=True, sharey=True, constrained_layout=True
    )
    mesh = None
    summary_rows = []

    for j, stype in enumerate(STAR_ORDER):
        p = ppop[ppop["stype_clean"] == stype].copy()
        r = rocky[rocky["stype_clean"] == stype].copy()

        det_grid, support = fraction_grid(p, p["detected_case"], p["denominator_case"])

        ax = axes[j]
        mesh = ax.pcolormesh(
            INSOLATION_BINS, PLANET_RADIUS_BINS, det_grid,
            shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED,
        )
        _add_contours(ax, det_grid)
        _overlay_rocky(
            ax, r,
            label="Rocky PSCompPars (≤ rocky threshold)" if j == 0 else None,
        )
        _setup_axis(ax, f"{stype} stars\nN_rocky = {len(r)}")

        summary_rows.append({
            "case": case_name,
            "host_type": stype,
            "ppop_transiting_support": int(support.sum()),
            "rocky_planets_overlaid": len(r),
        })

    # Legend
    legend_handle = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="crimson",
        markersize=7,
        label=(
            f"Rocky PSCompPars (radius ≤ rocky threshold)\n"
            f"Rocky threshold shift: {shift:+.3f} R⊕\n"
            f"Anchor: LHS 1140 b  ({LHS1140B_MASS_MEARTH} M⊕, {LHS1140B_RADIUS_REARTH} R⊕)"
        ),
    )
    axes[0].legend(handles=[legend_handle], loc="upper right", fontsize=6.5)

    fig.colorbar(mesh, ax=axes, label="Detected fraction", shrink=0.92)
    fig.suptitle(
        f"Kepler FGKM — P-Pop detected-fraction background + rocky PSCompPars overlay  [{case_name}]\n"
        f"Rocky threshold = ref.ddat pure-rock curve shifted {shift:+.3f} R⊕  "
        f"(anchored at LHS 1140 b)",
        fontsize=12,
    )

    out = OUT_DIR / f"rocky_threshold_fgkm_insolation_radius_{case_name}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FGKM figure: {out}")
    return pd.DataFrame(summary_rows), out


# ── Mass-radius diagnostic ────────────────────────────────────────────────────


def plot_mr_diagnostic(
    m_ref: np.ndarray,
    r_ref: np.ndarray,
    shift: float,
    nasa_all: pd.DataFrame,
    rocky: pd.DataFrame,
) -> Path:
    """
    Mass-radius plot showing:
      - original pure-rock curve (dashed black)
      - rocky threshold (solid red)
      - all quality-filtered PSCompPars (grey)
      - rocky-filtered subset (crimson)
      - LHS 1140 b anchor (blue star)
    """
    XLIM = (0.0, 12.0)
    YLIM = (0.5, 2.2)

    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)
    r_rocky     = np.interp(m_line, m_ref, r_ref,         left=np.nan, right=np.nan)
    r_threshold = np.interp(m_line, m_ref, r_ref + shift, left=np.nan, right=np.nan)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    ax.scatter(
        nasa_all["mass_p"], nasa_all["radius_p"],
        s=10, c="0.65", alpha=0.35, linewidths=0,
        label=f"PSCompPars quality-filtered (N={len(nasa_all):,})",
        zorder=1,
    )
    ax.scatter(
        rocky["mass_p"], rocky["radius_p"],
        s=14, c="crimson", alpha=0.75, linewidths=0,
        label=f"Rocky (≤ rocky threshold, N={len(rocky):,})",
        zorder=2,
    )

    ax.plot(m_line, r_rocky,     "k--", lw=1.5, label="Pure-rock curve (ref.ddat)", zorder=3)
    ax.plot(m_line, r_threshold, color="crimson", lw=2.0,
            label=f"Rocky threshold (+{shift:.3f} R⊕)", zorder=3)

    ax.scatter(
        [LHS1140B_MASS_MEARTH], [LHS1140B_RADIUS_REARTH],
        s=120, marker="*", color="dodgerblue", edgecolors="navy",
        linewidths=0.8, zorder=5, label="LHS 1140 b (anchor)",
    )
    ax.annotate(
        "LHS 1140 b",
        xy=(LHS1140B_MASS_MEARTH, LHS1140B_RADIUS_REARTH),
        xytext=(0.22, 0.04), textcoords="offset points",
        fontsize=8.5, color="navy",
    )

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    ax.set_title(
        "Rocky threshold diagnostic — pure-rock curve shifted to pass through LHS 1140 b",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")

    out = OUT_DIR / "rocky_threshold_diagnostic_mass_radius.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved M-R diagnostic: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("35_rocky_threshold_fgkm_insolation_radius.py")
    print("=" * 70)
    print(f"Project root : {ROOT}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"LHS 1140 b anchor: {LHS1140B_MASS_MEARTH} M⊕, {LHS1140B_RADIUS_REARTH} R⊕")
    print()

    # 1. Rocky threshold
    m_ref, r_ref = load_rocky_reference_curve()
    shift = compute_rocky_threshold_shift(m_ref, r_ref)
    print()

    # 2. NASA PSCompPars → quality filter → rocky filter
    nasa_all, rocky = load_and_filter_nasa(m_ref, r_ref, shift)
    rocky.to_csv(OUT_DIR / "rocky_pscomppars_below_threshold.csv", index=False)
    nasa_all.to_csv(OUT_DIR / "nasa_pscomppars_quality_filtered.csv", index=False)
    print(f"Saved {len(rocky):,} rocky planets to rocky_pscomppars_below_threshold.csv")
    print()

    # 3. Mass-radius diagnostic
    plot_mr_diagnostic(m_ref, r_ref, shift, nasa_all, rocky)
    print()

    # 4. P-Pop FGKM background + rocky overlay
    ppop = load_ppop()
    print()

    all_summaries = []
    for case_name in RUN_CASES:
        print(f"Running case: {case_name}")
        summary, fig_path = plot_rocky_fgkm(ppop, rocky, case_name, shift)
        all_summaries.append(summary)
        print()

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_csv = OUT_DIR / "rocky_fgkm_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved to:\n  {OUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print("\nSCRIPT FAILED")
        traceback.print_exc()
        sys.exit(1)
