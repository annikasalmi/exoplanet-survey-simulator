"""
54_tess_rocky_fgkm_detection.py

TESS version of script 35.  Same rocky-threshold logic (LHS 1140 b anchor,
ref.ddat pure-rock boundary), same NASA PSCompPars quality filter, but the
detection-fraction background uses the TESS P-Pop catalogue
(Gaia_C_F_K_combined_cdpp_v1) instead of the Kepler one.

Detection rule (per planet, per P-Pop run):
    tess_observed AND tess_transiting_geometric AND tess_star_bright_enough
    AND tess_enough_transits AND (tess_snr / noise_scale) >= snr_threshold

Denominator: tess_observed AND tess_transiting_geometric
  (different from Kepler, where the whole field is always observed)

Figures produced:
  rocky_threshold_diagnostic_mass_radius.png   — M-R diagnostic
  rocky_threshold_fgkm_insolation_radius_<case>.png — 1×4 FGKM heatmap

Rocky threshold anchor (Cadieux et al. 2024, JWST era):
    LHS 1140 b:  M_p = 5.60 M_earth,  R_p = 1.730 R_earth

Run from repo root:
    python scripts/36_tess_rocky_threshold_fgkm_insolation_radius.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Project root ──────────────────────────────────────────────────────────────

def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "tess").exists():
            return p
    return start_path.parents[2]


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

PPOP_DATA_DIR = ROOT / "run" / "tess" / "data" / "Gaia_C_F_K_combined_cdpp_v1"

REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)
NASA_FLAGS_CACHE = (
    NASA_DATA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)

OUT_DIR = ROOT / "my_outputs" / "54_tess_rocky_fgkm_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LHS 1140 b anchor ────────────────────────────────────────────────────────

LHS1140B_MASS_MEARTH   = 5.60
LHS1140B_RADIUS_REARTH = 1.730

# ── NASA quality settings (mirrors script 35) ─────────────────────────────────

FORCE_REDOWNLOAD_NASA    = False
DOWNLOAD_NASA_IF_MISSING = True

EXCLUDE_MASS_LIMITS       = True
EXCLUDE_RADIUS_LIMITS     = True
EXCLUDE_CALCULATED_MASSES = True   # drops M-R relationship masses
REQUIRE_TWO_SIDED_MASS    = True
REQUIRE_TWO_SIDED_RADIUS  = True

MAX_MASS_REL_UNCERTAINTY   = 0.25
MAX_RADIUS_REL_UNCERTAINTY = 0.08

# ── TESS detection / grid settings ────────────────────────────────────────────

STAR_ORDER = ["F", "G", "K", "M"]

CASES = {
    "optimistic":   {"snr_threshold": 6.5, "noise_scale": 0.8},
    "baseline":     {"snr_threshold": 7.1, "noise_scale": 1.0},
    "conservative": {"snr_threshold": 7.7, "noise_scale": 1.2},
}
RUN_CASES = ["baseline"]

RADIUS_LIMITS     = (0.5, 4.0)
INSOLATION_LIMITS = (0.1, 1e5)

INSOLATION_BINS    = np.logspace(np.log10(INSOLATION_LIMITS[0]), np.log10(INSOLATION_LIMITS[1]), 18)
PLANET_RADIUS_BINS = np.logspace(np.log10(RADIUS_LIMITS[0]),     np.log10(RADIUS_LIMITS[1]), 13)

MIN_BIN_COUNT = 2
CMAP_DETECTED = "viridis"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 250,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})


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
    teff_col = first_col(df, ["teff_s", "temp_s", "st_teff"])
    if teff_col:
        df["stype_clean"] = infer_star_type_from_teff(df[teff_col])
    else:
        df["stype_clean"] = "Unknown"
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
        return m, m ** 0.27
    ref = np.loadtxt(REF_CURVE_PATH, comments="#")
    m_ref = ref[:, 0].astype(float)
    r_ref = ref[:, 1].astype(float)
    good = np.isfinite(m_ref) & np.isfinite(r_ref) & (m_ref > 0) & (r_ref > 0)
    m_ref, r_ref = m_ref[good], r_ref[good]
    order = np.argsort(m_ref)
    return m_ref[order], r_ref[order]


def compute_rocky_threshold_shift(m_ref: np.ndarray, r_ref: np.ndarray) -> float:
    r_at_lhs = float(np.interp(LHS1140B_MASS_MEARTH, m_ref, r_ref))
    shift = LHS1140B_RADIUS_REARTH - r_at_lhs
    print(
        f"Rocky curve at LHS 1140 b mass ({LHS1140B_MASS_MEARTH} M⊕): {r_at_lhs:.4f} R⊕\n"
        f"LHS 1140 b radius: {LHS1140B_RADIUS_REARTH} R⊕  →  rocky threshold shift = {shift:+.4f} R⊕"
    )
    return shift


def rocky_threshold_at_mass(masses: np.ndarray, m_ref, r_ref, shift: float) -> np.ndarray:
    return np.interp(
        np.asarray(masses, dtype=float),
        m_ref, r_ref + shift,
        left=np.nan, right=np.nan,
    )


# ── NASA PSCompPars ───────────────────────────────────────────────────────────

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
        raise FileNotFoundError(f"NASA cache missing: {NASA_FLAGS_CACHE}")
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


def load_and_filter_nasa(m_ref, r_ref, shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all_quality_filtered, rocky_filtered) DataFrames."""
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
        "mass_limit_flag", "radius_limit_flag",
        "teff_s",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()
    df = df[(df["mass_p"] > 0) & (df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()
    print(f"NASA rows with valid M / R / flux: {n0:,} → {len(df):,}")

    if EXCLUDE_MASS_LIMITS:
        df = df[~df["mass_limit_flag"].fillna(0).ne(0)].copy()
        print(f"  remove mass limit flags: → {len(df):,}")

    if EXCLUDE_CALCULATED_MASSES and "mass_provider" in df.columns:
        calc = df["mass_provider"].astype(str).str.contains("M-R relationship|Calculated", case=False, na=False)
        df = df[~calc].copy()
        print(f"  remove M-R relationship / calculated masses: → {len(df):,}")
    elif EXCLUDE_CALCULATED_MASSES and "mass_reference" in df.columns:
        calc = df["mass_reference"].astype(str).str.contains("CALCULATED_VALUE|Calculated Value", case=False, na=False)
        df = df[~calc].copy()
        print(f"  remove calculated masses (via reflink): → {len(df):,}")

    if EXCLUDE_RADIUS_LIMITS:
        df = df[~df["radius_limit_flag"].fillna(0).ne(0)].copy()
        print(f"  remove radius limit flags: → {len(df):,}")

    if REQUIRE_TWO_SIDED_MASS:
        df = df[df["mass_err_plus"].notna() & df["mass_err_minus"].notna()].copy()
        print(f"  require two-sided mass errors: → {len(df):,}")

    if REQUIRE_TWO_SIDED_RADIUS:
        df = df[df["radius_err_plus"].notna() & df["radius_err_minus"].notna()].copy()
        print(f"  require two-sided radius errors: → {len(df):,}")

    mass_err = pd.concat([df["mass_err_plus"].abs(), df["mass_err_minus"].abs()], axis=1).max(axis=1)
    mass_rel = mass_err / df["mass_p"].abs()
    df = df[~(mass_rel.notna() & (mass_rel > MAX_MASS_REL_UNCERTAINTY))].copy()
    print(f"  mass uncertainty ≤ {MAX_MASS_REL_UNCERTAINTY:.0%}: → {len(df):,}")

    rad_err = pd.concat([df["radius_err_plus"].abs(), df["radius_err_minus"].abs()], axis=1).max(axis=1)
    rad_rel = rad_err / df["radius_p"].abs()
    df = df[~(rad_rel.notna() & (rad_rel > MAX_RADIUS_REL_UNCERTAINTY))].copy()
    print(f"  radius uncertainty ≤ {MAX_RADIUS_REL_UNCERTAINTY:.0%}: → {len(df):,}")

    df = add_stype_clean(df)
    df["planet_label"] = df.get("planet_name", pd.Series("", index=df.index)).fillna("").astype(str)
    df["mass_rel_uncertainty"]   = mass_rel.reindex(df.index)
    df["radius_rel_uncertainty"] = rad_rel.reindex(df.index)
    print(f"NASA after all quality filters: {len(df):,} planets")

    threshold_r = rocky_threshold_at_mass(df["mass_p"].to_numpy(), m_ref, r_ref, shift)
    is_rocky = (df["radius_p"].to_numpy() <= threshold_r) & np.isfinite(threshold_r)
    df["rocky_threshold_radius"] = threshold_r
    df["below_rocky_threshold"]  = is_rocky
    rocky = df[is_rocky].copy()
    print(f"Rocky threshold filter: {len(df):,} → {len(rocky):,} rocky planets")
    return df, rocky


# ── TESS P-Pop ────────────────────────────────────────────────────────────────

def load_ppop() -> pd.DataFrame:
    files = sorted(PPOP_DATA_DIR.glob("tess_catalog_*.csv"))
    if not files:
        raise FileNotFoundError(f"No tess_catalog_*.csv found in:\n  {PPOP_DATA_DIR}")
    frames = []
    for p in files:
        d = pd.read_csv(p)
        if "run" not in d.columns:
            m = re.search(r"tess_catalog_(\d+)\.csv", p.name)
            d["run"] = int(m.group(1)) if m else len(frames)
        d["source_file"] = p.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} TESS P-Pop file(s) from {PPOP_DATA_DIR.name}: {len(df):,} rows")
    return _prepare_ppop(df)


def _prepare_ppop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column aliases
    if "flux_p" not in df.columns:
        c = first_col(df, ["pl_insol", "insolation"])
        if c:
            df = df.rename(columns={c: "flux_p"})
    if "radius_p" not in df.columns:
        c = first_col(df, ["pl_rade", "planet_radius"])
        if c:
            df = df.rename(columns={c: "radius_p"})

    # Backward-compatible detection columns
    if "tess_transiting_geometric" not in df.columns and "transiting_geometric" in df.columns:
        df["tess_transiting_geometric"] = df["transiting_geometric"]
    if "tess_star_bright_enough" not in df.columns:
        df["tess_star_bright_enough"] = True
    if "tess_observed" not in df.columns:
        df["tess_observed"] = True
    if "tess_enough_transits" not in df.columns:
        if "tess_n_transits" in df.columns:
            df["tess_enough_transits"] = pd.to_numeric(df["tess_n_transits"], errors="coerce") >= 2
        else:
            df["tess_enough_transits"] = True

    required = ["radius_p", "flux_p", "tess_snr",
                "tess_observed", "tess_transiting_geometric",
                "tess_star_bright_enough", "tess_enough_transits"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"TESS P-Pop missing required columns: {missing}")

    for col in ["flux_p", "radius_p", "tess_snr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tess_observed", "tess_transiting_geometric",
                "tess_star_bright_enough", "tess_enough_transits"]:
        df[col] = as_bool(df[col])

    df = add_stype_clean(df)
    df = df.dropna(subset=["flux_p", "radius_p", "tess_snr"]).copy()
    df = restrict_science_window(df)

    print(f"TESS P-Pop in science window: {len(df):,} rows")
    print("  host types:", df["stype_clean"].value_counts().reindex(STAR_ORDER).fillna(0).astype(int).to_dict())
    obs_tr = df["tess_observed"] & df["tess_transiting_geometric"]
    print(f"  observed + transiting (denominator): {obs_tr.sum():,}")
    return df


def _add_case_columns(df: pd.DataFrame, case_name: str) -> pd.DataFrame:
    case = CASES[case_name]
    df = df.copy()
    snr = pd.to_numeric(df["tess_snr"], errors="coerce").fillna(0.0) / case["noise_scale"]
    df["detected_case"] = (
        df["tess_observed"]
        & df["tess_transiting_geometric"]
        & df["tess_star_bright_enough"]
        & df["tess_enough_transits"]
        & (snr >= case["snr_threshold"])
    )
    # Denominator: planets TESS could in principle see transiting
    df["denominator_case"] = df["tess_observed"] & df["tess_transiting_geometric"]
    return df


# ── Grid helpers ──────────────────────────────────────────────────────────────

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
        cs = ax.contour(X, Y, grid, levels=[0.2, 0.5, 0.8],
                        colors="white", linewidths=0.8, alpha=0.8)
        ax.clabel(cs, fmt="%.1f", fontsize=7)
    except Exception:
        pass


def _setup_axis(ax, title: str):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(INSOLATION_LIMITS)
    ax.set_ylim(RADIUS_LIMITS)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"Insolation flux [$I_\oplus$]")
    ax.set_ylabel(r"Planet radius [$R_\oplus$]")
    ax.grid(True, which="both", alpha=0.18)


# ── Overlay helpers ───────────────────────────────────────────────────────────

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

    lhs_mask = sub["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False, na=False, regex=True)
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


# ── Main plots ────────────────────────────────────────────────────────────────

def plot_rocky_fgkm(ppop: pd.DataFrame, rocky: pd.DataFrame, case_name: str, shift: float) -> pd.DataFrame:
    ppop = _add_case_columns(ppop, case_name)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    summary_rows = []

    for j, stype in enumerate(STAR_ORDER):
        p = ppop[ppop["stype_clean"] == stype].copy()
        r = rocky[rocky["stype_clean"] == stype].copy()

        det_grid, support = fraction_grid(p, p["detected_case"], p["denominator_case"])

        ax = axes[j]
        if np.isfinite(det_grid).any():
            mesh = ax.pcolormesh(
                INSOLATION_BINS, PLANET_RADIUS_BINS, det_grid,
                shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED,
            )
            _add_contours(ax, det_grid)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

        _overlay_rocky(ax, r,
                       label="Rocky PSCompPars (≤ rocky threshold)" if j == 0 else None)
        _setup_axis(ax, f"{stype} stars\nN_ppop={len(p):,}  N_rocky={len(r)}")

        summary_rows.append({
            "case": case_name,
            "host_type": stype,
            "ppop_rows": len(p),
            "ppop_denominator": int(p["denominator_case"].sum()),
            "ppop_detected": int(p["detected_case"].sum()),
            "rocky_planets_overlaid": len(r),
        })

    legend_handle = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=7,
        label=(
            f"Rocky PSCompPars (radius ≤ rocky threshold)\n"
            f"Rocky threshold shift: {shift:+.3f} R⊕\n"
            f"Anchor: LHS 1140 b  ({LHS1140B_MASS_MEARTH} M⊕, {LHS1140B_RADIUS_REARTH} R⊕)"
        ),
    )
    axes[0].legend(handles=[legend_handle], loc="upper right", fontsize=6.5)

    if mesh:
        fig.colorbar(mesh, ax=axes.tolist(), label="Detected fraction (observed-transiting P-Pop)", shrink=0.92)
    fig.suptitle(
        f"TESS FGKM — P-Pop detected-fraction background + rocky PSCompPars overlay  [{case_name}]\n"
        f"Rocky threshold = ref.ddat pure-rock curve shifted {shift:+.3f} R⊕  "
        f"(anchored at LHS 1140 b)  |  P-Pop: {PPOP_DATA_DIR.name}",
        fontsize=11,
    )

    out = OUT_DIR / f"rocky_threshold_fgkm_insolation_radius_{case_name}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FGKM figure: {out}")
    return pd.DataFrame(summary_rows)


def plot_mr_diagnostic(m_ref, r_ref, shift: float, nasa_all: pd.DataFrame, rocky: pd.DataFrame) -> None:
    XLIM = (0.0, 12.0)
    YLIM = (0.5, 2.2)
    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)
    r_rocky     = np.interp(m_line, m_ref, r_ref,         left=np.nan, right=np.nan)
    r_threshold = np.interp(m_line, m_ref, r_ref + shift, left=np.nan, right=np.nan)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(nasa_all["mass_p"], nasa_all["radius_p"],
               s=10, c="0.65", alpha=0.35, linewidths=0,
               label=f"PSCompPars quality-filtered (N={len(nasa_all):,})", zorder=1)
    ax.scatter(rocky["mass_p"], rocky["radius_p"],
               s=14, c="crimson", alpha=0.75, linewidths=0,
               label=f"Rocky (≤ rocky threshold, N={len(rocky):,})", zorder=2)
    ax.plot(m_line, r_rocky,     "k--", lw=1.5, label="Pure-rock curve (ref.ddat)", zorder=3)
    ax.plot(m_line, r_threshold, color="crimson", lw=2.0,
            label=f"Rocky threshold (+{shift:.3f} R⊕)", zorder=3)
    ax.scatter([LHS1140B_MASS_MEARTH], [LHS1140B_RADIUS_REARTH],
               s=120, marker="*", color="dodgerblue", edgecolors="navy",
               linewidths=0.8, zorder=5, label="LHS 1140 b (anchor)")
    ax.annotate("LHS 1140 b",
                xy=(LHS1140B_MASS_MEARTH, LHS1140B_RADIUS_REARTH),
                xytext=(0.22, 0.04), textcoords="offset points",
                fontsize=8.5, color="navy")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    ax.set_title("Rocky threshold diagnostic — pure-rock curve shifted to pass through LHS 1140 b", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")

    out = OUT_DIR / "rocky_threshold_diagnostic_mass_radius.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved M-R diagnostic: {out}")


# ── Combined FGKM plot ────────────────────────────────────────────────────────

def plot_rocky_all_fgkm(ppop: pd.DataFrame, rocky: pd.DataFrame, case_name: str, shift: float) -> None:
    """Single panel: all FGKM detection fraction + all rocky dots + trend fit."""
    ppop = _add_case_columns(ppop, case_name)

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    # Combined detection fraction across all FGKM
    det_grid, _ = fraction_grid(ppop, ppop["detected_case"], ppop["denominator_case"])
    if np.isfinite(det_grid).any():
        mesh = ax.pcolormesh(
            INSOLATION_BINS, PLANET_RADIUS_BINS, det_grid,
            shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED,
        )
        _add_contours(ax, det_grid)
        fig.colorbar(mesh, ax=ax, label="Detected fraction (observed-transiting P-Pop)", shrink=0.92)

    # Restrict rocky to science window
    rocky_win = rocky[
        rocky["flux_p"].between(INSOLATION_LIMITS[0], INSOLATION_LIMITS[1])
        & rocky["radius_p"].between(RADIUS_LIMITS[0], RADIUS_LIMITS[1])
    ].copy()

    # Color by star type
    stype_colors = {"F": "#e6ab02", "G": "#66a61e", "K": "#7570b3", "M": "#d95f02"}
    for stype in STAR_ORDER:
        sub = rocky_win[rocky_win["stype_clean"] == stype]
        if len(sub) == 0:
            continue
        color = stype_colors.get(stype, "crimson")
        yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
        yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
        ax.errorbar(
            sub["flux_p"], sub["radius_p"],
            yerr=[yerr_lo, yerr_hi],
            fmt="o", ms=4, color=color, alpha=0.80,
            elinewidth=0.8, capsize=2, ecolor=color,
            label=f"{stype} stars (N={len(sub)})", zorder=4,
        )

    # OLS trend line in log-log space
    if len(rocky_win) >= 3:
        log_f = np.log10(rocky_win["flux_p"].values)
        log_r = np.log10(rocky_win["radius_p"].values)
        valid = np.isfinite(log_f) & np.isfinite(log_r)
        if valid.sum() >= 3:
            coeffs = np.polyfit(log_f[valid], log_r[valid], 1)
            # Fit range: within the data extent
            f_lo = rocky_win["flux_p"][valid.values if hasattr(valid, "values") else valid].min()
            f_hi = rocky_win["flux_p"][valid.values if hasattr(valid, "values") else valid].max()
            x_fit = np.logspace(np.log10(max(f_lo * 0.5, INSOLATION_LIMITS[0])),
                                np.log10(min(f_hi * 2.0, INSOLATION_LIMITS[1])), 300)
            y_fit = 10 ** np.polyval(coeffs, np.log10(x_fit))
            slope_sign = "+" if coeffs[0] >= 0 else ""
            ax.plot(x_fit, y_fit, "w--", lw=1.8, zorder=5,
                    label=f"OLS fit (log-log): slope = {coeffs[0]:.3f}")

            # 90th-percentile upper envelope (running window in log-flux)
            order = np.argsort(log_f[valid])
            lf_s = log_f[valid][order]
            lr_s = log_r[valid][order]
            win = max(5, len(lf_s) // 5)
            upper_lf, upper_lr = [], []
            for i in range(len(lf_s)):
                i0, i1 = max(0, i - win // 2), min(len(lf_s), i + win // 2 + 1)
                upper_lf.append(lf_s[i])
                upper_lr.append(np.percentile(lr_s[i0:i1], 90))
            ax.plot(10 ** np.array(upper_lf), 10 ** np.array(upper_lr),
                    color="white", lw=1.2, ls=":", alpha=0.75, zorder=5,
                    label="90th-pct upper envelope")

    # LHS 1140 b annotation
    lhs_mask = rocky_win["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False, na=False, regex=True)
    for _, row in rocky_win[lhs_mask].iterrows():
        ax.scatter([row["flux_p"]], [row["radius_p"]],
                   s=80, marker="*", color="gold", edgecolors="darkred",
                   linewidths=0.9, zorder=7)
        ax.annotate("LHS 1140 b",
                    xy=(row["flux_p"], row["radius_p"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, color="darkred", fontweight="bold", zorder=8)

    _setup_axis(ax, f"All FGKM rocky PSCompPars  (N={len(rocky_win)})  —  combined FGKM P-Pop background")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.75)

    fig.suptitle(
        f"TESS FGKM — Combined detection fraction + all rocky PSCompPars  [{case_name}]\n"
        f"Rocky threshold = ref.ddat shifted {shift:+.3f} R⊕ (anchored at LHS 1140 b)"
        f"  |  P-Pop: {PPOP_DATA_DIR.name}",
        fontsize=10,
    )

    out = OUT_DIR / f"rocky_all_fgkm_combined_{case_name}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined FGKM figure: {out}")


# ── Standalone rocky scatter (no background) ─────────────────────────────────

def plot_rocky_scatter_standalone(rocky: pd.DataFrame, shift: float) -> None:
    """Insolation vs radius for confirmed rocky planets only — no P-Pop background."""
    rocky_win = rocky[
        rocky["flux_p"].between(INSOLATION_LIMITS[0], INSOLATION_LIMITS[1])
        & rocky["radius_p"].between(RADIUS_LIMITS[0], RADIUS_LIMITS[1])
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    # Insolation-bin shading (matches the bins used in the survival analysis)
    ax.axvspan(INSOLATION_LIMITS[0], 10.0,  color="#4575b4", alpha=0.06, zorder=0, label="I < 10 (cold)")
    ax.axvspan(10.0,                  50.0,  color="#74add1", alpha=0.06, zorder=0, label="10 ≤ I < 50")
    ax.axvspan(50.0, INSOLATION_LIMITS[1],   color="#fdae61", alpha=0.06, zorder=0, label="I ≥ 50 (hot)")
    for x_boundary in [10.0, 50.0]:
        ax.axvline(x_boundary, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    # Dots colored by host star type
    stype_colors = {"F": "#e6ab02", "G": "#66a61e", "K": "#7570b3", "M": "#d95f02"}
    for stype in STAR_ORDER:
        sub = rocky_win[rocky_win["stype_clean"] == stype]
        if len(sub) == 0:
            continue
        color = stype_colors.get(stype, "gray")
        yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
        yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
        ax.errorbar(
            sub["flux_p"], sub["radius_p"],
            yerr=[yerr_lo, yerr_hi],
            fmt="o", ms=5, color=color, alpha=0.85,
            elinewidth=0.9, capsize=2.5, ecolor=color,
            label=f"{stype} stars (N={len(sub)})", zorder=4,
        )

    # Running 90th-pct upper envelope in log-log space
    log_f = np.log10(rocky_win["flux_p"].values)
    log_r = np.log10(rocky_win["radius_p"].values)
    valid = np.isfinite(log_f) & np.isfinite(log_r)
    if valid.sum() >= 5:
        order = np.argsort(log_f[valid])
        lf_s = log_f[valid][order]
        lr_s = log_r[valid][order]
        win = max(5, len(lf_s) // 5)
        upper_lf, upper_lr = [], []
        for i in range(len(lf_s)):
            i0 = max(0, i - win // 2)
            i1 = min(len(lf_s), i + win // 2 + 1)
            upper_lf.append(lf_s[i])
            upper_lr.append(np.percentile(lr_s[i0:i1], 90))
        ax.plot(10 ** np.array(upper_lf), 10 ** np.array(upper_lr),
                "k--", lw=1.6, alpha=0.55, zorder=5,
                label="90th-pct upper envelope")

    # Mark LHS 1140 b (the rocky threshold anchor)
    lhs_mask = rocky_win["planet_label"].str.contains(
        r"LHS\s*1140\s*b", case=False, na=False, regex=True
    )
    for _, row in rocky_win[lhs_mask].iterrows():
        ax.scatter([row["flux_p"]], [row["radius_p"]],
                   s=100, marker="*", color="gold", edgecolors="darkred",
                   linewidths=1.0, zorder=7)
        ax.annotate(
            "LHS 1140 b",
            xy=(row["flux_p"], row["radius_p"]),
            xytext=(6, 5), textcoords="offset points",
            fontsize=8.5, color="darkred", fontweight="bold", zorder=8,
        )

    _setup_axis(ax, f"Confirmed rocky planets — insolation vs radius  (N={len(rocky_win)})")
    ax.set_facecolor("#fafafa")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.90)

    fig.suptitle(
        f"NASA PSCompPars confirmed rocky planets  (M–R below rocky threshold)\n"
        f"Rocky threshold = ref.ddat shifted {shift:+.3f} R⊕ (anchored at LHS 1140 b)\n"
        f"Shaded columns = insolation bins used in survival analysis",
        fontsize=10,
    )

    out = OUT_DIR / "rocky_scatter_standalone.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved standalone rocky scatter: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("36_tess_rocky_threshold_fgkm_insolation_radius.py")
    print("=" * 70)
    print(f"Project root  : {ROOT}")
    print(f"P-Pop dir     : {PPOP_DATA_DIR}")
    print(f"Output dir    : {OUT_DIR}")
    print(f"LHS 1140 b    : {LHS1140B_MASS_MEARTH} M⊕, {LHS1140B_RADIUS_REARTH} R⊕")
    print()

    m_ref, r_ref = load_rocky_reference_curve()
    shift = compute_rocky_threshold_shift(m_ref, r_ref)
    print()

    nasa_all, rocky = load_and_filter_nasa(m_ref, r_ref, shift)
    rocky.to_csv(OUT_DIR / "rocky_pscomppars_below_threshold.csv", index=False)
    nasa_all.to_csv(OUT_DIR / "nasa_pscomppars_quality_filtered.csv", index=False)
    print(f"Saved {len(rocky):,} rocky planets to rocky_pscomppars_below_threshold.csv")
    print()

    plot_mr_diagnostic(m_ref, r_ref, shift, nasa_all, rocky)
    plot_rocky_scatter_standalone(rocky, shift)
    print()

    ppop = load_ppop()
    print()

    all_summaries = []
    for case_name in RUN_CASES:
        print(f"Running case: {case_name}")
        summary = plot_rocky_fgkm(ppop, rocky, case_name, shift)
        all_summaries.append(summary)
        plot_rocky_all_fgkm(ppop, rocky, case_name, shift)
        print()

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv(OUT_DIR / "rocky_fgkm_summary.csv", index=False)
    print("Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved to:\n  {OUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
