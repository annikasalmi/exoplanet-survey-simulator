"""
56_kepler_tess_rocky_fgkm_gaia60pc.py

New-universe version of script 44. Reads the P-Pop catalogs generated from the
Gaia-60pc star catalog, stacking up to N_UNIVERSES seeded universes per mission
to smooth the (locally sparse) F/G/K detection background:
    Kepler : run/kepler/data/Gaia/kepler_catalog_{0..N-1}.csv
    TESS   : run/tess/data/Gaia_cdpp_v1/tess_catalog_{0..N-1}.csv
Files are addressed by explicit index (0..N_UNIVERSES-1), NOT globbed, so stale
higher-numbered catalogs left in the same folder by older runs are ignored.
Each universe shares the same fixed star catalog with a different RNG seed, so
stacking is a valid Monte-Carlo bootstrap that raises planets-per-bin without
distorting the underlying occurrence statistics.

Combines the Kepler (script 35) and TESS (script 36) rocky-threshold FGKM
detection-fraction panels into a single 2x4 figure:

    Row 0 (top)    : Kepler P-Pop detected-fraction background  (F G K M)
    Row 1 (bottom) : TESS   P-Pop detected-fraction background  (F G K M)

The same set of NASA PSCompPars "rocky" planets (radius <= rocky threshold,
anchored at LHS 1140 b) is overlaid on every panel.  Each real planet is
drawn with two-sided radius error bars and colored by the telescope / mission
that discovered it (disc_facility).  A shared legend lists every facility that
contributed more than 2 rocky planets inside the plotted science window;
everything else is collapsed into a single "Other" entry.

Rocky threshold anchor (Cadieux et al. 2024, JWST era):
    LHS 1140 b:  M_p = 5.60 M_earth,  R_p = 1.730 R_earth

Run from repo root:
    python scripts/56_kepler_tess_rocky_fgkm_gaia60pc.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter


# ── Project root ──────────────────────────────────────────────────────────────

def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists() and (p / "run" / "tess").exists():
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

# Gaia-60pc P-Pop universes. Each seeded run i writes <stem>_<i>.csv. We stack
# the first N_UNIVERSES of them to smooth the (locally sparse) FGK detection
# background. We build the file list explicitly from indices 0..N-1 rather than
# globbing, so any stale higher-numbered catalogs left in the folder from older
# multi-universe runs are ignored.
N_UNIVERSES = 10

# Top-up universes produced by run/generate_gk_8000.py are numbered from here.
# They are single-spectral-type (G-only or K-only) Gaia-60pc / Bergsten2022
# universes generated to push the (locally sparse) G and K detection background
# past ~3000 transiting planets so the per-type panels smooth out. They are
# stacked IN ADDITION to the original universes 0..N_UNIVERSES-1; because each
# only contains one spectral type, it adds counts solely to that type's panel.
EXTRA_START_INDEX = 8001

KEPLER_PPOP_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
TESS_PPOP_DIR   = ROOT / "run" / "tess"   / "data" / "Gaia_cdpp_v1"


def _ppop_files(directory: Path, stem: str) -> list[Path]:
    """Return <stem>_<i>.csv for i in 0..N_UNIVERSES-1 PLUS any top-up universes
    with index >= EXTRA_START_INDEX, in order."""
    wanted = [directory / f"{stem}_{i}.csv" for i in range(N_UNIVERSES)]
    present = [f for f in wanted if f.exists()]
    if not present:
        raise FileNotFoundError(
            f"No P-Pop catalogs found in {directory} for stem '{stem}' "
            f"(expected {stem}_0.csv .. {stem}_{N_UNIVERSES - 1}.csv)"
        )
    missing = [f.name for f in wanted if not f.exists()]
    if missing:
        print(f"  [warn] {len(missing)} expected universe(s) missing, "
              f"stacking {len(present)}: missing {missing}")

    # Additively stack any top-up (G/K-only) universes numbered >= EXTRA_START_INDEX.
    extra = []
    for f in directory.glob(f"{stem}_*.csv"):
        try:
            idx = int(f.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if idx >= EXTRA_START_INDEX:
            extra.append((idx, f))
    extra.sort()
    if extra:
        print(f"  + stacking {len(extra)} top-up universe(s) "
              f"(idx >= {EXTRA_START_INDEX}) for stem '{stem}'")
    return present + [f for _, f in extra]

# Pure-rock reference curve (kept as the BLACK comparison line in the M-R
# diagnostic only; it no longer defines the threshold).
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

# Rocky / silicate mass-radius curve supplied by the professor (Hongyi-silicon.ddat).
# Column 0 = mass [M_earth], column 1 = radius [R_earth]; the rest are unused
# interior-model outputs. THIS curve defines the RED rocky threshold used to
# filter out puffy (non-rocky) planets everywhere in this script. The loader
# reads cols 0,1 and the threshold is anchored to LHS 1140 b via an automatic
# (here ~0) vertical shift.
ROCKY_CURVE_PATH  = ROOT / "Hongyi-silicon.ddat"
ROCKY_CURVE_LABEL = "silicate rocky curve (Hongyi-silicon.ddat)"

NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)
NASA_FLAGS_CACHE = (
    NASA_DATA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)

OUT_DIR = ROOT / "output/plots" / "56_kepler_tess_rocky_fgkm_gaia60pc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LHS 1140 b anchor ────────────────────────────────────────────────────────

LHS1140B_MASS_MEARTH   = 5.60
LHS1140B_RADIUS_REARTH = 1.730

# ── NASA quality settings (mirror scripts 35 / 36) ───────────────────────────

FORCE_REDOWNLOAD_NASA    = False
DOWNLOAD_NASA_IF_MISSING = True

EXCLUDE_MASS_LIMITS       = True
EXCLUDE_RADIUS_LIMITS     = True
EXCLUDE_CALCULATED_MASSES = True
REQUIRE_TWO_SIDED_MASS    = True
REQUIRE_TWO_SIDED_RADIUS  = True

MAX_MASS_REL_UNCERTAINTY   = 0.25
MAX_RADIUS_REL_UNCERTAINTY = 0.08

# ── Grid / detection settings ─────────────────────────────────────────────────

STAR_ORDER = ["F", "G", "K", "M"]

# Baseline detection case for both missions.
KEPLER_MES_THRESHOLD = 7.1
KEPLER_CDPP_SCALE    = 1.0
TESS_SNR_THRESHOLD   = 7.1
TESS_NOISE_SCALE     = 1.0
CASE_NAME            = "baseline"

RADIUS_LIMITS     = (0.6, 2.2)
INSOLATION_LIMITS = (0.1, 1e4)

INSOLATION_BINS    = np.logspace(np.log10(INSOLATION_LIMITS[0]), np.log10(INSOLATION_LIMITS[1]), 16)
PLANET_RADIUS_BINS = np.logspace(np.log10(RADIUS_LIMITS[0]),     np.log10(RADIUS_LIMITS[1]), 12)

MIN_BIN_COUNT = 2
CMAP_DETECTED = "viridis"

# Facilities with at least this many rocky planets in-window get their own color.
FACILITY_MIN_COUNT = 2   # "> 2" planets  →  strictly greater than 2

# Pretty short names for legend labels.
FACILITY_RELABEL = {
    "Transiting Exoplanet Survey Satellite (TESS)": "TESS",
    "Next-Generation Transit Survey (NGTS)": "NGTS",
}


def short_facility(name: str) -> str:
    return FACILITY_RELABEL.get(name, name)


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
    """Load the rocky-threshold curve (professor's silicate curve, Hongyi-silicon.ddat).

    Returns (mass, radius) sorted by mass. This curve defines the red rocky
    threshold used to separate rocky from puffy planets.
    """
    if not ROCKY_CURVE_PATH.exists():
        print(f"WARNING: {ROCKY_CURVE_PATH} not found — using toy power-law rocky curve.")
        m = np.linspace(0.05, 30.0, 600)
        return m, m ** 0.27
    ref = np.loadtxt(ROCKY_CURVE_PATH, comments="#")
    m_ref = ref[:, 0].astype(float)
    r_ref = ref[:, 1].astype(float)
    good = np.isfinite(m_ref) & np.isfinite(r_ref) & (m_ref > 0) & (r_ref > 0)
    m_ref, r_ref = m_ref[good], r_ref[good]
    order = np.argsort(m_ref)
    return m_ref[order], r_ref[order]


def compute_anchor_shift(m_ref: np.ndarray, r_ref: np.ndarray) -> float:
    """Vertical offset that would move the curve to pass exactly through LHS 1140 b."""
    r_at_lhs = float(np.interp(LHS1140B_MASS_MEARTH, m_ref, r_ref))
    return LHS1140B_RADIUS_REARTH - r_at_lhs


def compute_rocky_threshold_shift(m_ref: np.ndarray, r_ref: np.ndarray) -> float:
    """Operational rocky/puffy cutoff shift.

    We now use the UNSHIFTED silicate curve (Hongyi-silicon.ddat) as the rocky
    cutoff, so this returns 0.0. The LHS 1140 b anchor offset is only reported
    for reference (it is ~0 because the silicate curve already passes through
    LHS 1140 b). The anchored curves are still drawn in the comparison figure
    via compute_anchor_shift().
    """
    r_at_lhs = float(np.interp(LHS1140B_MASS_MEARTH, m_ref, r_ref))
    anchor = compute_anchor_shift(m_ref, r_ref)
    print(
        f"Rocky curve at LHS 1140 b mass ({LHS1140B_MASS_MEARTH} M⊕): {r_at_lhs:.4f} R⊕\n"
        f"Using UNSHIFTED silicate curve as rocky cutoff (shift = +0.000 R⊕; "
        f"LHS 1140 b anchor offset would be {anchor:+.4f} R⊕)"
    )
    return 0.0


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
    if "discovery_facility" not in df.columns:
        df["discovery_facility"] = "Unknown"
    df["discovery_facility"] = df["discovery_facility"].fillna("Unknown").astype(str)
    print(f"NASA after all quality filters: {len(df):,} planets")

    threshold_r = rocky_threshold_at_mass(df["mass_p"].to_numpy(), m_ref, r_ref, shift)
    is_rocky = (df["radius_p"].to_numpy() <= threshold_r) & np.isfinite(threshold_r)
    df["rocky_threshold_radius"] = threshold_r
    df["below_rocky_threshold"]  = is_rocky
    rocky = df[is_rocky].copy()
    print(f"Rocky threshold filter: {len(df):,} → {len(rocky):,} rocky planets")
    return df, rocky


# ── P-Pop loading ─────────────────────────────────────────────────────────────

def load_kepler_ppop() -> pd.DataFrame:
    files = _ppop_files(KEPLER_PPOP_DIR, "kepler_catalog")
    frames = []
    for i, f in enumerate(files):
        d = pd.read_csv(f)
        d["run"] = i                 # distinct universe id (overrides per-file run=0)
        d["source_file"] = f.name
        frames.append(d)
        print(f"  + {f.name}  ({len(d):,} rows)")
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} Kepler P-Pop universe(s): {len(df):,} rows total")
    return _prepare_kepler_ppop(df)


def _standardize_flux_radius(df: pd.DataFrame) -> pd.DataFrame:
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


def _prepare_kepler_ppop(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_flux_radius(df)
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
        raise ValueError(f"Kepler P-Pop missing required columns: {missing}")

    df["transiting_geometric"]   = as_bool(df["transiting_geometric"])
    df["bright_enough_kepler"]   = as_bool(df["bright_enough_kepler"])
    df["kepler_enough_transits"] = as_bool(df["kepler_enough_transits"])

    df = df.dropna(subset=["flux_p", "radius_p", "kepler_mes"]).copy()
    df = restrict_science_window(df)

    mes = pd.to_numeric(df["kepler_mes"], errors="coerce").fillna(0.0) / KEPLER_CDPP_SCALE
    df["detected_case"] = (
        df["transiting_geometric"]
        & df["bright_enough_kepler"]
        & df["kepler_enough_transits"]
        & (mes >= KEPLER_MES_THRESHOLD)
    )
    df["denominator_case"] = df["transiting_geometric"]

    print(f"Kepler P-Pop in science window: {len(df):,} rows")
    return df


def load_tess_ppop() -> pd.DataFrame:
    files = _ppop_files(TESS_PPOP_DIR, "tess_catalog")
    frames = []
    for i, f in enumerate(files):
        d = pd.read_csv(f)
        d["run"] = i                 # distinct universe id (overrides per-file run=0)
        d["source_file"] = f.name
        frames.append(d)
        print(f"  + {f.name}  ({len(d):,} rows)")
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} TESS P-Pop universe(s): {len(df):,} rows total")
    return _prepare_tess_ppop(df)


def _prepare_tess_ppop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "flux_p" not in df.columns:
        c = first_col(df, ["pl_insol", "insolation"])
        if c:
            df = df.rename(columns={c: "flux_p"})
    if "radius_p" not in df.columns:
        c = first_col(df, ["pl_rade", "planet_radius"])
        if c:
            df = df.rename(columns={c: "radius_p"})

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

    snr = pd.to_numeric(df["tess_snr"], errors="coerce").fillna(0.0) / TESS_NOISE_SCALE
    df["detected_case"] = (
        df["tess_observed"]
        & df["tess_transiting_geometric"]
        & df["tess_star_bright_enough"]
        & df["tess_enough_transits"]
        & (snr >= TESS_SNR_THRESHOLD)
    )
    df["denominator_case"] = df["tess_observed"] & df["tess_transiting_geometric"]

    print(f"TESS P-Pop in science window: {len(df):,} rows")
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


# Plain radius tick positions/labels (no 4x10^0 style scientific notation).
RADIUS_TICKS = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]


def _format_radius_axis(ax):
    """Label the log radius axis with plain numbers (0.6, 1, 2 …).

    Forces the numeric labels to show even on shared-y inner panels so the
    radius values are readable on every subplot.
    """
    ticks = [t for t in RADIUS_TICKS if RADIUS_LIMITS[0] <= t <= RADIUS_LIMITS[1]]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelleft=True)


def _setup_axis(ax, title: str, show_xlabel: bool = True, show_ylabel: bool = True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(INSOLATION_LIMITS)
    ax.set_ylim(RADIUS_LIMITS)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel(r"Insolation flux [$I_\oplus$]")
    if show_ylabel:
        ax.set_ylabel(r"Planet radius [$R_\oplus$]")
    _format_radius_axis(ax)
    ax.grid(True, which="both", alpha=0.18)


# ── 90% upper-bound fit ───────────────────────────────────────────────────────

def fit_90pct_line(flux: np.ndarray, radius: np.ndarray, quantile: float = 0.90):
    """
    Fit the q-quantile line through the NASA planets in log-log (flux, radius)
    space via quantile (pinball-loss) regression, i.e. the straight line that
    best follows the data while keeping ~`quantile` of the points below it.

    Returns (slope, intercept) in log10 space, or None if too few points.
    Falls back to an OLS-slope + percentile-offset line if the optimizer fails.
    """
    log_f = np.log10(np.asarray(flux, dtype=float))
    log_r = np.log10(np.asarray(radius, dtype=float))
    valid = np.isfinite(log_f) & np.isfinite(log_r)
    x, y = log_f[valid], log_r[valid]
    if x.size < 4:
        return None

    # OLS as a starting guess / fallback.
    ols_slope, ols_intercept = np.polyfit(x, y, 1)

    def pinball(params):
        a, b = params
        r = y - (a * x + b)
        return np.sum(np.where(r >= 0, quantile * r, (quantile - 1.0) * r))

    try:
        from scipy.optimize import minimize
        res = minimize(pinball, x0=[ols_slope, ols_intercept], method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000})
        if res.success or res.fun < pinball([ols_slope, np.quantile(y - ols_slope * x, quantile)]):
            return float(res.x[0]), float(res.x[1])
    except Exception:
        pass

    # Fallback: OLS slope, intercept raised to the q-th residual percentile.
    intercept = float(np.quantile(y - ols_slope * x, quantile))
    return float(ols_slope), intercept


def line_radius(slope: float, intercept: float, flux) -> np.ndarray:
    return 10 ** (slope * np.log10(np.asarray(flux, dtype=float)) + intercept)


def draw_90pct_line(ax, flux, radius, color="black", lw=2.0, label=None,
                    x_extent=None):
    """Fit and draw the 90% line.  Returns (slope, intercept) or None.

    x_extent=(lo, hi) overrides the flux range over which the line is drawn
    (default: the data's own flux extent).
    """
    fit = fit_90pct_line(flux, radius)
    if fit is None:
        return None
    slope, intercept = fit
    if x_extent is None:
        f = np.asarray(flux, dtype=float)
        f = f[np.isfinite(f) & (f > 0)]
        if f.size == 0:
            return None
        lo, hi = f.min(), f.max()
    else:
        lo, hi = x_extent
    x = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(x, line_radius(slope, intercept, x), color=color, lw=lw, ls="--",
            alpha=0.9, zorder=5, label=label)
    return slope, intercept


COLD_INSOLATION = 10.0  # "I < 10" cold boundary


def false_negative_prob(ppop_panel: pd.DataFrame, slope: float, intercept: float):
    """
    P-Pop false-negative probability in the region {I < 10  AND  radius above
    the 90% line}: the fraction of transiting P-Pop planets there that the
    survey fails to detect.  Returns (fn_prob, n_denominator, n_missed).
    """
    f = pd.to_numeric(ppop_panel["flux_p"], errors="coerce")
    rr = pd.to_numeric(ppop_panel["radius_p"], errors="coerce")
    in_region = (
        (f < COLD_INSOLATION)
        & (rr > line_radius(slope, intercept, f))
        & ppop_panel["denominator_case"].to_numpy()
    )
    n_denom = int(in_region.sum())
    if n_denom == 0:
        return None, 0, 0
    n_det = int((in_region & ppop_panel["detected_case"].to_numpy()).sum())
    n_missed = n_denom - n_det
    return n_missed / n_denom, n_denom, n_missed


# ── Facility colors ───────────────────────────────────────────────────────────

FACILITY_PALETTE = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)


def build_facility_styles(rocky_win: pd.DataFrame):
    """
    Assign a distinct color to every discovery facility that contributed more
    than FACILITY_MIN_COUNT rocky planets inside the plotted window.  Returns
    (color_map, major_facilities_in_order, counts_series).
    """
    counts = rocky_win["discovery_facility"].fillna("Unknown").value_counts()
    major = [f for f, c in counts.items() if c > FACILITY_MIN_COUNT]
    color_map = {f: FACILITY_PALETTE[i % len(FACILITY_PALETTE)] for i, f in enumerate(major)}
    return color_map, major, counts


OTHER_COLOR = "0.55"


def _overlay_rocky_by_facility(ax, sub: pd.DataFrame, color_map: dict, major: list[str]):
    """Overlay rocky planets in one panel, colored & error-barred by facility."""
    if len(sub) == 0:
        return

    yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
    yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
    fac = sub["discovery_facility"].fillna("Unknown").values

    # Major facilities (own color), then everything else as "Other".
    for f in major:
        m = fac == f
        if not m.any():
            continue
        ax.errorbar(
            sub["flux_p"].values[m], sub["radius_p"].values[m],
            yerr=[yerr_lo[m], yerr_hi[m]],
            fmt="o", ms=4, color=color_map[f], alpha=0.85,
            elinewidth=0.8, capsize=2, ecolor=color_map[f],
            zorder=4,
        )

    other = ~np.isin(fac, major)
    if other.any():
        ax.errorbar(
            sub["flux_p"].values[other], sub["radius_p"].values[other],
            yerr=[yerr_lo[other], yerr_hi[other]],
            fmt="o", ms=4, color=OTHER_COLOR, alpha=0.70,
            elinewidth=0.7, capsize=1.5, ecolor=OTHER_COLOR,
            zorder=3,
        )

    # Highlight LHS 1140 b if it landed in this panel.
    lhs_mask = sub["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False, na=False, regex=True)
    for _, row in sub[lhs_mask].iterrows():
        ax.scatter(
            [row["flux_p"]], [row["radius_p"]],
            s=70, marker="*", color="gold", edgecolors="darkred",
            linewidths=0.9, zorder=6,
        )
        ax.annotate(
            "LHS 1140 b",
            xy=(row["flux_p"], row["radius_p"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7, color="darkred", fontweight="bold", zorder=7,
        )


# ── Combined 2x4 figure ───────────────────────────────────────────────────────

def restrict_to_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only planets inside the plotted science (context) window."""
    return df[
        df["flux_p"].between(INSOLATION_LIMITS[0], INSOLATION_LIMITS[1])
        & df["radius_p"].between(RADIUS_LIMITS[0], RADIUS_LIMITS[1])
        & df["stype_clean"].isin(STAR_ORDER)
    ].copy()


def plot_combined(kepler: pd.DataFrame, tess: pd.DataFrame,
                  rocky_win: pd.DataFrame, shift: float,
                  color_map: dict, major: list[str], counts) -> Path:
    print("\nFacilities with > {} rocky planets in window:".format(FACILITY_MIN_COUNT))
    for f in major:
        print(f"  {short_facility(f):<10s} N={counts[f]}")
    n_other = int(len(rocky_win) - sum(counts[f] for f in major))
    print(f"  Other      N={n_other}")

    rows = [
        ("Kepler", kepler),
        ("TESS",   tess),
    ]

    fig, axes = plt.subplots(
        2, 4, figsize=(24, 11), sharex=True, sharey=True, constrained_layout=True
    )
    mesh = None

    print("\nCold-region false-negative probabilities (I<10, radius above 90% fit):")
    for i, (mission, ppop) in enumerate(rows):
        for j, stype in enumerate(STAR_ORDER):
            ax = axes[i, j]
            p = ppop[ppop["stype_clean"] == stype].copy()
            r = rocky_win[rocky_win["stype_clean"] == stype].copy()

            n_bg = int(p["denominator_case"].sum())  # P-Pop planets forming the background
            det_grid, _ = fraction_grid(p, p["detected_case"], p["denominator_case"])
            if np.isfinite(det_grid).any():
                mesh = ax.pcolormesh(
                    INSOLATION_BINS, PLANET_RADIUS_BINS, det_grid,
                    shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED,
                )
                _add_contours(ax, det_grid)
            else:
                ax.text(0.5, 0.5, "no P-Pop data", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")

            _overlay_rocky_by_facility(ax, r, color_map, major)

            # 90% upper-bound fit + cold false-negative region (G/K/M panels).
            fn_note = ""
            if stype in ("G", "K", "M"):
                fit = draw_90pct_line(
                    ax, r["flux_p"].values, r["radius_p"].values,
                    color="black", lw=2.0, x_extent=INSOLATION_LIMITS,
                )
                if fit is not None:
                    slope, intercept = fit
                    # Shade the cold (I<10), above-the-line region.
                    xreg = np.logspace(np.log10(INSOLATION_LIMITS[0]),
                                       np.log10(COLD_INSOLATION), 100)
                    ax.fill_between(xreg, line_radius(slope, intercept, xreg),
                                    RADIUS_LIMITS[1], color="red", alpha=0.12,
                                    zorder=1.5, lw=0)
                    fn, n_denom, n_missed = false_negative_prob(p, slope, intercept)
                    if fn is not None:
                        fn_note = (f"P(false neg | I<10, R>fit) = {fn:.0%}\n"
                                   f"({n_missed}/{n_denom} P-Pop missed)")
                        ax.text(0.03, 0.97, fn_note, transform=ax.transAxes,
                                va="top", ha="left", fontsize=7, color="darkred",
                                bbox=dict(boxstyle="round", fc="white",
                                          ec="darkred", alpha=0.85), zorder=8)
                        print(f"  {mission:6s} {stype}: "
                              f"P(false neg | I<10, R>fit) = {fn:.1%}  "
                              f"({n_missed}/{n_denom} P-Pop)")

            _setup_axis(
                ax,
                f"{mission} — {stype} stars\n"
                f"N_PPop(background) = {n_bg:,}   ·   N_rocky = {len(r)}",
                show_xlabel=(i == 1),
                show_ylabel=(j == 0),
            )

    # Shared facility legend.
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[f],
               markersize=7, label=f"{short_facility(f)}  (N={counts[f]})")
        for f in major
    ]
    if n_other > 0:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", color=OTHER_COLOR,
                   markersize=7, label=f"Other facilities  (N={n_other})")
        )
    handles.append(
        Line2D([0], [0], marker="*", linestyle="", color="gold",
               markeredgecolor="darkred", markersize=12, label="LHS 1140 b (anchor)")
    )
    handles.append(
        Line2D([0], [0], color="black", lw=2.0, ls="--",
               label="90% upper-bound fit (≈90% of planets below; G/K/M)")
    )
    handles.append(
        Patch(facecolor="red", alpha=0.12,
              label="Cold false-negative region (I<10, R>fit)")
    )
    # "outside" reserves space below the axes so the legend never overlaps the
    # bottom-row x-axis labels.
    fig.legend(
        handles=handles, loc="outside lower center",
        ncol=min(len(handles), 4), fontsize=9, framealpha=0.9,
        title="Rocky NASA planets — colored by discovery facility (telescope/mission)",
        title_fontsize=10,
    )

    fig.colorbar(mesh, ax=axes.ravel().tolist(),
                 label="Detected fraction", shrink=0.6, pad=0.01)

    fig.suptitle(
        "Kepler (top) & TESS (bottom) FGKM — P-Pop detected-fraction background "
        "+ rocky PSCompPars overlay  [baseline]\n"
        f"Rocky threshold = {ROCKY_CURVE_LABEL}, unshifted  |  "
        f"red error bars = two-sided radius uncertainty",
        fontsize=13,
    )

    out = OUT_DIR / "kepler_tess_2x4_rocky_fgkm_detection_baseline.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined 2x4 figure: {out}")
    return out


# ── Mass-radius diagnostic (from script 36) ──────────────────────────────────

STYPE_COLORS = {"F": "#e6ab02", "G": "#66a61e", "K": "#7570b3", "M": "#d95f02"}


def plot_mr_diagnostic(m_ref, r_ref, shift: float,
                       nasa_win: pd.DataFrame, rocky_win: pd.DataFrame,
                       color_map: dict, major: list[str], counts,
                       red_shift: float | None = None,
                       title: str | None = None,
                       out_name: str = "rocky_threshold_diagnostic_mass_radius.png") -> Path:
    """Mass-radius diagnostic — only planets inside the context window.

    Rocky planets are colored by discovery facility (same colors as the 2x4
    figure), with thin, cap-less two-sided mass/radius error bars.
    """
    XLIM = (0.0, 12.0)
    YLIM = (RADIUS_LIMITS[0] - 0.05, RADIUS_LIMITS[1])
    # red_shift lets callers draw the silicate curve unshifted (red_shift=0) or
    # with any other anchor offset; defaults to the LHS 1140 b shift.
    red_shift = shift if red_shift is None else red_shift

    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)
    # Red rocky threshold = professor's silicate curve (m_ref, r_ref), anchored to LHS 1140 b.
    r_threshold = np.interp(m_line, m_ref, r_ref + red_shift, left=np.nan, right=np.nan)
    # Black dashed reference = pure-rock curve (ref.ddat), drawn only for comparison.
    if REF_CURVE_PATH.exists():
        _ref = np.loadtxt(REF_CURVE_PATH, comments="#")
        _mr, _rr = _ref[:, 0].astype(float), _ref[:, 1].astype(float)
        _ord = np.argsort(_mr)
        r_rocky = np.interp(m_line, _mr[_ord], _rr[_ord], left=np.nan, right=np.nan)
    else:
        r_rocky = np.full_like(m_line, np.nan)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(nasa_win["mass_p"], nasa_win["radius_p"],
               s=10, c="0.85", alpha=0.30, linewidths=0,
               label=f"PSCompPars quality-filtered, in-window (N={len(nasa_win):,})", zorder=1)

    # Rocky planets colored by discovery facility, thin cap-less error bars.
    fac = rocky_win["discovery_facility"].fillna("Unknown")
    for f in major:
        sub = rocky_win[fac == f]
        if len(sub) == 0:
            continue
        ax.errorbar(
            sub["mass_p"], sub["radius_p"],
            xerr=[sub["mass_err_minus"].abs().fillna(0).values,
                  sub["mass_err_plus"].abs().fillna(0).values],
            yerr=[sub["radius_err_minus"].abs().fillna(0).values,
                  sub["radius_err_plus"].abs().fillna(0).values],
            fmt="o", ms=4, color=color_map[f], alpha=0.85,
            elinewidth=0.4, capsize=0, ecolor=color_map[f],
            label=f"{short_facility(f)}  (N={counts[f]})", zorder=4,
        )
    other = rocky_win[~fac.isin(major)]
    if len(other) > 0:
        ax.errorbar(
            other["mass_p"], other["radius_p"],
            xerr=[other["mass_err_minus"].abs().fillna(0).values,
                  other["mass_err_plus"].abs().fillna(0).values],
            yerr=[other["radius_err_minus"].abs().fillna(0).values,
                  other["radius_err_plus"].abs().fillna(0).values],
            fmt="o", ms=4, color=OTHER_COLOR, alpha=0.70,
            elinewidth=0.4, capsize=0, ecolor=OTHER_COLOR,
            label=f"Other facilities  (N={len(other)})", zorder=3,
        )

    _red_lbl = (f"Rocky threshold — {ROCKY_CURVE_PATH.name} (unshifted)"
                if abs(red_shift) < 1e-9
                else f"Rocky threshold — {ROCKY_CURVE_PATH.name} ({red_shift:+.3f} R⊕)")
    ax.plot(m_line, r_rocky,     "k--", lw=1.5, label=f"Pure-rock reference ({REF_CURVE_PATH.name})", zorder=5)
    ax.plot(m_line, r_threshold, color="crimson", lw=2.0, label=_red_lbl, zorder=5)
    ax.scatter([LHS1140B_MASS_MEARTH], [LHS1140B_RADIUS_REARTH],
               s=120, marker="*", color="dodgerblue", edgecolors="navy",
               linewidths=0.8, zorder=6, label="LHS 1140 b (anchor)")
    ax.annotate("LHS 1140 b",
                xy=(LHS1140B_MASS_MEARTH, LHS1140B_RADIUS_REARTH),
                xytext=(0.22, 0.04), textcoords="offset points",
                fontsize=8.5, color="navy")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    _default_title = "Rocky threshold diagnostic — silicate curve (unshifted rocky cutoff)"
    ax.set_title(title if title is not None else _default_title, fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.25, linestyle="--")

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved M-R diagnostic: {out}")
    return out


def plot_threshold_curve_comparison(m_ref, r_ref, nasa_win: pd.DataFrame) -> Path:
    """Overlay the three candidate rocky thresholds as thin solid curves so the
    differences are directly visible:
        1. pure-rock reference (ref.ddat) shifted UP to LHS 1140 b  (the old threshold)
        2. silicate curve (Hongyi-silicon.ddat), unshifted
        3. silicate curve shifted to LHS 1140 b                     (the current threshold)
    """
    XLIM = (0.0, 12.0)
    YLIM = (RADIUS_LIMITS[0] - 0.05, RADIUS_LIMITS[1])
    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)

    # Pure-rock reference (ref.ddat) with its own LHS 1140 b anchor shift.
    ref_shift = float("nan")
    r_ref_pure_shifted = np.full_like(m_line, np.nan)
    if REF_CURVE_PATH.exists():
        _ref = np.loadtxt(REF_CURVE_PATH, comments="#")
        _mr, _rr = _ref[:, 0].astype(float), _ref[:, 1].astype(float)
        _o = np.argsort(_mr); _mr, _rr = _mr[_o], _rr[_o]
        ref_shift = LHS1140B_RADIUS_REARTH - float(np.interp(LHS1140B_MASS_MEARTH, _mr, _rr))
        r_ref_pure_shifted = np.interp(m_line, _mr, _rr + ref_shift, left=np.nan, right=np.nan)

    sil_anchor    = compute_anchor_shift(m_ref, r_ref)
    r_sil_raw     = np.interp(m_line, m_ref, r_ref,              left=np.nan, right=np.nan)
    r_sil_shifted = np.interp(m_line, m_ref, r_ref + sil_anchor, left=np.nan, right=np.nan)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    if nasa_win is not None and len(nasa_win):
        ax.scatter(nasa_win["mass_p"], nasa_win["radius_p"], s=8, c="0.85",
                   alpha=0.35, linewidths=0, zorder=1,
                   label=f"PSCompPars in-window (N={len(nasa_win):,})")

    ax.plot(m_line, r_ref_pure_shifted, color="black", lw=1.3, zorder=5,
            label=f"Pure-rock ref ({REF_CURVE_PATH.name}) → LHS 1140 b ({ref_shift:+.3f} R⊕)")
    ax.plot(m_line, r_sil_raw, color="seagreen", lw=1.3, zorder=5,
            label=f"Silicate ({ROCKY_CURVE_PATH.name}) unshifted")
    ax.plot(m_line, r_sil_shifted, color="crimson", lw=1.3, zorder=5,
            label=f"Silicate → LHS 1140 b ({sil_anchor:+.3f} R⊕)")

    ax.scatter([LHS1140B_MASS_MEARTH], [LHS1140B_RADIUS_REARTH], s=130, marker="*",
               color="dodgerblue", edgecolors="navy", linewidths=0.8, zorder=6,
               label="LHS 1140 b (anchor)")
    ax.annotate("LHS 1140 b",
                xy=(LHS1140B_MASS_MEARTH, LHS1140B_RADIUS_REARTH),
                xytext=(6, 4), textcoords="offset points", fontsize=8.5, color="navy")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    ax.set_title("Rocky threshold comparison — three candidate boundaries", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")

    out = OUT_DIR / "rocky_threshold_curve_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved threshold curve comparison: {out}")
    return out


# ── Standalone rocky scatter (from script 36, with straight 90% line) ─────────

def plot_rocky_scatter_standalone(rocky_win: pd.DataFrame, shift: float) -> Path:
    """Insolation vs radius for in-window confirmed rocky planets, colored by
    host star type, with a straight 90% upper-bound fit line."""
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    # Insolation-bin shading (matches the survival-analysis bins).
    ax.axvspan(INSOLATION_LIMITS[0], 10.0, color="#4575b4", alpha=0.06, zorder=0, label="I < 10 (cold)")
    ax.axvspan(10.0, 50.0,                  color="#74add1", alpha=0.06, zorder=0, label="10 ≤ I < 50")
    ax.axvspan(50.0, INSOLATION_LIMITS[1],  color="#fdae61", alpha=0.06, zorder=0, label="I ≥ 50 (hot)")
    for x_boundary in [10.0, 50.0]:
        ax.axvline(x_boundary, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    for stype in STAR_ORDER:
        sub = rocky_win[rocky_win["stype_clean"] == stype]
        if len(sub) == 0:
            continue
        color = STYPE_COLORS.get(stype, "gray")
        yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
        yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
        ax.errorbar(
            sub["flux_p"], sub["radius_p"], yerr=[yerr_lo, yerr_hi],
            fmt="o", ms=5, color=color, alpha=0.85,
            elinewidth=0.9, capsize=2.5, ecolor=color,
            label=f"{stype} stars (N={len(sub)})", zorder=4,
        )

    # Straight 90% upper-bound fit over all in-window rocky planets.
    drew = draw_90pct_line(
        ax, rocky_win["flux_p"].values, rocky_win["radius_p"].values,
        color="black", lw=1.8,
        label="90% upper-bound fit (≈90% of planets below)",
    )

    lhs_mask = rocky_win["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False, na=False, regex=True)
    for _, row in rocky_win[lhs_mask].iterrows():
        ax.scatter([row["flux_p"]], [row["radius_p"]],
                   s=100, marker="*", color="gold", edgecolors="darkred",
                   linewidths=1.0, zorder=7)
        ax.annotate("LHS 1140 b",
                    xy=(row["flux_p"], row["radius_p"]),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=8.5, color="darkred", fontweight="bold", zorder=8)

    _setup_axis(ax, f"Confirmed rocky planets — insolation vs radius  (N={len(rocky_win)})")
    ax.set_facecolor("#fafafa")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.90)

    fig.suptitle(
        "NASA PSCompPars confirmed rocky planets  (M–R below rocky threshold)\n"
        f"Rocky threshold = {ROCKY_CURVE_LABEL}, unshifted\n"
        "Shaded columns = insolation bins used in survival analysis",
        fontsize=10,
    )

    out = OUT_DIR / "rocky_scatter_standalone.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved standalone rocky scatter: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("56_kepler_tess_rocky_fgkm_gaia60pc.py")
    print("=" * 70)
    print(f"Project root  : {ROOT}")
    print(f"Kepler P-Pop  : {KEPLER_PPOP_DIR}  (stacking up to {N_UNIVERSES} x kepler_catalog_i.csv)")
    print(f"TESS   P-Pop  : {TESS_PPOP_DIR}  (stacking up to {N_UNIVERSES} x tess_catalog_i.csv)")
    print(f"Output dir    : {OUT_DIR}")
    print(f"LHS 1140 b    : {LHS1140B_MASS_MEARTH} M⊕, {LHS1140B_RADIUS_REARTH} R⊕")
    print()

    m_ref, r_ref = load_rocky_reference_curve()
    shift = compute_rocky_threshold_shift(m_ref, r_ref)
    print()

    nasa_all, rocky = load_and_filter_nasa(m_ref, r_ref, shift)

    # Limit the planets used in ALL figures to the context window.
    rocky_win = restrict_to_window(rocky)
    nasa_win  = restrict_to_window(nasa_all)
    rocky_win.to_csv(OUT_DIR / "rocky_pscomppars_below_threshold_in_window.csv", index=False)
    print(f"Rocky planets in context window "
          f"(R {RADIUS_LIMITS[0]}–{RADIUS_LIMITS[1]} R⊕, I < {INSOLATION_LIMITS[1]:g}): "
          f"{len(rocky):,} → {len(rocky_win):,}")
    print()

    # Shared facility color map (same colors across all figures).
    color_map, major, counts = build_facility_styles(rocky_win)

    # Figures 2 & 3 (mass-radius diagnostic, confirmed rocky scatter).
    # Main M-R diagnostic — the cutoff is now the UNSHIFTED silicate curve (shift=0).
    plot_mr_diagnostic(m_ref, r_ref, shift, nasa_win, rocky_win, color_map, major, counts)
    # Comparison: the three candidate thresholds overlaid as thin solid curves.
    plot_threshold_curve_comparison(m_ref, r_ref, nasa_win)
    plot_rocky_scatter_standalone(rocky_win, shift)
    print()

    kepler = load_kepler_ppop()
    tess   = load_tess_ppop()
    print()

    plot_combined(kepler, tess, rocky_win, shift, color_map, major, counts)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
