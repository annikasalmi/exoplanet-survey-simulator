"""
compare_pscomppars_vs_ppop_insolation.py

Makes the graph you asked for:

    NASA observed planets split by insolation
    vs.
    P-Pop Kepler-detected planets split by insolation

Caveman picture:
    Left team  = fake planets that your Kepler detector says it can see.
    Right team = real confirmed NASA planets that humans already observed.
    Split both teams by sunlight/insolation.
    Put them on the same Mass-Radius map.
    Draw the rocky curve.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifesim.core.kepler_data import KeplerData


# ============================================================
# Settings
# ============================================================


def find_project_root(start_path: Path) -> Path:
    """Walk upward until we find the project root that has run/kepler."""
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists():
            return p
    # Fallback: your old style usually worked when script was one folder below project root.
    return start_path.parents[1]


ROOT = find_project_root(Path(__file__).resolve())

PPOP_DATA_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

OUT_DIR = ROOT / "my_outputs" / "w2_nasa_pscomppars_vs_ppop_insolation"
NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PSCOMPARS_CACHE = NASA_DATA_DIR / "NASA_PSCompPars_Kepler_only_transiting_confirmed_RM_insolation.csv"
# What P-Pop planets to compare against observed NASA planets.
PPOP_DETECTION_COLUMN = "detected"

# If True, keep only planets in the same M-R window as your rocky-curve-focused plots.
APPLY_FOCUS_WINDOW = True
MASS_MIN = 0.0
MASS_MAX = 12.0
RADIUS_MIN = 0.5
RADIUS_MAX = 2.2

# Use your attached-result style:
#   I < 10, I < 50, I > 50
# Note: I < 50 includes I < 10. This is cumulative, not disjoint.
USE_REFERENCE_STYLE_CUMULATIVE_BINS = True

# NASA mass caveat:
# PSCompPars may include calculated masses. If you want a stricter measured-mass comparison,
# keep EXCLUDE_CALCULATED_NASA_MASSES=True. If this removes too many planets, set False.
EXCLUDE_CALCULATED_NASA_MASSES = True

POINT_SIZE = 25
POINT_ALPHA = 0.80
CMAP = "plasma"
COLOR_PERCENTILES = (1, 99)


# ============================================================
# Basic helpers
# ============================================================


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        return ss.isin(["true", "1", "yes", "y", "t"])
    return s.fillna(0).astype(bool)


def force_numeric_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def require_columns(df: pd.DataFrame, columns, label="dataframe"):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )


# ============================================================
# Load P-Pop detected planets
# ============================================================


def load_all_ppop_kepler_catalogs(data_dir: Path) -> pd.DataFrame:
    pattern = str(data_dir / "kepler_catalog_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No P-Pop Kepler catalog files found here:\n{data_dir}\n\n"
            "Expected files like kepler_catalog_0.csv, kepler_catalog_1.csv, etc."
        )

    dfs = []
    for f in files:
        df_i = pd.read_csv(f)
        stem = Path(f).stem
        try:
            run_number = int(stem.split("_")[-1])
        except ValueError:
            run_number = len(dfs)
        df_i["run"] = run_number
        dfs.append(df_i)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} P-Pop Kepler catalog files.")
    print(f"P-Pop planets before filtering: {len(df)}")
    return df


def prepare_ppop_detected() -> pd.DataFrame:
    # Caveman: P-Pop is fake universe. We only want fake planets Kepler says it can see.
    df = load_all_ppop_kepler_catalogs(PPOP_DATA_DIR)
    df = KeplerData(df, source="ppop", validate_for_detection=False).catalog

    # Older files may use detected_best instead of detected.
    detected_col = PPOP_DETECTION_COLUMN
    if detected_col not in df.columns:
        if "detected_best" in df.columns:
            detected_col = "detected_best"
        else:
            raise ValueError(
                "Could not find detected or detected_best in P-Pop catalogs. "
                "Regenerate P-Pop Kepler CSVs after running KeplerData.determine_detectable()."
            )

    require_columns(df, ["mass_p", "radius_p", "flux_p", detected_col], "P-Pop catalog")
    df = force_numeric_columns(df, ["mass_p", "radius_p", "flux_p"])
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()

    detected = to_bool_series(df[detected_col])
    df = df[detected].copy()
    df["comparison_group"] = "P-Pop detected by toy Kepler"

    print(f"P-Pop detected planets kept: {len(df)}")
    return df


# ============================================================
# Download/load NASA PSCompPars
# ============================================================


def download_pscomppars_if_needed(cache_path: Path = PSCOMPARS_CACHE, force: bool = False) -> pd.DataFrame:
    """
    Download transiting confirmed planets from PSCompPars.

    Caveman:
        Ask NASA: give me real confirmed transiting planets with radius, mass, and sunlight.
        Save it locally so we do not download again every run.
    """
    if cache_path.exists() and not force:
        print(f"Loading cached NASA PSCompPars table:\n{cache_path}")
        return pd.read_csv(cache_path)

    query = """
    SELECT
        pl_name,
        hostname,
        discoverymethod,
        disc_facility,
        disc_telescope,
        tran_flag,
        pl_orbper,
        pl_orbsmax,
        pl_rade,
        pl_bmasse,
        pl_bmasse_reflink,
        pl_rade_reflink,
        pl_insol,
        pl_eqt,
        pl_orbincl,
        pl_trandep,
        pl_trandur,
        st_rad,
        st_mass,
        st_teff,
        sy_dist,
        sy_kepmag,
        sy_gaiamag
    FROM pscomppars
    WHERE tran_flag = 1
        AND disc_facility = 'Kepler'
        AND pl_rade IS NOT NULL
        AND pl_bmasse IS NOT NULL
        AND pl_insol IS NOT NULL
    """

    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + quote(query) + "&format=csv"
    print("Downloading NASA PSCompPars from Exoplanet Archive TAP...")
    df = pd.read_csv(url)
    df.to_csv(cache_path, index=False)
    print(f"Saved NASA PSCompPars cache:\n{cache_path}")
    print(f"NASA rows downloaded: {len(df)}")
    return df


def prepare_nasa_observed() -> pd.DataFrame:
    # Caveman: NASA table speaks NASA names. KeplerData renames them to our plot names.
    raw = download_pscomppars_if_needed()
    df = KeplerData(raw, source="pscomppars", validate_for_detection=False).catalog

    require_columns(df, ["mass_p", "radius_p", "flux_p"], "NASA PSCompPars")
    df = force_numeric_columns(df, ["mass_p", "radius_p", "flux_p"])
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()

    # Caveman: Some NASA masses may be calculated from a formula. If we compare to a curve,
    # formula-made masses can secretly bake in theory. We can remove them.
    if EXCLUDE_CALCULATED_NASA_MASSES and "mass_reference" in df.columns:
        ref = df["mass_reference"].astype(str).str.lower()
        before = len(df)
        df = df[~ref.str.contains("calculated", na=False)].copy()
        print(f"Removed likely calculated NASA masses: {before - len(df)}")

    df["comparison_group"] = "NASA confirmed Kepler planets"
    print(f"NASA observed planets kept for M-R plot: {len(df)}")
    return df


# ============================================================
# Rocky curve helpers
# ============================================================


def load_rocky_reference_curve(ref_path: Path = REF_CURVE_PATH):
    ref_path = Path(ref_path)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Could not find rocky reference curve:\n{ref_path}\n"
            "Expected run/kepler/reference_curves/ref.ddat"
        )

    ref = np.loadtxt(ref_path, comments="#")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if ref.shape[1] < 2:
        raise ValueError("ref.ddat must have at least two columns: mass, radius")

    m_ref = ref[:, 0].astype(float)
    r_ref = ref[:, 1].astype(float)
    good = np.isfinite(m_ref) & np.isfinite(r_ref) & (m_ref > 0) & (r_ref > 0)
    m_ref = m_ref[good]
    r_ref = r_ref[good]
    order = np.argsort(m_ref)
    return m_ref[order], r_ref[order]


def rocky_radius_from_mass(m_planet, m_ref=None, r_ref=None):
    if m_ref is None or r_ref is None:
        m_ref, r_ref = load_rocky_reference_curve()
    return np.interp(np.asarray(m_planet, dtype=float), m_ref, r_ref, left=np.nan, right=np.nan)


def add_rocky_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m_ref, r_ref = load_rocky_reference_curve()
    df["rocky_radius_ref"] = rocky_radius_from_mass(df["mass_p"], m_ref, r_ref)
    df["radius_ratio_to_rocky"] = df["radius_p"] / df["rocky_radius_ref"]
    df["radius_excess_to_rocky"] = df["radius_p"] - df["rocky_radius_ref"]
    return df


def add_rocky_curve(ax):
    m_ref, r_ref = load_rocky_reference_curve()
    if APPLY_FOCUS_WINDOW:
        m_min = max(MASS_MIN, np.nanmin(m_ref))
        m_max = min(MASS_MAX, np.nanmax(m_ref))
    else:
        m_min = np.nanmin(m_ref)
        m_max = np.nanmax(m_ref)

    if m_max <= m_min:
        return

    m_line = np.linspace(m_min, m_max, 500)
    r_line = rocky_radius_from_mass(m_line, m_ref, r_ref)
    ax.plot(
        m_line,
        r_line,
        linestyle="--",
        linewidth=1.7,
        color="black",
        alpha=0.92,
        zorder=3,
        label="Earth-like rocky curve",
    )


# ============================================================
# Plot helpers
# ============================================================


def apply_focus_window(df: pd.DataFrame) -> pd.DataFrame:
    if not APPLY_FOCUS_WINDOW:
        return df.copy()
    mask = (
        (df["mass_p"] > MASS_MIN)
        & (df["mass_p"] <= MASS_MAX)
        & (df["radius_p"] >= RADIUS_MIN)
        & (df["radius_p"] <= RADIUS_MAX)
    )
    return df[mask].copy()


def setup_rm_axis(ax):
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    if APPLY_FOCUS_WINDOW:
        ax.set_xlim(MASS_MIN, MASS_MAX)
        ax.set_ylim(RADIUS_MIN, RADIUS_MAX)
    ax.grid(True, alpha=0.28)


def log_flux_values(df_sub: pd.DataFrame):
    flux = pd.to_numeric(df_sub["flux_p"], errors="coerce").clip(lower=1e-12)
    return np.log10(flux.to_numpy(dtype=float))


def compute_log_flux_limits(df: pd.DataFrame, fallback=(-1.0, 2.0)):
    if len(df) == 0 or "flux_p" not in df.columns:
        return fallback
    finite = log_flux_values(df)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return fallback
    vmin = np.nanpercentile(finite, COLOR_PERCENTILES[0])
    vmax = np.nanpercentile(finite, COLOR_PERCENTILES[1])
    if np.isclose(vmin, vmax):
        vmin -= 0.1
        vmax += 0.1
    return vmin, vmax


def split_insolation_panels(df: pd.DataFrame):
    if USE_REFERENCE_STYLE_CUMULATIVE_BINS:
        return [
            (r"I < 10 $I_\oplus$", df[df["flux_p"] < 10].copy()),
            (r"I < 50 $I_\oplus$", df[df["flux_p"] < 50].copy()),
            (r"I > 50 $I_\oplus$", df[df["flux_p"] > 50].copy()),
        ]
    return [
        (r"I < 10 $I_\oplus$", df[df["flux_p"] < 10].copy()),
        (r"10 $\leq$ I < 50 $I_\oplus$", df[(df["flux_p"] >= 10) & (df["flux_p"] < 50)].copy()),
        (r"I $\geq$ 50 $I_\oplus$", df[df["flux_p"] >= 50].copy()),
    ]


def scatter_rm(ax, df_sub: pd.DataFrame, title: str, vmin: float, vmax: float):
    setup_rm_axis(ax)
    add_rocky_curve(ax)

    if len(df_sub) == 0:
        ax.text(0.5, 0.5, "No planets", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{title}\nN = 0", fontsize=12)
        return None

    sc = ax.scatter(
        df_sub["mass_p"],
        df_sub["radius_p"],
        c=log_flux_values(df_sub),
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
        zorder=2,
    )
    ax.set_title(f"{title}\nN = {len(df_sub)}", fontsize=12)
    return sc


def add_colorbar(fig, sc, axes):
    if sc is None:
        return
    cbar = fig.colorbar(sc, ax=axes, location="right", shrink=0.92, pad=0.02)
    cbar.set_label(r"log$_{10}$(Insolation Flux [$I_\oplus$])", fontsize=11)


# ============================================================
# Main
# ============================================================


def main():
    print("Project root:", ROOT)
    print("P-Pop data directory:", PPOP_DATA_DIR)
    print("NASA data directory:", NASA_DATA_DIR)
    print("Rocky reference table:", REF_CURVE_PATH)

    ppop = prepare_ppop_detected()
    nasa = prepare_nasa_observed()

    ppop = add_rocky_diagnostics(apply_focus_window(ppop))
    nasa = add_rocky_diagnostics(apply_focus_window(nasa))

    print("\nAfter focus window:")
    print("P-Pop detected:", len(ppop))
    print("NASA observed:", len(nasa))

    # Save plotting data, because debugging plots is easier when the plotted rows are saved.
    ppop_path = OUT_DIR / "plot_data_ppop_detected.csv"
    nasa_path = OUT_DIR / "plot_data_nasa_observed_pscomppars.csv"
    ppop.to_csv(ppop_path, index=False)
    nasa.to_csv(nasa_path, index=False)
    print("\nSaved plot data:")
    print(ppop_path)
    print(nasa_path)

    both = pd.concat([ppop, nasa], ignore_index=True)
    vmin, vmax = compute_log_flux_limits(both)

    ppop_panels = split_insolation_panels(ppop)
    nasa_panels = split_insolation_panels(nasa)

    # Caveman: top row fake-detected planets, bottom row real-observed planets.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2), constrained_layout=True)
    shared_sc = None

    for j, (bin_title, df_sub) in enumerate(ppop_panels):
        sc = scatter_rm(
            axes[0, j],
            df_sub,
            title=f"P-Pop detected\n{bin_title}",
            vmin=vmin,
            vmax=vmax,
        )
        if shared_sc is None and sc is not None:
            shared_sc = sc

    for j, (bin_title, df_sub) in enumerate(nasa_panels):
        sc = scatter_rm(
            axes[1, j],
            df_sub,
            title=f"NASA observed PSCompPars\n{bin_title}",
            vmin=vmin,
            vmax=vmax,
        )
        if shared_sc is None and sc is not None:
            shared_sc = sc

    add_colorbar(fig, shared_sc, axes.ravel().tolist())

    bin_style = "cumulative bins: I<10, I<50, I>50" if USE_REFERENCE_STYLE_CUMULATIVE_BINS else "disjoint bins"
    fig.suptitle(
        "NASA observed planets vs. P-Pop Kepler-detected planets split by insolation\n"
        f"Mass-Radius plane; black dashed line = ref.ddat rocky curve; {bin_style}",
        fontsize=16,
    )

    png_path = OUT_DIR / "nasa_pscomppars_vs_ppop_detected_rm_insolation_bins.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved comparison plot:")
    print(png_path)

    counts = []
    for group_name, panels in [
        ("P-Pop detected by toy Kepler", ppop_panels),
        ("NASA confirmed Kepler planets", nasa_panels),
    ]:
        for bin_title, df_sub in panels:
            counts.append({"group": group_name, "bin": bin_title, "count": len(df_sub)})

    counts_df = pd.DataFrame(counts)
    counts_path = OUT_DIR / "nasa_vs_ppop_insolation_bin_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print("Saved bin counts:")
    print(counts_path)

    # Second quick plot: whole populations side-by-side without bins.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    sc1 = scatter_rm(axes[0], ppop, "P-Pop detected by toy Kepler", vmin, vmax)
    sc2 = scatter_rm(axes[1], nasa, "NASA confirmed Kepler planets", vmin, vmax)
    add_colorbar(fig, sc1 or sc2, axes.tolist())
    fig.suptitle("Whole-population comparison before insolation splitting", fontsize=15)

    png2 = OUT_DIR / "nasa_pscomppars_vs_ppop_detected_rm_all.png"
    fig.savefig(png2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved whole-population plot:")
    print(png2)


if __name__ == "__main__":
    main()
