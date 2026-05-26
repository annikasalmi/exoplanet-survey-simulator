"""
09_plot_pscomppars_vs_ppop_insolation.py

Goal:
    Make the comparison figure:
        NASA observed planets split by insolation
        vs.
        P-Pop Kepler-detected planets split by insolation

Output:
    A 2 x 3 Mass-Radius figure.
        Top row    = NASA PSCompPars observed/confirmed planets
        Bottom row = your P-Pop planets that your Kepler detector marked as detected
        Columns    = insolation bins

Caveman version:
    NASA rock pile = real observed planets.
    P-Pop rock pile = fake planets your fake Kepler saw.
    Put both piles into same hotness boxes.
    Draw mass vs radius.
    Compare shapes.
"""

from pathlib import Path
from urllib.parse import quote
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Your simulated P-Pop -> Kepler output files.
PPOP_DATA_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"

# NASA PSCompPars cache file. If missing, this script downloads it.
NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)
NASA_PSCOMPPARS_PATH = NASA_DATA_DIR / "pscomppars_transiting_confirmed_mr_insolation.csv"

REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

OUT_DIR = ROOT / "my_outputs" / "w2_nasa_vs_kepler_rm_Insolation_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NASA dataset choices.
DOWNLOAD_NASA_IF_MISSING = True
NASA_TRANSITING_ONLY = True      # Recommended for Kepler/TESS-like comparison.
NASA_KEPLER_ONLY = True         # Set True if you want only disc_facility = 'Kepler'.

# Which simulated P-Pop planets should be compared against NASA?
PPOP_DETECTED_COL_PREFERENCE = ["detected", "detected_best"]

# Plot window: same physical range as your current M-R plots.
APPLY_FOCUS_WINDOW = True
MASS_MIN = 0.0
MASS_MAX = 12.0
RADIUS_MIN = 0.5
RADIUS_MAX = 2.2

DRAW_ROCKY_CURVE = True

# If True, bins mimic your reference style: I < 10, I < 50, I > 50.
# Note: first two bins overlap on purpose.
USE_REFERENCE_STYLE_CUMULATIVE_BINS = True

POINT_SIZE_NASA = 26
POINT_SIZE_PPOP = 22
POINT_ALPHA = 0.82
CMAP = "plasma"
COLOR_PERCENTILES = (1, 99)

SAVE_COMBINED_TABLES = True


# ============================================================
# General helpers
# ============================================================

def to_bool_series(s):
    """Robustly convert a pandas Series to boolean."""
    if s.dtype == bool:
        return s.fillna(False)
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        return s.isin(["true", "1", "yes", "y", "t"])
    return s.fillna(0).astype(bool)


def force_numeric_columns(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )


# ============================================================
# NASA PSCompPars download and normalization
# ============================================================

def build_pscomppars_query():
    """
    Build a NASA Exoplanet Archive TAP query.

    Caveman version:
        Ask NASA for only planets with mass, radius, and sunlight number.
        No mass -> cannot draw M-R.
        No radius -> cannot draw M-R.
        No insolation -> cannot put into hotness bins.
    """
    where_parts = [
        "pl_rade IS NOT NULL",
        "pl_bmasse IS NOT NULL",
        "pl_insol IS NOT NULL",
    ]

    if NASA_TRANSITING_ONLY:
        where_parts.append("tran_flag = 1")

    if NASA_KEPLER_ONLY:
        where_parts.append("disc_facility = 'Kepler'")

    where_clause = "\n  AND ".join(where_parts)

    return f"""
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
    pl_insol,
    pl_eqt,
    st_rad,
    st_mass,
    st_teff,
    st_lum,
    sy_dist,
    sy_kepmag,
    sy_gaiamag
FROM pscomppars
WHERE {where_clause}
"""


def download_pscomppars_csv(out_path=NASA_PSCOMPPARS_PATH):
    query = build_pscomppars_query()
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + quote(query) + "&format=csv"

    print("Downloading NASA PSCompPars from TAP...")
    print("NASA_TRANSITING_ONLY =", NASA_TRANSITING_ONLY)
    print("NASA_KEPLER_ONLY     =", NASA_KEPLER_ONLY)

    df = pd.read_csv(url)
    df.to_csv(out_path, index=False)

    print(f"Saved NASA PSCompPars cache: {out_path}")
    print(f"NASA rows downloaded: {len(df)}")
    return df


def load_pscomppars():
    if NASA_PSCOMPPARS_PATH.exists():
        print("Loading cached NASA PSCompPars:")
        print(NASA_PSCOMPPARS_PATH)
        return pd.read_csv(NASA_PSCOMPPARS_PATH)

    if not DOWNLOAD_NASA_IF_MISSING:
        raise FileNotFoundError(
            f"NASA PSCompPars file not found:\n{NASA_PSCOMPPARS_PATH}\n\n"
            "Set DOWNLOAD_NASA_IF_MISSING = True or place the CSV there."
        )

    return download_pscomppars_csv(NASA_PSCOMPPARS_PATH)


def standardize_nasa_pscomppars(df):
    """
    Rename PSCompPars columns to match your P-Pop plotting names.

    Caveman version:
        NASA word pl_bmasse -> our word mass_p.
        NASA word pl_rade   -> our word radius_p.
        NASA word pl_insol  -> our word flux_p.
        Now both tables speak one language.
    """
    df = df.copy()

    rename_map = {
        "pl_bmasse": "mass_p",
        "pl_rade": "radius_p",
        "pl_insol": "flux_p",
        "pl_orbper": "p_orb",
        "pl_orbsmax": "semimajor_p",
        "st_rad": "radius_s",
        "st_mass": "mass_s",
        "sy_dist": "distance_s",
        "sy_kepmag": "kepmag",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # PSCompPars st_lum is log10(L/Lsun). Convert to normal L/Lsun if useful later.
    if "st_lum" in df.columns and "l_sun" not in df.columns:
        st_lum = pd.to_numeric(df["st_lum"], errors="coerce")
        df["l_sun"] = 10 ** st_lum

    df["population"] = "NASA observed PSCompPars"
    df["observed"] = True

    df = force_numeric_columns(
        df,
        ["mass_p", "radius_p", "flux_p", "p_orb", "semimajor_p", "radius_s", "mass_s", "distance_s", "kepmag"],
    )

    require_columns(df, ["mass_p", "radius_p", "flux_p"], "NASA PSCompPars")
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()

    return df


# ============================================================
# P-Pop simulated Kepler catalog loading
# ============================================================

def load_all_ppop_kepler_catalogs(data_dir=PPOP_DATA_DIR):
    pattern = os.path.join(data_dir, "kepler_catalog_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No P-Pop Kepler catalog files found here:\n{data_dir}\n\n"
            "Expected files like kepler_catalog_0.csv, kepler_catalog_1.csv, etc.\n"
            "Run your Kepler simulation first."
        )

    dfs = []
    for f in files:
        df_i = pd.read_csv(f)
        name = Path(f).stem
        try:
            run_number = int(name.split("_")[-1])
        except ValueError:
            run_number = len(dfs)
        df_i["run"] = run_number
        df_i["source_file"] = str(f)
        dfs.append(df_i)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} P-Pop Kepler catalog files.")
    print(f"P-Pop rows before filtering: {len(df)}")
    return df


def choose_ppop_detected_column(df):
    for col in PPOP_DETECTED_COL_PREFERENCE:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a detected column in P-Pop catalog. Tried: "
        f"{PPOP_DETECTED_COL_PREFERENCE}\nAvailable columns:\n{df.columns.tolist()}"
    )


def standardize_ppop_detected(df):
    """
    Keep only P-Pop planets detected by your toy Kepler detector.

    Caveman version:
        P-Pop made many fake planets.
        Kepler toy saw only some.
        Compare NASA only to the fake planets Kepler saw.
    """
    df = df.copy()

    # If older/newer code calls insolation differently, normalize to flux_p.
    if "pl_insol" in df.columns and "flux_p" not in df.columns:
        df = df.rename(columns={"pl_insol": "flux_p"})
    if "insolation" in df.columns and "flux_p" not in df.columns:
        df = df.rename(columns={"insolation": "flux_p"})

    detected_col = choose_ppop_detected_column(df)
    detected = to_bool_series(df[detected_col])
    df = df[detected].copy()

    df["population"] = "P-Pop Kepler-detected"
    df["observed"] = False
    df["detected_col_used"] = detected_col

    df = force_numeric_columns(df, ["mass_p", "radius_p", "flux_p"])

    require_columns(df, ["mass_p", "radius_p", "flux_p"], "P-Pop detected catalog")
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()

    print(f"P-Pop detected column used: {detected_col}")
    print(f"P-Pop detected rows after filtering: {len(df)}")
    return df


# ============================================================
# Rocky reference curve helpers
# ============================================================

def load_rocky_reference_curve(ref_path=REF_CURVE_PATH):
    ref_path = Path(ref_path)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Could not find ref.ddat at:\n{ref_path}\n\n"
            "Expected path relative to project root:\n"
            "run/kepler/reference_curves/ref.ddat"
        )

    ref = np.loadtxt(ref_path, comments="#")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if ref.shape[1] < 2:
        raise ValueError("ref.ddat needs at least two columns: mass and radius.")

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


def add_rocky_diagnostics(df):
    df = df.copy()
    m_ref, r_ref = load_rocky_reference_curve(REF_CURVE_PATH)
    df["rocky_radius_ref"] = rocky_radius_from_mass(df["mass_p"], m_ref, r_ref)
    df["radius_ratio_to_rocky"] = df["radius_p"] / df["rocky_radius_ref"]
    df["radius_excess_to_rocky"] = df["radius_p"] - df["rocky_radius_ref"]
    return df


def add_rocky_curve(ax):
    if not DRAW_ROCKY_CURVE:
        return

    m_ref, r_ref = load_rocky_reference_curve(REF_CURVE_PATH)

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
        label="Earth-like rocky curve from ref.ddat",
    )


# ============================================================
# Plotting helpers
# ============================================================

def apply_focus_window(df):
    df = df.copy()
    if not APPLY_FOCUS_WINDOW:
        return df

    mask = (
        (df["mass_p"] > MASS_MIN)
        & (df["mass_p"] <= MASS_MAX)
        & (df["radius_p"] >= RADIUS_MIN)
        & (df["radius_p"] <= RADIUS_MAX)
    )
    return df[mask].copy()


def make_insolation_panels(df):
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


def log_flux_values(df):
    flux = pd.to_numeric(df["flux_p"], errors="coerce").clip(lower=1e-12)
    return np.log10(flux.to_numpy(dtype=float))


def compute_shared_color_limits(*dfs, fallback=(-1.0, 2.0)):
    values = []
    for df in dfs:
        if len(df) > 0 and "flux_p" in df.columns:
            values.append(log_flux_values(df))

    if not values:
        return fallback

    all_values = np.concatenate(values)
    finite = all_values[np.isfinite(all_values)]
    if len(finite) == 0:
        return fallback

    lo, hi = COLOR_PERCENTILES
    vmin = np.nanpercentile(finite, lo)
    vmax = np.nanpercentile(finite, hi)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return fallback
    if np.isclose(vmin, vmax):
        vmin -= 0.1
        vmax += 0.1
    return vmin, vmax


def setup_rm_axis(ax):
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    if APPLY_FOCUS_WINDOW:
        ax.set_xlim(MASS_MIN, MASS_MAX)
        ax.set_ylim(RADIUS_MIN, RADIUS_MAX)
    ax.grid(True, alpha=0.28)


def scatter_rm(ax, df, title, vmin, vmax, point_size):
    setup_rm_axis(ax)
    add_rocky_curve(ax)

    if len(df) == 0:
        ax.text(0.5, 0.5, "No planets", transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_title(f"{title}\nN = 0", fontsize=12)
        return None

    sc = ax.scatter(
        df["mass_p"],
        df["radius_p"],
        c=log_flux_values(df),
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        alpha=POINT_ALPHA,
        edgecolors="none",
        zorder=2,
    )

    ax.set_title(f"{title}\nN = {len(df)}", fontsize=12)
    return sc


# ============================================================
# Main script
# ============================================================

def main():
    print("Project root:", ROOT)
    print("P-Pop Kepler data:", PPOP_DATA_DIR)
    print("NASA PSCompPars path:", NASA_PSCOMPPARS_PATH)
    print("Rocky reference table:", REF_CURVE_PATH)

    # Load NASA observed planets.
    nasa_raw = load_pscomppars()
    nasa = standardize_nasa_pscomppars(nasa_raw)
    nasa = add_rocky_diagnostics(nasa)
    nasa_plot = apply_focus_window(nasa)

    # Load P-Pop detected planets.
    ppop_raw = load_all_ppop_kepler_catalogs(PPOP_DATA_DIR)
    ppop_detected = standardize_ppop_detected(ppop_raw)
    ppop_detected = add_rocky_diagnostics(ppop_detected)
    ppop_plot = apply_focus_window(ppop_detected)

    print("\nAfter M-R focus window:")
    print("NASA observed rows:", len(nasa_plot))
    print("P-Pop detected rows:", len(ppop_plot))

    # Save cleaned tables so you can inspect exactly what was plotted.
    if SAVE_COMBINED_TABLES:
        nasa_out = OUT_DIR / "cleaned_NASA_PSCompPars_for_MR_insolation.csv"
        ppop_out = OUT_DIR / "cleaned_PPop_detected_for_MR_insolation.csv"
        nasa_plot.to_csv(nasa_out, index=False)
        ppop_plot.to_csv(ppop_out, index=False)
        print("\nSaved cleaned plotting tables:")
        print(nasa_out)
        print(ppop_out)

    # Make panels.
    nasa_panels = make_insolation_panels(nasa_plot)
    ppop_panels = make_insolation_panels(ppop_plot)

    vmin, vmax = compute_shared_color_limits(nasa_plot, ppop_plot)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)

    scatter_for_colorbar = None

    for j, (bin_title, df_sub) in enumerate(nasa_panels):
        sc = scatter_rm(
            axes[0, j],
            df_sub,
            title=f"NASA observed PSCompPars\n{bin_title}",
            vmin=vmin,
            vmax=vmax,
            point_size=POINT_SIZE_NASA,
        )
        if scatter_for_colorbar is None and sc is not None:
            scatter_for_colorbar = sc

    for j, (bin_title, df_sub) in enumerate(ppop_panels):
        sc = scatter_rm(
            axes[1, j],
            df_sub,
            title=f"P-Pop Kepler-detected\n{bin_title}",
            vmin=vmin,
            vmax=vmax,
            point_size=POINT_SIZE_PPOP,
        )
        if scatter_for_colorbar is None and sc is not None:
            scatter_for_colorbar = sc

    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=axes.ravel().tolist(), location="right", shrink=0.92, pad=0.02)
        cbar.set_label(r"log$_{10}$(Insolation Flux [$I_\oplus$])", fontsize=11)

    source_note = "transiting confirmed planets" if NASA_TRANSITING_ONLY else "all confirmed planets with M/R/insolation"
    if NASA_KEPLER_ONLY:
        source_note += "; Kepler discovery facility only"

    fig.suptitle(
        "NASA observed planets vs. P-Pop Kepler-detected planets\n"
        f"Mass-Radius diagram split by insolation; NASA sample = {source_note}",
        fontsize=16,
    )

    png_path = OUT_DIR / "rm_NASA_PSCompPars_vs_PPop_detected_insolation_bins.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved comparison figure:")
    print(png_path)

    # Counts table.
    counts = []
    for population, panels in [
        ("NASA observed PSCompPars", nasa_panels),
        ("P-Pop Kepler-detected", ppop_panels),
    ]:
        for title, df_sub in panels:
            counts.append({"population": population, "insolation_bin": title, "count": len(df_sub)})

    counts_df = pd.DataFrame(counts)
    counts_path = OUT_DIR / "rm_NASA_vs_PPop_insolation_bin_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print("\nSaved bin counts:")
    print(counts_path)


if __name__ == "__main__":
    main()
