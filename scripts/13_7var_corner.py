from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import corner
except ImportError as e:
    raise ImportError(
        "Missing package 'corner'. Install it with: pip install corner"
    ) from e


# ============================================================
# Settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Use all Kepler catalogues in the same format.
DATA_GLOB = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_*.csv"

OUT_DIR = ROOT / "my_outputs" / "w2_kepler_corner_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "w2_corner_kepler_detectability_parameters.png"

# Pick which detection label to use.
# Usually use "detected". If you want optimistic/pessimistic versions,
# change this to "detected_best" or "detected_worst".
DETECTED_COL = "detected"


# ============================================================
# Helper functions
# ============================================================

def load_catalogues(data_glob):
    files = sorted(glob.glob(str(data_glob)))

    if not files:
        raise FileNotFoundError(f"No files found at: {data_glob}")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = Path(file).name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} file(s)")
    print(f"Total planets: {len(combined)}")
    return combined


def to_bool(series):
    """
    Handles boolean columns saved as True/False or as strings.
    """
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def add_corner_columns(df):
    """
    Make log-transformed columns for a cleaner corner plot.
    Dynamic-range parameters look better in log space.
    """

    required = [
        "radius_p",
        "mass_p",
        "radius_s",
        "p_orb",
        "flux_p",
        "distance_s",
        "kepler_mag_used",
        DETECTED_COL,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only physically positive values before log10.
    positive_cols = [
        "radius_p",
        "mass_p",
        "radius_s",
        "p_orb",
        "flux_p",
        "distance_s",
    ]

    for col in positive_cols:
        df = df[df[col] > 0].copy()

    df["log10_radius_p"] = np.log10(df["radius_p"])
    df["log10_mass_p"] = np.log10(df["mass_p"])
    df["log10_radius_s"] = np.log10(df["radius_s"])
    df["log10_p_orb"] = np.log10(df["p_orb"])
    df["log10_flux_p"] = np.log10(df["flux_p"])
    df["log10_distance_s"] = np.log10(df["distance_s"])

    # Kepler magnitude is already logarithmic by definition,
    # so do not apply log10 again.
    df["kepler_mag"] = df["kepler_mag_used"]

    df["is_detected"] = to_bool(df[DETECTED_COL])

    return df


# ============================================================
# Main plotting
# ============================================================

def main():
    df = load_catalogues(DATA_GLOB)
    df = add_corner_columns(df)

    plot_cols = [
        "log10_radius_p",
        "log10_mass_p",
        "log10_radius_s",
        "log10_p_orb",
        "log10_flux_p",
        "log10_distance_s",
        "kepler_mag",
    ]

    labels = [
        r"$\log_{10}(R_p)$",
        r"$\log_{10}(M_p)$",
        r"$\log_{10}(R_\star)$",
        r"$\log_{10}(P_{\rm orb})$",
        r"$\log_{10}(F_p)$",
        r"$\log_{10}(d_\star)$",
        r"$Kp$",
    ]

    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=plot_cols)

    all_planets = clean[plot_cols]
    detected_planets = clean.loc[clean["is_detected"], plot_cols]

    print(f"Planets used in plot: {len(all_planets)}")
    print(f"Detected planets: {len(detected_planets)}")

    if len(detected_planets) == 0:
        raise ValueError(
            f"No detected planets found using detection column: {DETECTED_COL}"
        )

    # Plot all planets first.
    fig = corner.corner(
        all_planets,
        labels=labels,
        color="0.75",
        plot_datapoints=True,
        plot_density=True,
        fill_contours=False,
        show_titles=True,
        title_fmt=".2f",
        hist_kwargs={"density": True},
    )

    # Overlay Kepler-detected planets.
    corner.corner(
        detected_planets,
        labels=labels,
        fig=fig,
        color="C1",
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        fill_contours=False,
        show_titles=False,
        hist_kwargs={"density": True},
    )

    fig.suptitle(
        "Kepler Detectability Corner Plot\n"
        "Gray = all P-Pop planets, Orange = Kepler-detectable planets",
        fontsize=16,
        y=1.02,
    )

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved corner plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()