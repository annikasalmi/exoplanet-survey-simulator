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

# If this file is inside something like scripts/, parents[1] should be repo root.
# If not, fallback to current working directory.
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "run").exists():
    ROOT = Path.cwd()

DATA_GLOB = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_*.csv"

OUT_DIR = ROOT / "my_outputs" / "14_w2_v2.0_corner_and_heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DETECTED_COL = "detected"

# Histograms:
# False = raw counts, so orange should NOT be taller than gray just from normalization.
# True = compare distribution shapes.
NORMALIZE_HISTOGRAMS = False

# Heatmap settings
BINS = 30
MIN_PLANETS_PER_BIN = 5

# If you have too many planets, this keeps the corner plot from becoming slow.
MAX_GRAY_POINTS = 30000


# ============================================================
# Load and clean
# ============================================================

def load_catalogues():
    files = sorted(glob.glob(str(DATA_GLOB)))

    if not files:
        raise FileNotFoundError(f"No files found at: {DATA_GLOB}")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = Path(file).name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(files)} file(s)")
    print(f"Total planets before cleaning: {len(df)}")

    return df


def to_bool(series):
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


def prepare_columns(df):
    required = [
        "radius_p",
        "radius_s",
        "p_orb",
        "flux_p",
        "kepler_mag_used",
        DETECTED_COL,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    positive_cols = [
        "radius_p",
        "radius_s",
        "p_orb",
        "flux_p",
    ]

    for col in positive_cols:
        df = df[df[col] > 0].copy()

    df["log10_radius_p"] = np.log10(df["radius_p"])
    df["log10_radius_s"] = np.log10(df["radius_s"])
    df["log10_p_orb"] = np.log10(df["p_orb"])
    df["log10_flux_p"] = np.log10(df["flux_p"])
    df["kepler_mag"] = df["kepler_mag_used"]

    df["is_detected"] = to_bool(df[DETECTED_COL])

    plot_cols = [
        "log10_radius_p",
        "log10_radius_s",
        "log10_p_orb",
        "log10_flux_p",
        "kepler_mag",
    ]

    clean = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=plot_cols + ["is_detected"])
        .copy()
    )

    print(f"Total planets after cleaning: {len(clean)}")
    print(f"Detected planets: {clean['is_detected'].sum()}")

    return clean, plot_cols


# ============================================================
# 1. No-contour 5-variable corner plot
# ============================================================

def make_nocontour_corner(clean, plot_cols):
    labels = [
        r"$\log_{10}(R_p)$",
        r"$\log_{10}(R_\star)$",
        r"$\log_{10}(P_{\rm orb})$",
        r"$\log_{10}(F_p)$",
        r"$Kp$",
    ]

    all_planets = clean.copy()
    detected_planets = clean[clean["is_detected"]].copy()

    if len(all_planets) > MAX_GRAY_POINTS:
        all_planets = all_planets.sample(MAX_GRAY_POINTS, random_state=42)

    hist_kwargs = {
        "density": NORMALIZE_HISTOGRAMS,
        "histtype": "step",
        "linewidth": 1.2,
    }

    # Gray background: all planets, no contours.
    fig = corner.corner(
        all_planets[plot_cols].to_numpy(),
        labels=labels,
        color="0.65",
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        fill_contours=False,
        show_titles=False,
        data_kwargs={
            "alpha": 0.08,
            "ms": 1.5,
            "mew": 0,
        },
        hist_kwargs=hist_kwargs,
    )

    # Orange overlay: detected planets, points only.
    corner.corner(
        detected_planets[plot_cols].to_numpy(),
        labels=labels,
        fig=fig,
        color="C1",
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        fill_contours=False,
        show_titles=False,
        data_kwargs={
            "alpha": 0.85,
            "ms": 3.5,
            "mew": 0,
        },
        hist_kwargs=hist_kwargs,
    )

    hist_note = "raw-count histograms" if not NORMALIZE_HISTOGRAMS else "normalized histograms"

    fig.suptitle(
        "Kepler Detectability Corner Plot, 5 Variables, No Contours\n"
        f"Gray = all P-Pop planets, Orange = Kepler-detectable planets, {hist_note}",
        fontsize=15,
        y=1.02,
    )

    out_path = OUT_DIR / "corner_5var_nocontour.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved no-contour corner plot to: {out_path}")


# ============================================================
# 2. Easy single detection-fraction heatmap
# ============================================================

def compute_detection_fraction(clean, x_col, y_col, bins=BINS):
    x = clean[x_col].to_numpy()
    y = clean[y_col].to_numpy()
    detected = clean["is_detected"].to_numpy()

    total_counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    detected_counts, _, _ = np.histogram2d(x[detected], y[detected], bins=[x_edges, y_edges])

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = detected_counts / total_counts

    # Hide bins with too few planets, because those fractions are noisy.
    frac[total_counts < MIN_PLANETS_PER_BIN] = np.nan

    return frac, x_edges, y_edges, total_counts


def make_single_detection_fraction_heatmap(clean):
    x_col = "log10_p_orb"
    y_col = "log10_radius_p"

    frac, x_edges, y_edges, total_counts = compute_detection_fraction(clean, x_col, y_col)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        frac.T,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Detection fraction")

    ax.set_xlabel(r"$\log_{10}(P_{\rm orb})$")
    ax.set_ylabel(r"$\log_{10}(R_p)$")
    ax.set_title(
        "Kepler Detection Fraction Heatmap\n"
        "Short-period + large-radius planets should be easiest"
    )

    out_path = OUT_DIR / "heatmap_detection_fraction_period_vs_radius.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved single detection-fraction heatmap to: {out_path}")


# ============================================================
# 3. Bonus: pairwise detection-fraction heatmap grid
# ============================================================

def make_pairwise_detection_fraction_grid(clean, plot_cols):
    labels = {
        "log10_radius_p": r"$\log_{10}(R_p)$",
        "log10_radius_s": r"$\log_{10}(R_\star)$",
        "log10_p_orb": r"$\log_{10}(P_{\rm orb})$",
        "log10_flux_p": r"$\log_{10}(F_p)$",
        "kepler_mag": r"$Kp$",
    }

    n = len(plot_cols)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))

    last_im = None

    for i, y_col in enumerate(plot_cols):
        for j, x_col in enumerate(plot_cols):
            ax = axes[i, j]

            # Upper triangle: empty, like a corner plot.
            if i < j:
                ax.axis("off")
                continue

            # Diagonal: simple histograms.
            if i == j:
                ax.hist(
                    clean[x_col],
                    bins=BINS,
                    histtype="step",
                    linewidth=1.2,
                    label="All",
                )
                ax.hist(
                    clean.loc[clean["is_detected"], x_col],
                    bins=BINS,
                    histtype="step",
                    linewidth=1.2,
                    label="Detected",
                )
                ax.set_yticks([])

            # Lower triangle: detection-fraction heatmap.
            else:
                frac, x_edges, y_edges, total_counts = compute_detection_fraction(
                    clean,
                    x_col,
                    y_col,
                    bins=BINS,
                )

                last_im = ax.imshow(
                    frac.T,
                    origin="lower",
                    aspect="auto",
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    vmin=0,
                    vmax=1,
                )

            # Only label outside edges to reduce visual chaos.
            if i == n - 1:
                ax.set_xlabel(labels[x_col], fontsize=10)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(labels[y_col], fontsize=10)
            else:
                ax.set_yticklabels([])

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.6)
        cbar.set_label("Detection fraction")

    fig.suptitle(
        "Pairwise Kepler Detection-Fraction Heatmaps\n"
        f"Each colored bin = detected planets / all planets, hidden if fewer than {MIN_PLANETS_PER_BIN} planets",
        fontsize=16,
        y=0.92,
    )

    out_path = OUT_DIR / "heatmap_pairgrid_5var_detection_fraction.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved pairwise detection-fraction grid to: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    df = load_catalogues()
    clean, plot_cols = prepare_columns(df)

    make_nocontour_corner(clean, plot_cols)
    make_single_detection_fraction_heatmap(clean)
    make_pairwise_detection_fraction_grid(clean, plot_cols)

    print("\nDone.")
    print(f"Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()