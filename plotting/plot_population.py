"""Population and detector summaries for a generated universe."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.paths import PLOTS_DIR


def plot_population(df, *, name):
    """Plot the supplied catalog without generating or modifying its population."""
    out_dir = os.path.join(PLOTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Plotting {name} ({len(df):,} planets)")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(df["radius_p"], bins=48, color="#4c78a8", alpha=0.85)
    axs[0].set_xlabel("Planet radius [R_earth]")
    axs[0].set_ylabel("Planets")
    axs[0].set_title("Planet radii")

    axs[1].hist(df["mass_p"], bins=np.logspace(np.log10(df["mass_p"].min()),
                                               np.log10(df["mass_p"].max()), 50),
                color="#f58518", alpha=0.85)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Planet mass [M_earth]")
    axs[1].set_title("Planet masses")

    axs[2].hist(df["flux_p"], bins=np.logspace(np.log10(df["flux_p"].min()),
                                               np.log10(df["flux_p"].max()), 50),
                color="#54a24b", alpha=0.85)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("Insolation [I_earth]")
    axs[2].set_title("Derived insolation")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "population.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    detector_options = [
        ("kepler_detected", "Kepler", "#4c78a8"),
        ("tess_detected", "TESS", "#e45756"),
        ("rv_detected", "RV", "#72b7b2"),
        ("hwo_detected", "HWO", "#b279a2"),
    ]
    detector_cols = [item for item in detector_options if item[0] in df]
    if not detector_cols:
        print(f"Saved plots: {out_dir}")
        return

    labels = [item[1] for item in detector_cols]
    colors = [item[2] for item in detector_cols]
    counts = [int(df[col].astype(bool).sum()) for col, _, _ in detector_cols]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black")
    ax.set_ylabel("Detected planets")
    ax.set_title("Detections")
    for bar, count in zip(bars, counts):
        pct = 100 * count / len(df) if len(df) else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "detections.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plots: {out_dir}")
