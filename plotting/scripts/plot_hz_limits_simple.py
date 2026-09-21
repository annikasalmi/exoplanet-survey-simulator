"""Where HWO can see habitable-zone planets in stellar luminosity vs distance.
Run: python plotting/scripts/plot_hz_limits_simple.py
-> results/figures/other/distance_luminosity_simple.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from tools import physics_constants as const
from tools.paths import OTHER_FIGURES_DIR

OUT_PATH = OTHER_FIGURES_DIR / "distance_luminosity_simple.png"

XLIM_MIN = 0.01
XLIM_MAX = 15.0
YLIM_MIN = 1.0
YLIM_MAX = 35.0

hwo = const.HWOConstants("best")
FLUX_LIMIT = float(hwo.min_planet_flux_star_ratio)
IWA_LIMIT = float(hwo.iwa)
THETA_LIMIT_RAD = IWA_LIMIT * const.arcsec_to_radians


def plot_luminosity_distance():
    """Plot detectability for habitable planets."""

    L_vals = np.logspace(np.log10(XLIM_MIN), np.log10(XLIM_MAX), 100)
    D_vals = np.linspace(YLIM_MIN, YLIM_MAX, 100)
    L_grid, D_grid = np.meshgrid(L_vals, D_vals)

    a_hz_m = np.sqrt(L_grid) * const.au_to_m
    distance_m = D_grid * const.pc_to_m
    theta_arcsec = (a_hz_m / distance_m) * const.rad_to_arcsec

    Rp_small = const.R_earth_min_habitable * const.R_earth
    Rp_large = const.R_earth_max_habitable * const.R_earth
    T_star = (L_grid * const.temp_sun**4) ** 0.25
    T_planet = const.T_earth

    flux_ratio_small = (T_planet * Rp_small**2) / (T_star * const.R_sun**2) / (distance_m / const.pc_to_m) ** 2
    flux_ratio_large = (T_planet * Rp_large**2) / (T_star * const.R_sun**2) / (distance_m / const.pc_to_m) ** 2

    angular_sep_ok = theta_arcsec >= THETA_LIMIT_RAD * const.rad_to_arcsec
    detect_small = (flux_ratio_small >= FLUX_LIMIT) & angular_sep_ok
    detect_large = (flux_ratio_large >= FLUX_LIMIT) & angular_sep_ok

    region = np.zeros_like(L_grid, dtype=int)
    region[detect_small | detect_large] = 1

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_betweenx([0, YLIM_MAX], XLIM_MIN, 0.1, color="pink", alpha=0.3, label="Too faint to detect")

    L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
    D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)

    ax.pcolormesh(L_edges, D_edges, region, cmap=plt.cm.RdYlGn, shading="auto", vmin=0, vmax=1)

    ax.set_xscale("log")
    ax.set_xlabel("Stellar Luminosity [L☉]")
    ax.set_ylabel("Distance [pc]")
    ax.set_title(f"Detectable Radius Range ({const.R_earth_min_habitable}–{const.R_earth_max_habitable} R⊕)")
    ax.set_ylim(YLIM_MIN, YLIM_MAX)
    ax.set_xlim(XLIM_MIN, 2.0)

    D_theta_boundary = (
        np.sqrt(L_vals) * const.au_to_m * const.rad_to_arcsec
        / (THETA_LIMIT_RAD * const.rad_to_arcsec * const.pc_to_m)
    )
    mask = D_theta_boundary <= YLIM_MAX

    ax.plot(L_vals[mask], D_theta_boundary[mask], color="black", linestyle="--", linewidth=2, label="HWO Angular Sep. Limit")

    for L, color in zip([const.L_m_dwarf_max, const.L_g_star_min, const.L_g_star_max], ["red", "gold", "gold"]):
        ax.axvline(L, color=color, linestyle="--", linewidth=2)

    L_m_dwarf_range = np.linspace(0, const.L_m_dwarf_max, 100)
    D_theta_boundary_mdwarf = (
        np.sqrt(L_m_dwarf_range) * const.au_to_m * const.rad_to_arcsec
        / (THETA_LIMIT_RAD * const.rad_to_arcsec * const.pc_to_m)
    )

    mask = (D_theta_boundary_mdwarf <= YLIM_MAX) & (D_theta_boundary_mdwarf >= 0)

    if np.any(mask):
        ax.fill_between(L_m_dwarf_range[mask], 0, D_theta_boundary_mdwarf[mask], color="darkgreen", alpha=1, label="M dwarfs observable by HWO")

    specific_planets = {
        "Proxima Cen b": (0.0016, 1.3),
        "TOI-700 d": (0.023, 31.1),
        "TOI-700 e": (0.023, 31.1),
    }

    for planet_name, (lum, dist) in specific_planets.items():
        if XLIM_MIN <= lum <= XLIM_MAX and YLIM_MIN <= dist <= YLIM_MAX:
            ax.annotate(
                planet_name, (lum, dist), xytext=(5, 5), textcoords="offset points",
                fontsize=8, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    legend_elements = [
        Line2D([0], [0], color="red", linestyle="--", label="M dwarf Region"),
        Line2D([0], [0], color="gold", linestyle="--", label="G Star Region"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="HWO Angular Sep. Limit"),
        Patch(facecolor="darkgreen", alpha=1, label="M dwarfs observable by HWO"),
        Patch(facecolor="pink", alpha=0.3, label="Too faint to detect"),
    ]

    ax.legend(handles=legend_elements, loc="lower right", fontsize=12)

    plt.tight_layout()
    OTHER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_luminosity_distance()
    print(f"Plot saved to {OUT_PATH}")