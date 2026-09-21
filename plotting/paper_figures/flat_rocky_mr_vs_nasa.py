"""Makes the 2x1 plot of the sub-Neptune and super-Earth vs just super-Earth scenario,
with the resulting histogram after running through the telescope detections. (fig 4 in the paper)

Which rocky M-R relation (Chen & Kipping 2017, Otegi 2020, Edmondson 2023, Müller 2024), imposed
on flat_nonphysical, best matches NASA's volatile (sub-Neptune) fraction?
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from science.catalogs import load_measured_planets
from science.physics import extend_mass_radius_curve_power_law, load_mass_radius_curve
from science.comparison import (
    mock_survey_volatile_fractions,
    monte_carlo_observed_fraction,
    noised_detected_population_sample,
    observed_measurement_arrays,
    true_detected_population_sample,
)
from science.universes.flat_baseline import (
    MR_SCATTER_DEX,
    flat_nonphysical,
)
from science.universes.flat_curves import flat_radii_curves
from science.telescopes.detection import run_transit_rv_selection
from science.statistics import gaussian_density
from science.statistics import (
    NASA_MEASUREMENT_ERROR, SIMULATED_MEASUREMENT_ERROR,
)
from science.physics_constants import MR_RELATIONS as RELATIONS
from tools.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR, PSCOMPPARS_CSV

OUT_DIR = ANALYSIS_DIR / "flat_rocky_mr_vs_nasa"

MASS_LIMS = (0.0, 12.0)
RADIUS_LIMS = (0.5, 2.4)
FLAT_N = 150000
SEED = 0
MC_REPEATS = 4000
RV_MAG_TARGET = 12.0


CUTS = [("all (no cut)", {}, "flat_rocky_mr_relations_2x4.png"),
        ("insol<50 I⊕", dict(insol_max=50.0), "flat_rocky_mr_relations_2x4_s50.png"),
        ("mass>2 M⊕ & insol<50 I⊕ (low-insolation super-Earth selection)",
         dict(mass_min=2.0, insol_max=50.0), "flat_rocky_mr_relations_2x4_low-insolation_corner.png")]

# The 2x2 paper figure shows the default relation (Otegi) alone, before/after the low-insolation selection.
OTEGI_2X2_CUTS = [("All detected planets", {}),
                  (r"Insolation < 50 $I_\oplus$, mass > 2 $M_\oplus$",
                   dict(mass_min=2.0, insol_max=50.0))]


SCATTER_LABELS = ("Sub-Neptunes", "Sub-Neptunes and super-Earths")


def _draw_scatter(ax, arr, cut, nasa, m_sil, r_sil, rng, title,
                  labels=SCATTER_LABELS, true_values=False, sil_range=None):
    """Top-row panel: one detected, noise-perturbed mass-radius draw. With
    true_values, the planets are drawn at their true masses and radii with the
    simulated measurement errors as error bars instead."""
    nmc, nrc, nme1, nme2, nre1, nre2 = observed_measurement_arrays(nasa, cut)
    # sil_range=(lo, hi) draws the curve across that whole mass range.
    m_c, r_c = (
        extend_mass_radius_curve_power_law(m_sil, r_sil, *sil_range)
        if sil_range else (m_sil, r_sil)
    )
    ax.fill_between(m_c, r_c, 2.6, color="0.965", zorder=0)
    ax.plot(m_c, r_c, "k-", lw=1.2, zorder=6, label="Pure silicate")
    if true_values:
        # One illustrative survey for each flat_radii_curves variant.
        for above_only, n, colour, lbl, z in [
                (True, N_SURVEY_BLUE, "tab:blue", labels[0], 4),
                (False, N_SURVEY_ORANGE, "tab:orange", labels[1], 3)]:
            mt, rt = true_detected_population_sample(
                arr, cut, n, rng, exclude_super_earths=above_only
            )
            ax.errorbar(mt, rt,
                        xerr=np.array([mt * (1 - np.exp(-SIMULATED_MEASUREMENT_ERROR["mass"])),
                                       mt * (np.exp(SIMULATED_MEASUREMENT_ERROR["mass"]) - 1)]),
                        yerr=np.array([rt * (1 - np.exp(-SIMULATED_MEASUREMENT_ERROR["radius"])),
                                       rt * (np.exp(SIMULATED_MEASUREMENT_ERROR["radius"]) - 1)]),
                        fmt="o", ms=4, color=colour, alpha=0.6, elinewidth=0.6,
                        capsize=0, zorder=z, label=lbl)
    else:
        mo, ro, dropped = noised_detected_population_sample(
            arr, cut, rng, window=(MASS_LIMS[1], *RADIUS_LIMS)
        )
        ax.scatter(mo[~dropped], ro[~dropped], s=15, color="tab:blue", alpha=0.45, lw=0,
                   zorder=3, label=labels[0])
        ax.scatter(mo[dropped], ro[dropped], s=15, color="tab:orange", alpha=0.5, lw=0,
                   zorder=4, label=labels[1])
    ax.errorbar(nmc, nrc, xerr=np.array([nme2, nme1]), yerr=np.array([nre2, nre1]),
                fmt="o", mfc="none", mec="k", ecolor="k", ms=5, mew=1.0,
                elinewidth=0.6, capsize=1.5, alpha=0.8, zorder=5,
                label="Measured exoplanets")
    ax.set_xlim(*MASS_LIMS); ax.set_ylim(*RADIUS_LIMS)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xlabel(r"Planet mass [$M_\oplus$]")
    ax.set_ylabel(r"Planet radius [$R_\oplus$]")


# Planets per simulated survey in the 1x2: blue ("Sub-Neptunes only") and orange
# ("Sub-Neptunes and super-Earths"). The left panel shows one survey of each size.
N_SURVEY_BLUE = 50
N_SURVEY_ORANGE = 100


def make_figure(cut_label, cut, fname, pools, nasa, m_sil, r_sil, rng):
    print(f"\n--> [{cut_label}]")
    fig, axes = plt.subplots(2, 4, figsize=(23, 11))
    for ci, (name, eq, applies, arr) in enumerate(pools):
        _draw_scatter(axes[0, ci], arr, cut, nasa, m_sil, r_sil, rng, name)
        _draw_density_1x2(axes[1, ci], arr, cut, nasa, m_sil, r_sil, rng, tag=f"[{cut_label}] {name}")
    fig.tight_layout()
    out_png = OUT_DIR / fname
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    if fname == "flat_rocky_mr_relations_2x4_low-insolation_corner.png":
        PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIGURES_DIR / fname, dpi=170, bbox_inches="tight")
        print(f"--> Saved paper copy: {PAPER_FIGURES_DIR / fname}")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


def _draw_density_1x2(ax, arr, cut, nasa, m_sil, r_sil, rng, tag=""):
    """Right panel of the Otegi 1x2: probability densities of the volatile
    fraction. Blue / orange: histograms of the 50- / 100-planet surveys with
    their fitted normals. NASA: its fitted normal only, filled (its 27 planets
    are the same in every draw, so only its mean and spread matter)."""
    nv, _ = monte_carlo_observed_fraction(
        nasa, cut, m_sil, r_sil, rng, repeats=MC_REPEATS)
    n_mu, n_sd = nv.mean(), nv.std()
    lo, hi = -np.inf, cut.get("insol_max", np.inf)
    mass_min = cut.get("mass_min", 0.0)
    sA = mock_survey_volatile_fractions(
        arr, lo, hi, m_sil, r_sil, rng, N_SURVEY_BLUE,
        repeats=MC_REPEATS, mass_min=mass_min,
        error=SIMULATED_MEASUREMENT_ERROR, exclude_super_earths=True,
    )
    sB = mock_survey_volatile_fractions(
        arr, lo, hi, m_sil, r_sil, rng, N_SURVEY_ORANGE,
        repeats=MC_REPEATS, mass_min=mass_min,
        error=SIMULATED_MEASUREMENT_ERROR,
    )
    cat = np.concatenate([nv, sA, sB])
    lo, hi = cat.min(), cat.max(); pad = 0.05 * (hi - lo)
    gx = np.linspace(lo - pad, hi + pad, 400)
    # Surveys of 50 / 100 planets give fractions on a 0.02 / 0.01 grid, so 0.02 is
    # the narrowest bin that leaves no blue bar empty; edges at half-hundredths
    # keep every grid value off a bin edge.
    edges = np.arange(-0.005, 1.02 + 1e-9, 0.02)
    y_max = 0.0
    for lbl, s, colour in [("Sub-Neptunes", sA, "tab:blue"),
                           ("Sub-Neptunes and super-Earths", sB, "tab:orange")]:
        heights, _, _ = ax.hist(s, bins=edges, density=True, color=colour, alpha=0.30)
        pdf = gaussian_density(gx, s.mean(), s.std())
        ax.plot(gx, pdf, color=colour, lw=2.0, label=lbl)
        y_max = max(y_max, heights.max(), pdf.max())
        tens = abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
        print(f"    {tag} {lbl}: mu={s.mean():.3f} sd={s.std():.3f} "
              f"tension={tens:.1f}sigma N_draws={s.size}")
    pdf = gaussian_density(gx, n_mu, n_sd)
    ax.fill_between(gx, pdf, color="tab:green", alpha=0.30, lw=0)
    ax.plot(gx, pdf, color="tab:green", lw=2.4, label="Measured exoplanets")
    y_max = max(y_max, pdf.max())
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(0, y_max * 1.4)   # headroom for the legend
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_xlabel("Volatile fraction")
    ax.set_ylabel("Probability density")


def make_otegi_1x2(nasa, m_sil, r_sil, rng, show_full_population=False):
    """Volatile-fraction density beside the mass-radius draw.
    show_full_population adds the same pair for the uncut population as a row on top.
    Orange = superearths_supneptunes; blue = only_subneptunes. The mass-radius panel shows
    true values with simulated error bars; the densities use noisy values, so some blue can fall
    below the line."""
    cuts = list(OTEGI_2X2_CUTS) if show_full_population else [OTEGI_2X2_CUTS[1]]
    print(f"\n--> Otegi 1x2 ({len(cuts)} column(s); flat_radii_curves variants):")
    arr = run_transit_rv_selection(
        flat_radii_curves(
            FLAT_N, seed=SEED, variant="superearths_supneptunes"
        ),
        rv_mag_target=RV_MAG_TARGET,
    )
    fig, axes = plt.subplots(len(cuts), 2, figsize=(12.5, 5.6 * len(cuts)), squeeze=False)
    for ci, (cut_label, cut) in enumerate(cuts):
        ax_hist, ax_mr = axes[ci, 0], axes[ci, 1]
        # The mass-radius draw runs first so the density draws use the same random
        # stream as before the panels were swapped.
        _draw_scatter(ax_mr, arr, cut, nasa, m_sil, r_sil, rng, "",
                      labels=("Sub-Neptunes", "Sub-Neptunes and super-Earths"),
                      true_values=True, sil_range=(1e-3, MASS_LIMS[1]))
        _draw_density_1x2(ax_hist, arr, cut, nasa, m_sil, r_sil, rng, tag=f"[1x2] {cut_label}")
        ax_mr.set_xlabel(r"Planet mass [$M_\oplus$]")
        ax_mr.set_ylabel(r"Planet radius [$R_\oplus$]")
        if show_full_population:
            ax_hist.set_title(cut_label)
            ax_mr.set_title(cut_label)
    fig.tight_layout()
    if not show_full_population:
        # Figure-wide title over both panels (placed above the axes; the tight
        # bounding box on save keeps it).
        fig.suptitle("Simulated detections of low-insolation massive planets", y=1.01, va="bottom")
    fname = ("flat_otegi_1x2_with_full_population.png" if show_full_population
             else "flat_otegi_1x2_low-insolation_selection.png")
    out_png = OUT_DIR / fname
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURES_DIR / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")
    print(f"--> Saved paper copy: {PAPER_FIGURES_DIR / fname}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m_sil, r_sil = load_mass_radius_curve()
    rng = np.random.default_rng(SEED)
    nasa = load_measured_planets(
        PSCOMPPARS_CSV,
        mass_bounds=(0.1, 12.0), radius_bounds=(0.5, 2.2),
        insolation_bounds=None,
        max_relative_error=NASA_MEASUREMENT_ERROR,
        missing_relative_error=SIMULATED_MEASUREMENT_ERROR,
    )

    print(f"--> building {len(RELATIONS)} rocky-relation pools + detectors "
          f"(mass scatter = {MR_SCATTER_DEX} dex)...")
    pools = [(name, eq, applies, run_transit_rv_selection(
                 flat_nonphysical(FLAT_N, seed=SEED, **kw),
                 rv_mag_target=RV_MAG_TARGET))
             for name, eq, applies, kw in RELATIONS]

    for cut_label, cut, fname in CUTS:
        make_figure(cut_label, cut, fname, pools, nasa, m_sil, r_sil, rng)

    make_otegi_1x2(nasa, m_sil, r_sil, rng)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
