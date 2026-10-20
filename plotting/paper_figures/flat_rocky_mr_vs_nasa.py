"""Makes the 2x1 plot of the sub-Neptune and super-Earth vs just super-Earth scenario,
with the resulting histogram after running through the telescope detections. (fig 4 in the paper)

Which rocky M-R relation (Chen & Kipping 2017, Otegi 2020, Edmondson 2023, Müller 2024), imposed
on flat_nonphysical, best matches NASA's volatile (sub-Neptune) fraction?
"""

from __future__ import annotations

import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from science.population_comparison import ComparisonConfig, run_comparison, load_sample, zone_mask, rng_for
from science.physics import extend_mass_radius_curve_power_law, load_mass_radius_curve
from science.comparison import (
    observed_measurement_arrays,
)
from science.universes.flat_baseline import (
    MR_SCATTER_DEX,
    flat_nonphysical,
)
from science.universes.flat_curves import (
    OTEGI_VOLATILE, SUB_NEPTUNE_FRAC_SD, SUPER_EARTH_FRAC_SD,
)
from science.physics import SUPER_EARTH_MIN_MASS, is_super_earth
from science.telescopes.detection import run_transit_rv_selection
from science.statistics import perturb_asymmetric, gaussian_density
from science.physics_constants import MR_RELATIONS as RELATIONS
from tools.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR
from tools.plotting_constants import PAPER_STYLE

OUT_DIR = ANALYSIS_DIR / "flat_rocky_mr_vs_nasa"

MASS_LIMS = (0.0, 12.0)
RADIUS_LIMS = (0.5, 2.4)
CONFIG = ComparisonConfig()
FLAT_N = CONFIG.pool_size
SEED = CONFIG.pool_seed
MC_REPEATS = CONFIG.draws
RV_MAG_TARGET = CONFIG.rv_mag_target


CUTS = [("M>2.5, R>1.4, I<50", dict(mass_min=CONFIG.mass_min, radius_min=CONFIG.radius_min, insol_max=50.0),
         "flat_rocky_mr_relations_2x4_low-insolation_corner.png")]

# The 2x2 paper figure shows the default relation (Otegi) alone, before/after the low-insolation selection.
OTEGI_2X2_CUTS = [("All detected planets", {}),
                  (r"Insolation < 50 $I_\oplus$, mass > 2.5 $M_\oplus$, radius > 1.4 $R_\oplus$",
                   dict(mass_min=CONFIG.mass_min, radius_min=CONFIG.radius_min, insol_max=50.0))]


SCATTER_LABELS = ("Sub-Neptunes", "Sub-Neptunes and super-Earths")


def _draw_scatter(ax, arr, cut, nasa, m_sil, r_sil, rng, title,
                  labels=SCATTER_LABELS, true_values=False, sil_range=None,
                  show_outside_box=False):
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
        # Underlying flat_radii_curves draw, shaded as its normal radius scatter.
        m = np.linspace(max(cut.get("mass_min", 0.0), 1e-3), MASS_LIMS[1], 600)
        r = np.linspace(*RADIUS_LIMS, 600)[:, None]
        r_sil_m = np.interp(m, m_c, r_c)
        r_vol = OTEGI_VOLATILE["mr_C"] * m ** OTEGI_VOLATILE["mr_beta"]
        extent = (m[0], m[-1], *RADIUS_LIMS)
        g_vol = np.exp(-0.5 * ((r / r_vol - 1) / SUB_NEPTUNE_FRAC_SD) ** 2)
        g_sil = np.exp(-0.5 * ((r / r_sil_m - 1) / SUPER_EARTH_FRAC_SD) ** 2)
        # Orange: Otegi + silicate (super-Earths); blue: Otegi above the silicate line.
        # Dash offsets alternate the colours where the 1 sigma edges coincide.
        for g, colour, z, dash in [
                (np.maximum(g_vol, g_sil * (m > SUPER_EARTH_MIN_MASS)), "tab:orange", 3, (4, (4, 4))),
                (g_vol * (r >= r_sil_m), "tab:blue", 4, (0, (4, 4)))]:
            if cut.get("mass_min") is not None:
                g = g * (m > cut["mass_min"]) * (r > cut.get("radius_min", 1.4))
            rgba = np.zeros((r.size, m.size, 4))
            rgba[..., :3] = to_rgb(colour)
            rgba[..., 3] = 0.7 * g
            ax.imshow(rgba, extent=extent, origin="lower", aspect="auto",
                      interpolation="bilinear", zorder=z)
            ax.contour(m, r.ravel(), g, levels=[np.exp(-0.5)], colors=colour,
                       linestyles=[dash], linewidths=1.5, zorder=5)
        ax.fill_between([], [], color="tab:blue", alpha=0.7, lw=0, label=labels[0])
        ax.fill_between([], [], color="tab:orange", alpha=0.7, lw=0, label=labels[1])
    else:
        # Illustrative detected-pool scatter, using the primary empirical errors.
        # Inference uses the saved N-sized surveys, independently of this rendering.
        cfg = CONFIG
        local_rng = rng_for(cfg, title, "scatter")
        zone = nasa.loc[zone_mask(nasa, .01, cut.get("insol_max", 10000.), cfg)]
        pool = arr.loc[arr.joint_detected & arr.insolation.ge(.01) & arr.insolation.lt(cut.get("insol_max",10000.))]
        mass, radius = pool.mass.to_numpy(), pool.radius.to_numpy()
        e = np.column_stack([zone[f"{q}_error_{side}"].to_numpy()/zone[q].to_numpy()
                             for q in ("mass","radius") for side in ("plus","minus")])
        error = e[local_rng.integers(len(e),size=len(pool))]
        mo = perturb_asymmetric(mass,mass*error[:,0],mass*error[:,1],local_rng)
        ro = perturb_asymmetric(radius,radius*error[:,2],radius*error[:,3],local_rng)
        dropped = is_super_earth(mass,radius)
        keep = (mo>cfg.mass_min)&(ro>cfg.radius_min)&(mo<=12)&(ro<=2.2)
        mo,ro,dropped = mo[keep],ro[keep],dropped[keep]
        if len(mo)>400:
            idx=local_rng.choice(len(mo),400,replace=False)
            mo,ro,dropped=mo[idx],ro[idx],dropped[idx]
        ax.scatter(mo[~dropped], ro[~dropped], s=15, color="tab:blue", alpha=0.45, lw=0,
                   zorder=3, label=labels[0])
        ax.scatter(mo[dropped], ro[dropped], s=15, color="tab:orange", alpha=0.5, lw=0,
                   zorder=4, label=labels[1])
    if show_outside_box:
        context = nasa[nasa["insolation"] < cut.get("insol_max", np.inf)]
        outside = ((context["mass"] <= cut["mass_min"])
                   | (context["radius"] <= cut["radius_min"]))
        context = context.loc[outside]
        ax.errorbar(context["mass"], context["radius"],
                    xerr=context[["mass_error_minus", "mass_error_plus"]].to_numpy().T,
                    yerr=context[["radius_error_minus", "radius_error_plus"]].to_numpy().T,
                    fmt="o", mfc="none", mec="0.5", ecolor="0.65", ms=5,
                    mew=0.8, elinewidth=0.6, capsize=1.5, alpha=0.6, zorder=4,
                    label="_nolegend_")
    ax.errorbar(nmc, nrc, xerr=np.array([nme2, nme1]), yerr=np.array([nre2, nre1]),
                fmt="o", mfc="none", mec="k", ecolor="k", ms=5, mew=1.0,
                elinewidth=0.6, capsize=1.5, alpha=0.8, zorder=5,
                label="Measured exoplanets")
    ax.set_xlim(*MASS_LIMS); ax.set_ylim(*RADIUS_LIMS)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xlabel(r"Planet mass [$M_\oplus$]")
    ax.set_ylabel(r"Planet radius [$R_\oplus$]")


def make_figure(cut_label, cut, fname, pools, nasa, m_sil, r_sil, rng):
    print(f"\n--> [{cut_label}]")
    fig, axes = plt.subplots(2, 4, figsize=(23, 11))
    summaries = {}
    for ci, (name, eq, applies, arr) in enumerate(pools):
        _draw_scatter(axes[0, ci], arr, cut, nasa, m_sil, r_sil, rng, name)
        summaries[name] = _draw_density_1x2(axes[1, ci], arr, cut, nasa, m_sil, r_sil, rng, tag=f"[{cut_label}] {name}", scenario=name)
    fig.tight_layout()
    if fname == "flat_rocky_mr_relations_2x4_low-insolation_corner.png":
        PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_png = PAPER_FIGURES_DIR / fname
    else:
        out_png = OUT_DIR / fname
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    out_png.with_suffix(".json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False)+"\n")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


def _draw_density_1x2(ax, arr, cut, nasa, m_sil, r_sil, rng, tag="", scenario="primary"):
    """Original density styling, using the shared observed-size survey draws."""
    from science.telescopes.detection import split_universes
    if scenario == "primary":
        summary, arrays = run_comparison()
    else:
        summary, arrays = run_comparison(populations=split_universes(arr), scenario=scenario.replace(" ", "_"),
                                         bins=(("I < 50", .01, 50.),))
    item = summary["bins"]["I < 50"]
    index = list(summary["bins"]).index("I < 50")
    n = item["n_observed"]
    edges = (np.arange(n+2)-.5)/n
    samples = [arrays[f"b{index}_{model}"] for model in
               ("escape_only", "rocky_formation", "observed")]
    low, high = min(v.min() for v in samples), max(v.max() for v in samples)
    pad = .05 * (high-low)
    gx = np.linspace(low-pad, high+pad, 400)
    y_max = 0.
    for (label, color), values in zip([
            ("Sub-Neptunes", "tab:blue"),
            ("Sub-Neptunes and super-Earths", "tab:orange"),
            ("Measured exoplanets", "tab:green")], samples):
        if color != "tab:green":
            heights, _, _ = ax.hist(values, bins=edges, density=True, color=color, alpha=.30)
            y_max = max(y_max, heights.max())
        pdf = gaussian_density(gx, values.mean(), values.std())
        if color == "tab:green":
            ax.fill_between(gx, pdf, color=color, alpha=.30, lw=0)
        ax.plot(gx, pdf, color=color, lw=2.4 if color == "tab:green" else 2., label=label)
        y_max = max(y_max, pdf.max())
    ax.set(xlim=(gx[0],gx[-1]), ylim=(0,y_max*1.4),
           xlabel="Volatile fraction", ylabel="Probability density")
    ax.legend(loc="upper left", framealpha=.95)
    print(f"{tag}: N={n}; model weights={item['posterior']}")
    return summary


def make_otegi_1x2(nasa, m_sil, r_sil, rng):
    """Volatile-fraction density beside the mass-radius draw.
    Orange = superearths_supneptunes; blue = only_subneptunes. The mass-radius panel shades
    the underlying radius distributions; the densities use noisy values."""
    plt.rcParams.update(PAPER_STYLE)
    cuts = [OTEGI_2X2_CUTS[1]]
    print(f"\n--> Otegi 1x2 ({len(cuts)} column(s); flat_radii_curves variants):")
    from science.population_comparison import CACHE_DIR
    from science.telescopes.detection import make_detected_pool
    from science.catalogs import COMPARISON_PARAMETER_BOX
    cfg = ComparisonConfig()
    arr = make_detected_pool("superearths_supneptunes", pool_size=cfg.pool_size,
                            chunk_size=cfg.chunk_size, cache_dir=CACHE_DIR, mission=cfg.mission,
                            seed=cfg.pool_seed, box=COMPARISON_PARAMETER_BOX, rv_mag_target=cfg.rv_mag_target)
    fig, axes = plt.subplots(len(cuts), 2, figsize=(12.5, 5.6 * len(cuts)), squeeze=False)
    for ci, (cut_label, cut) in enumerate(cuts):
        ax_mr, ax_hist = axes[ci, 0], axes[ci, 1]
        # Draw mass-radius first so the density panel's random stream is unchanged.
        _draw_scatter(ax_mr, arr, cut, nasa, m_sil, r_sil, rng, "",
                      labels=("Sub-Neptunes", "Sub-Neptunes and super-Earths"),
                      true_values=True, sil_range=(1e-3, MASS_LIMS[1]),
                      show_outside_box=True)
        summary = _draw_density_1x2(ax_hist, arr, cut, nasa, m_sil, r_sil, rng, tag=f"[1x2] {cut_label}")
        ax_mr.set_xlabel(r"Planet mass [$M_\oplus$]")
        ax_mr.set_ylabel(r"Planet radius [$R_\oplus$]")
    fig.tight_layout()
    fig.suptitle(r"Simulated Planet Detections at $I < 50\,I_\oplus$, $M>2.5\,M_\oplus$, $R>1.4\,R_\oplus$",
                     y=1.01, va="bottom")
    fname = "flat_otegi_1x2_low-insolation_selection.png"
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PAPER_FIGURES_DIR / fname
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    out_png.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m_sil, r_sil = load_mass_radius_curve()
    rng = np.random.default_rng(SEED)
    nasa, _ = load_sample()

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
