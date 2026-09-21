"""Sub-Neptune fraction of flat and P-Pop universes (comparison each) vs NASA under four cuts: all, M > 2,
I < 50, both. Detection is transit+RV with NASA-like measurement error. Also loaded by
flat_rocky_mr_vs_nasa.py.
Run: python plotting/scripts/analysis/puffy_cuts_flat.py
Needs Kepler Gaia-60pc universe 0 from the Kepler/TESS lines in `sim.py`
(~3-4 h for all 20; see README). Not a paper figure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tools.paths import REPO_ROOT, PSCOMPPARS_CSV, KEPLER_DATA_DIR, ANALYSIS_DIR
ROOT = Path(REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from science.physics import load_mass_radius_curve
from science.catalogs import load_measured_planets
from science.comparison import (
    monte_carlo_observed_fraction,
    monte_carlo_population_fraction,
)
from science.telescopes.detection import build_detected_population
from science.statistics import gaussian_density
from science.statistics import (
    NASA_MEASUREMENT_ERROR,
    SIMULATED_MEASUREMENT_ERROR,
)

PPOP_CATALOG = Path(KEPLER_DATA_DIR) / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = Path(PSCOMPPARS_CSV)
OUT_DIR = os.path.join(ANALYSIS_DIR, "puffy_cuts_flat")

N_SAMPLE = 20000
N_REPEATS = 10000
MASS_FRAC_ERR = SIMULATED_MEASUREMENT_ERROR["mass"]
RAD_FRAC_ERR = SIMULATED_MEASUREMENT_ERROR["radius"]
NASA_MASS_PREC = NASA_MEASUREMENT_ERROR["mass"]
NASA_RAD_PREC = NASA_MEASUREMENT_ERROR["radius"]
FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

ROWS = [("flat", "FLAT universe", [(True, "flat only_subneptunes", "tab:orange"), (False, "flat superearths_supneptunes", "tab:blue")]),
        ("ppop", "P-Pop universe", [(True, "P-Pop only_subneptunes", "tab:red"), (False, "P-Pop superearths_supneptunes", "tab:purple")])]
CUTS = [("all (no cut)", {}),
        ("mass > 2 M⊕", dict(mass_min=2.0)),
        ("insolation < 50 I⊕", dict(insol_max=50.0)),
        ("mass>2 & insol<50", dict(mass_min=2.0, insol_max=50.0))]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_mass_radius_curve()
    rng = np.random.default_rng(RNG_SEED)
    print("--> building pools + detectors...")
    pools = {
        population: build_detected_population(
            population, ppop_catalog=PPOP_CATALOG,
            flat_size=FLAT_N_POOL, seed=RNG_SEED, rv_mag_target=RV_MAG_TARGET,
            box=BOX,
        )
        for population in ("flat", "ppop")
    }
    nasa = load_measured_planets(
        NASA_FILE,
        mass_bounds=(BOX["m_lo"], BOX["m_hi"]),
        radius_bounds=(BOX["r_lo"], BOX["r_hi"]),
        insolation_bounds=None,
        max_relative_error=NASA_MEASUREMENT_ERROR,
        missing_relative_error=SIMULATED_MEASUREMENT_ERROR,
    )

    # precompute NASA bells per cut (shared by both rows)
    nasa_cut = {}
    for cut_label, cut in CUTS:
        nasa_cut[cut_label] = monte_carlo_observed_fraction(
            nasa, cut, m_sil, r_sil, rng, repeats=N_REPEATS
        )

    # ---- pass 1: compute every panel, collect all values for shared axes ----
    panels = {}
    all_vals = []
    print(f"\n  {'row':<7}{'cut':<20}{'universe':<9}{'μ':>7}{'σ':>7}{'tension':>9}{'N_eff':>7}")
    for ri, (pop, row_label, models) in enumerate(ROWS):
        for ci, (cut_label, cut) in enumerate(CUTS):
            nv, n_nasa = nasa_cut[cut_label]
            n_mu, n_sd = nv.mean(), nv.std()
            series = []
            for drop, label, colour in models:
                s, neff = monte_carlo_population_fraction(
                    pools[pop], cut, m_sil, r_sil, rng,
                    drop_super_earths=drop, sample_size=N_SAMPLE,
                    repeats=N_REPEATS, error=SIMULATED_MEASUREMENT_ERROR,
                )
                tens = (abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
                        if s.size and (s.std() + n_sd) > 0 else np.nan)
                series.append((label, colour, s, neff, tens))
                if s.size:
                    all_vals.append(s)
                    print(f"  {row_label.split()[0]:<7}{cut_label:<20}{label:<9}"
                          f"{s.mean():>7.3f}{s.std():>7.3f}{tens:>8.1f}σ{neff:>7.0f}")
            all_vals.append(nv)
            panels[(ri, ci)] = dict(series=series, nv=nv, n_nasa=n_nasa, n_mu=n_mu, n_sd=n_sd)
        print()

    # shared x-range + bins (so every panel is on the same scale and the bells are directly comparable)
    cat = np.concatenate(all_vals)
    lo, hi = cat.min(), cat.max(); pad = 0.04 * (hi - lo)
    gx_lo, gx_hi = lo - pad, hi + pad
    edges = np.linspace(gx_lo, gx_hi, 60)
    xs = np.linspace(gx_lo, gx_hi, 400)

    # shared y-max: tallest histogram bar / gaussian peak across all panels (so nothing clips)
    y_max = 0.0
    for P in panels.values():
        for _, _, s, _, _ in P["series"]:
            if s.size:
                y_max = max(y_max, np.histogram(s, bins=edges, density=True)[0].max(),
                            gaussian_density(xs, s.mean(), s.std()).max())
        y_max = max(y_max, np.histogram(P["nv"], bins=edges, density=True)[0].max(),
                    gaussian_density(xs, P["n_mu"], P["n_sd"]).max())
    gy_hi = y_max * 1.06

    # ---- pass 2: plot every panel on the shared x/y axes ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
    for ri, (pop, row_label, models) in enumerate(ROWS):
        for ci, (cut_label, cut) in enumerate(CUTS):
            ax = axes[ri, ci]
            P = panels[(ri, ci)]
            for label, colour, s, neff, tens in P["series"]:
                if s.size == 0:
                    continue
                ax.hist(s, bins=edges, density=True, color=colour, alpha=0.30)
                ax.plot(xs, gaussian_density(xs, s.mean(), s.std()), color=colour, lw=2.0,
                        label=f"{label}: μ={s.mean():.2f} σ={s.std():.3f} ({tens:.1f}σ)")
            ax.hist(P["nv"], bins=edges, density=True, color="tab:green", alpha=0.34)
            ax.plot(xs, gaussian_density(xs, P["n_mu"], P["n_sd"]), color="tab:green", lw=2.4,
                    label=f"NASA: μ={P['n_mu']:.2f} σ={P['n_sd']:.3f} (N={P['n_nasa']})")
            ax.set_xlim(gx_lo, gx_hi); ax.set_ylim(0, gy_hi)
            ax.set_title((f"{cut_label}\n" if ri == 0 else "") + f"{row_label} — sub-Neptune fraction", fontsize=10)
            ax.grid(alpha=0.2); ax.legend(fontsize=7.6, loc="upper left")
            if ci == 0:
                ax.set_ylabel("density over repeated draws")
            if ri == 1:
                ax.set_xlabel("sub-Neptune fraction")

    fig.suptitle("Sub-Neptune fraction: which universe does NASA imply, under four cuts — FLAT (top) vs P-Pop (bottom)\n"
                 "transit+RV detected, measurement-error propagated; closest bell to green NASA wins",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "puffy_cuts_flat_ppop.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
