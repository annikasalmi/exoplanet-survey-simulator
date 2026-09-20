"""Bayesian comparison in the low-insolation super-Earth desert over three insolation panels. Priors:
rocky_formation (superearths_supneptunes), escape_only (only_subneptunes); flat_nonphysical only maps
detectability. Likelihood = transit+RV detection fraction. Scores NASA's volatile fraction per bin (binomial).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tools.paths import REPO_ROOT, PSCOMPPARS_CSV, ANALYSIS_DIR
ROOT = Path(REPO_ROOT)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

from science.physics import (
    SUPER_EARTH_MIN_MASS,
    load_mass_radius_curve,
    radius_on_curve,
)
from science.catalogs import COMPARISON_PARAMETER_BOX, load_measured_planets
from science.comparison import (
    COLD_DESERT_MAX_INSOLATION,
    INSOLATION_BINS,
    detection_fraction_map,
    observed_fraction_uncertainty,
    observed_volatile_count,
    predicted_volatile_fraction,
)
from science.telescopes.detection import make_detected_pool, split_universes
from science.statistics import binomial_model_posterior
from science.statistics import (
    NASA_MEASUREMENT_ERROR,
    SIMULATED_MEASUREMENT_ERROR,
)

# Transit leg of the joint detector; main("tess") switches it to TESS.
MISSION = "kepler"
_MISSION_LABEL = {"kepler": "Kepler", "tess": "TESS"}
_OUT_NAME = {"kepler": "bayesian_low-insolation_rocky_desert",
             "tess": "bayesian_low-insolation_rocky_desert_tess"}


NASA_FILE = Path(PSCOMPPARS_CSV)

BOX = COMPARISON_PARAMETER_BOX
FLAT_N_POOL = 10_000_000       # 10x: the low-insolation among-transiting denominator is thin (~2% transit);
                               # this fills every MR cell in the I<10 panel above MIN_CELL.
CHUNK = 2_000_000              # generate+detect in chunks to bound peak memory (~1.5 GB/chunk)
RNG_SEED = 0
RV_MAG_TARGET = 12.0
MASS_MIN = SUPER_EARTH_MIN_MASS  # low-insolation super-Earth desert cut
LOW_INSOLATION_MAX = COLD_DESERT_MAX_INSOLATION
MASS_FRAC_ERR = SIMULATED_MEASUREMENT_ERROR["mass"]
RAD_FRAC_ERR = SIMULATED_MEASUREMENT_ERROR["radius"]
NASA_MASS_PREC = NASA_MEASUREMENT_ERROR["mass"]
NASA_RAD_PREC = NASA_MEASUREMENT_ERROR["radius"]
N_FRAC_REP = 4000               # noise realizations for the predicted volatile fraction
                                # (as in the main paper's Section-4.2 MC procedure)

UNIVERSES = [
    ("rocky_formation", "Primordial-rocky", "tab:blue"),
    ("escape_only", "Escape-only", "tab:orange"),
]

# LINEAR mass grid (paper Fig.1 axis). With the 10M pool every cell in every panel clears
# MIN_CELL, so the detectability field is fully colored (no blanks).
M_EDGES = np.linspace(BOX["m_lo"], BOX["m_hi"], 17)
R_EDGES = np.linspace(BOX["r_lo"], BOX["r_hi"], 15)
M_CENT = 0.5 * (M_EDGES[:-1] + M_EDGES[1:])
R_CENT = 0.5 * (R_EDGES[:-1] + R_EDGES[1:])
MIN_CELL = 5                   # display gate; at 10M even the sparsest low-insolation cell has >=~10 samples


def build_universes():
    print(f"--> building superearths_supneptunes pool N={FLAT_N_POOL} ...")
    cache_dir = os.path.join(ANALYSIS_DIR, _OUT_NAME[MISSION])
    univ = split_universes(make_detected_pool(
        "superearths_supneptunes", pool_size=FLAT_N_POOL, chunk_size=CHUNK,
        cache_dir=cache_dir, mission=MISSION, seed=RNG_SEED,
        box=BOX, rv_mag_target=RV_MAG_TARGET,
    ))
    print(f"--> building flat_nonphysical pool N={FLAT_N_POOL} ...")
    univ["flat_nonphysical"] = make_detected_pool(
        "flat_nonphysical", pool_size=FLAT_N_POOL, chunk_size=CHUNK,
        cache_dir=cache_dir, mission=MISSION, seed=RNG_SEED,
        box=BOX, rv_mag_target=RV_MAG_TARGET,
    )
    for key, _, _ in UNIVERSES:
        u = univ[key]
        print(f"    {key:<16} pool={u['mass'].size:>8}  "
              f"transiting={int(u['transit_eligible'].sum()):>7}  "
              f"detected={int(u['joint_detected'].sum()):>6}")
    return univ


# ----------------------------------------------------------------------- reporting
def desert_table(univ, nasa, m_sil, r_sil, rng, tag):
    print(f"\n  ================ Bayesian model comparison — {tag} ================")
    print("  volatile fraction f_k, NASA v/n, binomial likelihood L_k, and likelihood odds O=L/sum L\n")
    segments = [(lbl, lo, hi, MASS_MIN) for lbl, lo, hi in INSOLATION_BINS]
    segments += [("all insolation", BOX["f_lo"], BOX["f_hi"], None)]
    rows = []
    for lbl, lo, hi, mcut in segments:
        p_raw = {key: predicted_volatile_fraction(
                     univ[key], lo, hi, m_sil, r_sil, rng, mass_min=mcut,
                     repeats=N_FRAC_REP, error=SIMULATED_MEASUREMENT_ERROR)
                 for key, _, _ in UNIVERSES}
        p_by = {key: v[0] for key, v in p_raw.items()}
        p_sigma = {key: v[1] for key, v in p_raw.items()}
        k, n = observed_volatile_count(
            nasa, lo, hi, m_sil, r_sil, mass_min=mcut
        )
        if n == 0:
            print(f"  {lbl:<24} NASA n=0 — skipped")
            continue
        _, nasa_sigma = observed_fraction_uncertainty(
            nasa, lo, hi, m_sil, r_sil, rng,
            mass_min=mcut, repeats=N_FRAC_REP,
        )
        logl, post = binomial_model_posterior(p_by, k, n)
        L = {key: float(np.exp(logl[key])) for key in logl}
        ftxt = "  ".join(f"{lbl2[:5]}={p_by[key]:.3f}+-{p_sigma[key]:.3f}" for key, lbl2, _ in UNIVERSES)
        Ltxt = "  ".join(f"{lbl2[:5]}={L[key]:.2e}" for key, lbl2, _ in UNIVERSES)
        postxt = "  ".join(f"{lbl2[:5]}={post[key]:.2f}" for key, lbl2, _ in UNIVERSES)
        cut = " (M>2)" if mcut else ""
        print(f"  {lbl + cut:<24} NASA vol {k}/{n} (v/n={k/n:.3f}+-{nasa_sigma:.3f})")
        print(f"      f_k : {ftxt}")
        print(f"      L_k : {Ltxt}")
        print(f"      odds O   : {postxt}")
        rows.append(dict(label=lbl + cut, k=k, n=n, p=p_by, p_sigma=p_sigma,
                          nasa_sigma=nasa_sigma, post=post, logl=logl, L=L))
    return rows


def save_stats_table(rows, tag):
    """Per-panel f_k, L_k, O_esc, and Bayes factor to CSV, so the paper table traces to code.
    rocky_formation is the paper's Primordial-rocky (f_prim); escape_only is Escape-only (f_esc)."""
    recs = []
    for r in rows:
        Le, Lp = r["L"]["escape_only"], r["L"]["rocky_formation"]
        recs.append(dict(
            panel=r["label"], v=r["k"], n=r["n"], v_over_n=round(r["k"] / r["n"], 3),
            v_over_n_sigma=round(r["nasa_sigma"], 3),
            f_esc=round(r["p"]["escape_only"], 3), f_esc_sigma=round(r["p_sigma"]["escape_only"], 3),
            f_prim=round(r["p"]["rocky_formation"], 3), f_prim_sigma=round(r["p_sigma"]["rocky_formation"], 3),
            L_esc=Le, L_prim=Lp,
            O_esc=round(r["post"]["escape_only"], 3),
            BF_esc_over_prim=Le / max(Lp, 1e-300)))
    df = pd.DataFrame(recs)
    out = os.path.join(ANALYSIS_DIR, _OUT_NAME[MISSION], "model_stats.csv")
    df.to_csv(out, index=False)
    print(f"--> Saved: {out}")
    return df


def print_bayes_factors(rows):
    print("\n  ---- pairwise Bayes factors (headline: I<50 low-insolation super-Earth desert, M>2) ----")
    desert = next((r for r in rows if r["label"].startswith("I < 50")), None)
    if desert is None:
        return
    logl = desert["logl"]
    for a, la, _ in UNIVERSES:
        for b, lb, _ in UNIVERSES:
            if a < b:
                d = (logl[a] - logl[b]) / np.log(10)
                print(f"    {la} vs {lb}: BF = {np.exp(logl[a] - logl[b]):.2g}  (log10 = {d:+.2f})")


# -------------------------------------------------------------------------- figures
def _nasa_overlay(ax, nasa, lo, hi, norm, mass_min=None):
    """NASA planets in insolation bin: grey error bars + points colored by log insolation.
    mass_min, if given, applies the same super-Earth cut used by the panel it is drawn on."""
    nsel = nasa["insolation"].between(lo, hi, inclusive="left")
    if mass_min:
        nsel &= nasa["mass"] > mass_min
    if nsel.sum() == 0:
        return None
    ax.errorbar(nasa.loc[nsel, "mass"], nasa.loc[nsel, "radius"],
                xerr=[nasa.loc[nsel, "mass_error_minus"],
                      nasa.loc[nsel, "mass_error_plus"]],
                yerr=[nasa.loc[nsel, "radius_error_minus"],
                      nasa.loc[nsel, "radius_error_plus"]],
                fmt="none", ecolor="0.8", elinewidth=0.7, capsize=0, zorder=4)
    return ax.scatter(nasa.loc[nsel, "mass"], nasa.loc[nsel, "radius"],
                      c=np.log10(nasa.loc[nsel, "insolation"]),
                      cmap="plasma", norm=norm, s=34, edgecolor="k", lw=0.5, zorder=5)


def _field_image(ax, field, vmax, levels):
    """Figure-3-style filled background: smooth viridis image + white contour lines. `field` is
    a rate/intensity on the (M,R) grid; nan cells (none at 10M) render transparent."""
    ax.imshow(field.T, origin="lower",
              extent=[BOX["m_lo"], BOX["m_hi"], BOX["r_lo"], BOX["r_hi"]],
              aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="bilinear", zorder=0)
    sm = gaussian_filter(np.nan_to_num(field, nan=0.0), 0.8)
    X, Y = np.meshgrid(M_CENT, R_CENT)
    ax.contour(X, Y, sm.T, levels=levels, colors="white", linewidths=0.7, alpha=0.75, zorder=2)


def fig_likelihood_maps(univ, nasa):
    """1x3 by insolation: the detectability field l_b(M,R) (detected fraction among transiting,
    flat_nonphysical universe, ALL masses) in Figure-3 format — filled viridis + white contours,
    NASA planets (mass > MASS_MIN only) colored by log insolation with error bars."""
    Ds = [detection_fraction_map(
        univ["flat_nonphysical"], lo, hi, M_EDGES, R_EDGES, min_count=MIN_CELL
    )[0] for _, lo, hi in INSOLATION_BINS]
    vmax = max((np.nanmax(D) for D in Ds if np.isfinite(D).any()), default=1.0)
    levels = np.round(np.linspace(0.2, vmax, 4), 2)
    norm = Normalize(vmin=np.log10(BOX["f_lo"]), vmax=np.log10(BOX["f_hi"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True, constrained_layout=True)
    sc = None
    for ax, (lbl, lo, hi), D in zip(axes, INSOLATION_BINS, Ds):
        _field_image(ax, D, vmax, levels)
        s = _nasa_overlay(ax, nasa, lo, hi, norm, mass_min=MASS_MIN)
        if s is not None:
            sc = s
        ax.set_xlim(BOX["m_lo"], BOX["m_hi"]); ax.set_ylim(BOX["r_lo"], BOX["r_hi"])
        ax.set_xlabel(r"planet mass [$M_\oplus$]")
        ax.set_title(f"{lbl} $I_\\oplus$", fontsize=12)
    axes[0].set_ylabel(r"planet radius [$R_\oplus$]")
    cb = fig.colorbar(axes[0].images[0], ax=axes, location="right", shrink=0.9)
    cb.set_label(f"detected fraction among transiting  $\\ell_b(M,R)$  ({_MISSION_LABEL[MISSION]} transit + RV)")
    if sc is not None:
        cb2 = fig.colorbar(sc, ax=axes, location="bottom", shrink=0.45, pad=0.02, aspect=45)
        cb2.set_label(r"NASA planets: log(Insolation Flux [$I_\oplus$])")
    fig.suptitle(f"Detectability (likelihood) on the MR plane — {_MISSION_LABEL[MISSION]} transit + RV, "
                 "transiting only\n"
                 "viridis = detected fraction among transiting (flat_nonphysical); "
                 "white contours",
                 fontsize=13)
    out = os.path.join(ANALYSIS_DIR, _OUT_NAME[MISSION], "likelihood_detection_maps.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def fig_posterior_predictive(univ, nasa, m_sil, r_sil):
    """3x3 (universe x insolation) in Figure-3 format: filled viridis predicted-detected
    (transiting, mass > MASS_MIN) density per panel (self-normalized, so every cell carries a
    color), white contours, NASA planets colored by log insolation with error bars, silicate
    line. Linear mass axis; no red highlight (per request)."""
    ms = np.linspace(BOX["m_lo"], BOX["m_hi"], 250)
    r_sil_line = radius_on_curve(ms, m_sil, r_sil)
    norm = Normalize(vmin=np.log10(BOX["f_lo"]), vmax=np.log10(BOX["f_hi"]))
    X, Y = np.meshgrid(M_CENT, R_CENT)
    fig, axes = plt.subplots(len(UNIVERSES), 3, figsize=(17, 9.5), sharex=True, sharey=True,
                             constrained_layout=True)
    sc = None
    for i, (key, plabel, _) in enumerate(UNIVERSES):
        u = univ[key]
        for j, (blabel, lo, hi) in enumerate(INSOLATION_BINS):
            ax = axes[i, j]
            sel = (u["joint_detected"]
                   & u["insolation"].between(lo, hi, inclusive="left")
                   & (u["mass"] > MASS_MIN))
            H, _, _ = np.histogram2d(u["mass"][sel], u["radius"][sel], bins=[M_EDGES, R_EDGES])
            # log stretch: the flat_nonphysical track dams a density spike at the M=12 box wall that a
            # linear+clip scale saturates into a flat slab; log1p renders it as a smooth gradient.
            Hs = gaussian_filter(H, 1.0)
            L = np.log1p(Hs)
            field = L / L.max() if L.max() > 0 else L
            ax.imshow(field.T, origin="lower",
                      extent=[BOX["m_lo"], BOX["m_hi"], BOX["r_lo"], BOX["r_hi"]],
                      aspect="auto", cmap="viridis", vmin=0, vmax=1,
                      interpolation="bilinear", zorder=0)
            ax.contour(X, Y, field.T, levels=[0.25, 0.5, 0.75], colors="white",
                       linewidths=0.7, alpha=0.7, zorder=2)
            ax.plot(ms, r_sil_line, "k--", lw=1.3, zorder=3)
            s = _nasa_overlay(ax, nasa, lo, hi, norm)
            if s is not None:
                sc = s
            ax.set_xlim(BOX["m_lo"], BOX["m_hi"]); ax.set_ylim(BOX["r_lo"], BOX["r_hi"])
            if i == 0:
                ax.set_title(f"{blabel} $I_\\oplus$", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{plabel}\n" + r"radius [$R_\oplus$]", fontsize=11)
            if i == len(UNIVERSES) - 1:
                ax.set_xlabel(r"planet mass [$M_\oplus$]")
    cb = fig.colorbar(axes[0, 0].images[0], ax=axes, location="right", shrink=0.85)
    cb.set_label("predicted-detected density (per-panel normalized)")
    if sc is not None:
        cb2 = fig.colorbar(sc, ax=axes, location="bottom", shrink=0.4, pad=0.02, aspect=50)
        cb2.set_label(r"NASA planets: log(Insolation Flux [$I_\oplus$])")
    fig.suptitle(f"Posterior-predictive detected density ({_MISSION_LABEL[MISSION]} transit + RV, viridis, "
                 f"transiting only, $M>{MASS_MIN:.0f}\\,M_\\oplus$) vs confirmed NASA planets "
                 "(points colored by log insolation, with error bars)\n"
                 "rows = universes (priors); columns = insolation panels; "
                 "black dashed = silicate line", fontsize=13)
    out = os.path.join(ANALYSIS_DIR, _OUT_NAME[MISSION], "posterior_predictive_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def fig_model_odds(rows):
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for i, (key, plabel, color) in enumerate(UNIVERSES):
        ax.bar(x + (i - (len(UNIVERSES) - 1) / 2) * w, [r["post"][key] for r in rows], w,
               color=color, label=plabel)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], fontsize=8.5)
    ax.set_ylabel(r"likelihood odds  $O = L / \sum_j L_j$")
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / len(UNIVERSES), color="0.6", ls=":", lw=1,
               label=f"even split (1/{len(UNIVERSES)})")
    ax.set_title(f"Which universe does NASA prefer? ({_MISSION_LABEL[MISSION]} transit + RV; normalized "
                 "binomial composition likelihood)\n"
                 "headline = I<50 low-insolation super-Earth desert (M>2); transiting planets only", fontsize=11)
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(ANALYSIS_DIR, _OUT_NAME[MISSION], "model_odds.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def fig_model_stats(rows):
    """Companion to the odds bars: (top) each universe's predicted volatile fraction f_k with NASA's
    v/n marked; (bottom) the binomial likelihoods L_k on a log axis, showing how a small f gap becomes
    orders of magnitude in L."""
    esc, prim = "escape_only", "rocky_formation"
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.26
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.0), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    for i, (key, plabel, color) in enumerate(UNIVERSES):
        xb = x + (i - (len(UNIVERSES) - 1) / 2) * w
        vals = [r["p"][key] for r in rows]
        ax1.bar(xb, vals, w, color=color, label=f"$f$  {plabel}")
        for xi, v in zip(xb, vals):
            ax1.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 2),
                         ha="center", fontsize=8, color=color)
    ax1.plot(x, [r["k"] / r["n"] for r in rows], "kD", ms=8, zorder=5, label=r"observed $v/n$")
    for xi, r in zip(x, rows):
        vn = r["k"] / r["n"]
        ytop = max([vn] + [r["p"][key] for key, _, _ in UNIVERSES]) + 0.09
        ax1.annotate(f"{r['k']}/{r['n']} = {vn:.2f}", (xi, ytop),
                     ha="center", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax1.set_ylabel("volatile fraction")
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(x)
    ax1.grid(alpha=0.2, axis="y"); ax1.legend(fontsize=9, ncol=3)
    ax1.set_title(r"Predicted volatile fractions $f_k$ vs observed $v/n$ "
                  "(the two universes sit close)", fontsize=11)
    for i, (key, plabel, color) in enumerate(UNIVERSES):
        xb = x + (i - (len(UNIVERSES) - 1) / 2) * w
        vals = [max(r["L"][key], 1e-300) for r in rows]
        ax2.bar(xb, vals, w, color=color, label=f"$L$  {plabel}")
        for xi, v in zip(xb, vals):
            ax2.annotate(f"{v:.1e}", (xi, v), textcoords="offset points", xytext=(0, 2),
                         ha="center", fontsize=8, color=color)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"binomial likelihood $L_k$")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(" (M>2)", "\n(M>2)") for l in labels])
    ax2.grid(alpha=0.2, axis="y"); ax2.legend(fontsize=9)
    ax2.set_title(r"Binomial likelihoods $L_k$ (log axis): the small $f$ gap becomes orders "
                  "of magnitude", fontsize=11)
    fig.suptitle(f"Why the odds saturate — {_MISSION_LABEL[MISSION]} transit + RV, precision-cut sample",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(ANALYSIS_DIR, _OUT_NAME[MISSION], "model_stats.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def main(mission="kepler"):
    global MISSION
    MISSION = mission
    os.makedirs(os.path.join(ANALYSIS_DIR, _OUT_NAME[MISSION]), exist_ok=True)
    m_sil, r_sil = load_mass_radius_curve()
    rng = np.random.default_rng(RNG_SEED)

    print(f"=== Low-insolation super-Earth desert Bayesian comparison — transit mission: {_MISSION_LABEL[MISSION]} ===")
    univ = build_universes()

    # Table-6 reconciliation targets are Kepler-derived; only meaningful for the Kepler run.
    if MISSION == "kepler":
        print("\n--> reconciliation vs paper Table 6 (Flat All / Flat Vol detected f_vol):")
        for lbl, lo, hi, mcut, ref in [("all bins, no cut", BOX["f_lo"], BOX["f_hi"], None,
                                        "All 0.43 / Vol 0.70"),
                                       ("low-insolation I<50, M>2", BOX["f_lo"], LOW_INSOLATION_MAX, MASS_MIN,
                                        "All 0.44 / Vol 0.74")]:
            pa = predicted_volatile_fraction(
                univ["rocky_formation"], lo, hi, m_sil, r_sil, rng,
                mass_min=mcut, repeats=N_FRAC_REP,
                error=SIMULATED_MEASUREMENT_ERROR,
            )[0]
            pv = predicted_volatile_fraction(
                univ["escape_only"], lo, hi, m_sil, r_sil, rng,
                mass_min=mcut, repeats=N_FRAC_REP,
                error=SIMULATED_MEASUREMENT_ERROR,
            )[0]
            print(f"    {lbl:<20} rocky-formation f_vol={pa:.2f}  escape-only f_vol={pv:.2f}   "
                  f"[paper Table 6: {ref}]")

    rows_prec, nasa_prec = None, None
    for precision, tag in [(True, f"{_MISSION_LABEL[MISSION]} — precision-cut NASA (primary)"),
                           (False, f"{_MISSION_LABEL[MISSION]} — full measured-mass NASA (sensitivity)")]:
        nasa = load_measured_planets(
            NASA_FILE,
            mass_bounds=(BOX["m_lo"], BOX["m_hi"]),
            radius_bounds=(BOX["r_lo"], BOX["r_hi"]),
            insolation_bounds=(BOX["f_lo"], BOX["f_hi"]),
            max_relative_error=NASA_MEASUREMENT_ERROR if precision else None,
            missing_relative_error=SIMULATED_MEASUREMENT_ERROR,
        )
        print(f"\n--> {tag}: N={len(nasa)} in box")
        rows = desert_table(univ, nasa, m_sil, r_sil, rng, tag)
        print_bayes_factors(rows)
        if precision:
            rows_prec, nasa_prec = rows, nasa

    fig_likelihood_maps(univ, nasa_prec)
    fig_posterior_predictive(univ, nasa_prec, m_sil, r_sil)
    fig_model_odds(rows_prec)
    fig_model_stats(rows_prec)
    save_stats_table(rows_prec, "precision-cut (primary)")

    print("\n  CAVEATS:")
    print("  - Figures consider TRANSITING planets only (detector geometric transit flag); the")
    print("    likelihood map is the detected fraction among transiting (paper fig:flat).")
    print("  - Blanks were the MIN_CELL quality mask, not empty cells: low-insolation planets transit ~2%,")
    print(f"    so the I<10 transiting subsample was thin. N={FLAT_N_POOL:,} fills every MR cell.")
    print("    Posterior-predictive dark regions for the two track universes are PHYSICAL (the MR")
    print("    relation is a curve); only flat_nonphysical's fill needed more simulations.")
    print("  - Normalization-free: the binomial conditions on n_b, so absolute occurrence and")
    print("    survey effort cancel. No absolute-rate prior is used.")
    print("  - Point-estimate binomial => overconfident; beta-binomial + NASA error propagation")
    print("    is the honest next step (I<10 especially is data-poor).")
    print("  - escape_only removes TRUE rocky M>2; measurement noise keeps p<1 so one confirmed")
    print("    low-insolation rocky planet (LHS 1140 b) does not send its likelihood to 0.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
