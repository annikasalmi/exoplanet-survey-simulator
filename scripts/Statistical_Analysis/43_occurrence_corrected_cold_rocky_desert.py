"""
43_occurrence_corrected_cold_rocky_desert.py — occurrence-corrected companion to
41_bayesian_cold_rocky_desert.py, in the SAME (M,R) 3x3 universe x insolation format.

Question (same as script 88): is the cold rocky desert corner empty because planets are
not there (physical) or because we cannot see them (selection)? Script 88 answers it with a
composition-ratio Bayes comparison. This script answers the map-level version directly by
UNDOING the detection selection with an inverse-completeness correction.

Method (conditional completeness, per the methodology decision):
    completeness  C_b(M,R) = detected / TRANSITING  per cell  (= script 88's likelihood d_b)
    corrected     N_true(M,R) ~ N_detected(M,R) / C_b(M,R)
    -> this recovers the INTRINSIC TRANSITING density on the MR plane, WITHIN each insolation
       panel. It removes detection incompleteness (SNR, baseline, #transits) but NOT the
       geometric transit probability. That is fine here: transit probability depends on a/R*,
       set by insolation and star, ~independent of planet M,R at fixed insolation, so the
       geometric factor cancels in the MR SHAPE within a panel. It does NOT make cold vs hot
       panels comparable in absolute normalization (that would need detected/GENERATED).
    target        the MODEL's detected map (rigorous, self-consistent): dividing the model's
       detected counts by the model's own completeness must recover the model's transiting
       density. NASA planets are OVERLAID as points only (NOT corrected — their mixed
       Kepler/TESS/RV selection function has no single valid completeness field).

Read the cold rocky desert corner (red ellipse, I<50, M>2) AFTER correction:
    still dark -> the transiting population genuinely lacks large cold rocky planets (physical).
    fills in   -> the emptiness was detection selection.
Contrast rocky_formation (keeps born-rocky super-Earths) vs escape_only (removes them): the
difference in the corrected cold corner is the signal script 88 turns into model odds.

Shares 88's box, grids, universes, silicate line, NASA loader, and cached 10M pools (loaded,
not regenerated). Only the transit mission (Kepler default; pass mission="tess") changes the
detector, exactly as in 88 / 88b.

Run:
    python scripts/Statistical_Analysis/43_occurrence_corrected_cold_rocky_desert.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Reuse script 88's machinery (universes, pools, silicate, NASA overlay, grids) verbatim.
_spec = importlib.util.spec_from_file_location(
    "bayes_cold_rocky_desert", _HERE / "41_bayesian_cold_rocky_desert.py")
_bayes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bayes)

_OUT_NAME = {"kepler": "43_occurrence_corrected_cold_rocky_desert",
             "tess": "43b_occurrence_corrected_cold_rocky_desert_tess"}

C_FLOOR = 0.02   # completeness below this -> correction amplifies noise unboundedly => UNCONSTRAINED
                 # (mirrors ETA_MIN in scripts/Statistical_Analysis/30_occurrence_correction.py)


def _out_dir():
    return os.path.join(ROOT, "my_outputs", _OUT_NAME[_bayes.MISSION])


def _smooth_nan(field, sigma=1.0):
    """Gaussian-smooth a field with nan cells ignored (fill 0, smooth, divide by smoothed mask).
    Used to model the completeness as a smooth field rather than dividing by the exact per-cell
    ratio -- the honest inverse-completeness estimator (you never know completeness cell-exact)."""
    m = np.isfinite(field).astype(float)
    v = np.where(np.isfinite(field), field, 0.0)
    vs = gaussian_filter(v, sigma)
    ms = gaussian_filter(m, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ms > 1e-6, vs / ms, np.nan)
    return out


def corrected_maps(u, lo, hi):
    """Return (H_det, C, C_smooth, corrected, H_transit) on the 88 MR grid for one universe/bin.
        C          = detected / transiting  (conditional completeness = 88's d_b)
        corrected  = detected / C_smooth, blanked where C_smooth < C_FLOOR (unconstrained)
    corrected recovers the intrinsic TRANSITING density (H_transit) within this insolation panel.
    """
    inb = (u["flux"] >= lo) & (u["flux"] < hi)
    tr = inb & u["transit"]
    dt = inb & u["det"]                       # detected planets all transit
    edges = [_bayes.M_EDGES, _bayes.R_EDGES]
    H_tr, _, _ = np.histogram2d(u["mass"][tr], u["radius"][tr], bins=edges)
    H_det, _, _ = np.histogram2d(u["mass"][dt], u["radius"][dt], bins=edges)
    C = np.where(H_tr >= _bayes.MIN_CELL, H_det / np.maximum(H_tr, 1), np.nan)
    C_s = _smooth_nan(C, sigma=1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = np.where((H_det > 0) & (C_s >= C_FLOOR), H_det / C_s, np.nan)
    return H_det, C, C_s, corrected, H_tr


def _panel_field(corrected):
    """Per-panel max-normalized log1p stretch, matching 88's posterior_predictive rendering so
    the two figures are visually comparable. nan cells stay transparent."""
    Hs = gaussian_filter(np.nan_to_num(corrected, nan=0.0), 1.0)
    L = np.log1p(Hs)
    field = L / L.max() if L.max() > 0 else L
    field[~np.isfinite(gaussian_filter(np.isfinite(corrected).astype(float), 1.0) > 0.05)] = np.nan
    return field


def fig_corrected_maps(univ, nasa, m_sil, r_sil):
    """3x3 (universe x insolation): occurrence-corrected (transiting) density in 88's Figure-3
    style -- filled viridis + white contours, NASA overlaid (uncorrected) colored by log
    insolation, silicate line, red super-Earth cut + cold rocky desert ellipse."""
    ms = np.linspace(_bayes.BOX["m_lo"], _bayes.BOX["m_hi"], 250)
    r_sil_line = np.interp(ms, m_sil, r_sil)
    norm = Normalize(vmin=np.log10(_bayes.BOX["f_lo"]), vmax=np.log10(_bayes.BOX["f_hi"]))
    X, Y = np.meshgrid(_bayes.M_CENT, _bayes.R_CENT)
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), sharex=True, sharey=True,
                             constrained_layout=True)
    sc = None
    for i, (key, plabel, _) in enumerate(_bayes.UNIVERSES):
        u = univ[key]
        for j, (blabel, lo, hi) in enumerate(_bayes.INSOL_BINS):
            ax = axes[i, j]
            corrected = corrected_maps(u, lo, hi)[3]
            field = _panel_field(corrected)
            ax.imshow(np.where(np.isfinite(field), field, 0.0).T, origin="lower",
                      extent=[_bayes.BOX["m_lo"], _bayes.BOX["m_hi"],
                              _bayes.BOX["r_lo"], _bayes.BOX["r_hi"]],
                      aspect="auto", cmap="viridis", vmin=0, vmax=1,
                      interpolation="bilinear", zorder=0)
            ax.contour(X, Y, np.nan_to_num(field, nan=0.0).T, levels=[0.25, 0.5, 0.75],
                       colors="white", linewidths=0.7, alpha=0.7, zorder=2)
            ax.plot(ms, r_sil_line, "k--", lw=1.3, zorder=3)
            ax.axvline(_bayes.MASS_MIN, color="red", ls="--", lw=1.0, alpha=0.7, zorder=3)
            if hi <= _bayes.COLD_MAX:
                _bayes._desert_ellipse(ax)
            s = _bayes._nasa_overlay(ax, nasa, lo, hi, norm)
            if s is not None:
                sc = s
            ax.set_xlim(_bayes.BOX["m_lo"], _bayes.BOX["m_hi"])
            ax.set_ylim(_bayes.BOX["r_lo"], _bayes.BOX["r_hi"])
            if i == 0:
                ax.set_title(f"{blabel} $I_\\oplus$", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{plabel}\n" + r"radius [$R_\oplus$]", fontsize=11)
            if i == 2:
                ax.set_xlabel(r"planet mass [$M_\oplus$]")
    cb = fig.colorbar(axes[0, 0].images[0], ax=axes, location="right", shrink=0.85)
    cb.set_label("occurrence-corrected transiting density (per-panel normalized)")
    if sc is not None:
        cb2 = fig.colorbar(sc, ax=axes, location="bottom", shrink=0.4, pad=0.02, aspect=50)
        cb2.set_label(r"NASA planets (uncorrected overlay): log(Insolation Flux [$I_\oplus$])")
    fig.suptitle(
        f"Occurrence-corrected density on the MR plane ({_bayes._mlabel()} transit + RV) -- "
        "detected / (detected-among-transiting completeness)\n"
        "recovers the intrinsic TRANSITING density per panel; rows = universes, columns = "
        "insolation; black dashed = silicate line, red = M=2 cut + cold rocky desert",
        fontsize=13)
    out = os.path.join(_out_dir(), "occurrence_corrected_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def fig_completeness_maps(univ):
    """3x3 (universe x insolation): the conditional completeness C_b(M,R)=detected/transiting
    that was divided out. Dark = large correction (cold, near/above the curve) = where the
    'occurrence-corrected' claim leans hardest on the model."""
    X, Y = np.meshgrid(_bayes.M_CENT, _bayes.R_CENT)
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), sharex=True, sharey=True,
                             constrained_layout=True)
    vmax = 0.0
    Cs = {}
    for key, _, _ in _bayes.UNIVERSES:
        for blabel, lo, hi in _bayes.INSOL_BINS:
            C = corrected_maps(univ[key], lo, hi)[1]
            Cs[(key, blabel)] = C
            if np.isfinite(C).any():
                vmax = max(vmax, np.nanmax(C))
    vmax = vmax or 1.0
    for i, (key, plabel, _) in enumerate(_bayes.UNIVERSES):
        for j, (blabel, lo, hi) in enumerate(_bayes.INSOL_BINS):
            ax = axes[i, j]
            C = Cs[(key, blabel)]
            ax.imshow(np.nan_to_num(C, nan=0.0).T, origin="lower",
                      extent=[_bayes.BOX["m_lo"], _bayes.BOX["m_hi"],
                              _bayes.BOX["r_lo"], _bayes.BOX["r_hi"]],
                      aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                      interpolation="bilinear", zorder=0)
            ax.contour(X, Y, gaussian_filter(np.nan_to_num(C, nan=0.0), 0.8).T,
                       levels=np.round(np.linspace(0.2 * vmax, vmax, 4), 2),
                       colors="white", linewidths=0.7, alpha=0.7, zorder=2)
            ax.axvline(_bayes.MASS_MIN, color="red", ls="--", lw=1.0, alpha=0.7, zorder=3)
            if hi <= _bayes.COLD_MAX:
                _bayes._desert_ellipse(ax)
            ax.set_xlim(_bayes.BOX["m_lo"], _bayes.BOX["m_hi"])
            ax.set_ylim(_bayes.BOX["r_lo"], _bayes.BOX["r_hi"])
            if i == 0:
                ax.set_title(f"{blabel} $I_\\oplus$", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{plabel}\n" + r"radius [$R_\oplus$]", fontsize=11)
            if i == 2:
                ax.set_xlabel(r"planet mass [$M_\oplus$]")
    cb = fig.colorbar(axes[0, 0].images[0], ax=axes, location="right", shrink=0.85)
    cb.set_label("conditional completeness  C(M,R) = detected / transiting")
    fig.suptitle(
        f"Completeness field divided out ({_bayes._mlabel()} transit + RV) -- detected fraction "
        "among transiting\ndark below the curve for the two track universes = physically EMPTY "
        "(no transiting planets to measure), not low completeness;\nread the Uniform row for true "
        "detectability everywhere. red = M=2 cut + cold rocky desert",
        fontsize=12)
    out = os.path.join(_out_dir(), "completeness_field_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def report(univ):
    """Console diagnostics: recovery check (corrected/transiting ~ 1) and the cold rocky desert
    corrected density, rocky_formation vs escape_only."""
    print("\n  ===== recovery check: median (corrected / transiting) per universe, I<50 =====")
    lo, hi = _bayes.BOX["f_lo"], _bayes.COLD_MAX
    for key, plabel, _ in _bayes.UNIVERSES:
        _, _, _, corr, H_tr = corrected_maps(univ[key], lo, hi)
        ok = np.isfinite(corr) & (H_tr > 0)
        ratio = np.nanmedian(corr[ok] / H_tr[ok]) if ok.any() else np.nan
        print(f"    {plabel:<16} median corrected/transiting = {ratio:.3f}  (1.0 = exact recovery)")

    # cold rocky desert corner: M>2 (mass-center cut) & near/above the silicate curve, I<50
    m_hi_cut = _bayes.M_CENT > _bayes.MASS_MIN
    print("\n  ===== cold rocky desert (I<50, M>2): corrected transiting density summary =====")
    print("  higher corrected density in the large-radius corner = more large cold planets in")
    print("  the intrinsic transiting population (physical), independent of detection selection.\n")
    for key, plabel, _ in _bayes.UNIVERSES:
        _, _, _, corr, H_tr = corrected_maps(univ[key], lo, hi)
        sub = corr[np.ix_(m_hi_cut, np.ones(len(_bayes.R_CENT), bool))]
        total = np.nansum(sub)
        # split by radius: large (R>1.4) vs small in the M>2 column
        big_r = _bayes.R_CENT > 1.4
        big = np.nansum(corr[np.ix_(m_hi_cut, big_r)])
        small = np.nansum(corr[np.ix_(m_hi_cut, ~big_r)])
        frac_big = big / total if total > 0 else np.nan
        print(f"    {plabel:<16} M>2 corrected total={total:>10.0f}   large-R(>1.4) fraction={frac_big:.3f}")

    print("\n  CAVEATS:")
    print("  - Conditional completeness (detected/transiting) recovers the MR SHAPE within a")
    print("    panel only; it does NOT correct transit geometry, so cold vs hot ABSOLUTE rates")
    print("    are not comparable. For that, completeness must be detected/GENERATED.")
    print("  - NASA points are an UNCORRECTED overlay: NASA's mixed survey selection has no")
    print("    single valid completeness field. The corrected field is the MODEL's, not NASA's.")
    print(f"  - Cells with completeness < {C_FLOOR:g} are blanked (correction is unconstrained).")


def main(mission="kepler"):
    _bayes.MISSION = mission
    os.makedirs(_out_dir(), exist_ok=True)
    m_sil, r_sil = _bayes.load_silicate()

    print(f"=== Occurrence-corrected cold rocky desert (conditional completeness) — "
          f"transit mission: {_bayes._mlabel()} ===")
    univ = _bayes.build_universes(m_sil, r_sil)
    nasa = _bayes.load_nasa(precision=True)
    print(f"--> NASA precision-cut overlay: N={nasa['n']} in box")

    fig_corrected_maps(univ, nasa, m_sil, r_sil)
    fig_completeness_maps(univ)
    report(univ)


if __name__ == "__main__":
    main()
