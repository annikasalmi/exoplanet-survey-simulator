"""
34_cold_window_power.py

Forward-model the joint (transit x RV) completeness and test the cold red window
ONLY where the model actually has statistical power.

THE LOGIC
---------
A confirmed *rocky* NASA planet must (a) transit, (b) have its transit detected,
and (c) have an RV-measurable mass.  The stacked rocky P-Pop, run through the
transit model AND the RV model (script 53), already folds occurrence x geometry
x transit-detection x RV-measurability into a single prediction of WHERE
confirmed rocky planets should land:

    M_pred(cell) = # P-Pop rocky planets in the cell that pass transit AND RV.

This predicts the *shape* of the confirmed sample up to one unknown scale A that
maps "P-Pop universe" to "the real sky" (different star samples, distances,
survey footprints).  We CALIBRATE A in a high-completeness CONTROL region where
the model is trustworthy and NASA is well sampled:

    A_hat = sum_ctrl N_obs / sum_ctrl M_pred           (per star type)

then PREDICT the expected confirmed count in the cold red window:

    mu_redwin = A_hat * sum_redwin M_pred
    obs_redwin = # confirmed rocky NASA planets in the red window
    p_deficit = Poisson.cdf(obs_redwin, mu_redwin)     (one-sided: fewer than expected)

THE POINT ABOUT POWER
---------------------
Where joint completeness -> 0, M_pred -> 0, so mu_redwin -> 0 and observing zero
is unsurprising under BOTH "physical limit" and "pure bias": the test has NO
power and must be reported as inconclusive, not as evidence.  mu_redwin IS the
power.  Only when mu_redwin is appreciable (>= --power-floor) and obs is much
smaller is there a real, physical deficit beyond selection.  For M dwarfs the
red-window completeness is partial (not zero), so that panel is where the test
can actually discriminate.

CAVEATS (printed in the summary too)
  * Some confirmed M-dwarf rocky masses (e.g. TRAPPIST-1) come from TTV, not RV;
    those inflate obs vs an RV-only prediction.  Use --rv-only to drop the most
    likely TTV-mass hosts as a sensitivity check.
  * The transit-detection model is TESS-based; --transit geometric replaces it
    with perfect transit detection (isolates the RV bottleneck, upper bound on
    completeness).
  * A is assumed spatially constant; that is the null we are testing departures
    from.  Two control definitions are reported for robustness.

Run from repo root:
    python scripts/34_cold_window_power.py
    python scripts/34_cold_window_power.py --transit geometric --rv-only
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.stats import poisson

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.paths import LIFESIM_OUTER_DIR
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S44 = _load_module("s44", ROOT / "plot" / "script plots" / "56_kepler_tess_rocky_fgkm_gaia60pc.py")
S53 = _load_module("s53", ROOT / "plot" / "script plots" / "58_rv_rocky_fgkm_test.py")

OUT_DIR = ROOT / "output/plots" / "34_cold_window_power"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAR_ORDER         = S44.STAR_ORDER
INSOLATION_BINS    = S44.INSOLATION_BINS
PLANET_RADIUS_BINS = S44.PLANET_RADIUS_BINS
INSOLATION_LIMITS  = S44.INSOLATION_LIMITS
RADIUS_LIMITS      = S44.RADIUS_LIMITS
COLD_INSOLATION    = S44.COLD_INSOLATION


def _centers(edges):
    return np.sqrt(edges[:-1] * edges[1:])


FLUX_C = _centers(INSOLATION_BINS)      # (nx,)
RAD_C  = _centers(PLANET_RADIUS_BINS)   # (ny,)


# ── TTV heuristic (drop likely TTV-mass hosts for the --rv-only check) ──────────

# Hosts whose small-planet masses in the archive come predominantly from transit
# timing rather than RV.  Conservative, name-based; only used under --rv-only.
TTV_HOST_PATTERNS = [
    r"TRAPPIST-1", r"Kepler-", r"TOI-178", r"K2-138", r"L 98-59",
]


def drop_ttv_mass_hosts(rocky: pd.DataFrame) -> pd.DataFrame:
    host = rocky.get("host_name", pd.Series("", index=rocky.index)).fillna("").astype(str)
    name = rocky.get("planet_label", pd.Series("", index=rocky.index)).fillna("").astype(str)
    mask = pd.Series(False, index=rocky.index)
    for pat in TTV_HOST_PATTERNS:
        mask |= host.str.contains(pat, case=False, regex=True)
        mask |= name.str.contains(pat, case=False, regex=True)
    print(f"  --rv-only: dropping {int(mask.sum())} likely-TTV-mass planets "
          f"({len(rocky)} -> {int((~mask).sum())})")
    return rocky[~mask].copy()


# ── grids ───────────────────────────────────────────────────────────────────────

def hist2d(df: pd.DataFrame, mask=None) -> np.ndarray:
    x = pd.to_numeric(df["flux_p"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["radius_p"], errors="coerce").to_numpy()
    if mask is not None:
        x, y = x[mask], y[mask]
    h, _, _ = np.histogram2d(x, y, bins=[INSOLATION_BINS, PLANET_RADIUS_BINS])
    return h  # (nx, ny)


def redwindow_cell_mask(fit) -> np.ndarray:
    """Boolean (nx, ny): cell center is cold (I<10) and above the 90% line."""
    if fit is None:
        return np.zeros((len(FLUX_C), len(RAD_C)), dtype=bool)
    slope, intercept = fit
    FX, RY = np.meshgrid(FLUX_C, RAD_C, indexing="ij")
    return (FX < COLD_INSOLATION) & (RY > S44.line_radius(slope, intercept, FX))


# ── per-star-type forward model ─────────────────────────────────────────────────

def analyze_startype(p_st: pd.DataFrame, nasa_st: pd.DataFrame, fit,
                     transit_mode: str, c_ctrl: float, min_pred: float,
                     hot_thresh: float):
    """Return a dict of grids + the red-window power-test numbers."""
    denom = p_st["denominator_case"].to_numpy()
    transit_pass = denom if transit_mode == "geometric" else p_st["detected_case"].to_numpy()
    joint = transit_pass & p_st["rv_ok"].to_numpy()

    M_trans = hist2d(p_st, denom)              # underlying transiting population
    M_pred  = hist2d(p_st, joint)              # predicted confirmed (occ x geom x det x RV)
    N_obs   = hist2d(nasa_st)                  # real confirmed rocky NASA

    with np.errstate(invalid="ignore", divide="ignore"):
        C_joint = np.divide(M_pred, M_trans, out=np.full_like(M_pred, np.nan),
                            where=M_trans > 0)

    red = redwindow_cell_mask(fit)

    # --- control region A: high completeness cells (model trustworthy) ---
    ctrl_hi = (C_joint >= c_ctrl) & (M_pred >= min_pred)
    # --- control region B: hot cells (sensitivity) ---
    FX, _ = np.meshgrid(FLUX_C, RAD_C, indexing="ij")
    ctrl_hot = (FX > hot_thresh) & (M_pred >= min_pred)

    def calibrate(ctrl):
        denom_pred = M_pred[ctrl].sum()
        obs_ctrl = N_obs[ctrl].sum()
        if denom_pred <= 0 or obs_ctrl <= 0:
            return np.nan, obs_ctrl, denom_pred
        return obs_ctrl / denom_pred, obs_ctrl, denom_pred

    A_hi, obs_hi, predsum_hi = calibrate(ctrl_hi)
    A_hot, obs_hot, predsum_hot = calibrate(ctrl_hot)

    pred_red = M_pred[red].sum()
    obs_red = int(N_obs[red].sum())

    def power_test(A):
        if not np.isfinite(A):
            return np.nan, np.nan
        mu = A * pred_red
        p = poisson.cdf(obs_red, mu) if mu > 0 else np.nan
        return mu, p

    mu_hi, p_hi = power_test(A_hi)
    mu_hot, p_hot = power_test(A_hot)

    # mean joint completeness in the red window (transiting-weighted)
    red_trans = M_trans[red].sum()
    red_pred = M_pred[red].sum()
    C_red = (red_pred / red_trans) if red_trans > 0 else np.nan

    return dict(
        M_trans=M_trans, M_pred=M_pred, N_obs=N_obs, C_joint=C_joint, red=red,
        ctrl_hi=ctrl_hi, ctrl_hot=ctrl_hot,
        A_hi=A_hi, obs_ctrl_hi=obs_hi, predsum_hi=predsum_hi,
        A_hot=A_hot, obs_ctrl_hot=obs_hot, predsum_hot=predsum_hot,
        pred_red=pred_red, obs_red=obs_red, C_red=C_red,
        mu_hi=mu_hi, p_hi=p_hi, mu_hot=mu_hot, p_hot=p_hot,
        n_trans_total=int(M_trans.sum()),
    )


# ── figure ───────────────────────────────────────────────────────────────────────

def plot_forward(results: dict, rocky_win: pd.DataFrame, lines: dict,
                 power_floor: float, args) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.2), sharex=True, sharey=True,
                             constrained_layout=True)
    mesh = None
    for j, st in enumerate(STAR_ORDER):
        ax = axes[j]
        R = results[st]
        A = R["A_hi"]
        expected = (A * R["M_pred"]) if np.isfinite(A) else R["M_pred"] * np.nan
        grid = np.ma.masked_invalid(np.where(expected > 0, expected, np.nan)).T

        if np.isfinite(expected).any() and (expected > 0).any():
            mesh = ax.pcolormesh(INSOLATION_BINS, PLANET_RADIUS_BINS, grid,
                                 shading="auto", cmap="viridis",
                                 norm=LogNorm(vmin=max(expected[expected > 0].min(), 1e-3),
                                              vmax=max(expected.max(), 1e-2)))
        else:
            ax.text(0.5, 0.5, "no calibration\n(no power)", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

        # NASA overlay (warm, outlined) reusing script 53 styling.
        sub = rocky_win[rocky_win["stype_clean"] == st]
        S53._overlay_rocky_outlined(ax, sub, {}, [])

        # 90% line + red-window box.
        fit = lines[st]
        if fit is not None:
            slope, intercept = fit
            x = np.logspace(np.log10(INSOLATION_LIMITS[0]), np.log10(INSOLATION_LIMITS[1]), 200)
            ax.plot(x, S44.line_radius(slope, intercept, x), "k--", lw=1.8, zorder=5)
            xr = np.logspace(np.log10(INSOLATION_LIMITS[0]), np.log10(COLD_INSOLATION), 100)
            ax.fill_between(xr, S44.line_radius(slope, intercept, xr), RADIUS_LIMITS[1],
                            color="red", alpha=0.10, zorder=1.5, lw=0)

        # annotate the power test (high-completeness calibration).
        mu, p, C_red = R["mu_hi"], R["p_hi"], R["C_red"]
        if np.isfinite(mu):
            powered = mu >= power_floor
            if not powered:
                verdict = "NO POWER (mu<%.0f)" % power_floor
                vc = "0.35"
            elif p < 0.05:
                verdict = "DEFICIT (physical-leaning)"
                vc = "darkred"
            else:
                verdict = "consistent w/ bias"
                vc = "darkgreen"
            txt = (f"red-window completeness ~ {C_red:.0%}\n"
                   f"expected mu = {mu:.1f}   observed = {R['obs_red']}\n"
                   f"Poisson p(deficit) = {p:.3f}\n{verdict}")
        else:
            txt = "no control calibration\n(insufficient NASA / P-Pop)"
            vc = "0.35"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.6, color=vc,
                bbox=dict(boxstyle="round", fc="white", ec=vc, alpha=0.9), zorder=9)

        S44._setup_axis(ax, f"{st} stars   (transiting P-Pop N={R['n_trans_total']:,}, "
                            f"NASA rocky={len(sub)})",
                        show_xlabel=True, show_ylabel=(j == 0))

    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(),
                     label="expected confirmed rocky planets per cell (A_hat x M_pred)",
                     shrink=0.7, pad=0.01)
    fig.suptitle(
        "Forward model: expected confirmed rocky planets (occurrence x transit x RV, "
        f"calibrated to NASA in high-completeness cells)  —  transit={args.transit}, "
        f"RV={args.instrument} N={args.n_obs}\n"
        "Red box = cold window.  mu = expected count there under uniform occurrence; "
        "obs = real count.  mu<power-floor => the data cannot tell physical limit from "
        "selection bias (no power).",
        fontsize=12,
    )
    out = OUT_DIR / f"forward_power_test_{args.transit}_{args.instrument.lower()}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


# ── summary ─────────────────────────────────────────────────────────────────────

def write_summary(results, power_floor, args):
    lines = [
        "=" * 96,
        "FORWARD-MODEL POWER TEST — cold red window, rocky confirmed sample",
        "=" * 96,
        f"transit model = {args.transit}   |   RV = {args.instrument}, N={args.n_obs}, "
        f"K/sigmaK>={args.snr_threshold}, band-mag<={args.mag_target}   |   "
        f"rv_only={args.rv_only}",
        f"control A = cells with joint completeness >= {args.c_ctrl:.2f} (and M_pred >= "
        f"{args.min_pred:g});  control B = hot cells I > {args.hot_thresh:g}",
        f"power floor = expected mu >= {power_floor:g} to call the test informative",
        "",
        f"{'stype':<6}{'C_red':>7}{'pred_red':>10}{'obs_red':>8}"
        f"{'A(hi)':>9}{'mu(hi)':>9}{'p(hi)':>8}{'mu(hot)':>9}{'p(hot)':>8}{'verdict':>16}",
        "-" * 96,
    ]
    for st in STAR_ORDER:
        R = results[st]
        mu, p = R["mu_hi"], R["p_hi"]
        if not np.isfinite(mu):
            verdict = "no calibration"
        elif mu < power_floor:
            verdict = "NO POWER"
        elif p < 0.05:
            verdict = "DEFICIT"
        else:
            verdict = "bias-consistent"
        C_red = R["C_red"]
        lines.append(
            f"{st:<6}{(C_red if np.isfinite(C_red) else float('nan')):>7.0%}"
            f"{R['pred_red']:>10.1f}{R['obs_red']:>8d}"
            f"{(R['A_hi'] if np.isfinite(R['A_hi']) else float('nan')):>9.3f}"
            f"{(mu if np.isfinite(mu) else float('nan')):>9.2f}"
            f"{(p if np.isfinite(p) else float('nan')):>8.3f}"
            f"{(R['mu_hot'] if np.isfinite(R['mu_hot']) else float('nan')):>9.2f}"
            f"{(R['p_hot'] if np.isfinite(R['p_hot']) else float('nan')):>8.3f}"
            f"{verdict:>16}"
        )
    lines += [
        "-" * 96,
        "",
        "HOW TO READ",
        "  C_red    = modeled joint (transit x RV) completeness inside the cold window.",
        "  pred_red = sum of M_pred over red-window cells (stacked-universe units).",
        "  obs_red  = real confirmed rocky NASA planets in the red window.",
        "  mu       = expected confirmed count there IF cold occurrence == calibrated",
        "             occurrence (the no-atmosphere-stripping null).",
        "  p(hi/hot)= one-sided Poisson prob of seeing <= obs_red given mu, for the two",
        "             control-region calibrations.",
        "",
        "  VERDICT LOGIC",
        "    NO POWER        mu < power floor: completeness is so low that observing few/zero",
        "                    confirmed planets is expected under BOTH hypotheses. The cold",
        "                    emptiness here is consistent with pure mass-confirmation bias;",
        "                    a real radius limit cannot be established or excluded.",
        "    DEFICIT         mu is appreciable but obs is significantly smaller (p<0.05):",
        "                    fewer cold rocky planets than selection alone predicts -> a",
        "                    residual physical suppression (atmosphere-stripping-leaning).",
        "    bias-consistent mu appreciable and obs ~ mu: the cold count matches what",
        "                    selection predicts -> no extra physics needed.",
        "",
        "  CAVEATS",
        "   * TTV masses (TRAPPIST-1 etc.) are RV-independent and inflate obs; rerun with",
        "     --rv-only to see if any DEFICIT strengthens.",
        "   * transit=geometric is the most generous transit case (perfect detection);",
        "     if the verdict survives both transit settings it is robust.",
        "   * A is assumed spatially uniform; agreement between p(hi) and p(hot) supports it.",
        "   * Monte-Carlo precision of mu depends on stack depth — see script 60.",
        "=" * 96,
    ]
    text = "\n".join(lines)
    (OUT_DIR / f"summary_{args.transit}_{args.instrument.lower()}.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-catalogs", type=int, default=None)
    ap.add_argument("--instrument", default="best", help="HARPS | NIRPS | best")
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--mag-target", type=float, default=12.0)
    ap.add_argument("--transit", choices=["detected", "geometric"], default="detected",
                    help="transit-detection model: TESS detected_case, or perfect (geometric)")
    ap.add_argument("--rv-only", action="store_true",
                    help="drop likely TTV-mass hosts from the observed sample")
    ap.add_argument("--c-ctrl", type=float, default=0.5,
                    help="min joint completeness for control-region calibration")
    ap.add_argument("--min-pred", type=float, default=2.0,
                    help="min M_pred per control cell")
    ap.add_argument("--hot-thresh", type=float, default=50.0,
                    help="insolation above which cells form control B (hot)")
    ap.add_argument("--power-floor", type=float, default=3.0,
                    help="min expected mu to call the red-window test informative")
    args = ap.parse_args()

    print("=" * 70)
    print("34_cold_window_power.py")
    print("=" * 70)

    m_ref, r_ref = S44.load_rocky_reference_curve()
    shift = S44.compute_rocky_threshold_shift(m_ref, r_ref)
    _, rocky = S44.load_and_filter_nasa(m_ref, r_ref, shift)
    rocky_win = S44.restrict_to_window(rocky)
    if args.rv_only:
        rocky_win = drop_ttv_mass_hosts(rocky_win)
    lines = {st: S44.fit_90pct_line(
                rocky_win[rocky_win["stype_clean"] == st]["flux_p"].values,
                rocky_win[rocky_win["stype_clean"] == st]["radius_p"].values)
             for st in STAR_ORDER}

    ppop = S53.load_tess_ppop_for_rv(args.n_catalogs)
    ppop = S53.restrict_ppop_to_rocky(ppop, m_ref, r_ref, shift)
    ppop = S53.add_rv_ok(ppop, args.instrument, args.snr_threshold, args.n_obs, args.mag_target)

    print("\nRunning forward-model power test per star type...")
    results = {}
    for st in STAR_ORDER:
        p_st = ppop[ppop["stype_clean"] == st].copy()
        nasa_st = rocky_win[rocky_win["stype_clean"] == st].copy()
        results[st] = analyze_startype(
            p_st, nasa_st, lines[st], args.transit, args.c_ctrl, args.min_pred, args.hot_thresh)

    plot_forward(results, rocky_win, lines, args.power_floor, args)
    write_summary(results, args.power_floor, args)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
