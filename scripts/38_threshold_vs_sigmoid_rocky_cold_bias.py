"""
38_threshold_vs_sigmoid_rocky_cold_bias.py

Compares hard-threshold vs sigmoid detection efficiency, conditioned on
geometrically transiting planets.  The transit-or-not factor (~2% geometric
probability) is NOT the question here and drowns everything when left in the
denominator.  Conditioning on transiting isolates what we actually care about:

    P(detected | transiting, observed, ...) vs SNR model

Science question:
    Is the rocky-cold planet detection deficit robust to the detection model?
    Cold rocky planets (HZ, long period) have fewer transits AND lower SNR --
    both effects show up in the conditional completeness.  The sigmoid/threshold
    difference is the SNR contribution only.

Outputs (in my_outputs/38_threshold_vs_sigmoid_rocky_cold_bias/):
    fig1_conditional_completeness.png  -- heatmap conditioned on transiting
    fig2_completeness_vs_insolation.png -- cold-bias slope for rocky planets
    summary.txt

Usage:
    python scripts/38_threshold_vs_sigmoid_rocky_cold_bias.py
    python scripts/38_threshold_vs_sigmoid_rocky_cold_bias.py \\
        --ppop-csv run/tess/data/Gaia_C_F_K_combined/ppop_output.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "lifesim" / "core" / "tess_data.py").exists():
            return p
    return start.parents[1]


ROOT = find_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "my_outputs" / "38_threshold_vs_sigmoid_rocky_cold_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── TESSData import (bypass package __init__ to avoid 'git' dependency) ───────

def _load_tess_data():
    spec = importlib.util.spec_from_file_location(
        "tess_data", ROOT / "lifesim" / "core" / "tess_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TESSData

TESSData = _load_tess_data()

# ── Science bins ──────────────────────────────────────────────────────────────

# Insolation bins: cold HZ / warm HZ / warm / hot / very hot
FLUX_EDGES   = np.array([0.1, 0.5, 1.5, 5.0, 20.0, 100.0])
FLUX_LABELS  = ["0.1-0.5\n(cold HZ)", "0.5-1.5\n(warm HZ)", "1.5-5\n(warm)",
                 "5-20\n(hot)", "20-100\n(very hot)"]

# Radius bins: sub-Earth, rocky, super-rocky, sub-Neptune, Neptune
RADIUS_EDGES  = np.array([0.5, 1.0, 1.5, 2.0, 2.8, 3.5])
RADIUS_LABELS = ["0.5-1", "1-1.5", "1.5-2", "2-2.8", "2.8-3.5"]

# "Rocky" = R < 2 R_earth.  "Cold" = F < 1.5 S_earth (includes both HZ bins).
ROCKY_R_MAX = 2.0
COLD_F_MAX  = 1.5

# For the line plot: finer insolation grid
FLUX_FINE_EDGES = np.array([0.1, 0.3, 0.6, 1.0, 1.8, 3.5, 8.0, 25.0, 100.0])
FLUX_FINE_CENTERS = np.sqrt(FLUX_FINE_EDGES[:-1] * FLUX_FINE_EDGES[1:])  # geometric midpoint

# ── Constants ─────────────────────────────────────────────────────────────────
R_SUN_AU    = 0.00465047
R_EARTH_AU  = 4.26352e-5

# ── Synthetic P-Pop catalog ───────────────────────────────────────────────────

def make_synthetic_ppop(n_planets: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Synthetic M-dwarf P-Pop.  Inclinations set so ~all planets transit
    (uniform in b from 0 to 1 per planet, clipped by geometry).
    This gives maximum sample size for the conditional-completeness comparison.
    """
    rng = np.random.default_rng(seed)

    # Stellar properties (M dwarfs, Teff 2700-4200 K)
    teff_s   = rng.uniform(2700.0, 4200.0, n_planets)
    t_frac   = (teff_s - 2700.0) / (4200.0 - 2700.0)
    mass_s   = 0.10 + t_frac * 0.55          # 0.10-0.65 Msun
    radius_s = 0.14 + t_frac * 0.48          # 0.14-0.62 Rsun
    l_sun    = radius_s**2 * (teff_s / 5778.0)**4

    distance_s = rng.uniform(10.0, 120.0, n_planets)
    gaiamag    = 4.74 - 2.5 * np.log10(l_sun.clip(1e-12)) + 5.0 * np.log10(distance_s / 10.0) + 0.5

    ra  = rng.uniform(0.0, 360.0, n_planets)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n_planets)))

    # Planet properties
    radius_p    = rng.uniform(0.5, 3.5, n_planets)
    p_orb       = np.exp(rng.uniform(np.log(2.0), np.log(600.0), n_planets))
    p_yr        = p_orb / 365.25
    semimajor_p = (mass_s * p_yr**2)**(1.0 / 3.0)
    flux_p      = l_sun / semimajor_p**2

    # Force all planets to transit: sample b uniformly in [0, 1).
    # Maximum b for transit is 1 + Rp/Rs ≈ 1.01; keeping b < 1 is conservative
    # and makes all planets transit with a realistic b distribution.
    b        = rng.uniform(0.0, 0.999, n_planets)
    rs_au    = radius_s * R_SUN_AU
    cos_i    = b * rs_au / semimajor_p.clip(min=rs_au * 1.001)
    inc_p    = np.degrees(np.arccos(np.clip(cos_i, 0.0, 1.0)))

    return pd.DataFrame({
        "radius_p": radius_p, "radius_s": radius_s, "mass_s": mass_s,
        "teff_s": teff_s, "p_orb": p_orb, "semimajor_p": semimajor_p,
        "inc_p": inc_p, "flux_p": flux_p, "l_sun": l_sun,
        "gaiamag": gaiamag, "distance_s": distance_s, "ra": ra, "dec": dec,
    })

# ── Run detector ──────────────────────────────────────────────────────────────

TESS_BASE_KWARGS = dict(
    source="ppop",
    use_tesspoint=False,
    default_n_sectors=3,        # ~82 days; more coverage for cold planets
    min_transits=2,
    snr_threshold=7.1,
    phase_mode="random",
    random_seed=42,
    smooth_noise_ref_ppm_1hr=60.0,
    smooth_noise_floor_ppm_1hr=30.0,
    apply_b_to_duration=True,
    apply_mdwarf_tmag_correction=True,
    sigmoid_steepness=1.5,
    validate_for_detection=True,
)


def run_both_models(df: pd.DataFrame):
    cat_hard = TESSData(df, detection_model="threshold", **TESS_BASE_KWARGS).determine_detectable()
    cat_soft = TESSData(df, detection_model="sigmoid",   **TESS_BASE_KWARGS).determine_detectable()
    return cat_hard, cat_soft


# ── Completeness helpers ──────────────────────────────────────────────────────

def _get_arrays(cat_hard, cat_soft):
    r = pd.to_numeric(cat_hard["radius_p"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(cat_hard["flux_p"],   errors="coerce").to_numpy(float)
    transiting = cat_hard["tess_transiting_geometric"].astype(bool).to_numpy()
    det_hard   = cat_hard["tess_detected"].astype(int).to_numpy()
    p_soft     = pd.to_numeric(cat_soft["tess_p_detect"], errors="coerce").fillna(0.0).to_numpy()
    return r, f, transiting, det_hard, p_soft


def completeness_grid(cat_hard, cat_soft, r_edges, f_edges, min_n=5):
    """Completeness conditioned on transiting. Returns (hard, soft, n_transiting)."""
    r, f, transiting, det_hard, p_soft = _get_arrays(cat_hard, cat_soft)
    nr, nf = len(r_edges) - 1, len(f_edges) - 1
    comp_h = np.full((nr, nf), np.nan)
    comp_s = np.full((nr, nf), np.nan)
    n_trans = np.zeros((nr, nf), dtype=int)

    for i in range(nr):
        for j in range(nf):
            in_bin = ((r >= r_edges[i]) & (r < r_edges[i+1]) &
                      (f >= f_edges[j]) & (f < f_edges[j+1]))
            sel = in_bin & transiting
            n   = sel.sum()
            n_trans[i, j] = n
            if n >= min_n:
                comp_h[i, j] = det_hard[sel].sum() / n
                comp_s[i, j] = p_soft[sel].sum()   / n

    return comp_h, comp_s, n_trans


def completeness_vs_insolation(cat_hard, cat_soft, f_edges, r_lo, r_hi, min_n=3):
    """Completeness in insolation bins for a given radius slice, conditioned on transiting."""
    r, f, transiting, det_hard, p_soft = _get_arrays(cat_hard, cat_soft)
    r_mask = (r >= r_lo) & (r < r_hi)
    nf = len(f_edges) - 1
    comp_h = np.full(nf, np.nan)
    comp_s = np.full(nf, np.nan)
    err_h  = np.full(nf, np.nan)   # 1-sigma binomial
    n_arr  = np.zeros(nf, dtype=int)

    for j in range(nf):
        sel = r_mask & ((f >= f_edges[j]) & (f < f_edges[j+1])) & transiting
        n   = sel.sum()
        n_arr[j] = n
        if n >= min_n:
            p = det_hard[sel].sum() / n
            comp_h[j] = p
            comp_s[j] = p_soft[sel].sum() / n
            err_h[j]  = np.sqrt(p * (1 - p) / n) if n > 0 else np.nan

    return comp_h, comp_s, err_h, n_arr


# ── Figure 1: conditional completeness heatmap ───────────────────────────────

def plot_conditional_completeness(cat_hard, cat_soft, out_dir: Path) -> None:
    comp_h, comp_s, n_trans = completeness_grid(
        cat_hard, cat_soft, RADIUS_EDGES, FLUX_EDGES, min_n=5
    )
    diff = comp_s - comp_h

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    fig.suptitle(
        "Detection completeness  |  conditioned on geometrically transiting planets\n"
        "M-dwarf synthetic P-Pop  |  TESS toy detector  |  cell annotation = n_detected / n_transiting",
        fontsize=10,
    )

    ext = [0, len(FLUX_EDGES) - 1, 0, len(RADIUS_EDGES) - 1]

    for ax, grid, title, vmin, vmax, cmap in [
        (axes[0], comp_h, "Hard threshold  (sum(detected)/N_trans)", 0.0, 1.0, "viridis"),
        (axes[1], comp_s, "Sigmoid weight k=1.5  (sum(p_detect)/N_trans)", 0.0, 1.0, "viridis"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax, cmap=cmap, extent=ext)
        plt.colorbar(im, ax=ax, label="Completeness  (given transiting)")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(len(FLUX_LABELS)) + 0.5)
        ax.set_xticklabels(FLUX_LABELS, fontsize=7)
        ax.set_yticks(np.arange(len(RADIUS_LABELS)) + 0.5)
        ax.set_yticklabels(RADIUS_LABELS, fontsize=8)
        ax.set_xlabel("Insolation (S_earth)")
        ax.set_ylabel("Radius (R_earth)")

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                n = n_trans[i, j]
                if not np.isnan(grid[i, j]):
                    n_det = int(round(grid[i, j] * n))
                    ax.text(j + 0.5, i + 0.5, f"{grid[i,j]:.2f}\n({n_det}/{n})",
                            ha="center", va="center", fontsize=6, color="white")
                elif n > 0:
                    ax.text(j + 0.5, i + 0.5, f"n={n}", ha="center", va="center",
                            fontsize=6, color="gray")

    # Difference panel
    vabs = max(abs(np.nanmax(diff)), abs(np.nanmin(diff)), 0.005)
    im2 = axes[2].imshow(diff, aspect="auto", origin="lower",
                          vmin=-vabs, vmax=vabs, cmap="RdBu_r", extent=ext)
    plt.colorbar(im2, ax=axes[2], label="Sigmoid - Threshold")
    axes[2].set_title("Difference  (sigmoid - threshold)\nPositive = sigmoid gives more weight", fontsize=9)
    axes[2].set_xticks(np.arange(len(FLUX_LABELS)) + 0.5)
    axes[2].set_xticklabels(FLUX_LABELS, fontsize=7)
    axes[2].set_yticks(np.arange(len(RADIUS_LABELS)) + 0.5)
    axes[2].set_yticklabels(RADIUS_LABELS, fontsize=8)
    axes[2].set_xlabel("Insolation (S_earth)")
    axes[2].set_ylabel("Radius (R_earth)")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if not np.isnan(diff[i, j]):
                axes[2].text(j + 0.5, i + 0.5, f"{diff[i,j]:+.3f}",
                             ha="center", va="center", fontsize=7)

    out = out_dir / "fig1_conditional_completeness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: completeness vs insolation (cold-bias slope) ───────────────────

def plot_efficiency_vs_insolation(cat_hard, cat_soft, out_dir: Path) -> None:
    """Show the cold-bias slope for rocky and sub-Neptune planets under both models."""

    radius_slices = [
        (0.5, 2.0, "Rocky  (R < 2 R_earth)",        "C0",  "C1"),
        (2.0, 3.5, "Sub-Neptune  (2-3.5 R_earth)",   "C2",  "C3"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(
        "Detection completeness vs insolation  |  conditioned on transiting\n"
        "Solid = hard threshold    Dashed = sigmoid k=1.5    Shading = 1-sigma binomial",
        fontsize=10,
    )

    fc = FLUX_FINE_CENTERS

    for ax_idx, (r_lo, r_hi, label, col_h, col_s) in enumerate(radius_slices):
        comp_h, comp_s, err_h, n_arr = completeness_vs_insolation(
            cat_hard, cat_soft, FLUX_FINE_EDGES, r_lo, r_hi, min_n=3
        )
        ax = axes[ax_idx]

        valid = ~np.isnan(comp_h)

        if valid.any():
            ax.plot(fc[valid], comp_h[valid], "-o", color=col_h, lw=2.0, ms=5,
                    label="Hard threshold")
            ax.fill_between(fc[valid],
                            np.clip(comp_h[valid] - err_h[valid], 0, 1),
                            np.clip(comp_h[valid] + err_h[valid], 0, 1),
                            color=col_h, alpha=0.15)
            ax.plot(fc[valid], comp_s[valid], linestyle="dashed", marker="s",
                    color=col_s, lw=1.8, ms=5, label="Sigmoid k=1.5")

        # Annotate with N per bin
        for j in range(len(fc)):
            if valid[j] and n_arr[j] > 0:
                ax.annotate(f"n={n_arr[j]}", (fc[j], comp_h[j]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=6.5, color="gray")

        # Shade the habitable zone
        ax.axvspan(0.25, 1.5, color="green", alpha=0.07, label="HZ (approx)")

        ax.set_xscale("log")
        ax.set_xlim(FLUX_FINE_EDGES[0], FLUX_FINE_EDGES[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Insolation (S_earth)  [log scale]")
        ax.set_ylabel("Completeness  (given transiting)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

        ax.set_xticks(FLUX_FINE_CENTERS)
        ax.set_xticklabels([f"{v:.1f}" for v in FLUX_FINE_CENTERS], rotation=30, fontsize=7)
        ax.grid(axis="y", alpha=0.3)

        # Print whether slope differs between models
        if valid.sum() >= 3:
            hot_h  = comp_h[valid & (fc > 5.0)]
            cold_h = comp_h[valid & (fc < 2.0)]
            hot_s  = comp_s[valid & (fc > 5.0)]
            cold_s = comp_s[valid & (fc < 2.0)]
            if len(hot_h) and len(cold_h):
                bias_h = np.nanmean(hot_h) - np.nanmean(cold_h)
                bias_s = np.nanmean(hot_s) - np.nanmean(cold_s)
                ax.set_xlabel(
                    f"Insolation (S_earth)  [log scale]\n"
                    f"Cold bias: hard={bias_h:+.3f}  sigmoid={bias_s:+.3f}  "
                    f"(positive = hot > cold = cold deficit exists)",
                    fontsize=8,
                )

    out = out_dir / "fig2_completeness_vs_insolation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(cat_hard, cat_soft, out_dir: Path) -> None:
    r, f, transiting, det_hard, p_soft = _get_arrays(cat_hard, cat_soft)

    masks = {
        "All transiting":                       transiting,
        "Rocky cold  (R<2, F<1.5, transiting)": transiting & (r < 2.0) & (f < 1.5),
        "Rocky hot   (R<2, F>5,   transiting)": transiting & (r < 2.0) & (f > 5.0),
        "Sub-Nept cold (2-3.5, F<1.5, trans)":  transiting & (r >= 2.0) & (r < 3.5) & (f < 1.5),
        "Sub-Nept hot  (2-3.5, F>5,   trans)":  transiting & (r >= 2.0) & (r < 3.5) & (f > 5.0),
    }

    lines = [
        "=" * 72,
        "ROCKY-COLD BIAS (conditional on transiting)",
        "Hard threshold vs sigmoid weight",
        "=" * 72,
        f"{'Region':<40} {'N':>5} {'Hard%':>7} {'Soft%':>7} {'Diff':>7} {'Sensitive?':>12}",
        "-" * 72,
    ]

    for label, mask in masks.items():
        n = mask.sum()
        if n < 5:
            lines.append(f"{label:<40} {n:>5} {'< 5 planets':>28}")
            continue
        comp_h = det_hard[mask].sum() / n
        comp_s = p_soft[mask].sum()   / n
        diff   = comp_s - comp_h
        rel    = abs(diff) / max(comp_h, 1e-9)
        sens   = "YES(!)" if (abs(diff) > 0.02 and rel > 0.15) else "no"
        lines.append(
            f"{label:<40} {n:>5} {100*comp_h:>6.1f}% {100*comp_s:>6.1f}% "
            f"{100*diff:>+6.1f}% {sens:>12}"
        )

    lines += [
        "-" * 72,
        "",
        "Cold bias = rocky-hot completeness  minus  rocky-cold completeness.",
        "Model-sensitive if the cold-bias magnitude differs between hard and sigmoid.",
        "",
        "Interpretation:",
        "  'no'    -> bias is robust; both models agree on the cold deficit.",
        "  'YES(!)' -> near-threshold cold planets drive the result; report both.",
        "             (Also: calibrate sigmoid_steepness against TESS injection-recovery.)",
        "=" * 72,
    ]

    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(f"\n  Saved: {out_dir / 'summary.txt'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ppop-csv",    type=Path,  default=None)
    parser.add_argument("--n-planets",   type=int,   default=12000)
    parser.add_argument("--sigmoid-k",   type=float, default=1.5)
    parser.add_argument("--n-sectors",   type=int,   default=3)
    args = parser.parse_args()

    TESS_BASE_KWARGS["sigmoid_steepness"]  = args.sigmoid_k
    TESS_BASE_KWARGS["default_n_sectors"]  = args.n_sectors

    if args.ppop_csv is not None and args.ppop_csv.exists():
        print(f"Loading P-Pop catalog: {args.ppop_csv}")
        df = pd.read_csv(args.ppop_csv, low_memory=False)
    else:
        if args.ppop_csv is not None:
            print(f"Warning: {args.ppop_csv} not found -- using synthetic catalog.")
        print(f"Generating synthetic M-dwarf P-Pop (N={args.n_planets}, "
              f"all-transiting, {args.n_sectors} sectors)...")
        df = make_synthetic_ppop(n_planets=args.n_planets)

    print(f"  Catalog: {len(df)} planets,  columns: {list(df.columns[:8])}...")

    print("Running detector (threshold + sigmoid)...")
    cat_hard, cat_soft = run_both_models(df)

    n_trans = cat_hard["tess_transiting_geometric"].astype(bool).sum()
    n_det   = cat_hard["tess_detected"].astype(bool).sum()
    n_soft  = pd.to_numeric(cat_soft["tess_p_detect"], errors="coerce").fillna(0).sum()
    print(f"  Transiting: {n_trans}  |  Hard detected: {n_det}  |  Sigmoid-weighted: {n_soft:.1f}")
    print(f"  Conditional completeness (given transiting): "
          f"hard={n_det/max(n_trans,1):.2%}  sigmoid={n_soft/max(n_trans,1):.2%}")

    print(f"\nPlotting -> {OUT_DIR}/")
    plot_conditional_completeness(cat_hard, cat_soft, OUT_DIR)
    plot_efficiency_vs_insolation(cat_hard, cat_soft, OUT_DIR)
    print()
    print_summary(cat_hard, cat_soft, OUT_DIR)


if __name__ == "__main__":
    main()
