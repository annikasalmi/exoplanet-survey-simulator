"""
44_power_analysis_puffy.py — Method 1: analytic two-proportion power on the puffy fraction.

Question: how many NASA precision-passing planets are needed before the puffy fraction can
separate flat A ("no rocky super-Earths": drop rocky true-M>2 M⊕) from flat B (keep all),
for each published rocky M-R relation?

Machinery is reused, not reimplemented: script 77 supplies the relation pools
(build_arrays, RELATIONS), script 72 supplies detection + noise MC (mc_universe, mc_nasa).
Per relation and cut we measure:
    p_A, p_B — mean detected+noised puffy fractions under each hypothesis,
    κ_X      — variance inflation over pure binomial: κ = σ²_MC · N_eff / (p·(1-p)),
               calibrating bootstrap sampling + measurement noise into the analytic formula.
Truth is taken as flat A (where NASA currently sits); we ask when flat B is rejected:
    Δ = |p_A − p_B|
    N ≥ [z_disc·√(κ_B p_B q_B) + z_power·√(κ_A p_A q_A)]² / Δ²
Median expected significance at sample size N:  z_med(N) = Δ·√N / √(κ_B p_B q_B).

Note: this forecast includes future-sample bootstrap variance (via κ), so z_med at today's
N is LOWER than script 77's quoted tensions, whose NASA bell deliberately has no bootstrap.

Outputs (my_outputs/44_power_analysis_puffy/):
    power_table.csv + printed table, significance_vs_n.png

Run:
    python "scripts/Statistical_Analysis/44_power_analysis_puffy.py"
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S77 = _load("s77", str(ROOT / "scripts" / "Statistical_Analysis" / "20_flat_rocky_mr_vs_nasa.py"))
S72 = S77.S72                    # already loaded by S77, N_REPEATS = 4000

OUT_DIR = os.path.join(ROOT, "my_outputs", "44_power_analysis_puffy")
Z_POWER_90 = float(norm.ppf(0.90))
CUTS = [("all (no cut)", {}),
        ("mass>2 M⊕ & insol<50 I⊕ (cold super-Earth corner)", dict(mass_min=2.0, insol_max=50.0))]
N_GRID = np.logspace(1, 3.5, 200)


def kappa(s, p, n_eff):
    pq = p * (1.0 - p)
    if s.size == 0 or n_eff <= 0 or pq <= 0:
        return np.nan
    return s.std() ** 2 * n_eff / pq


def n_required(pA, kA, pB, kB, delta, z_disc, z_power):
    if delta <= 0:
        return np.inf
    num = (z_disc * np.sqrt(kB * pB * (1 - pB)) + z_power * np.sqrt(kA * pA * (1 - pA))) ** 2
    return num / delta ** 2


def z_median(N, delta, pB, kB):
    return delta * np.sqrt(N) / np.sqrt(kB * pB * (1 - pB))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = S72.load_silicate()
    rng = np.random.default_rng(S77.SEED)
    nasa = S72.load_nasa()

    print(f"--> building {len(S77.RELATIONS)} rocky-relation pools + detectors "
          f"(mass scatter = {S77.MR_SCATTER_DEX} dex, N_REPEATS = {S72.N_REPEATS})...")
    pools = [(name, S77.build_arrays(kw, m_sil, r_sil)) for name, eq, applies, kw in S77.RELATIONS]

    rows = []
    fig, axes = plt.subplots(1, len(CUTS), figsize=(8 * len(CUTS), 6), sharey=True)
    for ci, (cut_label, cut) in enumerate(CUTS):
        nv, n_now = S72.mc_nasa(nasa, cut, m_sil, r_sil, rng)
        print(f"\n--> [{cut_label}]  NASA today: N_eff={n_now}, puffy μ={nv.mean():.3f}")
        print(f"  {'relation':<24}{'p_A':>7}{'p_B':>7}{'Δ':>7}{'κ_A':>6}{'κ_B':>6}"
              f"{'z_now':>7}{'N 3σ/50%':>10}{'N 3σ/90%':>10}{'N 5σ/90%':>10}{'extra 5σ':>10}")

        ax = axes[ci]
        for name, arr in pools:
            sA, nA = S72.mc_universe(arr, True, cut, m_sil, r_sil, rng)
            sB, nB = S72.mc_universe(arr, False, cut, m_sil, r_sil, rng)
            pA, pB = sA.mean(), sB.mean()
            kA, kB = kappa(sA, pA, nA), kappa(sB, pB, nB)
            delta = abs(pA - pB)
            z_now = z_median(n_now, delta, pB, kB)
            n3_50 = n_required(pA, kA, pB, kB, delta, 3.0, 0.0)
            n3_90 = n_required(pA, kA, pB, kB, delta, 3.0, Z_POWER_90)
            n5_90 = n_required(pA, kA, pB, kB, delta, 5.0, Z_POWER_90)
            extra = max(0, int(np.ceil(n5_90)) - n_now) if np.isfinite(n5_90) else np.nan
            print(f"  {name:<24}{pA:>7.3f}{pB:>7.3f}{delta:>7.3f}{kA:>6.2f}{kB:>6.2f}"
                  f"{z_now:>6.1f}σ{n3_50:>10.0f}{n3_90:>10.0f}{n5_90:>10.0f}{extra:>10.0f}")
            rows.append(dict(cut=cut_label, relation=name, p_A=pA, p_B=pB, delta=delta,
                             kappa_A=kA, kappa_B=kB, N_now=n_now, z_now=z_now,
                             N_3sig_50pct=n3_50, N_3sig_90pct=n3_90, N_5sig_90pct=n5_90,
                             extra_planets_5sig=extra))
            ax.plot(N_GRID, z_median(N_GRID, delta, pB, kB), lw=2,
                    label=f"{name}: 3σ at N={n3_50:.0f}, 5σ/90% at N={n5_90:.0f}")

        ax.axhline(3, color="0.4", ls="--", lw=1)
        ax.axhline(5, color="0.4", ls=":", lw=1)
        ax.axvline(n_now, color="tab:green", lw=1.5)
        ax.text(n_now * 1.05, 0.3, f"today N_eff={n_now}", color="tab:green", fontsize=8)
        ax.text(N_GRID[-1], 3.1, "3σ", ha="right", fontsize=8, color="0.4")
        ax.text(N_GRID[-1], 5.1, "5σ", ha="right", fontsize=8, color="0.4")
        ax.set_xscale("log")
        ax.set_xlim(N_GRID[0], N_GRID[-1]); ax.set_ylim(0, 12)
        ax.set_xlabel("NASA precision-passing planets N")
        ax.set_title(cut_label, fontsize=10)
        ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="upper left")
        if ci == 0:
            ax.set_ylabel("median significance rejecting flat B (truth = flat A)")

    fig.suptitle("How many NASA planets to reject flat B (rocky super-Earths exist)?\n"
                 "two-proportion power, κ-calibrated against the script-72 MC (bootstrap + noise included)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(OUT_DIR, "significance_vs_n.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(OUT_DIR, "power_table.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n--> Saved: {out_png}")
    print(f"--> Saved: {out_csv}")


if __name__ == "__main__":
    main()
