"""
46_systematic_floor.py — Method 6: systematic floor on the puffy-fraction significance.

Question: at what N does adding more NASA planets stop helping, because the choice of rocky
M-R relation (systematic) dominates over counting noise (statistical)?

Machinery: reads script 85's power_table.csv directly (cut, relation, p_A, p_B, delta,
kappa_A, kappa_B, N_now, z_now, N_3sig_50pct, N_3sig_90pct, N_5sig_90pct, extra_planets_5sig).
No pools are rebuilt — this script is cheap (seconds).

Per cut, across the 4 relations:
    sigma_syst   = std_ddof1(p_B)                      — spread of the flat-B prediction itself
    sigma_stat(N)= sqrt(mean_kappa_B * mean_p_B*(1-mean_p_B) / N)   — counting noise at size N
    N_cross      : sigma_stat(N_cross) = sigma_syst    — beyond this N, the M-R systematic
                   dominates over counting noise
    z_floor(N)   = delta / sqrt(kappa_B*p_B*(1-p_B)/N + sigma_syst^2)   (per relation)
    z(N)         = delta*sqrt(N) / sqrt(kappa_B*p_B*(1-p_B))            (no floor, script-85 formula)
    z_max        = delta / sigma_syst                  — asymptotic ceiling of z_floor as N -> inf
    z_max_alt    = delta / sigma_delta, sigma_delta = std_ddof1(delta) — alternative ceiling
                   if the floor is put on the SEPARATION p_A-p_B instead of on p_B alone

Floor-definition caveat: sigma_syst is the ddof=1 std of FOUR published relations — a rough
proxy — and adding it in quadrature treats a fixed model choice as if it were Gaussian noise.
The two floor definitions give very different ceilings (z_max ~3-4 sigma vs z_max_alt ~6-9
sigma in the cold corner), so the binary verdict below is stated as CONDITIONAL on the floor
definition, not as a theorem; both are reported.

Outputs (my_outputs/46_systematic_floor/):
    floor_table.csv + printed table, systematic_floor.png

Run:
    python "scripts/Statistical_Analysis/46_systematic_floor.py"
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

POWER_CSV = ROOT / "my_outputs" / "44_power_analysis_puffy" / "power_table.csv"
OUT_DIR = os.path.join(ROOT, "my_outputs", "46_systematic_floor")
N_GRID = np.logspace(1, 3.5, 200)


def z_floor(N, delta, kB, pB, sigma_syst):
    return delta / np.sqrt(kB * pB * (1 - pB) / N + sigma_syst ** 2)


def z_nofloor(N, delta, kB, pB):
    return delta * np.sqrt(N) / np.sqrt(kB * pB * (1 - pB))


def n_cross(sigma_syst, kB_mean, pB_mean):
    if sigma_syst <= 0:
        return np.inf
    return kB_mean * pB_mean * (1 - pB_mean) / sigma_syst ** 2


def main():
    if not POWER_CSV.exists():
        sys.exit("run scripts/Statistical_Analysis/44_power_analysis_puffy.py first")
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(POWER_CSV)

    cuts = list(df["cut"].unique())
    rows = []
    fig, axes = plt.subplots(1, len(cuts), figsize=(9 * len(cuts), 6.2), sharey=True)
    if len(cuts) == 1:
        axes = [axes]

    for ci, cut_label in enumerate(cuts):
        sub = df[df["cut"] == cut_label].reset_index(drop=True)
        sigma_syst = float(sub["p_B"].std(ddof=1))
        sigma_delta = float(sub["delta"].std(ddof=1))
        kB_mean, pB_mean = float(sub["kappa_B"].mean()), float(sub["p_B"].mean())
        ncross = n_cross(sigma_syst, kB_mean, pB_mean)
        n_now = float(sub["N_now"].iloc[0])

        print(f"\n--> [{cut_label}]")
        print(f"    sigma_syst(p_B across relations) = {sigma_syst:.4f}   "
              f"sigma(delta across relations) = {sigma_delta:.4f}")
        print(f"    sigma_stat(N_now={n_now:.0f}) = {np.sqrt(kB_mean * pB_mean * (1 - pB_mean) / n_now):.4f}   "
              f"N_cross = {ncross:.1f}")
        print(f"  {'relation':<24}{'delta':>7}{'kappa_B':>9}{'p_B':>7}{'z_max':>8}{'z_max_alt':>10}")

        ax = axes[ci]
        zmax_list, zmax_alt_list = [], []
        for _, r in sub.iterrows():
            zmax = r["delta"] / sigma_syst if sigma_syst > 0 else np.inf
            zmax_alt = r["delta"] / sigma_delta if sigma_delta > 0 else np.inf
            zmax_list.append(zmax)
            zmax_alt_list.append(zmax_alt)
            zf = z_floor(N_GRID, r["delta"], r["kappa_B"], r["p_B"], sigma_syst)
            zn = z_nofloor(N_GRID, r["delta"], r["kappa_B"], r["p_B"])
            line, = ax.plot(N_GRID, zf, lw=2, label=f"{r['relation']}: z_max={zmax:.1f}")
            ax.plot(N_GRID, zn, lw=1.2, ls="--", color=line.get_color(), alpha=0.7)
            print(f"  {r['relation']:<24}{r['delta']:>7.3f}{r['kappa_B']:>9.3f}{r['p_B']:>7.3f}"
                  f"{zmax:>7.1f}σ{zmax_alt:>9.1f}σ")
            rows.append(dict(cut=cut_label, relation=r["relation"], sigma_syst=sigma_syst,
                             sigma_delta=sigma_delta, N_cross=ncross, z_max=zmax, z_max_alt=zmax_alt,
                             kappa_B=r["kappa_B"], p_B=r["p_B"], delta=r["delta"], N_now=n_now))

        ax.axhline(3, color="0.4", ls="--", lw=1)
        ax.axhline(5, color="0.4", ls=":", lw=1)
        ax.axvline(n_now, color="tab:green", lw=1.5)
        ax.text(n_now * 1.05, 0.4, f"today N_now={n_now:.0f}", color="tab:green", fontsize=8)
        if np.isfinite(ncross):
            ax.axvline(ncross, color="tab:red", lw=1.5, ls="-.")
            ax.text(ncross * 1.05, 1.2, f"N_cross={ncross:.0f}", color="tab:red", fontsize=8)
        ax.text(N_GRID[-1], 3.1, "3σ", ha="right", fontsize=8, color="0.4")
        ax.text(N_GRID[-1], 5.1, "5σ", ha="right", fontsize=8, color="0.4")
        ax.set_xscale("log")
        ax.set_xlim(N_GRID[0], N_GRID[-1]); ax.set_ylim(0, 12)
        ax.set_xlabel("NASA precision-passing planets N")
        ax.set_title(f"{cut_label}\nsolid = with relation-spread floor, dashed = counting noise only",
                     fontsize=9.5)
        ax.grid(alpha=0.2); ax.legend(fontsize=7.5, loc="upper left")
        if ci == 0:
            ax.set_ylabel("significance ruling out Primordial-rocky (truth = Escape-only)")

        min_zmax = min(zmax_list)
        sigma_stat_now = np.sqrt(kB_mean * pB_mean * (1 - pB_mean) / n_now)
        print(f"\n  VERDICT [{cut_label}]:")
        min_zmax_alt = min(zmax_alt_list)
        if min_zmax > 5:
            print(f"    (1) binary (rocky super-Earths yes/no): min z_max = {min_zmax:.1f}σ > 5σ across "
                  f"all relations -> statistics-limited up to N_cross ~ {ncross:.0f}; "
                  f"5σ is reachable there.")
        else:
            print(f"    (1) binary (rocky super-Earths yes/no): CONDITIONAL on the floor definition. "
                  f"Under the p_B-spread floor the overlap statistic saturates at min z_max = "
                  f"{min_zmax:.1f}σ (< 5σ); under the delta-spread floor the ceiling is min z_max_alt = "
                  f"{min_zmax_alt:.1f}σ. Quote as: current rocky M-R calibration spread, not sample "
                  f"size, is projected to limit this test beyond N ~ {ncross:.0f} — not as '5σ is "
                  f"impossible at any N'.")
        if sigma_syst > sigma_stat_now:
            print(f"    (2) fine (which relation / rocky fraction): sigma_syst={sigma_syst:.4f} > "
                  f"sigma_stat(N_now)={sigma_stat_now:.4f} -> ALREADY systematics-limited; better rocky "
                  f"M-R calibration is worth as much as new detections.")
        else:
            print(f"    (2) fine (which relation / rocky fraction): sigma_syst={sigma_syst:.4f} <= "
                  f"sigma_stat(N_now)={sigma_stat_now:.4f} -> still statistics-limited at N_now; more "
                  f"detections help before calibration does.")

    fig.suptitle("Systematic floor: rocky mass-radius relation choice vs counting noise on the detected volatile fraction\n"
                 "significance flattens once the relation spread beats the counting noise",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_png = os.path.join(OUT_DIR, "systematic_floor.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(OUT_DIR, "floor_table.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n--> Saved: {out_png}")
    print(f"--> Saved: {out_csv}")


if __name__ == "__main__":
    main()
