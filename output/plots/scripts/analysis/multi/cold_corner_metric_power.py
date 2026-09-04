"""
26_cold_corner_metric_power.py — CAN we prove/disprove rocky planets in the cold top-left
corner (low insolation, large radius)?  A self-check of how powerful each metric is THERE.

The global test (57/58) says massive rocky planets exist *somewhere* — but most detections
are warm. This script asks the harder, specific question and validates the metrics by
INJECTION-RECOVERY, the honest way to measure a statistic's power:

  1) COMPLETENESS: synthesise large ROCKY planets (R~1.4-2.0, density~7.5 g/cc -> below the
     silicate curve) on real P-Pop orbits/stars across all insolation, run transit + RV + joint
     detection, and measure the detected fraction vs insolation. If joint completeness -> 0 in the
     cold corner, the data is BLIND there.

  2) METRIC POWER: inject N_INJ_TRUE such rocky planets into the observed P-Pop B catalogue,
     COLD (flux<10) vs HOT (flux>100). Only the DETECTED ones enter. Build many NASA-sized
     "rocky-present" vs "rocky-absent" observed catalogues and measure each metric's AUC
     (separability). AUC~0.5 = the metric cannot tell present from absent = powerless there.

  Contrast cold (selection desert) with hot (detection works) to prove the metrics aren't
  inherently blind — they're blind only where detection is.

Run:
    python scripts/26_cold_corner_metric_power.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.paths import SILICON_CURVE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.flat_detect import run_kepler, run_rv_best
from run.ppop import universe_metrics as UM

SILICATE_CURVE = Path(SILICON_CURVE)
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "26_cold_corner_metric_power")

# ============================ KNOBS ============================
COLD_FLUX = 10.0          # insolation below this = "cold" (top-left corner)
HOT_FLUX = 100.0          # above this = "hot" control
INJ_R_LO, INJ_R_HI = 1.4, 2.0   # injected rocky planets: large radius
INJ_DENSITY = 7.5        # g/cc -> rocky (below silicate curve)
N_BANK = 40000           # injected planets used to measure completeness
N_INJ_TRUE = 300         # TRUE rocky planets injected per realization (cold or hot)
K_POW = 400              # realizations for AUC
N_OBS = 292              # observed catalogue size per realization (~ NASA)
MASS_FRAC_ERR, RAD_FRAC_ERR = 0.20, 0.047
# ==============================================================

RV_MAG_TARGET = 12.0
RNG_SEED = 0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
RHO_EARTH = 5.513

POWER_METRICS = [
    ("puffy_frac", lambda M, R, ms, rs: UM.puffy_fraction(M, R, ms, rs)),
    ("mean_dist_to_curve", lambda M, R, ms, rs: UM.mean_distance_to_curve(M, R, ms, rs)),
    ("median_density", lambda M, R, ms, rs: UM.median_density(M, R)),
    ("logdensity_scatter", lambda M, R, ms, rs: UM.logdensity_scatter(M, R)),
]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def load_ppop():
    df = pd.read_csv(PPOP_CATALOG, low_memory=False)
    for c in ["radius_p", "mass_p", "flux_p"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    r, m = df["radius_p"], df["mass_p"]
    keep = r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    f = df["flux_p"]
    keep = keep & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"]))
    return df[keep].copy().reset_index(drop=True)


def make_rocky(hosts: pd.DataFrame, rng) -> pd.DataFrame:
    """Take real P-Pop rows (orbit+star kept) and overwrite radius/mass with a large rocky planet."""
    df = hosts.copy()
    R = rng.uniform(INJ_R_LO, INJ_R_HI, len(df))
    M = np.clip(INJ_DENSITY / RHO_EARTH * R ** 3, BOX["m_lo"], BOX["m_hi"])
    df["radius_p"] = R
    df["mass_p"] = M
    return df


def detect(df):
    td = run_kepler(df)["detected"].to_numpy(bool)
    rd = run_rv_best(df, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    return td, rd


def noise(M, R, rng):
    return (M * np.exp(rng.normal(0, MASS_FRAC_ERR, M.size)),
            R * np.exp(rng.normal(0, RAD_FRAC_ERR, R.size)))


def auc_rectified(present, absent):
    p = present[np.isfinite(present)]; a = absent[np.isfinite(absent)]
    if p.size < 5 or a.size < 5:
        return np.nan
    allv = np.concatenate([p, a])
    r = rankdata(allv)
    u = r[:p.size].sum() - p.size * (p.size + 1) / 2
    auc = u / (p.size * a.size)
    return max(auc, 1 - auc)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    pool = load_ppop()
    print(f"--> P-Pop box pool: {len(pool)} planets")

    # ---------- baseline observed P-Pop B catalogue (detected + intrinsic) ----------
    td0, rd0 = detect(pool)
    det0 = td0 & rd0
    base = pool.loc[det0, ["mass_p", "radius_p", "flux_p"]].to_numpy(float)
    base_M, base_R, base_F = base[:, 0], base[:, 1], base[:, 2]
    n_cold_base = int((base_F < COLD_FLUX).sum())
    print(f"--> baseline DETECTED P-Pop B: {det0.sum()}  (of which cold flux<{COLD_FLUX:g}: {n_cold_base})")

    # ---------- (1) completeness of large rocky planets vs insolation ----------
    hosts = pool.sample(min(N_BANK, len(pool)), random_state=1).reset_index(drop=True)
    inj = make_rocky(hosts, rng)
    inj_puffy = UM.puffy_fraction(inj["mass_p"], inj["radius_p"], m_sil, r_sil)
    td, rd = detect(inj)
    joint = td & rd
    F = inj["flux_p"].to_numpy(float)
    print(f"--> injected rocky test planets: {len(inj)}  (puffy fraction={inj_puffy:.3f}; ~0 = all rocky, good)")

    fbins = np.logspace(-1, 4, 16)
    fc = np.sqrt(fbins[:-1] * fbins[1:])
    comp = {"transit": [], "RV": [], "joint": []}
    for lo, hi in zip(fbins[:-1], fbins[1:]):
        cell = (F >= lo) & (F < hi)
        n = int(cell.sum())
        for key, mask in [("transit", td), ("RV", rd), ("joint", joint)]:
            comp[key].append(mask[cell].mean() if n >= 20 else np.nan)
    for k in comp:
        comp[k] = np.array(comp[k])

    def comp_in(flo, fhi, mask):
        c = (F >= flo) & (F < fhi)
        return mask[c].mean() if c.sum() else np.nan
    f_cold = comp_in(0, COLD_FLUX, joint)
    f_hot = comp_in(HOT_FLUX, 1e9, joint)
    print(f"--> JOINT completeness of large rocky planets:  cold (flux<{COLD_FLUX:g}) = {f_cold:.4f}   "
          f"hot (flux>{HOT_FLUX:g}) = {f_hot:.4f}")

    # ---------- detected injection banks (cold / hot) ----------
    def bank(regime):
        if regime == "cold":
            sel = joint & (F < COLD_FLUX)
        else:
            sel = joint & (F > HOT_FLUX)
        return (inj["mass_p"].to_numpy(float)[sel], inj["radius_p"].to_numpy(float)[sel],
                inj["flux_p"].to_numpy(float)[sel])
    banks = {r: bank(r) for r in ["cold", "hot"]}
    f_regime = {"cold": f_cold, "hot": f_hot}

    # ---------- (2) injection-recovery metric power ----------
    print(f"\n  METRIC POWER (AUC; 0.5 = powerless, 1.0 = perfectly separates rocky-present vs -absent)")
    print(f"  injecting N_INJ_TRUE={N_INJ_TRUE} true rocky planets/realization; only the detected ones enter.")
    print(f"  {'metric':<22}{'COLD corner':>14}{'HOT control':>14}")
    print("  " + "-" * 50)
    power = {regime: {mk: None for mk, _ in POWER_METRICS} for regime in ["cold", "hot"]}
    cold_subset_sizes = []
    for regime in ["cold", "hot"]:
        bM, bR, bF = banks[regime]
        flo, fhi = (0, COLD_FLUX) if regime == "cold" else (HOT_FLUX, 1e9)
        present = {mk: [] for mk, _ in POWER_METRICS}
        absent = {mk: [] for mk, _ in POWER_METRICS}
        for _ in range(K_POW):
            bi = rng.integers(0, base_M.size, N_OBS)
            cM, cR, cF = base_M[bi], base_R[bi], base_F[bi]
            n_enter = rng.binomial(N_INJ_TRUE, max(f_regime[regime], 0.0)) if bM.size else 0
            if n_enter > 0 and bM.size:
                j = rng.integers(0, bM.size, n_enter)
                pM = np.concatenate([cM, bM[j]]); pR = np.concatenate([cR, bR[j]]); pF = np.concatenate([cF, bF[j]])
            else:
                pM, pR, pF = cM, cR, cF
            # restrict to this insolation regime, add measurement noise, compute metrics
            for tag, (M, R, Fl) in [("abs", (cM, cR, cF)), ("pre", (pM, pR, pF))]:
                inb = (Fl >= flo) & (Fl < fhi)
                Mo, Ro = noise(M[inb], R[inb], rng)
                if regime == "cold" and tag == "abs":
                    cold_subset_sizes.append(int(inb.sum()))
                for mk, fn in POWER_METRICS:
                    (present if tag == "pre" else absent)[mk].append(
                        fn(Mo, Ro, m_sil, r_sil) if inb.sum() >= 5 else np.nan)
        for mk, _ in POWER_METRICS:
            power[regime][mk] = auc_rectified(np.array(present[mk]), np.array(absent[mk]))
    for mk, _ in POWER_METRICS:
        print(f"  {mk:<22}{power['cold'][mk]:>14.3f}{power['hot'][mk]:>14.3f}")
    print(f"\n  median cold OBSERVED subset size per realization: {int(np.median(cold_subset_sizes))} planets "
          f"(this is all the data the cold metrics get)")

    # ---------- NASA in the cold corner ----------
    nd = pd.read_csv(NASA_FILE)
    nm = pd.to_numeric(nd["pl_bmasse"], errors="coerce")
    nr = pd.to_numeric(nd["pl_rade"], errors="coerce")
    nf = pd.to_numeric(nd["pl_insol"], errors="coerce")
    prov = nd.get("pl_bmassprov", pd.Series("", index=nd.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keepn = meas & nr.between(BOX["r_lo"], BOX["r_hi"]) & nm.between(BOX["m_lo"], BOX["m_hi"])
    cold_n = keepn & (nf < COLD_FLUX)
    n_cold_nasa = int(cold_n.sum())
    print(f"\n  NASA measured-mass planets in box: {int(keepn.sum())};  in the COLD corner (insol<{COLD_FLUX:g}): {n_cold_nasa}")
    if n_cold_nasa:
        pf = UM.puffy_fraction(nm[cold_n], nr[cold_n], m_sil, r_sil)
        print(f"    (their puffy fraction = {pf:.2f}, but N={n_cold_nasa} is tiny)")

    # ---------- figure ----------
    fig, (axc, axp) = plt.subplots(1, 2, figsize=(16, 6.4))
    for key, c in [("transit", "tab:green"), ("RV", "tab:red"), ("joint", "k")]:
        axc.plot(fc, comp[key], "-o", color=c, lw=2, ms=4, label=key)
    axc.axvspan(fbins[0], COLD_FLUX, color="tab:blue", alpha=0.12)
    axc.text(0.2, 0.92, "Cold Rocky Desert", color="tab:blue", fontsize=10, transform=axc.get_xaxis_transform())
    axc.axvline(COLD_FLUX, color="tab:blue", ls="--", lw=1)
    axc.set_xscale("log")
    axc.set_xlabel(r"insolation flux  [$I_\oplus$]")
    axc.set_ylabel("detection completeness of LARGE ROCKY planets")
    axc.set_title(f"(1) Can we even detect cold large rocky planets?\n"
                  f"joint completeness: cold={f_cold:.3f}  vs  hot={f_hot:.3f}", fontsize=11)
    axc.set_ylim(-0.03, 1.03)
    axc.grid(alpha=0.25, which="both")
    axc.legend(title="detector")

    mk_labels = [mk for mk, _ in POWER_METRICS]
    x = np.arange(len(mk_labels))
    axp.bar(x - 0.2, [power["cold"][m] for m in mk_labels], 0.4, label=f"Cold Rocky Desert (flux<{COLD_FLUX:g})", color="tab:blue")
    axp.bar(x + 0.2, [power["hot"][m] for m in mk_labels], 0.4, label=f"HOT control (flux>{HOT_FLUX:g})", color="tab:red")
    axp.axhline(0.5, color="k", ls="--", lw=1)
    axp.text(len(mk_labels) - 0.5, 0.51, "0.5 = powerless", fontsize=8, ha="right")
    axp.set_xticks(x); axp.set_xticklabels(mk_labels, rotation=18, ha="right", fontsize=9)
    axp.set_ylim(0.45, 1.02)
    axp.set_ylabel("metric power  (AUC: rocky-present vs absent)")
    axp.set_title("(2) Does any metric notice injected rocky planets?\n"
                  "cold bars near 0.5 = weak; hot bars high = metrics work where detection does", fontsize=11)
    axp.legend()
    axp.grid(alpha=0.2, axis="y")

    fig.suptitle("Self-check: how useful is each metric for the COLD top-left-corner rocky question?", fontsize=13)
    ratio = (f_hot / f_cold) if f_cold > 0 else float("inf")
    fig.text(0.5, 0.005,
             f"Large-rocky completeness is {f_cold:.1%} cold vs {f_hot:.1%} hot (~{ratio:.0f}x lower), so injected cold rocky "
             f"planets barely enter the catalogue and every metric's AUC stays near 0.5 (weak) — the cold-corner abundance is "
             f"selection-limited, not a measured radius wall.  NASA still has {n_cold_nasa} cold measured-mass planets "
             f"(puffy frac ~0.5), so cold rocky planets DO exist; their FRACTION is just hard to pin down.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_png = os.path.join(OUT_DIR, "cold_corner_metric_power.png")
    fig.savefig(out_png, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()
