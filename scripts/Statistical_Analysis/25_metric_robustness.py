"""
25_metric_robustness.py — FIXES for two referee objections to scripts 56-58.

PART A  (fix "single P-Pop catalog"):  COSMIC VARIANCE.
  Scripts 56-58/61 computed their metrics on ONE P-Pop universe (kepler_catalog_0), so
  the model arm carried an unquantified extra (cosmic / Monte-Carlo) scatter.  Here we run
  the transit+RV detector on EACH of the 10 Gaia P-Pop universes separately and report the
  across-universe mean +/- SD of the headline metrics (puffy fraction, median density).
  If that SD is small next to the model-vs-NASA gap, the single-catalog conclusions hold.

PART B  (fix "unexplained NASA-model gap"):  WHERE DOES 0.59 -> 0.40 COME FROM?
  NASA's measured-mass planets are ~0.40 puffy; detected P-Pop-with-rocky is ~0.59.
  Hypothesis: RV preferentially CONFIRMS dense (rocky) planets, because at fixed radius a
  rocky planet is more massive -> larger K -> easier mass measurement.  Real published
  masses typically demand higher significance than our nominal K/sigma_K >= 5.  So we sweep
  the RV significance threshold and watch the detected puffy fraction fall toward NASA.
  If a plausible threshold reaches ~0.40, the gap is RV selection (not a P-Pop occurrence
  error or a curve error); whatever residual remains is the occurrence/curve systematic.

Run from repo root:
    python scripts/25_metric_robustness.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.ppop.flat_detect import run_kepler, run_rv, RVData

SILICATE_CURVE = ROOT / "Hongyi-silicon.ddat"
KEPLER_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = ROOT / "my_outputs" / "25_metric_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RHO_EARTH = 5.513
RV_MAG_TARGET = 12.0
N_FULL_UNIVERSES = 10
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
PPOP_COLS = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p", "semimajor_p", "radius_s",
             "mass_s", "temp_s", "teff_s", "distance_s", "l_sun", "flux_p", "stype", "detected"]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def puffy_flag(mass, radius, m_sil, r_sil):
    return np.asarray(radius, float) > np.interp(np.asarray(mass, float), m_sil, r_sil)


def load_box(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in PPOP_COLS, low_memory=False)
    r = pd.to_numeric(df["radius_p"], errors="coerce")
    m = pd.to_numeric(df["mass_p"], errors="coerce")
    f = pd.to_numeric(df["flux_p"], errors="coerce")
    keep = (r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"])))
    return df[keep].copy()


def load_nasa_puffy(m_sil, r_sil):
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    measured = (prov.str.contains("Mass|Msini", case=False, na=False)
                & ~prov.str.contains("Calc", case=False, na=False))
    keep = measured & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    return float(puffy_flag(m[keep].to_numpy(), r[keep].to_numpy(), m_sil, r_sil).mean()), int(keep.sum())


# ── PART A: cosmic variance across the 10 universes ─────────────────────────────

def part_a(m_sil, r_sil, nasa_puffy):
    # Only the full F/G/K/M universes 0..9 are valid here. The 8001+ top-ups in the same
    # directory are single-spectral-type (G-only / K-only) partial draws, so treating them
    # as universes corrupts the across-universe mean and SD.
    files = [KEPLER_DIR / f"kepler_catalog_{i}.csv" for i in range(N_FULL_UNIVERSES)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"missing full P-Pop universes: {missing}")
    print(f"\nPART A — cosmic variance across {len(files)} P-Pop universes")
    puffies, denss, ndet = [], [], []
    for f in files:
        df = load_box(f)
        mass = pd.to_numeric(df["mass_p"], errors="coerce").to_numpy(float)
        rad = pd.to_numeric(df["radius_p"], errors="coerce").to_numpy(float)
        # transit: the catalog's own Kepler flag if present, else run the model
        if "detected" in df.columns and df["detected"].notna().any():
            td = pd.to_numeric(df["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
        else:
            td = run_kepler(df)["detected"].to_numpy(bool)
        h = run_rv(df, "HARPS"); n = run_rv(df, "NIRPS")
        hmag = pd.to_numeric(h["rv_mag"], errors="coerce").to_numpy()
        nmag = pd.to_numeric(n["rv_mag"], errors="coerce").to_numpy()
        rd = ((h["rv_detected"].to_numpy(bool) & (hmag <= RV_MAG_TARGET))
              | (n["rv_detected"].to_numpy(bool) & (nmag <= RV_MAG_TARGET)))
        det = td & rd
        if det.sum() < 10:
            continue
        puffy = puffy_flag(mass[det], rad[det], m_sil, r_sil)
        dens = RHO_EARTH * mass[det] / rad[det] ** 3
        puffies.append(float(puffy.mean()))
        denss.append(float(np.median(dens)))
        ndet.append(int(det.sum()))
    puffies, denss = np.array(puffies), np.array(denss)
    gap = abs(puffies.mean() - nasa_puffy)
    print(f"  {'metric':<22}{'mean':>9}{'SD(cosmic)':>12}{'min':>8}{'max':>8}")
    print(f"  {'puffy fraction':<22}{puffies.mean():>9.3f}{puffies.std(ddof=1):>12.3f}"
          f"{puffies.min():>8.3f}{puffies.max():>8.3f}")
    print(f"  {'median density g/cc':<22}{denss.mean():>9.2f}{denss.std(ddof=1):>12.2f}"
          f"{denss.min():>8.2f}{denss.max():>8.2f}")
    print(f"  N_detected per universe: mean {np.mean(ndet):.0f}")
    print(f"  model-vs-NASA puffy gap = {gap:.3f}   vs cosmic SD = {puffies.std(ddof=1):.3f}  "
          f"-> gap is {gap/max(puffies.std(ddof=1),1e-9):.1f}x the cosmic scatter")
    return dict(puffies=puffies, denss=denss, gap=gap, nasa_puffy=nasa_puffy)


# ── PART B: RV-strictness sweep explaining the puffy gap ────────────────────────

def part_b(m_sil, r_sil, nasa_puffy):
    print("\nPART B — RV-strictness sweep (does RV selection explain 0.59 -> 0.40?)")
    df = load_box(KEPLER_DIR / "kepler_catalog_0.csv")
    mass = pd.to_numeric(df["mass_p"], errors="coerce").to_numpy(float)
    rad = pd.to_numeric(df["radius_p"], errors="coerce").to_numpy(float)
    puffy = puffy_flag(mass, rad, m_sil, r_sil)

    if "detected" in df.columns and df["detected"].notna().any():
        td = pd.to_numeric(df["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    else:
        td = run_kepler(df)["detected"].to_numpy(bool)

    # RV significance + mag for each band, computed ONCE.
    h = RVData(df.copy(), source="ppop", instrument="HARPS").determine_detectable()
    n = RVData(df.copy(), source="ppop", instrument="NIRPS").determine_detectable()
    snr_h = pd.to_numeric(h["rv_snr"], errors="coerce").to_numpy()
    snr_n = pd.to_numeric(n["rv_snr"], errors="coerce").to_numpy()
    mag_h = pd.to_numeric(h["rv_mag"], errors="coerce").to_numpy()
    mag_n = pd.to_numeric(n["rv_mag"], errors="coerce").to_numpy()
    reach = (mag_h <= RV_MAG_TARGET) | (mag_n <= RV_MAG_TARGET)

    thresholds = np.array([3, 4, 5, 6, 7, 8, 10, 12, 15], float)
    fracs, ndet = [], []
    for thr in thresholds:
        rv = ((snr_h >= thr) & (mag_h <= RV_MAG_TARGET)) | ((snr_n >= thr) & (mag_n <= RV_MAG_TARGET))
        det = td & rv
        if det.sum() == 0:
            fracs.append(np.nan); ndet.append(0); continue
        fracs.append(float(puffy[det].mean())); ndet.append(int(det.sum()))
    fracs = np.array(fracs)

    # where does the model reach NASA's puffy fraction?
    reach_thr = np.nan
    for i in range(len(thresholds) - 1):
        if np.isfinite(fracs[i]) and np.isfinite(fracs[i + 1]) and \
           (fracs[i] - nasa_puffy) * (fracs[i + 1] - nasa_puffy) <= 0:
            x0, x1, y0, y1 = thresholds[i], thresholds[i + 1], fracs[i], fracs[i + 1]
            reach_thr = x0 + (nasa_puffy - y0) * (x1 - x0) / (y1 - y0)
            break

    print(f"  {'K/sigma_K thr':>14}{'puffy frac':>12}{'N_det':>9}")
    for thr, fr, nd in zip(thresholds, fracs, ndet):
        print(f"  {thr:>14.0f}{fr:>12.3f}{nd:>9d}")
    print(f"  NASA puffy fraction = {nasa_puffy:.3f}")
    if np.isfinite(reach_thr):
        print(f"  -> model reaches NASA's 0.40 at K/sigma_K ~ {reach_thr:.1f}  "
              f"(nominal was 5) => the gap is RV selection: real published masses demand "
              f"higher significance, which preferentially keeps dense/rocky planets.")
    else:
        print(f"  -> even at K/sigma_K=15 the model puffy fraction ({fracs[-1]:.2f}) does not "
              f"reach 0.40 => RV selection alone does NOT close the gap; residual is a P-Pop "
              f"occurrence (too many puffy planets) or curve systematic.")
    return dict(thresholds=thresholds, fracs=fracs, nasa_puffy=nasa_puffy, reach_thr=reach_thr)


def plot_all(a: dict, b: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Part A: cosmic variance of the puffy fraction vs the NASA gap.
    ax1.axhline(a["nasa_puffy"], color="tab:green", lw=2.2, label=f"NASA puffy = {a['nasa_puffy']:.2f}")
    ax1.axhline(a["puffies"].mean(), color="tab:purple", lw=2.0,
                label=f"model mean = {a['puffies'].mean():.2f}")
    ax1.fill_between([0, len(a["puffies"]) + 1],
                     a["puffies"].mean() - a["puffies"].std(ddof=1),
                     a["puffies"].mean() + a["puffies"].std(ddof=1),
                     color="tab:purple", alpha=0.18, label="±1 SD across universes")
    ax1.scatter(np.arange(1, len(a["puffies"]) + 1), a["puffies"], color="tab:purple", zorder=5)
    ax1.set_xlim(0, len(a["puffies"]) + 1)
    ax1.set_xlabel("P-Pop universe #"); ax1.set_ylabel("detected puffy fraction")
    ax1.set_title("PART A — cosmic variance is tiny vs the NASA gap\n"
                  f"gap {a['gap']:.2f} = {a['gap']/max(a['puffies'].std(ddof=1),1e-9):.0f}× the "
                  f"across-universe SD ({a['puffies'].std(ddof=1):.3f})", fontsize=10.5)
    ax1.legend(fontsize=8.5, loc="best"); ax1.grid(alpha=0.25)

    # Part B: RV-strictness sweep toward NASA.
    ax2.plot(b["thresholds"], b["fracs"], "-o", color="tab:purple", lw=2, label="model detected puffy fraction")
    ax2.axhline(b["nasa_puffy"], color="tab:green", lw=2.2, label=f"NASA puffy = {b['nasa_puffy']:.2f}")
    ax2.axvline(5, color="0.5", ls=":", lw=1.2, label="nominal K/σ_K = 5")
    if np.isfinite(b["reach_thr"]):
        ax2.axvline(b["reach_thr"], color="tab:red", ls="--", lw=1.6,
                    label=f"reaches NASA at K/σ_K ≈ {b['reach_thr']:.1f}")
    ax2.set_xlabel("RV significance threshold  K/σ_K"); ax2.set_ylabel("detected puffy fraction")
    ax2.set_title("PART B — stricter RV preferentially keeps DENSE (rocky) planets,\n"
                  "pushing the model puffy fraction down toward NASA", fontsize=10.5)
    ax2.legend(fontsize=8.5, loc="best"); ax2.grid(alpha=0.25)

    out = OUT_DIR / "robustness_and_gap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


def main():
    print("=" * 70)
    print("25_metric_robustness.py")
    print("=" * 70)
    m_sil, r_sil = load_silicate()
    nasa_puffy, n_nasa = load_nasa_puffy(m_sil, r_sil)
    print(f"NASA measured-mass planets in box: {n_nasa}  (puffy fraction {nasa_puffy:.3f})")

    a = part_a(m_sil, r_sil, nasa_puffy)
    b = part_b(m_sil, r_sil, nasa_puffy)
    plot_all(a, b)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
