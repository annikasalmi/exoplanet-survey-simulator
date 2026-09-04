"""
rv_detector_check.py

RV detector verification & calibration — the RV sibling of:
  * script 33  (toy TESS SNR vs official SPOC/TCE MES calibration), and
  * scripts 45/46  (P-Pop pipeline stage checks).

K is the RV model's "MES": every detection decision flows from
    K  ->  sigma_RV (instr + photon + jitter)  ->  K/sigma_K  ->  threshold.

FIGURE 1 — K calibration against PUBLISHED semi-amplitudes (script-33 mirror)
    Downloads NASA PSCompPars planets with a published pl_rvamp (the K actually
    measured by RV instruments) and compares the model K computed from the same
    planet's (M sin i, P, e, M_star):
      A. model K vs published K, log-log with 1:1 line (colored by stype)
      B. RVData pass fraction vs published-K bin (recovery rises with signal)
      C. calibration ratio K_model / K_published per bin (1 = perfect physics)

FIGURE 2 — pipeline stage check on P-Pop (45/46 mirror)
      A. K distribution, detected vs missed, with the effective K floor band
      B. recovery fraction vs K/sigma_K bin (threshold marked)
      C. sigma_RV decomposition by spectral type (instr / photon / jitter)
      D. loss budget: reason categories overall + by stype
      E. band magnitude vs sigma_RV, detected vs missed (the brightness wall)

Run from repo root:
    python scripts/rv_detector_check.py
    python scripts/rv_detector_check.py --instrument NIRPS --n-catalogs 80
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from urllib.parse import quote

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
PPOP_DIR = ROOT / "run" / "tess" / "data" / "Gaia_C_F_K_combined_cdpp_v1"
RVAMP_CACHE = ROOT / "run" / "kepler" / "data" / "NASA" / "NASA_PSCompPars_rvamp_calibration.csv"
OUT_DIR = ROOT / "my_outputs" / "rv_detector_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FIG_DIR = ROOT / "paper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("rv_data", ROOT / "detectors" / "rv_data.py")
_rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rv)
RVData = _rv.RVData

STYPE_COLORS = {"F": "#e6ab02", "G": "#66a61e", "K": "#7570b3", "M": "#d95f02", "A": "0.6", "Unknown": "0.8"}
PPOP_COLS = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p",
             "radius_s", "mass_s", "teff_s", "l_sun", "distance_s", "flux_p", "stype"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 260,
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
})


# ───────────────────────── Figure 1: K vs published pl_rvamp ──────────────────

RVAMP_QUERY = """
SELECT pl_name, hostname, discoverymethod, tran_flag,
       pl_orbper, pl_orbeccen, pl_orbincl,
       pl_rvamp, pl_rvamperr1, pl_rvamperr2,
       pl_bmasse, pl_bmassprov, pl_msinie,
       pl_rade, pl_insol,
       st_mass, st_rad, st_teff, st_lum, sy_dist, sy_vmag
FROM pscomppars
WHERE pl_rvamp IS NOT NULL AND pl_rvamp > 0
  AND pl_orbper IS NOT NULL AND st_mass IS NOT NULL
"""


def load_rvamp_sample() -> pd.DataFrame:
    if RVAMP_CACHE.exists():
        print(f"Loading published-K sample from cache: {RVAMP_CACHE.name}")
        df = pd.read_csv(RVAMP_CACHE)
    else:
        url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
               + quote(RVAMP_QUERY) + "&format=csv")
        print("Downloading PSCompPars planets with published pl_rvamp ...")
        df = pd.read_csv(url)
        RVAMP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RVAMP_CACHE, index=False)
        print(f"Saved cache: {RVAMP_CACHE}")
    for c in df.columns:
        if c not in ("pl_name", "hostname", "discoverymethod", "pl_bmassprov"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  Published-K sample: {len(df):,} planets")
    return df


def stype_from_teff(t):
    if pd.isna(t): return "Unknown"
    t = float(t)
    if t >= 7500: return "A"
    if t >= 6000: return "F"
    if t >= 5200: return "G"
    if t >= 3700: return "K"
    if t > 0: return "M"
    return "Unknown"


def model_k_from_published(df: pd.DataFrame) -> pd.Series:
    """K from the canonical formula using the planet's own published parameters.

    Uses M sin i directly when present (it already includes the projection RV saw);
    falls back to best mass (most of those are transiting, i ~ 90 deg).
    """
    msini = df["pl_msinie"].where(df["pl_msinie"].notna(), df["pl_bmasse"])
    m_jup = msini / 317.828
    p_yr = df["pl_orbper"] / 365.25
    ecc = df["pl_orbeccen"].fillna(0.0).clip(0, 0.95)
    return (28.4329 * m_jup * df["st_mass"] ** (-2 / 3) * p_yr ** (-1 / 3)
            / np.sqrt(1 - ecc ** 2))


def plot_k_calibration(df: pd.DataFrame, args) -> dict:
    df = df.dropna(subset=["pl_rvamp", "pl_orbper", "st_mass"]).copy()
    df = df[(df["pl_rvamp"] > 0) & (df["pl_msinie"].notna() | df["pl_bmasse"].notna())]
    df["k_model"] = model_k_from_published(df)
    df = df[np.isfinite(df["k_model"]) & (df["k_model"] > 0)].copy()
    df["stype"] = df["st_teff"].apply(stype_from_teff)
    ratio = df["k_model"] / df["pl_rvamp"]
    print(f"  K calibration sample: {len(df):,}  |  median K_model/K_pub = {ratio.median():.3f}  "
          f"(16-84%: {ratio.quantile(0.16):.3f}-{ratio.quantile(0.84):.3f})")

    # Run the full RV detector on the same planets (for panel B recovery).
    rv = RVData(df.rename(columns={
        "pl_orbper": "p_orb", "pl_orbeccen": "ecc_p", "pl_orbincl": "inc_p",
        "pl_msinie": "msini_p", "pl_bmasse": "mass_p", "pl_rade": "radius_p",
        "st_mass": "mass_s", "st_rad": "radius_s", "st_teff": "teff_s",
        "st_lum": "st_lum_log10", "sy_dist": "distance_s", "sy_vmag": "vmag",
    }), source="pscomppars", instrument=args.instrument if args.instrument.lower() != "best" else "HARPS",
        apply_sini=True, n_obs=args.n_obs, snr_threshold=args.snr_threshold,
        validate_for_detection=False)
    cat = rv.determine_detectable()
    df["rv_pass"] = cat["rv_detected"].to_numpy()
    df["sigma_k_model"] = pd.to_numeric(cat["rv_sigma_K_ms"], errors="coerce").to_numpy()
    df["snr_model"] = pd.to_numeric(cat["rv_snr"], errors="coerce").to_numpy()

    # Same 3-in-1 template as the Kepler (47) / TESS (48) calibration figures:
    # scatter spans the full-height left column; recovery + ratio stack on the right.
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.42, wspace=0.32)

    # A. model K vs published K
    axA = fig.add_subplot(gs[:, 0])
    for st in ["F", "G", "K", "M", "A", "Unknown"]:
        s = df[df["stype"] == st]
        if len(s) == 0:
            continue
        axA.scatter(s["pl_rvamp"], s["k_model"], s=7, alpha=0.45,
                    color=STYPE_COLORS[st], label=f"{st} (N={len(s)})", linewidths=0)
    lims = [0.3, df["pl_rvamp"].quantile(0.999) * 2]
    axA.plot(lims, lims, "k--", lw=1.2, label="1:1 (perfect physics)")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlim(lims); axA.set_ylim(lims)
    axA.set_xlabel("Published K  [m/s]")
    axA.set_ylabel("Model K  [m/s]")
    axA.set_title("A. Model K vs published K")
    axA.legend(fontsize=12, loc="upper left"); axA.grid(alpha=0.2, which="both")

    # B. recovery vs published K bin
    axB = fig.add_subplot(gs[0, 1])
    kbins = [0, 1, 2, 5, 10, 30, 100, 1e5]
    klabels = ["<1", "1-2", "2-5", "5-10", "10-30", "30-100", ">100"]
    df["kbin"] = pd.cut(df["pl_rvamp"], kbins, labels=klabels)
    s = df.groupby("kbin", observed=True).agg(n=("rv_pass", "size"), f=("rv_pass", "mean"))
    axB.plot(range(len(s)), s["f"], "o-", color="#2060c0")
    for i, (lab, row) in enumerate(s.iterrows()):
        axB.annotate(f"{row['f']:.0%}\nN={int(row['n'])}", (i, row["f"]),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    axB.set_xticks(range(len(s))); axB.set_xticklabels(s.index, rotation=25)
    axB.set_ylim(-0.05, 1.15); axB.set_xlabel("Published K bin [m/s]")
    axB.set_ylabel("Model pass fraction")
    axB.set_title("B. Recovery vs signal strength")
    axB.grid(alpha=0.25)

    # C. calibration ratio per bin
    axC = fig.add_subplot(gs[1, 1])
    med = df.groupby("kbin", observed=True)["k_model"].median() / 1.0
    stats = df.groupby("kbin", observed=True).apply(
        lambda g: pd.Series({
            "med": (g["k_model"] / g["pl_rvamp"]).median(),
            "lo": (g["k_model"] / g["pl_rvamp"]).quantile(0.16),
            "hi": (g["k_model"] / g["pl_rvamp"]).quantile(0.84)}),
        include_groups=False)
    axC.errorbar(range(len(stats)), stats["med"],
                 yerr=[stats["med"] - stats["lo"], stats["hi"] - stats["med"]],
                 fmt="o-", color="#2060c0", capsize=4)
    axC.axhline(1.0, color="black", ls="--", lw=1)
    axC.set_yscale("log")
    axC.set_xticks(range(len(stats))); axC.set_xticklabels(stats.index, rotation=25)
    axC.set_xlabel("Published K bin [m/s]"); axC.set_ylabel("Model / published K")
    axC.set_title("C. K ratio (bars = 16-84%)")
    axC.grid(alpha=0.25, which="both")

    out = OUT_DIR / "rv_k_vs_published_rvamp.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "rv_k_vs_published_rvamp.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    print(f"  Saved paper copy: {PAPER_FIG_DIR / 'rv_k_vs_published_rvamp.png'}")
    df.to_csv(OUT_DIR / "rv_k_calibration_matched_rows.csv", index=False)

    plot_sigmak_validation(df, args)
    plot_sigmak_3in1(df, args)
    return {"n": len(df), "ratio_med": float(ratio.median()),
            "ratio_16": float(ratio.quantile(0.16)), "ratio_84": float(ratio.quantile(0.84))}


def plot_sigmak_validation(df: pd.DataFrame, args) -> None:
    """Noise-side calibration: model sigma_K (sqrt(2/N)*sigma_RV) vs the PUBLISHED
    K uncertainty, plus a signal-to-noise calibration that folds K and sigma_K.

    The model predicts the BEST-CASE achievable precision of a single uniform
    modern campaign (fixed N_obs, one instrument, per-stype jitter floor).  The
    published pl_rvamperr, by contrast, records the actual precision of hugely
    heterogeneous historical campaigns (N ~ 10-1000, ELODIE -> ESPRESSO eras,
    quiet -> active hosts).  So model <= published (ratio < 1) is the EXPECTED,
    correct relationship, not "too optimistic": a best-case floor should sit
    below real achieved error bars.  A raw 1:1 is the wrong target because
    pl_rvamperr mixes populations the model was never meant to reproduce.

    Three panels:
      A. FLOOR test — fraction with model sigma_K <= published sigma_K (a valid
         lower bound should hold for nearly all planets).
      B. sigma_K ratio per K bin, highlighting the CALIBRATION SET (small,
         near-threshold planets with a real RV M sin i mass) where it should hug 1.
      C. S/N calibration — model K/sigma_K vs published pl_rvamp/pl_rvamperr.
         This is the decision-relevant quantity (detection = S/N > threshold);
         because K spans orders of magnitude it does NOT collapse into the flat
         band that pure sigma_K-vs-sigma_K does.
    """
    # Calibration set: small near-threshold semi-amplitude (K small) + a real RV
    # M sin i mass (pl_msinie present) -- the modern small-planet programmes the
    # HARPS/NIRPS constants are tuned to, not decades-old bright-giant discoveries
    # whose error bars were never minimised.
    K_CAL_CUT_MS = 15.0

    d = df.copy()
    err = pd.concat([d["pl_rvamperr1"].abs(), d["pl_rvamperr2"].abs()], axis=1).mean(axis=1)
    d["sigma_k_pub"] = pd.to_numeric(err, errors="coerce")
    d = d[(d["sigma_k_pub"] > 0) & np.isfinite(d["sigma_k_model"]) & (d["sigma_k_model"] > 0)].copy()

    d["cal_set"] = d["pl_rvamp"].le(K_CAL_CUT_MS) & d["pl_msinie"].notna()
    floor_ok = (d["sigma_k_model"] <= d["sigma_k_pub"])
    cal = d[d["cal_set"]]
    ratio_cal = cal["sigma_k_model"] / cal["sigma_k_pub"]
    floor_frac_all = float(floor_ok.mean())
    floor_frac_cal = float((cal["sigma_k_model"] <= cal["sigma_k_pub"]).mean()) if len(cal) else float("nan")
    print(f"  sigma_K validation: N={len(d):,}  floor-pass (model<=published) all={floor_frac_all:.0%}  "
          f"cal-set={floor_frac_cal:.0%}  (N_cal={len(cal):,})")
    print(f"    cal-set sigma_K ratio median = {ratio_cal.median():.2f} "
          f"(16-84%: {ratio_cal.quantile(0.16):.2f}-{ratio_cal.quantile(0.84):.2f})  "
          f"[K<={K_CAL_CUT_MS:g} m/s & M sin i]")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # ── A. scatter: floor region shaded; calibration-set points highlighted ────
    axA = axes[0]
    lims = [max(1e-3, d["sigma_k_pub"].quantile(0.002)), d["sigma_k_pub"].quantile(0.998)]
    grid = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 100)
    # Shade y <= x (model <= published) = floor holds.
    axA.fill_between(grid, lims[0], grid, color="#2060c0", alpha=0.06, zorder=0,
                     label="model ≤ published (floor holds)")
    # Context points (giants / non-RV mass): grey.
    o = d[~d["cal_set"]]
    axA.scatter(o["sigma_k_pub"], o["sigma_k_model"], s=6, alpha=0.18,
                color="0.6", linewidths=0, label=f"context: giants / non-RV mass (N={len(o)})")
    # Calibration-set points: coloured by stype.
    for st in ["F", "G", "K", "M"]:
        s = cal[cal["stype"] == st]
        if len(s):
            axA.scatter(s["sigma_k_pub"], s["sigma_k_model"], s=9, alpha=0.55,
                        color=STYPE_COLORS[st], label=f"{st} calibration set (N={len(s)})", linewidths=0)
    axA.plot(lims, lims, "k--", lw=1.2, label="1:1")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlim(lims); axA.set_ylim(lims)
    axA.set_xlabel("Published σ_K  [m/s]  (pl_rvamperr)")
    axA.set_ylabel(f"Model σ_K  [m/s]  (√(2/{args.n_obs})·σ_RV)")
    axA.set_title("A. σ_K floor test: model = best-case achievable precision")
    axA.text(0.03, 0.97,
             f"floor-pass (model ≤ published)\n  all: {floor_frac_all:.0%}   calibration set: {floor_frac_cal:.0%}",
             transform=axA.transAxes, va="top", ha="left", fontsize=7.5,
             bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.6"))
    axA.legend(fontsize=6.5, loc="lower right"); axA.grid(alpha=0.2, which="both")

    # ── B. sigma_K ratio per K bin: calibration-set bins are the claim ────────
    axB = axes[1]
    kbins = [0, 1, 2, 5, 10, 30, 100, 1e5]
    klabels = ["<1", "1-2", "2-5", "5-10", "10-30", "30-100", ">100"]
    d["kbin"] = pd.cut(d["pl_rvamp"], kbins, labels=klabels)
    stats = d.groupby("kbin", observed=True).apply(
        lambda g: pd.Series({"med": (g["sigma_k_model"] / g["sigma_k_pub"]).median(),
                             "lo": (g["sigma_k_model"] / g["sigma_k_pub"]).quantile(0.16),
                             "hi": (g["sigma_k_model"] / g["sigma_k_pub"]).quantile(0.84)}),
        include_groups=False)
    x = np.arange(len(stats))
    # bins whose upper edge is within the calibration-set cut.
    in_cal_bin = [kbins[i + 1] <= K_CAL_CUT_MS for i in range(len(stats))]
    colors = ["#2060c0" if r else "0.6" for r in in_cal_bin]
    # <=1 band: this is the EXPECTED region (model is a best-case floor).
    axB.axhspan(1e-3, 1.0, color="#2060c0", alpha=0.06, zorder=0)
    for i in range(len(stats)):
        axB.errorbar(x[i], stats["med"].iloc[i],
                     yerr=[[stats["med"].iloc[i] - stats["lo"].iloc[i]],
                           [stats["hi"].iloc[i] - stats["med"].iloc[i]]],
                     fmt="o", color=colors[i], capsize=4)
    axB.plot(x, stats["med"], "-", color="0.4", lw=0.8, zorder=1)
    axB.axhline(1.0, color="black", ls="--", lw=1)
    axB.set_yscale("log")
    axB.set_xticks(x); axB.set_xticklabels(stats.index, rotation=25)
    axB.set_xlabel("Published K bin [m/s]")
    axB.set_ylabel("σ_K model / σ_K published")
    axB.set_title(f"B. σ_K ratio per K bin  (blue = calibration set, K≤{K_CAL_CUT_MS:g})\n"
                  "shaded ≤1 = expected: model is a best-case floor")
    axB.grid(alpha=0.25, which="both")

    # ── C. S/N calibration: model K/sigma_K vs published pl_rvamp/pl_rvamperr ──
    axC = axes[2]
    d["snr_pub"] = d["pl_rvamp"] / d["sigma_k_pub"]
    d["snr_mod"] = pd.to_numeric(d["snr_model"], errors="coerce")
    m = np.isfinite(d["snr_pub"]) & np.isfinite(d["snr_mod"]) & (d["snr_pub"] > 0) & (d["snr_mod"] > 0)
    dd = d[m]
    cc = dd[dd["cal_set"]]
    slims = [max(0.5, dd["snr_pub"].quantile(0.01)), dd["snr_pub"].quantile(0.99) * 1.5]
    ratio_snr = cc["snr_mod"] / cc["snr_pub"]
    print(f"  S/N calibration: cal-set median model/published S/N = {ratio_snr.median():.2f} "
          f"(16-84%: {ratio_snr.quantile(0.16):.2f}-{ratio_snr.quantile(0.84):.2f}, N={len(cc):,})")
    oo = dd[~dd["cal_set"]]
    axC.scatter(oo["snr_pub"], oo["snr_mod"], s=6, alpha=0.18, color="0.6",
                linewidths=0, label=f"context (N={len(oo)})")
    for st in ["F", "G", "K", "M"]:
        s = cc[cc["stype"] == st]
        if len(s):
            axC.scatter(s["snr_pub"], s["snr_mod"], s=9, alpha=0.55,
                        color=STYPE_COLORS[st], label=f"{st} calibration set (N={len(s)})", linewidths=0)
    axC.plot(slims, slims, "k--", lw=1.2, label="1:1")
    thr = args.snr_threshold
    axC.axhline(thr, color="red", ls=":", lw=1.0)
    axC.axvline(thr, color="red", ls=":", lw=1.0, label=f"{thr:g}σ threshold")
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlim(slims); axC.set_ylim(slims)
    axC.set_xlabel("Published S/N   (pl_rvamp / pl_rvamperr)")
    axC.set_ylabel("Model S/N   (K / σ_K)")
    axC.set_title("C. Signal-to-noise calibration (the detection quantity)\n"
                  "folds K and σ_K; not crushed into a flat band")
    axC.legend(fontsize=6.5, loc="lower right"); axC.grid(alpha=0.2, which="both")

    fig.suptitle(f"RV noise & S/N calibration  ({args.instrument}, N_obs={args.n_obs})",
                 fontsize=12)
    out = OUT_DIR / "rv_sigmaK_vs_published.png"
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


# ───────────────────────── Figure 2: P-Pop pipeline stage check ───────────────

def plot_sigmak_3in1(df: pd.DataFrame, args) -> dict:
    """Paper appendix noise_K (sigma_K) 3-in-1 block (A-C): (A) sigma_K floor test,
    (B) detection S/N + small-planet recovery, (C) per-spectral-type jitter constants
    grounded to Bellotti & Korhonen (2021).  Sits next to the K signal 3-in-1 in the
    2x2 calibration figure."""
    LEG = 11  # larger legend text (paper request)
    d = df.copy()
    err = pd.concat([d["pl_rvamperr1"].abs(), d["pl_rvamperr2"].abs()], axis=1).mean(axis=1)
    d["sigma_k_pub"] = pd.to_numeric(err, errors="coerce")
    d["stype"] = d["st_teff"].apply(stype_from_teff)
    thr = args.snr_threshold

    small = d[(pd.to_numeric(d["pl_rade"], errors="coerce") <= 2.2) & (d["pl_rvamp"] > 0)].copy()
    rec_model = float(small["rv_pass"].mean()) if len(small) else float("nan")
    snr_pub_small = small["pl_rvamp"] / small["sigma_k_pub"]
    rec_pub = float((snr_pub_small >= thr).mean()) if len(small) else float("nan")
    print(f"  small-planet recovery (R<=2.2, N={len(small)}): model {rec_model:.1%} at >={thr:g}sigma "
          f"vs published S/N>={thr:g} {rec_pub:.1%}")

    db = d[(d["sigma_k_pub"] > 0) & np.isfinite(d["sigma_k_model"]) & (d["sigma_k_model"] > 0)].copy()
    db["cal_set"] = db["pl_rvamp"].le(15.0) & db["pl_msinie"].notna()
    floor_all = float((db["sigma_k_model"] <= db["sigma_k_pub"]).mean())

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.42, wspace=0.32)

    # A. sigma_K floor test (noise)
    axA = fig.add_subplot(gs[:, 0])
    lims2 = [max(1e-3, db["sigma_k_pub"].quantile(0.002)), db["sigma_k_pub"].quantile(0.998)]
    grid = np.logspace(np.log10(lims2[0]), np.log10(lims2[1]), 100)
    axA.fill_between(grid, lims2[0], grid, color="#2060c0", alpha=0.06, zorder=0,
                     label="model ≤ published (floor holds)")
    o = db[~db["cal_set"]]
    axA.scatter(o["sigma_k_pub"], o["sigma_k_model"], s=6, alpha=0.15, color="0.6",
                linewidths=0, label=f"context (N={len(o)})")
    for st in ["F", "G", "K", "M"]:
        s = db[db["cal_set"] & (db["stype"] == st)]
        if len(s):
            axA.scatter(s["sigma_k_pub"], s["sigma_k_model"], s=10, alpha=0.55,
                        color=STYPE_COLORS[st], label=f"{st} cal-set (N={len(s)})", linewidths=0)
    axA.plot(lims2, lims2, "k--", lw=1.2, label="1:1")
    axA.set_xscale("log"); axA.set_yscale("log"); axA.set_xlim(lims2); axA.set_ylim(lims2)
    axA.set_xlabel("Published σ_K  [m/s]  (pl_rvamperr)")
    axA.set_ylabel(f"Model σ_K  [m/s]  (√(2/{args.n_obs})·σ_RV)")
    axA.set_title(f"A. σ_K floor test  (floor-pass {floor_all:.0%})")
    axA.legend(fontsize=LEG, loc="lower right"); axA.grid(alpha=0.2, which="both")

    # B. detection S/N + small-planet recovery
    axB = fig.add_subplot(gs[0, 1])
    db["snr_pub"] = db["pl_rvamp"] / db["sigma_k_pub"]
    db["snr_mod"] = pd.to_numeric(db["snr_model"], errors="coerce")
    m = np.isfinite(db["snr_pub"]) & np.isfinite(db["snr_mod"]) & (db["snr_pub"] > 0) & (db["snr_mod"] > 0)
    dd = db[m]; cc = dd[dd["cal_set"]]
    slims = [max(0.5, dd["snr_pub"].quantile(0.01)), dd["snr_pub"].quantile(0.99) * 1.5]
    oo = dd[~dd["cal_set"]]
    axB.scatter(oo["snr_pub"], oo["snr_mod"], s=6, alpha=0.15, color="0.6",
                linewidths=0, label=f"context (N={len(oo)})")
    for st in ["F", "G", "K", "M"]:
        s = cc[cc["stype"] == st]
        if len(s):
            axB.scatter(s["snr_pub"], s["snr_mod"], s=10, alpha=0.55, color=STYPE_COLORS[st],
                        label=f"{st} cal-set (N={len(s)})", linewidths=0)
    axB.plot(slims, slims, "k--", lw=1.2, label="1:1")
    axB.axhline(thr, color="red", ls=":", lw=1.0)
    axB.axvline(thr, color="red", ls=":", lw=1.0, label=f"{thr:g}σ threshold")
    axB.set_xscale("log"); axB.set_yscale("log"); axB.set_xlim(slims); axB.set_ylim(slims)
    axB.set_xlabel("Published S/N  (pl_rvamp / pl_rvamperr)")
    axB.set_ylabel("Model S/N  (K / σ_K)")
    axB.set_title(f"B. Detection S/N: small-planet recovery {rec_model:.0%} vs {rec_pub:.0%} pub.")
    axB.legend(fontsize=LEG, loc="lower right"); axB.grid(alpha=0.2, which="both")

    # C. per-spectral-type jitter constants (Bellotti & Korhonen 2021)
    axC = fig.add_subplot(gs[1, 1])
    jit = RVData.INSTRUMENT_PRESETS["HARPS"]["jitter_by_stype"]
    order = ["F", "G", "K", "M"]
    vals = [jit[s] for s in order]
    axC.bar(order, vals, color=[STYPE_COLORS[s] for s in order], alpha=0.85, edgecolor="k")
    for i, v in enumerate(vals):
        axC.annotate(f"{v:g} m/s", (i, v), textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=12)
    axC.set_ylim(0, max(vals) * 1.25)
    axC.set_ylabel("HARPS activity jitter σ_jit  [m/s]")
    axC.set_xlabel("Host spectral type")
    axC.set_title("C. Per-type jitter (Bellotti & Korhonen 2021,\nTable 3; p2p→RMS ÷2.8)")
    axC.grid(alpha=0.25, axis="y")

    fig.suptitle(f"RV noise ($\\sigma_K$) calibration  ({args.instrument}, N_obs={args.n_obs}, ≥{thr:g}σ)",
                 fontsize=13)
    out = OUT_DIR / "rv_sigmaK_3in1.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "rv_sigmaK_3in1.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper σ_K 3-in-1: {PAPER_FIG_DIR / 'rv_sigmaK_3in1.png'}")
    return {"rec_model": rec_model, "rec_pub": rec_pub, "floor_all": floor_all}


def _unused_plot_rv_2x2_paper(df: pd.DataFrame, args) -> dict:
    """Retired: superseded by the K 3-in-1 + plot_sigmak_3in1 (2x2 calibration figure)."""
    d = df.copy()
    err = pd.concat([d["pl_rvamperr1"].abs(), d["pl_rvamperr2"].abs()], axis=1).mean(axis=1)
    d["sigma_k_pub"] = pd.to_numeric(err, errors="coerce")
    d["stype"] = d["st_teff"].apply(stype_from_teff)
    thr = args.snr_threshold

    # Recovery on the real small-planet sample (R <= 2.2 Re, published K).
    small = d[(pd.to_numeric(d["pl_rade"], errors="coerce") <= 2.2) & (d["pl_rvamp"] > 0)].copy()
    rec_model = float(small["rv_pass"].mean()) if len(small) else float("nan")
    snr_pub_small = small["pl_rvamp"] / small["sigma_k_pub"]
    rec_pub = float((snr_pub_small >= thr).mean()) if len(small) else float("nan")
    print(f"  small-planet recovery (R<=2.2, N={len(small)}): model {rec_model:.1%} at >={thr:g}sigma "
          f"vs published S/N>={thr:g} {rec_pub:.1%}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # A. signal: model K vs published K
    axA = axes[0, 0]
    for st in ["F", "G", "K", "M", "A", "Unknown"]:
        s = d[d["stype"] == st]
        if len(s):
            axA.scatter(s["pl_rvamp"], s["k_model"], s=7, alpha=0.45, color=STYPE_COLORS[st],
                        label=f"{st} (N={len(s)})", linewidths=0)
    lims = [0.3, d["pl_rvamp"].quantile(0.999) * 2]
    axA.plot(lims, lims, "k--", lw=1.2, label="1:1")
    axA.set_xscale("log"); axA.set_yscale("log"); axA.set_xlim(lims); axA.set_ylim(lims)
    axA.set_xlabel("Published K  [m/s]"); axA.set_ylabel("Model K  [m/s]")
    axA.set_title(f"A. Signal: model K vs published K  (median ratio "
                  f"{(d['k_model'] / d['pl_rvamp']).median():.3f})")
    axA.legend(fontsize=8, loc="upper left"); axA.grid(alpha=0.2, which="both")

    # B. noise: sigma_K floor test
    axB = axes[0, 1]
    db = d[(d["sigma_k_pub"] > 0) & np.isfinite(d["sigma_k_model"]) & (d["sigma_k_model"] > 0)].copy()
    db["cal_set"] = db["pl_rvamp"].le(15.0) & db["pl_msinie"].notna()
    floor_all = float((db["sigma_k_model"] <= db["sigma_k_pub"]).mean())
    lims2 = [max(1e-3, db["sigma_k_pub"].quantile(0.002)), db["sigma_k_pub"].quantile(0.998)]
    grid = np.logspace(np.log10(lims2[0]), np.log10(lims2[1]), 100)
    axB.fill_between(grid, lims2[0], grid, color="#2060c0", alpha=0.06, zorder=0,
                     label="model ≤ published (floor holds)")
    o = db[~db["cal_set"]]
    axB.scatter(o["sigma_k_pub"], o["sigma_k_model"], s=6, alpha=0.15, color="0.6",
                linewidths=0, label=f"context (N={len(o)})")
    for st in ["F", "G", "K", "M"]:
        s = db[db["cal_set"] & (db["stype"] == st)]
        if len(s):
            axB.scatter(s["sigma_k_pub"], s["sigma_k_model"], s=9, alpha=0.55,
                        color=STYPE_COLORS[st], label=f"{st} cal-set (N={len(s)})", linewidths=0)
    axB.plot(lims2, lims2, "k--", lw=1.2, label="1:1")
    axB.set_xscale("log"); axB.set_yscale("log"); axB.set_xlim(lims2); axB.set_ylim(lims2)
    axB.set_xlabel("Published σ_K  [m/s]  (pl_rvamperr)")
    axB.set_ylabel(f"Model σ_K  [m/s]  (√(2/{args.n_obs})·σ_RV)")
    axB.set_title(f"B. Noise: σ_K floor test  (floor-pass {floor_all:.0%})")
    axB.text(0.03, 0.97, "model = best-case floor\n→ model ≤ published expected",
             transform=axB.transAxes, va="top", ha="left", fontsize=8,
             bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.6"))
    axB.legend(fontsize=7, loc="lower right"); axB.grid(alpha=0.2, which="both")

    # C. S/N calibration + small-planet recovery
    axC = axes[1, 0]
    db["snr_pub"] = db["pl_rvamp"] / db["sigma_k_pub"]
    db["snr_mod"] = pd.to_numeric(db["snr_model"], errors="coerce")
    m = np.isfinite(db["snr_pub"]) & np.isfinite(db["snr_mod"]) & (db["snr_pub"] > 0) & (db["snr_mod"] > 0)
    dd = db[m]; cc = dd[dd["cal_set"]]
    slims = [max(0.5, dd["snr_pub"].quantile(0.01)), dd["snr_pub"].quantile(0.99) * 1.5]
    oo = dd[~dd["cal_set"]]
    axC.scatter(oo["snr_pub"], oo["snr_mod"], s=6, alpha=0.15, color="0.6",
                linewidths=0, label=f"context (N={len(oo)})")
    for st in ["F", "G", "K", "M"]:
        s = cc[cc["stype"] == st]
        if len(s):
            axC.scatter(s["snr_pub"], s["snr_mod"], s=9, alpha=0.55, color=STYPE_COLORS[st],
                        label=f"{st} cal-set (N={len(s)})", linewidths=0)
    axC.plot(slims, slims, "k--", lw=1.2, label="1:1")
    axC.axhline(thr, color="red", ls=":", lw=1.0)
    axC.axvline(thr, color="red", ls=":", lw=1.0, label=f"{thr:g}σ threshold")
    axC.set_xscale("log"); axC.set_yscale("log"); axC.set_xlim(slims); axC.set_ylim(slims)
    axC.set_xlabel("Published S/N  (pl_rvamp / pl_rvamperr)")
    axC.set_ylabel("Model S/N  (K / σ_K)")
    axC.set_title(f"C. Detection S/N: small-planet recovery "
                  f"{rec_model:.0%} model vs {rec_pub:.0%} published")
    axC.legend(fontsize=7, loc="lower right"); axC.grid(alpha=0.2, which="both")

    # D. per-spectral-type jitter constants (Bellotti & Korhonen 2021)
    axD = axes[1, 1]
    jit = RVData.INSTRUMENT_PRESETS["HARPS"]["jitter_by_stype"]
    order = ["F", "G", "K", "M"]
    vals = [jit[s] for s in order]
    axD.bar(order, vals, color=[STYPE_COLORS[s] for s in order], alpha=0.85, edgecolor="k")
    for i, v in enumerate(vals):
        axD.annotate(f"{v:g} m/s", (i, v), textcoords="offset points", xytext=(0, 4),
                     ha="center", fontsize=11)
    axD.set_ylim(0, max(vals) * 1.25)
    axD.set_ylabel("HARPS activity jitter σ_jit  [m/s]")
    axD.set_xlabel("Host spectral type")
    axD.set_title("D. Per-type jitter (one constant per type)\n"
                  "Bellotti & Korhonen 2021, Table 3; p2p→RMS ÷2.8")
    axD.grid(alpha=0.25, axis="y")

    fig.suptitle(f"RV detector calibration: signal (K) and noise (σ_K)  "
                 f"({args.instrument}, N_obs={args.n_obs}, ≥{thr:g}σ)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "rv_calibration_2x2.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "rv_calibration_2x2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper 2x2: {PAPER_FIG_DIR / 'rv_calibration_2x2.png'}")
    return {"rec_model": rec_model, "rec_pub": rec_pub, "floor_all": floor_all}


def load_ppop(n_catalogs):
    files = sorted(PPOP_DIR.glob("tess_catalog_*.csv"))[:n_catalogs]
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in PPOP_COLS, low_memory=False)
                    for f in files], ignore_index=True)
    for c in ["radius_p", "mass_p", "p_orb", "flux_p"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  Loaded {len(df):,} P-Pop planets from {len(files)} catalogs")
    return df


def plot_pipeline_check(ppop: pd.DataFrame, args, inst: str) -> None:
    # The funnel decomposes sigma_RV by spectral type, which is inherently
    # per-instrument; "best" (per-planet max of HARPS/NIRPS) is not one
    # decomposable instrument, so the caller passes a concrete instrument here.
    rv = RVData(ppop, source="ppop", instrument=inst, apply_sini=True,
                n_obs=args.n_obs, snr_threshold=args.snr_threshold,
                validate_for_detection=False)
    c = rv.determine_detectable()
    for col in ["rv_k_ms", "rv_sigma_ms", "rv_sigma_phot_ms", "rv_sigma_jitter_ms",
                "rv_sigma_K_ms", "rv_snr", "rv_mag"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    det = c["rv_detected"].astype(bool)

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

    # A. K distribution det vs missed + effective floor band
    axA = fig.add_subplot(gs[0, 0])
    k = c["rv_k_ms"].clip(lower=1e-3)
    bins = np.logspace(np.log10(0.005), np.log10(k.quantile(0.999)), 55)
    axA.hist(k[~det], bins=bins, color="#e07020", alpha=0.6, label=f"Missed ({int((~det).sum()):,})")
    axA.hist(k[det], bins=bins, color="#2060c0", alpha=0.85, label=f"Detected ({int(det.sum()):,})")
    kfloor = args.snr_threshold * c.loc[det | ~det, "rv_sigma_K_ms"]
    lo, hi = np.nanpercentile(kfloor, [16, 84])
    axA.axvspan(lo, hi, color="red", alpha=0.12, label=f"K floor (16-84%): {lo:.1f}-{hi:.1f} m/s")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("Model K [m/s]"); axA.set_ylabel("Count (log)")
    axA.set_title(f"A. K — the RV 'MES'  ({inst}, P-Pop)")
    axA.legend(fontsize=7); axA.grid(alpha=0.2)

    # B. recovery vs K/sigma_K bins
    axB = fig.add_subplot(gs[0, 1])
    snrbins = [0, 1, 3, args.snr_threshold, 8, 15, 40, np.inf]
    snrlabels = ["<1", "1-3", f"3-{args.snr_threshold:g}", f"{args.snr_threshold:g}-8", "8-15", "15-40", ">40"]
    sb = pd.cut(c["rv_snr"], snrbins, labels=snrlabels)
    s = c.groupby(sb, observed=True).agg(n=("rv_detected", "size"), f=("rv_detected", "mean"))
    axB.bar(range(len(s)), s["f"], color="#2060c0", alpha=0.8, width=0.6)
    for i, (lab, row) in enumerate(s.iterrows()):
        axB.text(i, row["f"] + 0.02, f"{row['f']:.0%}\nN={int(row['n']):,}",
                 ha="center", va="bottom", fontsize=6.5)
    axB.set_xticks(range(len(s))); axB.set_xticklabels(s.index, rotation=25)
    axB.set_ylim(0, 1.18); axB.set_xlabel("K/σ_K bin"); axB.set_ylabel("Detected fraction")
    axB.set_title(f"B. Recovery vs K/σ_K (threshold {args.snr_threshold:g})")
    axB.grid(axis="y", alpha=0.2)

    # C. sigma decomposition by stype
    axC = fig.add_subplot(gs[0, 2])
    stypes = ["F", "G", "K", "M"]
    comp = {part: [] for part in ["instr", "photon", "jitter"]}
    for st in stypes:
        s = c[c["stype"] == st]
        comp["instr"].append(rv.sigma_instr_ms)
        comp["photon"].append(float(s["rv_sigma_phot_ms"].median()))
        comp["jitter"].append(float(s["rv_sigma_jitter_ms"].median()))
    x = np.arange(len(stypes)); w = 0.26
    for i, (part, col) in enumerate([("instr", "0.5"), ("photon", "#e07020"), ("jitter", "#7570b3")]):
        axC.bar(x + (i - 1) * w, comp[part], width=w, color=col, label=part)
    tot = [float(np.sqrt(rv.sigma_instr_ms**2 + p**2 + j**2))
           for p, j in zip(comp["photon"], comp["jitter"])]
    axC.plot(x, tot, "k.-", label="total σ_RV (median)")
    axC.set_yscale("log")
    axC.set_xticks(x); axC.set_xticklabels(stypes)
    axC.set_xlabel("Spectral type"); axC.set_ylabel("σ component [m/s] (median)")
    axC.set_title(f"C. Noise decomposition by stype ({inst} band {rv.band})")
    axC.legend(fontsize=7); axC.grid(axis="y", alpha=0.2, which="both")

    # D. loss budget overall + by stype
    axD = fig.add_subplot(gs[1, 0])
    order = ["detected", "signal_too_weak", "jitter_dominated", "host_star_too_faint"]
    colors = {"detected": "#2060c0", "signal_too_weak": "#c04040",
              "jitter_dominated": "#7570b3", "host_star_too_faint": "#806080"}
    piv = (c.groupby(["stype", "rv_reason_category"]).size().unstack(fill_value=0)
           .reindex(index=stypes).fillna(0))
    for col in order:
        if col not in piv.columns:
            piv[col] = 0
    piv = piv[order]; frac = piv.div(piv.sum(axis=1), axis=0)
    bottom = np.zeros(len(frac))
    for col in order:
        axD.bar(np.arange(len(frac)), frac[col], bottom=bottom,
                color=colors[col], label=col, width=0.6)
        bottom += frac[col].values
    axD.set_xticks(np.arange(len(frac)))
    axD.set_xticklabels([f"{st}\nN={int(piv.loc[st].sum()):,}" for st in frac.index])
    axD.set_ylim(0, 1); axD.set_ylabel("Fraction")
    axD.set_title("D. Loss budget by stype")
    axD.legend(fontsize=6.5, loc="lower right")

    # E. band mag vs sigma_RV
    axE = fig.add_subplot(gs[1, 1])
    m = c["rv_mag"]; sig = c["rv_sigma_ms"]
    axE.scatter(m[~det], sig[~det], s=3, alpha=0.12, color="#e07020", label=f"Missed")
    axE.scatter(m[det], sig[det], s=5, alpha=0.4, color="#2060c0", label=f"Detected")
    axE.axvline(rv.phot_full_mag, color="black", ls="--", lw=1.2,
                label=f"photon-floor limit ({rv.band}={rv.phot_full_mag:g})")
    axE.axvline(args.mag_target, color="red", ls=":", lw=1.2, label=f"target cut ({args.mag_target:g})")
    axE.set_yscale("log")
    axE.set_xlabel(f"Band magnitude ({rv.band})"); axE.set_ylabel("σ_RV [m/s] (log)")
    axE.set_title("E. Brightness wall: σ_RV vs band magnitude")
    axE.legend(fontsize=7); axE.grid(alpha=0.2)

    # F. funnel text panel
    axF = fig.add_subplot(gs[1, 2]); axF.axis("off")
    n = len(c)
    n_bright = int((c["rv_mag"] <= args.mag_target).sum())
    kfloor_all = args.snr_threshold * np.sqrt(2 / args.n_obs) * c["rv_sigma_ms"]
    n_k = int(((c["rv_k_ms"] >= kfloor_all) & (c["rv_mag"] <= args.mag_target)).sum())
    n_det = int(det.sum())
    txt = (
        f"PIPELINE FUNNEL ({inst})\n"
        f"{'─'*34}\n"
        f"P-Pop planets        {n:>10,}\n"
        f"  band-mag ≤ {args.mag_target:g}      {n_bright:>10,} ({n_bright/n:.0%})\n"
        f"  + K ≥ {args.snr_threshold:g}·σ_K       {n_k:>10,} ({n_k/n:.1%})\n"
        f"  detected           {n_det:>10,} ({n_det/n:.1%})\n\n"
        f"medians:\n"
        f"  K       = {c['rv_k_ms'].median():.2f} m/s\n"
        f"  σ_RV    = {c['rv_sigma_ms'].median():.2f} m/s\n"
        f"  σ_K     = {c['rv_sigma_K_ms'].median():.2f} m/s\n"
        f"  K/σ_K   = {c['rv_snr'].median():.2f}  (need {args.snr_threshold:g})"
    )
    axF.text(0.02, 0.98, txt, transform=axF.transAxes, va="top", ha="left",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="#f7f7f7", ec="0.6"))
    axF.set_title("F. Stage funnel")

    fig.suptitle(f"RV detector pipeline check — {inst}, N_obs={args.n_obs}, "
                 f"K/σ_K ≥ {args.snr_threshold:g}  (P-Pop universe)", fontsize=12)
    out = OUT_DIR / f"rv_pipeline_check_{inst.lower()}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


# ───────────────────────────────── Main ───────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-catalogs", type=int, default=60)
    ap.add_argument("--instrument", default="HARPS", help="HARPS | NIRPS")
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--mag-target", type=float, default=12.0)
    ap.add_argument("--fig1-only", action="store_true",
                    help="only build the published-K calibration figure (the paper figure)")
    args = ap.parse_args()

    print(f"Output dir: {OUT_DIR}")
    print("\n── Figure 1: K vs published pl_rvamp ──")
    try:
        rvamp = load_rvamp_sample()
        cal = plot_k_calibration(rvamp, args)
    except Exception as exc:
        print(f"  SKIPPED (download/cache failed): {exc}")
        cal = None

    if args.fig1_only:
        if cal:
            print(f"\nK calibration: median K_model/K_published = {cal['ratio_med']:.3f} "
                  f"(16-84%: {cal['ratio_16']:.3f}-{cal['ratio_84']:.3f}, N={cal['n']:,})")
        print("\nDone (fig1 only).")
        return

    print("\n── Figure 2: P-Pop pipeline stage check ──")
    ppop = load_ppop(args.n_catalogs)
    # "best" is per-planet max(HARPS, NIRPS); the noise decomposition is only
    # meaningful per-instrument, so emit both funnels rather than mislabel one.
    insts = ["HARPS", "NIRPS"] if args.instrument.lower() == "best" else [args.instrument]
    for inst in insts:
        plot_pipeline_check(ppop, args, inst)

    if cal:
        print(f"\nK calibration: median K_model/K_published = {cal['ratio_med']:.3f} "
              f"(16-84%: {cal['ratio_16']:.3f}-{cal['ratio_84']:.3f}, N={cal['n']:,})")
    print("\nDone.")


if __name__ == "__main__":
    main()
