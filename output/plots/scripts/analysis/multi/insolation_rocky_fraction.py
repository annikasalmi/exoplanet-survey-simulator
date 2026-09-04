"""
28_insolation_rocky_fraction.py — TEST C: insolation-controlled rocky fraction.

Not ONE puffy fraction: f_rocky(R | I-slice), cold (I<50 I⊕) vs hot (I≥50 I⊕), compared
  (i)  WITHIN NASA  — cold slice vs hot slice, two-proportion test (Agresti–Coull), and
  (ii) against the FLAT-SELECTED control — the same flat universe (M ⊥ R ⊥ I by
       construction, so its TRUE f_rocky is insolation-independent) pushed through the
       same joint transit+RV selection + measurement noise. Whatever cold-hot difference
       the control shows is pure SELECTION; NASA must deviate from THAT, not from zero,
       to indicate formation.

Direction: at fixed R, rocky = high mass = large RV amplitude, so selection favours rocky
planets MORE in the cold (long P, small K). If NASA's cold slice is LESS rocky than this
control predicts, the cold large-rocky deficit is real (formation/retention).

Stats: Agresti–Coull intervals per bin; AC-adjusted two-proportion z-tests (binomial,
vanilla). NASA primary = precision-cut sample (labels trustworthy); full sample printed
as sensitivity.

Run:
    python scripts/28_insolation_rocky_fraction.py
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

from tools.paths import SILICON_CURVE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = Path(SILICON_CURVE)
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "28_insolation_rocky_fraction")

FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
COLD_MAX = 50.0
MASS_FRAC_ERR = 0.20
RAD_FRAC_ERR = 0.046
NASA_MASS_PREC = 0.25
NASA_RAD_PREC = 0.08
N_NOISE = 20
Z_AC = 1.96
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

R_EDGES = np.array([1.0, 1.4, 1.8, 2.2])
R_CENT = 0.5 * (R_EDGES[:-1] + R_EDGES[1:])
BIN_LABELS = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(R_EDGES[:-1], R_EDGES[1:])] + \
             [f"pooled {R_EDGES[0]:.1f}–{R_EDGES[-1]:.1f}"]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def agresti_coull(k, n, z=Z_AC):
    if n <= 0:
        return np.nan, np.nan
    nt = n + z * z
    pt = (k + z * z / 2.0) / nt
    return pt, np.sqrt(pt * (1.0 - pt) / nt)


def two_prop_z(p1, se1, p2, se2):
    se = np.sqrt(se1 ** 2 + se2 ** 2)
    if not np.isfinite(se) or se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return z, 2.0 * norm.sf(abs(z))


def bin_fracs(r_obs, rocky):
    """k, n per R bin + pooled; returns lists of length len(R_CENT)+1."""
    ks, ns = [], []
    for lo, hi in zip(R_EDGES[:-1], R_EDGES[1:]):
        sel = (r_obs >= lo) & (r_obs < hi)
        ks.append(int(rocky[sel].sum())); ns.append(int(sel.sum()))
    sel = (r_obs >= R_EDGES[0]) & (r_obs < R_EDGES[-1])
    ks.append(int(rocky[sel].sum())); ns.append(int(sel.sum()))
    return ks, ns


def build_flat():
    pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=RNG_SEED)
    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    f = pd.to_numeric(pool["flux_p"], errors="coerce")
    keep = (r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()
    td = run_kepler(pool)["detected"].to_numpy(bool)
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    det = td & rd
    return dict(
        mass=pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float)[det],
        radius=pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float)[det],
        flux=pd.to_numeric(pool["flux_p"], errors="coerce").to_numpy(float)[det],
        n_pool=len(pool), n_det=int(det.sum()),
    )


def flat_control(flat, m_sil, r_sil, rng):
    """f_rocky per (slice, bin) for the flat-selected control: mean ± sd over noise draws."""
    out = {}
    for slice_name, sel in [("cold", flat["flux"] < COLD_MAX), ("hot", flat["flux"] >= COLD_MAX)]:
        m, r = flat["mass"][sel], flat["radius"][sel]
        fr = np.full((N_NOISE, len(R_CENT) + 1), np.nan)
        for i in range(N_NOISE):
            mo = m * np.exp(rng.normal(0, MASS_FRAC_ERR, m.size))
            ro = r * np.exp(rng.normal(0, RAD_FRAC_ERR, r.size))
            rocky = ro < np.interp(mo, m_sil, r_sil)
            ks, ns = bin_fracs(ro, rocky)
            fr[i] = [k / n if n >= 5 else np.nan for k, n in zip(ks, ns)]
        out[slice_name] = dict(mean=np.nanmean(fr, 0), sd=np.nanstd(fr, 0),
                               n=int(m.size))
    return out


def load_nasa(precision: bool):
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keep = (meas & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & ins.between(BOX["f_lo"], BOX["f_hi"]))
    if precision:
        me = np.maximum(pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs(),
                        pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs())
        re = np.maximum(pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs(),
                        pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs())
        keep = keep & (me / m <= NASA_MASS_PREC) & (re / r <= NASA_RAD_PREC)
    return dict(m=m[keep].to_numpy(), r=r[keep].to_numpy(), ins=ins[keep].to_numpy())


def nasa_fracs(nasa, m_sil, r_sil):
    out = {}
    for slice_name, sel in [("cold", nasa["ins"] < COLD_MAX), ("hot", nasa["ins"] >= COLD_MAX)]:
        m, r = nasa["m"][sel], nasa["r"][sel]
        rocky = r < np.interp(m, m_sil, r_sil)
        ks, ns = bin_fracs(r, rocky)
        pt, se = zip(*[agresti_coull(k, n) for k, n in zip(ks, ns)])
        out[slice_name] = dict(k=ks, n=ns, pt=np.array(pt), se=np.array(se))
    return out


def print_tables(tag, nf, fc):
    print(f"\n  ================ TEST C — {tag} ================")
    print(f"  {'R bin':<16}{'slice':<6}{'k/n':>9}{'p̂':>7}{'AC p̃±se':>15}{'flat ctrl':>12}")
    for b, lbl in enumerate(BIN_LABELS):
        for s in ["cold", "hot"]:
            k, n = nf[s]["k"][b], nf[s]["n"][b]
            pt, se = nf[s]["pt"][b], nf[s]["se"][b]
            fm, fs = fc[s]["mean"][b], fc[s]["sd"][b]
            raw = f"{k / n:.2f}" if n else "--"
            print(f"  {lbl:<16}{s:<6}{f'{k}/{n}':>9}{raw:>7}{f'{pt:.2f}±{se:.2f}':>15}"
                  f"{f'{fm:.3f}±{fs:.3f}':>12}")
    print(f"\n  {'R bin':<16}{'NASA cold-vs-hot':>22}  {'NASA-vs-flat (cold)':>22}  {'Δf: NASA vs flat':>28}")
    for b, lbl in enumerate(BIN_LABELS):
        z1, p1 = two_prop_z(nf["cold"]["pt"][b], nf["cold"]["se"][b],
                            nf["hot"]["pt"][b], nf["hot"]["se"][b])
        z2, p2 = two_prop_z(nf["cold"]["pt"][b], nf["cold"]["se"][b],
                            fc["cold"]["mean"][b], fc["cold"]["sd"][b])
        dN = nf["cold"]["pt"][b] - nf["hot"]["pt"][b]
        seN = np.sqrt(nf["cold"]["se"][b] ** 2 + nf["hot"]["se"][b] ** 2)
        dF = fc["cold"]["mean"][b] - fc["hot"]["mean"][b]
        sdF = np.sqrt(fc["cold"]["sd"][b] ** 2 + fc["hot"]["sd"][b] ** 2)
        z3, p3 = two_prop_z(dN, seN, dF, sdF)
        print(f"  {lbl:<16}{f'z={z1:+.2f} p={p1:.3f}':>22}  {f'z={z2:+.2f} p={p2:.3f}':>22}  "
              f"{f'ΔN={dN:+.2f} ΔF={dF:+.2f} p={p3:.3f}':>28}")
    b = len(BIN_LABELS) - 1
    dN = nf["cold"]["pt"][b] - nf["hot"]["pt"][b]
    dF = fc["cold"]["mean"][b] - fc["hot"]["mean"][b]
    z3, p3 = two_prop_z(dN, np.sqrt(nf["cold"]["se"][b] ** 2 + nf["hot"]["se"][b] ** 2),
                        dF, np.sqrt(fc["cold"]["sd"][b] ** 2 + fc["hot"]["sd"][b] ** 2))
    print(f"\n  HEADLINE (pooled {R_EDGES[0]:.1f}–{R_EDGES[-1]:.1f} R⊕): NASA Δf(cold−hot) = {dN:+.2f}, "
          f"selection-only control Δf = {dF:+.2f}, difference p = {p3:.3f}")
    if np.isfinite(p3) and p3 < 0.05 and dN < dF:
        print("  VERDICT: NASA's cold slice is LESS rocky than selection predicts → "
              "cold-rocky deficit beyond selection (formation/retention).")
    elif np.isfinite(p3):
        print("  VERDICT: NASA's cold-hot rocky-fraction contrast is consistent with the "
              "selection-only control at this precision — no formation signal isolated here.")
    return p3


def make_figure(nf, fc, n_nasa_cold, n_nasa_hot):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    xs = np.arange(len(R_CENT))

    for ax, s, n_nasa in [(axes[0], "cold", n_nasa_cold), (axes[1], "hot", n_nasa_hot)]:
        fm, fsd = fc[s]["mean"][:-1], fc[s]["sd"][:-1]
        ax.fill_between(xs, fm - fsd, fm + fsd, color="tab:blue", alpha=0.20)
        ax.plot(xs, fm, "-o", color="tab:blue", ms=5,
                label=f"flat-selected control (N={fc[s]['n']})")
        ax.errorbar(xs + 0.06, nf[s]["pt"][:-1], yerr=Z_AC * nf[s]["se"][:-1], fmt="s",
                    color="tab:green", ms=7, capsize=4, lw=1.6,
                    label=f"NASA (N={n_nasa}), AC 95%")
        for b in range(len(R_CENT)):
            ax.text(xs[b] + 0.06, nf[s]["pt"][b] + Z_AC * nf[s]["se"][b] + 0.03,
                    f"{nf[s]['k'][b]}/{nf[s]['n'][b]}", ha="center", fontsize=8, color="0.3")
        ax.set_xticks(xs); ax.set_xticklabels(BIN_LABELS[:-1])
        ax.set_ylim(-0.04, 1.09)
        ax.set_xlabel(r"planet radius bin [$R_\oplus$]"); ax.set_ylabel("rocky fraction")
        ax.set_title(("(a) COLD slice (I<%g I⊕)" if s == "cold" else "(b) HOT slice (I≥%g I⊕)")
                     % COLD_MAX, fontsize=11)
        ax.grid(alpha=0.2); ax.legend(fontsize=8.5, loc="lower left")

    ax = axes[2]
    dN = nf["cold"]["pt"][:-1] - nf["hot"]["pt"][:-1]
    seN = np.sqrt(nf["cold"]["se"][:-1] ** 2 + nf["hot"]["se"][:-1] ** 2)
    dF = fc["cold"]["mean"][:-1] - fc["hot"]["mean"][:-1]
    sdF = np.sqrt(fc["cold"]["sd"][:-1] ** 2 + fc["hot"]["sd"][:-1] ** 2)
    ax.axhline(0, color="0.5", lw=1)
    ax.fill_between(xs, dF - sdF, dF + sdF, color="tab:blue", alpha=0.20)
    ax.plot(xs, dF, "-o", color="tab:blue", ms=5, label="selection-only expectation (flat)")
    ax.errorbar(xs + 0.06, dN, yerr=Z_AC * seN, fmt="s", color="tab:green", ms=7, capsize=4,
                lw=1.6, label="NASA")
    b = len(BIN_LABELS) - 1
    dNp = nf["cold"]["pt"][b] - nf["hot"]["pt"][b]
    dFp = fc["cold"]["mean"][b] - fc["hot"]["mean"][b]
    z3, p3 = two_prop_z(dNp, np.sqrt(nf["cold"]["se"][b] ** 2 + nf["hot"]["se"][b] ** 2),
                        dFp, np.sqrt(fc["cold"]["sd"][b] ** 2 + fc["hot"]["sd"][b] ** 2))
    ax.text(0.03, 0.05, f"pooled: NASA Δf={dNp:+.2f}, flat Δf={dFp:+.2f}\n"
                        f"NASA-vs-flat p={p3:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="whitesmoke", ec="0.7"))
    ax.set_xticks(xs); ax.set_xticklabels(BIN_LABELS[:-1])
    ax.set_xlabel(r"planet radius bin [$R_\oplus$]")
    ax.set_ylabel("Δf_rocky = cold − hot")
    ax.set_title("(c) cold−hot contrast: NASA vs selection-only", fontsize=11)
    ax.grid(alpha=0.2); ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("TEST C — insolation-controlled rocky fraction f_rocky(R | I-slice): "
                 "NASA vs the flat-selected (selection-only) control\n"
                 "flat universe has M ⊥ R ⊥ I, so its TRUE f_rocky is I-independent — any cold-hot "
                 "contrast in blue is pure selection; NASA (green) must deviate from BLUE, not from zero",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_png = os.path.join(OUT_DIR, "insolation_controlled_rocky_fraction.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    print(f"--> building flat pool (N={FLAT_N_POOL}) + transit/RV detectors...")
    flat = build_flat()
    print(f"    pool in box: {flat['n_pool']}, joint-detected: {flat['n_det']}")
    fc = flat_control(flat, m_sil, r_sil, rng)

    for precision, tag in [(True, "precision-cut NASA (primary)"),
                           (False, "FULL measured-mass NASA (sensitivity)")]:
        nasa = load_nasa(precision)
        nf = nasa_fracs(nasa, m_sil, r_sil)
        n_cold = int((nasa["ins"] < COLD_MAX).sum()); n_hot = int((nasa["ins"] >= COLD_MAX).sum())
        print(f"    NASA sample: {len(nasa['m'])} planets ({n_cold} cold / {n_hot} hot)")
        print_tables(tag, nf, fc)
        if precision:
            make_figure(nf, fc, n_cold, n_hot)


if __name__ == "__main__":
    main()
