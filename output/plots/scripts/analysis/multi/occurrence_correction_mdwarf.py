"""
31_occurrence_correction_mdwarf.py — the completeness-corrected occurrence of cold rocky
planets, CONDITIONED ON M-DWARF HOSTS, where the red window may become measurable.

Why M dwarfs: insolation I = L*/a^2. Low insolation (cold) needs a LONG period around a bright
star, but a SHORT period around a faint M dwarf (the cold zone sits close in). So the same cold,
large, rocky planet that is invisible around an FGK star is short-period -> transit+RV detectable
around an M dwarf. Conditioning on M dwarfs should lift completeness in the red window above the
floor where the inverse-detection-efficiency correction becomes trustworthy.

Same method as script 65 (inject rocky planets across (insolation, radius), measure transit+RV
completeness eta, divide NASA rocky counts by eta), now with hosts and NASA restricted to
Teff < 3900 K. Red window = insolation < 10 and radius > 1.7 R_earth (rocky => M >~ 5 M_earth).
We compute eta for ALL stars and for M dwarfs so the improvement is visible side by side.

Run:
    python scripts/31_occurrence_correction_mdwarf.py
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
from matplotlib.colors import LogNorm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = Path(SILICON_CURVE)
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "31_occurrence_correction_mdwarf")

N_HOST = 40000
ROCKY_MASS_FACTOR = 1.3
ETA_MIN = 0.02
COLD_FLUX = 10.0
BIG_RADIUS = 1.7
MDWARF_TEFF = 3900.0
RV_MAG_TARGET = 12.0

RADIUS_EDGES = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0])
RADIUS_CENT = 0.5 * (RADIUS_EDGES[:-1] + RADIUS_EDGES[1:])
FLUX_EDGES = np.logspace(np.log10(0.3), np.log10(3000), 10)
FLUX_CENT = np.sqrt(FLUX_EDGES[:-1] * FLUX_EDGES[1:])


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def teff_of(df):
    for c in ["teff_s", "temp_s"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return np.full(len(df), np.nan)


def rocky_mass(R, m_sil, r_sil):
    return float(np.clip(np.interp(R, r_sil, m_sil) * ROCKY_MASS_FACTOR, 0.1, 12.0))


def compute_eta(hosts, m_sil, r_sil):
    nR, nF = len(RADIUS_CENT), len(FLUX_CENT)
    eta = np.full((nR, nF), np.nan)
    F = hosts["flux_p"].to_numpy(float)
    fidx = np.digitize(F, FLUX_EDGES) - 1
    for ri, Rc in enumerate(RADIUS_CENT):
        inj = hosts.copy()
        inj["radius_p"] = Rc
        inj["mass_p"] = rocky_mass(Rc, m_sil, r_sil)
        det = (run_kepler(inj)["detected"].to_numpy(bool)
               & run_rv_best(inj, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool))
        for fi in range(nF):
            cell = fidx == fi
            if int(cell.sum()) >= 30:
                eta[ri, fi] = det[cell].mean()
    return eta


def nasa_counts(sub_mask, m, r, fl, mass_arr, m_sil, r_sil):
    mm, rr, ff = mass_arr[sub_mask], r[sub_mask], fl[sub_mask]
    rocky = rr < np.interp(mm, m_sil, r_sil)
    N, _, _ = np.histogram2d(rr[rocky], ff[rocky], bins=[RADIUS_EDGES, FLUX_EDGES])
    return N, rr[rocky], ff[rocky], mm[rocky]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()

    # hosts
    pool = pd.read_csv(PPOP_CATALOG, low_memory=False)
    pool["flux_p"] = pd.to_numeric(pool["flux_p"], errors="coerce")
    pool = pool[pool["flux_p"].between(FLUX_EDGES[0], FLUX_EDGES[-1])].copy()
    pool["teff"] = teff_of(pool)
    rng = np.random.default_rng(1)
    hosts_all = pool.sample(min(N_HOST, len(pool)), random_state=1).reset_index(drop=True)
    md = pool[pool["teff"] < MDWARF_TEFF]
    hosts_md = md.sample(min(N_HOST, len(md)), random_state=1).reset_index(drop=True)
    print(f"--> hosts: all={len(hosts_all)}  M-dwarf(Teff<{MDWARF_TEFF:g})={len(hosts_md)} (of {len(md)} in pool)")

    print("--> completeness (all stars)...")
    eta_all = compute_eta(hosts_all, m_sil, r_sil)
    print("--> completeness (M dwarfs)...")
    eta_md = compute_eta(hosts_md, m_sil, r_sil)

    # NASA
    nd = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(nd["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(nd["pl_rade"], errors="coerce")
    fl = pd.to_numeric(nd["pl_insol"], errors="coerce")
    te = pd.to_numeric(nd["st_teff"], errors="coerce")
    prov = nd.get("pl_bmassprov", pd.Series("", index=nd.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    base = meas & r.between(RADIUS_EDGES[0], RADIUS_EDGES[-1]) & fl.between(FLUX_EDGES[0], FLUX_EDGES[-1]) & m.between(0.1, 12)
    m, r, fl, te = m.to_numpy(), r.to_numpy(), fl.to_numpy(), te.to_numpy()
    base = base.to_numpy()
    mask_md = base & (te < MDWARF_TEFF)
    N_all, _, _, _ = nasa_counts(base, m, r, fl, m, m_sil, r_sil)
    N_md, rR, flR, mR = nasa_counts(mask_md, m, r, fl, m, m_sil, r_sil)
    print(f"--> NASA: all measured-mass in grid={int(base.sum())}; M-dwarf-hosted={int(mask_md.sum())}")

    big = RADIUS_CENT > BIG_RADIUS
    cold = FLUX_CENT < COLD_FLUX
    rw = np.outer(big, cold)

    def summarize(tag, eta, N):
        n = int(N[rw].sum())
        e = np.nanmedian(eta[rw])
        ok = np.isfinite(e) and e >= ETA_MIN
        print(f"  [{tag:8}] red window: N_obs={n}  median η={e:.3f}  "
              f"{'TRUSTWORTHY -> corrected=%.0f±%.0f' % (n/e, np.sqrt(max(n,1))/e) if ok else 'UNCONSTRAINED (η<%.2f)' % ETA_MIN}")
        return e

    print(f"\n  ===== RED WINDOW (insol<{COLD_FLUX:g}, radius>{BIG_RADIUS:g}; rocky) =====")
    e_all = summarize("all stars", eta_all, N_all)
    e_md = summarize("M dwarfs", eta_md, N_md)
    # M>2 framing among M-dwarf cold rocky
    cold_big_md = (flR < COLD_FLUX) & (rR > BIG_RADIUS)
    print(f"  M-dwarf cold rocky planets (insol<{COLD_FLUX:g}, R>{BIG_RADIUS:g}): N={int(cold_big_md.sum())}, "
          f"all have mass {('%.1f-%.1f M⊕' % (mR[cold_big_md].min(), mR[cold_big_md].max())) if cold_big_md.any() else '(none)'}")
    print(f"  -> M-dwarf conditioning changed red-window completeness {e_all:.3f} -> {e_md:.3f} "
          f"({e_md/e_all:.1f}x)" if np.isfinite(e_all) and e_all > 0 else "")

    # large-rocky corrected vs insolation, both samples
    def big_curve(eta, N):
        Nb = N[big, :].sum(0)
        eb = np.nanmean(np.where(big[:, None], eta, np.nan), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(eb >= ETA_MIN, Nb / eb, np.nan)
            ce = np.where(eb >= ETA_MIN, np.sqrt(Nb) / eb, np.nan)
        return Nb, eb, c, ce
    Nb_all, eb_all, c_all, ce_all = big_curve(eta_all, N_all)
    Nb_md, eb_md, c_md, ce_md = big_curve(eta_md, N_md)

    # corrected map (M dwarf)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_md = np.where(eta_md >= ETA_MIN, N_md / eta_md, np.nan)
    unreliable = ~(eta_md >= ETA_MIN)

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    X, Y = np.meshgrid(FLUX_EDGES, RADIUS_EDGES)

    def redbox(ax):
        ax.add_patch(plt.Rectangle((FLUX_EDGES[0], BIG_RADIUS), COLD_FLUX - FLUX_EDGES[0],
                                   RADIUS_EDGES[-1] - BIG_RADIUS, fill=False, ec="red", lw=2, ls="--"))
        ax.set_xscale("log"); ax.set_xlabel(r"insolation flux [$I_\oplus$]"); ax.set_ylabel(r"planet radius [$R_\oplus$]")

    ax = axes[0, 0]
    pc = ax.pcolormesh(X, Y, eta_md, cmap="viridis", vmin=0, vmax=np.nanmax([np.nanmax(eta_md), np.nanmax(eta_all)]))
    fig.colorbar(pc, ax=ax, label="M-dwarf completeness η (transit+RV)")
    redbox(ax); ax.set_title("(a) M-dwarf completeness η\n(cold zone is short-period → far brighter than all-star)", fontsize=11)

    ax = axes[0, 1]
    pc = ax.pcolormesh(X, Y, np.where(N_md > 0, N_md, np.nan), cmap="magma")
    fig.colorbar(pc, ax=ax, label="NASA M-dwarf rocky detections")
    ax.scatter(flR, rR, s=18, c="cyan", ec="k", lw=0.3, zorder=5)
    redbox(ax); ax.set_title("(b) NASA M-dwarf rocky detections", fontsize=11)

    ax = axes[1, 0]
    cc = np.where(corr_md > 0, corr_md, np.nan)
    pc = ax.pcolormesh(X, Y, cc, cmap="cividis",
                       norm=LogNorm(vmin=max(np.nanmin(cc), 1), vmax=np.nanmax(cc)) if np.isfinite(np.nanmax(cc)) else None)
    fig.colorbar(pc, ax=ax, label="corrected count N/η (M dwarfs)")
    for ri in range(len(RADIUS_CENT)):
        for fi in range(len(FLUX_CENT)):
            if unreliable[ri, fi]:
                ax.add_patch(plt.Rectangle((FLUX_EDGES[fi], RADIUS_EDGES[ri]),
                                           FLUX_EDGES[fi + 1] - FLUX_EDGES[fi], RADIUS_EDGES[ri + 1] - RADIUS_EDGES[ri],
                                           hatch="xx", fill=False, ec="0.5", lw=0))
    redbox(ax); ax.set_title(f"(c) M-dwarf corrected occurrence (hatched = η<{ETA_MIN:g})", fontsize=11)

    ax = axes[1, 1]
    okA = np.isfinite(c_all); okM = np.isfinite(c_md)
    ax.errorbar(FLUX_CENT[okA], c_all[okA], yerr=ce_all[okA], fmt="o-", color="gray", capsize=3, label="all stars (corrected)")
    ax.errorbar(FLUX_CENT[okM], c_md[okM], yerr=ce_md[okM], fmt="s-", color="tab:blue", capsize=3, label="M dwarfs (corrected)")
    for fi in np.where(~okM)[0]:
        ax.axvspan(FLUX_EDGES[fi], FLUX_EDGES[fi + 1], color="0.88", alpha=0.5)
    ax.axvline(COLD_FLUX, color="red", ls="--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"insolation flux [$I_\oplus$]"); ax.set_ylabel(f"large-rocky count (R>{BIG_RADIUS:g})")
    ax.set_title("(d) Large-rocky corrected abundance vs insolation\nM dwarfs extend trustworthy correction to colder cells", fontsize=11)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9)

    fig.suptitle("Cold red window CONDITIONED ON M DWARFS — does the deficit become measurable when cold = short period?\n"
                 "transit+RV, completeness-corrected; red box = insol<%g, R>%g" % (COLD_FLUX, BIG_RADIUS), fontsize=13)
    fig.text(0.5, 0.005,
             f"Red-window completeness rose from {e_all:.1%} (all stars) to {e_md:.1%} (M dwarfs). If now ABOVE η_min and the "
             f"corrected M-dwarf count stays low toward low insolation, that is a PHYSICAL deficit; if it inflates to match warm "
             f"cells, the all-star deficit was bias. Hatched/grey = still unconstrained.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_png = os.path.join(OUT_DIR, "mdwarf_corrected_occurrence.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()
