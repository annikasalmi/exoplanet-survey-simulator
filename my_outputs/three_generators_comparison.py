"""Compare the three planet generators on (a) generation time, (b) the
UNIFORM-SAMPLED input parameters, and (c) the DERIVED quantities:

    old P-Pop   = SAG13 (Kopparapu 2018) for FGK   -- Annika's distribution
    new P-Pop   = Bergsten2022 for FGK             -- ours
    Flat        = uniform_generator (fully-flat box, no occurrence/mass priors)

Key point this figure makes: in the Flat generator, radius / mass / period /
Teff are drawn uniformly (flat, or flat-in-log), but INSOLATION is never sampled
-- it is derived as flux = L_star / a^2 (L_star from sampled Teff; a from sampled
period via Kepler's 3rd law). The catalogue is then rejection-filtered on
insolation in [1e-2, 1e4] S_earth, which removes long-period (low-flux) planets,
so even the KEPT period is no longer flat. Hence panels C/D of the old figure
("period", "insolation") were not uniform.

Run `python my_outputs/three_generators_comparison.py` (uses a cached .npz so the
plot can be re-rendered without re-timing the slow P-Pop generation;
`--rebuild` forces regeneration).
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import PPop.SystemGenerator as SystemGenerator
from PPop.StarCatalogs import gaia
from PPop.PlanetDistributions import Bergsten2022, SAG13, Dressing2015
from PPop.ScalingModels import BinarySuppression
from PPop.MassModels import Chen2017
from PPop.EccentricityModels import Circular
from PPop.StabilityModels import He2019
from PPop.OrbitModels import Random
from PPop.AlbedoModels import Uniform
from PPop.ExozodiModels import Ertel2020
from run.ppop.uniform_generator import generate_flat_catalog, DEFAULTS

N_STARS = 1200
SUBSET_SEED = 12345
DRAW_SEED = 0
TARGET = 1_000_000
N_FLAT = 200_000           # flat planets kept for the population panels
CACHE = os.path.join(ROOT, "my_outputs", "three_generators_data.npz")

COL = {"old": "#d1495b", "new": "#1f77b4", "flat": "#6c757d"}
LAB = {"old": "old P-Pop  (SAG13 / Kopparapu 2018, Annika)",
       "new": "new P-Pop  (Bergsten 2022, ours)",
       "flat": "Flat universe  (uniform box)"}
_SUN_T = 5772.0


def build_sysgen(fgk):
    s2m = {"A": SAG13, "F": fgk, "G": fgk, "K": fgk, "M": Dressing2015}
    return SystemGenerator.SystemGenerator(
        gaia, s2m, BinarySuppression, Chen2017, Circular, He2019, Random,
        Uniform, Ertel2020, ["A", "F", "G", "K", "M"], [0.0, 60.0],
        [-90.0, 90.0], "baseline", False, 1, None, True,
        rng=np.random.default_rng(DRAW_SEED))


def run_ppop(fgk, star_idx):
    sg = build_sysgen(fgk)
    sg.StarCatalog.SC = sg.StarCatalog.SC[star_idx]
    t0 = time.perf_counter()
    df = sg.SimulateUniverses("c", Nuniverses=1, write_to_file=False)
    dt = time.perf_counter() - t0
    stars = df["Star"].to_numpy()
    teff = np.array([s.Teff for s in stars], float)
    rad_s = np.array([s.Rad for s in stars], float)
    return dict(
        R=np.asarray(df["Rp"], float), P=np.asarray(df["Porb"], float),
        F=np.asarray(df["Fp"], float), a=np.asarray(df["ap"], float),
        teff=teff, lsun=rad_s ** 2 * (teff / _SUN_T) ** 4,
        sec_1e6=TARGET / (len(df) / dt), n=len(df), t=dt)


def flat_raw_sampled(n=N_FLAT, seed=1):
    """The genuine 'as-sampled' (pre-rejection) input draws of the flat box."""
    rng = np.random.default_rng(seed)
    lo, hi = DEFAULTS["radius_lims"]; R = rng.uniform(lo, hi, n)
    lo, hi = DEFAULTS["period_lims"]; P = 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n)
    lo, hi = DEFAULTS["teff_lims"]; T = rng.uniform(lo, hi, n)
    return R, P, T


def generate_all():
    probe = build_sysgen(Bergsten2022)
    n_full = len(probe.StarCatalog.SC)
    idx = np.random.default_rng(SUBSET_SEED).choice(n_full, size=N_STARS, replace=False)
    idx.sort()
    print(f"Generating + timing over {N_STARS} Gaia stars per P-Pop case ...")
    new = run_ppop(Bergsten2022, idx)
    old = run_ppop(SAG13, idx)

    t0 = time.perf_counter()
    df_flat = generate_flat_catalog(n_planets=TARGET, seed=0)
    t_flat = time.perf_counter() - t0
    df_flat = df_flat.iloc[:N_FLAT]
    R_raw, P_raw, T_raw = flat_raw_sampled()

    store = {}
    for k, d in (("new", new), ("old", old)):
        for f in ("R", "P", "F", "a", "teff", "lsun"):
            store[f"{k}_{f}"] = d[f]
        store[f"{k}_sec"] = d["sec_1e6"]; store[f"{k}_n"] = d["n"]; store[f"{k}_t"] = d["t"]
    store["flat_R"] = df_flat["radius_p"].to_numpy()
    store["flat_P"] = df_flat["p_orb"].to_numpy()
    store["flat_F"] = df_flat["flux_p"].to_numpy()
    store["flat_a"] = df_flat["semimajor_p"].to_numpy()
    store["flat_teff"] = df_flat["teff_s"].to_numpy()
    store["flat_lsun"] = df_flat["l_sun"].to_numpy()
    store["flat_R_raw"], store["flat_P_raw"], store["flat_T_raw"] = R_raw, P_raw, T_raw
    store["flat_sec"] = TARGET / (len(df_flat) / t_flat) * (len(df_flat) / TARGET)  # = t_flat/TARGET*1e6
    store["flat_sec"] = t_flat                     # t_flat already for 1e6 planets
    store["flat_n"] = TARGET; store["flat_t"] = t_flat
    np.savez(CACHE, **store)
    print(f"--> cached: {CACHE}")
    return {k: np.array(store[k]) if hasattr(store[k], "__len__") else store[k] for k in store}


def fmt_time(sec):
    if sec < 90:
        return f"{sec:.1f} s"
    if sec < 5400:
        return f"{sec/60:.1f} min"
    return f"{sec/3600:.1f} h"


def main(rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        d = generate_all()
    else:
        print(f"--> loading cache: {CACHE}  (use --rebuild to regenerate)")
        d = dict(np.load(CACHE, allow_pickle=True))

    print("\n--- measured generation time ---")
    for k in ("old", "new", "flat"):
        print(f"{LAB[k]:48s}  {int(d[f'{k}_n']):>8d} planets in {float(d[f'{k}_t']):8.1f}s "
              f"-> {fmt_time(float(d[f'{k}_sec']))} per 1e6")

    order = ["old", "new", "flat"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.2))

    def _logticks(ax, lo, hi):
        ticks = np.arange(np.ceil(np.log10(lo)), np.floor(np.log10(hi)) + 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([rf"$10^{{{int(t)}}}$" for t in ticks])

    def hist3(ax, key, lo, hi, nbins=55, logx=False, flat_key=None):
        # For log-x we histogram log10(x) on LINEAR bins -> density is dN/dlog x
        # (per dex). This is what makes a log-uniform draw appear FLAT; plotting
        # density on log-spaced bins normalizes by linear width and would make a
        # log-uniform look like a falling 1/x curve.
        def get(k):
            return np.asarray(d[f"{k}_{flat_key}" if (k == "flat" and flat_key)
                                else f"{k}_{key}"], float)
        if logx:
            bl = np.linspace(np.log10(lo), np.log10(hi), nbins)
            for k in order:
                x = np.log10(np.clip(get(k), 10 ** bl[0], 10 ** bl[-1]))
                ax.hist(x, bins=bl, density=True, histtype="step", lw=2.0, color=COL[k])
            ax.set_xlim(bl[0], bl[-1])
            _logticks(ax, lo, hi)
            ax.set_ylabel("density  (per dex)")
        else:
            bins = np.linspace(lo, hi, nbins)
            for k in order:
                ax.hist(np.clip(get(k), bins[0], bins[-1]), bins=bins, density=True,
                        histtype="step", lw=2.0, color=COL[k])
            ax.set_ylabel("density")
        ax.grid(ls=":", alpha=0.5)

    # ---- (A) generation time -------------------------------------------------
    ax = axes[0, 0]
    secs = [float(d[f"{k}_sec"]) for k in order]
    bars = ax.bar(range(3), secs, color=[COL[k] for k in order], width=0.62,
                  edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["old P-Pop\n(SAG13)", "new P-Pop\n(Bergsten)", "Flat\n(uniform)"])
    ax.set_ylabel("generation time per $10^6$ planets  (s, log)")
    ax.set_title("(A) Time to generate $10^6$ planets  (1 thread)", fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    for b, s in zip(bars, secs):
        ax.text(b.get_x() + b.get_width() / 2, s * 1.25, fmt_time(s),
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylim(1e-1, max(secs) * 40)
    # bracket over the two P-Pop bars (clear of everything else)
    ytop = max(secs[0], secs[1]) * 3.0
    ax.plot([0, 0, 1, 1], [ytop / 1.5, ytop, ytop, ytop / 1.5], color="#333", lw=1.0)
    ax.text(0.5, ytop * 1.15, r"$\approx$ equal (run-to-run noise)",
            ha="center", va="bottom", fontsize=8.5, color="#333")
    # slowdown note placed in the empty mid-band, well below the tall bars
    sp = round(0.5 * (secs[0] + secs[1]) / secs[2], -3)
    ax.text(1.5, 6.0, f"P-Pop\n~{sp:,.0f}×\nslower\nthan Flat", ha="center", va="center",
            fontsize=8.5, style="italic", color="#444",
            bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.9))

    # ---- row 1: UNIFORM-SAMPLED inputs --------------------------------------
    ax = axes[0, 1]
    hist3(ax, "R", 0.4, 4.0, flat_key="R_raw")
    ax.set_xlabel(r"planet radius $R_p\ [R_\oplus]$")
    ax.set_title("(B) radius — SAMPLED  $U(0.5,2.2)$", fontweight="bold")

    ax = axes[0, 2]
    hist3(ax, "teff", 2300, 9700, flat_key="T_raw")
    ax.set_xlabel(r"host $T_{\rm eff}$ [K]")
    ax.set_title(r"(C) host $T_{\rm eff}$ — SAMPLED  $U(2300,9700)$", fontweight="bold")

    ax = axes[0, 3]
    hist3(ax, "P", 0.4, 20000, logx=True, flat_key="P_raw")
    ax.set_xlabel(r"orbital period $P_{\rm orb}$ [d]")
    ax.set_title(r"(D) period — SAMPLED  $\log U(0.5,2\times10^4)$", fontweight="bold")
    ax.text(0.04, 0.94, "flat = flat in log\n(as drawn)", transform=ax.transAxes,
            va="top", fontsize=8.5, color=COL["flat"])

    # ---- row 2: DERIVED quantities ------------------------------------------
    ax = axes[1, 0]
    hist3(ax, "lsun", 1e-4, 1e2, logx=True)
    ax.set_xlabel(r"stellar luminosity $L_\star\ [L_\odot]$")
    ax.set_title(r"(E) luminosity — DERIVED  $(T_{\rm eff},R_\star)$", fontweight="bold")

    ax = axes[1, 1]
    hist3(ax, "a", 10 ** -2.3, 1e2, logx=True)
    ax.set_xlabel(r"semi-major axis $a$ [AU]")
    ax.set_title("(F) semi-major axis — DERIVED  (Kepler III)", fontweight="bold")

    ax = axes[1, 2]
    hist3(ax, "F", 1e-2, 1e4, logx=True)
    ax.axvspan(np.log10(0.32), np.log10(1.78), color="green", alpha=0.10)
    ax.set_xlabel(r"insolation $F_p\ [S_\oplus]$")
    ax.set_title(r"(G) insolation — DERIVED  $L_\star/a^2$", fontweight="bold")
    ax.text(0.04, 0.94, "never sampled →\nnot uniform", transform=ax.transAxes,
            va="top", fontsize=8.5, color="#444")

    ax = axes[1, 3]
    lo, hi = 0.4, 20000.0
    bl = np.linspace(np.log10(lo), np.log10(hi), 55)
    for k in order:
        x = np.log10(np.clip(np.asarray(d[f"{k}_P"], float), 10 ** bl[0], 10 ** bl[-1]))
        ax.hist(x, bins=bl, density=True, histtype="step", lw=2.0, color=COL[k])
    xr = np.log10(np.clip(np.asarray(d["flat_P_raw"], float), 10 ** bl[0], 10 ** bl[-1]))
    ax.hist(xr, bins=bl, density=True, histtype="step", lw=1.6, ls="--", color=COL["flat"])
    ax.set_xlim(bl[0], bl[-1])
    _logticks(ax, lo, hi)
    ax.grid(ls=":", alpha=0.5)
    ax.set_ylabel("density  (per dex)")
    ax.set_xlabel(r"orbital period $P_{\rm orb}$ [d]")
    ax.set_title("(H) period in catalogue — after insolation cut", fontweight="bold")
    ax.text(0.50, 0.94, "flat: dashed = sampled (flat)\nsolid = kept — long-$P$\nremoved by insolation cut",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color="#444")

    fig.legend([plt.Line2D([], [], color=COL[k], lw=2.5) for k in order],
               [LAB[k] for k in order], loc="lower center", ncol=3, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Planet generators: old P-Pop vs new P-Pop vs Flat universe — "
                 "generation time, sampled inputs, and derived quantities",
                 fontsize=14.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    out = os.path.join(ROOT, "my_outputs", "three_generators_comparison.png")
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main(rebuild=("--rebuild" in sys.argv))
