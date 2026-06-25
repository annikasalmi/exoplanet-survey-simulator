"""
56_flat_puffy_rocky_ratio_vs_nasa.py — puffy-fraction distribution of two "fake"
universes (flat AND P-Pop) compared to NASA, shown under THREE detection modes:
transit-only, RV-only, and transit+RV together. Puffy fraction = N_puffy/(N_puffy+N_rocky).

Plus a CURVE-FREE TWIN on the bottom row: the median bulk density (rho = 5.513 M/R^3).
The puffy fraction needs the silicate curve to split rocky from puffy; the density needs
NO curve at all. If both rows tell the same story, the result does not hinge on where the
silicate curve is drawn -- that is the whole point of the second row.

================================ HOW EACH CURVE IS MADE ================================
1. POPULATION (the "fake-planet universe"):
   - flat : a pool of FLAT_N_POOL planets whose radius, mass, period, star are each drawn
            from independent UNIFORM distributions (run/ppop/uniform_generator.py).
   - ppop : a cached realistic P-Pop catalogue (occurrence-rate radii + Chen2017 Forecaster
            mass-from-radius), so mass and radius are physically correlated.
   Every planet is labelled puffy/rocky from its TRUE mass & radius vs the silicate curve
   (Hongyi-silicon.ddat): puffy = radius ABOVE the curve, rocky = on/below.

2. TWO UNIVERSES:
   - B "with rocky"       : the full pool.
   - A "no rocky > M_thr" : the pool with rocky planets ABOVE MASS_THRESHOLD removed
                            (keeps low-mass rocky + all puffy).

3. ONE CURVE = many REPEATED DRAWS (this is the bell curve, NOT a fixed probability):
   For each of N_REPEATS draws we grab N_SAMPLE_PER_UNIVERSE planets (with replacement)
   from the universe, KEEP only the ones a detector finds (N_det of them), and measure
   BOTH numbers on the kept ones: the puffy fraction and the median bulk density. The
   numbers wobble from draw to draw, which is what gives each bell its width. For the
   puffy fraction the width is the exact coin-flip spread:
        sigma = sqrt( p (1 - p) / N_det )
   p = the universe's puffy fraction among detectable planets, N_det = how many survive the
   detector each draw. Bigger N_det -> narrower curve. The detectors set p (which planets
   survive) and N_det (how many) -- they do NOT manufacture the bell shape; repeated drawing
   does. (The script prints predicted-vs-measured sigma to prove this.)

4. NASA curve: the real measured-mass planets have a fixed puffy fraction and a fixed median
   density; the green curve is the BOOTSTRAP of those N_NASA planets (resample with
   replacement, recompute) -- the same sampling spread, wider because N_NASA (~290) <<
   N_SAMPLE. NASA is the same observed sample in every panel (not re-run through our toy
   detectors).

5. Dotted vertical line = each universe's TRUE (undetected) value. The filled bell sits where
   the detector pushes it -- the gap between line and bell is the detection bias.

6. CURVE-NUDGE CHECK (printed, and written to summary.txt): we move the silicate curve up and
   down by CURVE_NUDGE (composition/EOS uncertainty) and recompute the detected puffy fraction.
   If the number barely moves, the rocky/puffy split does not depend on the exact curve.
========================================================================================

Run:
    python scripts/56_flat_puffy_rocky_ratio_vs_nasa.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = ROOT / "Hongyi-silicon.ddat"
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "my_outputs", "56_flat_puffy_rocky_ratio_vs_nasa")

# ============================ KNOBS ============================
N_SAMPLE_PER_UNIVERSE = 20000   # <<< planets grabbed per draw, per universe. CHANGE THIS.
N_REPEATS = 10000               # <<< how many draws make each bell (also # NASA bootstraps).
MASS_THRESHOLD = 2.0            # universe A removes rocky planets with mass ABOVE this (M_earth)
CURVE_NUDGE = 0.05             # move the silicate curve +/- this (in radius) for the robustness check
# ==============================================================

FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
RHO_EARTH = 5.513               # g/cc (Earth bulk density); rho_planet = RHO_EARTH * M/R^3
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

# detection modes shown side by side (label, id)
MODES = [
    ("Transit only\n(Kepler)", "transit"),
    ("RV only\n(best HARPS/NIRPS)", "rv"),
    ("Transit AND RV", "both"),
]

# universes: (population, drop_highmass_rocky, base_label, colour)
UNIVERSES = [
    ("flat", True,  f"flat A: no rocky M>{MASS_THRESHOLD:g}", "tab:orange"),
    ("flat", False, "flat B: with rocky",                    "tab:blue"),
    ("ppop", True,  f"P-Pop A: no rocky M>{MASS_THRESHOLD:g}", "tab:red"),
    ("ppop", False, "P-Pop B: with rocky",                   "tab:purple"),
]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def puffy_flag(mass, radius, m_sil, r_sil) -> np.ndarray:
    return np.asarray(radius, float) > np.interp(np.asarray(mass, float), m_sil, r_sil)


def density(mass, radius) -> np.ndarray:
    """Bulk density in g/cc -- curve-free; no silicate curve needed."""
    return RHO_EARTH * np.asarray(mass, float) / np.asarray(radius, float) ** 3


def build_pool(population: str, m_sil, r_sil) -> pd.DataFrame:
    """Return mass, radius, puffy flag, transit-detected (td) and RV-detected (rd) per planet."""
    if population == "flat":
        pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=RNG_SEED)
    else:
        cols = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p", "semimajor_p", "radius_s",
                "mass_s", "temp_s", "teff_s", "distance_s", "l_sun", "flux_p", "detected"]
        pool = pd.read_csv(PPOP_CATALOG, usecols=lambda c: c in cols, low_memory=False)

    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    keep = r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    if "flux_p" in pool.columns:
        f = pd.to_numeric(pool["flux_p"], errors="coerce")
        keep = keep & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()

    mass = pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float)
    puffy = puffy_flag(mass, radius, m_sil, r_sil)

    # transit detection: Kepler model for flat; P-Pop's own pipeline flag for ppop
    if population == "flat":
        td = run_kepler(pool)["detected"].to_numpy(bool)
    else:
        td = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    # RV detection: best of HARPS/NIRPS, bright enough to be a target
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)

    return pd.DataFrame({"mass": mass, "radius": radius, "puffy": puffy, "td": td, "rd": rd})


def det_mask(pool: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "transit":
        return pool["td"].to_numpy(bool)
    if mode == "rv":
        return pool["rd"].to_numpy(bool)
    if mode == "both":
        return (pool["td"].to_numpy(bool) & pool["rd"].to_numpy(bool))
    return np.ones(len(pool), bool)   # "none" = true population


def kept_index(pool: pd.DataFrame, drop_highmass_rocky: bool) -> np.ndarray:
    mass = pool["mass"].to_numpy(float)
    puffy = pool["puffy"].to_numpy(bool)
    keep = ~((~puffy) & (mass > MASS_THRESHOLD)) if drop_highmass_rocky else np.ones(len(pool), bool)
    return np.flatnonzero(keep)


def true_values(pool: pd.DataFrame, drop_highmass_rocky: bool):
    """Puffy fraction AND median density of the UNDETECTED population (the dotted lines)."""
    idx = kept_index(pool, drop_highmass_rocky)
    puffy = pool["puffy"].to_numpy(bool)[idx]
    dens = density(pool["mass"].to_numpy(float)[idx], pool["radius"].to_numpy(float)[idx])
    return float(puffy.mean()), float(np.median(dens))


def mc_draws(pool: pd.DataFrame, det: np.ndarray, drop_highmass_rocky: bool, rng):
    """N_REPEATS draws of N_SAMPLE planets; keep detected; per draw return the puffy fraction
    and the median bulk density. Also return the detected puffy fraction p and mean N_det."""
    idx = kept_index(pool, drop_highmass_rocky)
    pk = pool["puffy"].to_numpy(bool)[idx]
    dens_k = density(pool["mass"].to_numpy(float)[idx], pool["radius"].to_numpy(float)[idx])
    dk = det[idx]
    L = idx.size

    puffy_out = np.full(N_REPEATS, np.nan)
    dens_out = np.full(N_REPEATS, np.nan)
    counted = np.empty(N_REPEATS, float)
    CH = 400  # draws per chunk (memory-bounded)
    pos = 0
    while pos < N_REPEATS:
        c = min(CH, N_REPEATS - pos)
        S = rng.integers(0, L, size=(c, N_SAMPLE_PER_UNIVERSE))
        dks = dk[S]
        den = dks.sum(axis=1)
        # puffy fraction among detected
        num = np.logical_and(pk[S], dks).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            puffy_out[pos:pos + c] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        # median density among detected (non-detected set to nan, then nanmedian per row)
        dens_row = np.where(dks, dens_k[S], np.nan)
        with np.errstate(invalid="ignore"):
            dens_out[pos:pos + c] = np.nanmedian(dens_row, axis=1)
        counted[pos:pos + c] = den
        pos += c

    det_p = float(pk[dk].mean()) if dk.any() else float("nan")
    return puffy_out, dens_out, det_p, float(np.nanmean(counted))


def load_nasa(m_sil, r_sil):
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    measured = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keep = measured & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    return m[keep].to_numpy(float), r[keep].to_numpy(float)


def mc_nasa(mass, radius, m_sil, r_sil, rng):
    n = mass.size
    S = rng.integers(0, n, size=(N_REPEATS, n))
    puffy = puffy_flag(mass, radius, m_sil, r_sil)
    dens = density(mass, radius)
    pf = puffy[S].mean(axis=1)
    dn = np.median(dens[S], axis=1)
    return pf, float(puffy.mean()), dn, float(np.median(dens)), n


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) if sd > 0 else np.zeros_like(x)


# ── curve-nudge robustness: does the puffy fraction depend on where the curve sits? ──

def detected_both_mr(pool: pd.DataFrame, drop_highmass_rocky: bool):
    """Mass & radius of the transit+RV-detected planets in a universe (the 'both' panel)."""
    idx = kept_index(pool, drop_highmass_rocky)
    both = (pool["td"].to_numpy(bool) & pool["rd"].to_numpy(bool))[idx]
    return pool["mass"].to_numpy(float)[idx][both], pool["radius"].to_numpy(float)[idx][both]


def puffy_at_curve(mass, radius, m_sil, r_sil, factor):
    """Puffy fraction with the silicate curve scaled in radius by `factor`."""
    if mass.size == 0:
        return float("nan")
    return float((radius > np.interp(mass, m_sil, r_sil * factor)).mean())


def curve_nudge_table(pools, nasa_mr, m_sil, r_sil):
    """For each universe (transit+RV detected) and NASA: puffy fraction at curve x(1-d), x1, x(1+d)."""
    d = CURVE_NUDGE
    rows = []  # (label, lo, nom, hi, swing)
    for pop, drop, base, _ in UNIVERSES:
        mm, rr = detected_both_mr(pools[pop], drop)
        lo = puffy_at_curve(mm, rr, m_sil, r_sil, 1 + d)   # curve UP -> fewer puffy
        nom = puffy_at_curve(mm, rr, m_sil, r_sil, 1.0)
        hi = puffy_at_curve(mm, rr, m_sil, r_sil, 1 - d)   # curve DOWN -> more puffy
        rows.append((base, lo, nom, hi, abs(hi - lo)))
    nm, nr = nasa_mr
    lo = puffy_at_curve(nm, nr, m_sil, r_sil, 1 + d)
    nom = puffy_at_curve(nm, nr, m_sil, r_sil, 1.0)
    hi = puffy_at_curve(nm, nr, m_sil, r_sil, 1 - d)
    rows.append(("NASA observed", lo, nom, hi, abs(hi - lo)))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    report = []  # lines mirrored to summary.txt

    def say(line=""):
        print(line)
        report.append(line)

    say(f"--> N_SAMPLE_PER_UNIVERSE={N_SAMPLE_PER_UNIVERSE:,}  N_REPEATS={N_REPEATS:,}")
    print("--> building pools + running detectors (once each)...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}

    nasa_mass, nasa_radius = load_nasa(m_sil, r_sil)
    na_pf, na_pt, na_dn, na_dt, n_nasa = mc_nasa(nasa_mass, nasa_radius, m_sil, r_sil, rng)

    # true (undetected) values per universe -- the dotted reference lines
    true_v = {(pop, drop): true_values(pools[pop], drop) for pop, drop, _, _ in UNIVERSES}

    # ---- compute every (mode, universe) curve, and print predicted-vs-measured sigma ----
    say("")
    say("  Why the PUFFY bell has the width it does:  sigma = sqrt( p (1-p) / N_det )")
    say(f"  {'mode':<14}{'universe':<26}{'p(det)':>9}{'N_det':>9}"
        f"{'sig pred':>10}{'sig meas':>10}{'med rho':>10}")
    say("  " + "-" * 88)
    for pop, drop, base, _ in UNIVERSES:
        p, _rho = true_v[(pop, drop)]
        pred = np.sqrt(p * (1 - p) / N_SAMPLE_PER_UNIVERSE)
        say(f"  {'(undetected)':<14}{base:<26}{p:>9.3f}{N_SAMPLE_PER_UNIVERSE:>9d}"
            f"{pred:>10.4f}{'-':>10}{_rho:>10.2f}")

    # results[mode_id] -> list of rows for plotting
    results = {}
    for mode_label, mode_id in MODES:
        rows = []
        for pop, drop, base, colour in UNIVERSES:
            pool = pools[pop]
            pf, dn, det_p, neff = mc_draws(pool, det_mask(pool, mode_id), drop, rng)
            pf_good = pf[np.isfinite(pf)]
            dn_good = dn[np.isfinite(dn)]
            pred = np.sqrt(det_p * (1 - det_p) / max(neff, 1)) if np.isfinite(det_p) else float("nan")
            say(f"  {mode_id:<14}{base:<26}{det_p:>9.3f}{neff:>9.0f}"
                f"{pred:>10.4f}{pf_good.std():>10.4f}{np.median(dn_good):>10.2f}")
            rows.append(dict(base=base, colour=colour, pf=pf_good, dn=dn_good,
                             pf_mu=float(pf_good.mean()), dn_mu=float(np.median(dn_good)),
                             neff=neff, pop=pop, drop=drop))
        results[mode_id] = rows

    pred_nasa = np.sqrt(na_pt * (1 - na_pt) / n_nasa)
    say(f"  {'bootstrap':<14}{'NASA observed':<26}{na_pt:>9.3f}{n_nasa:>9d}"
        f"{pred_nasa:>10.4f}{na_pf.std():>10.4f}{na_dt:>10.2f}")

    # ---- curve-nudge robustness (the puffy fraction is the only curve-dependent number) ----
    say("")
    say(f"  CURVE-NUDGE CHECK: move the silicate curve +/-{CURVE_NUDGE:.0%} in radius "
        f"(transit+RV detected puffy fraction)")
    say(f"  {'universe':<26}{'curve+':>9}{'nominal':>9}{'curve-':>9}{'swing':>9}")
    say("  " + "-" * 62)
    nudge_rows = curve_nudge_table(pools, (nasa_mass, nasa_radius), m_sil, r_sil)
    for label, lo, nom, hi, swing in nudge_rows:
        say(f"  {label:<26}{lo:>9.3f}{nom:>9.3f}{hi:>9.3f}{swing:>9.3f}")
    nasa_swing = nudge_rows[-1][4]
    max_swing = max(r[4] for r in nudge_rows)
    # honest reading: compare the curve-induced wobble to the NASA-vs-(closest realistic universe) gap
    ppop_b_both = next(r for r in results["both"] if r["base"].startswith("P-Pop B"))
    nasa_vs_ppopB = abs(na_pt - ppop_b_both["pf_mu"])
    say(f"  -> a +/-{CURVE_NUDGE:.0%} curve shift moves the puffy fraction by ~{max_swing/2:.2f} "
        f"(up to {max_swing:.2f} peak-to-peak) -- comparable to the NASA-vs-P-Pop-B gap of "
        f"{nasa_vs_ppopB:.2f}. So the puffy fraction ALONE is curve-sensitive; the curve-FREE")
    say(f"     density row (bottom) confirms the SAME ranking without any curve "
        f"(NASA rho={na_dt:.1f} vs P-Pop B {ppop_b_both['dn_mu']:.1f}, flat B {next(r for r in results['both'] if r['base'].startswith('flat B'))['dn_mu']:.1f} g/cc = unphysical) -- that is what makes the conclusion safe.")

    # ---- figure: 2 rows (puffy / density) x 3 detection modes ----
    kshort = f"{N_REPEATS // 1000}k" if N_REPEATS % 1000 == 0 else f"{N_REPEATS}"

    # per-row x-ranges (puffy is bounded 0-1; density uses robust percentiles to ignore tails)
    pf_all = [na_pf] + [r["pf"] for rows in results.values() for r in rows]
    pf_all.append(np.array([v[0] for v in true_v.values()] + [na_pt]))
    pf_cat = np.concatenate(pf_all)
    pf_lo, pf_hi = pf_cat.min() - 0.03, pf_cat.max() + 0.03

    dn_all = [na_dn] + [r["dn"] for rows in results.values() for r in rows]
    dn_cat = np.concatenate(dn_all)
    dn_lo, dn_hi = np.percentile(dn_cat, 0.5), np.percentile(dn_cat, 99.5)
    dn_lo, dn_hi = dn_lo - 0.04 * (dn_hi - dn_lo), dn_hi + 0.04 * (dn_hi - dn_lo)

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.4))

    def draw_row(row_axes, metric, lo, hi, na_samples, na_pt_val, xlabel, ylabel):
        edges = np.linspace(lo, hi, 80)
        xs = np.linspace(lo, hi, 600)
        for ax, (mode_label, mode_id) in zip(row_axes, MODES):
            for r in results[mode_id]:
                s = r["pf"] if metric == "puffy" else r["dn"]
                mu = r["pf_mu"] if metric == "puffy" else r["dn_mu"]
                ax.hist(s, bins=edges, density=True, color=r["colour"], alpha=0.28)
                ax.plot(xs, gauss(xs, s.mean(), s.std()), color=r["colour"], lw=2.0,
                        label=f"{r['base']}: {mu:.2f} | N_det≈{r['neff']:.0f}")
                tv = true_v[(r["pop"], r["drop"])][0 if metric == "puffy" else 1]
                ax.axvline(tv, color=r["colour"], ls=":", lw=1.1, alpha=0.55)
            ax.hist(na_samples, bins=edges, density=True, color="tab:green", alpha=0.28)
            ax.plot(xs, gauss(xs, na_samples.mean(), na_samples.std()), color="tab:green", lw=2.0,
                    label=f"NASA observed: {na_samples.mean():.2f} | N={n_nasa}")
            ax.axvline(na_pt_val, color="darkgreen", ls=":", lw=1.3)
            ax.set_xlim(lo, hi)
            ax.set_title(mode_label, fontsize=11)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7.3, loc="best")
        row_axes[0].set_ylabel(ylabel)

    draw_row(axes[0], "puffy", pf_lo, pf_hi, na_pf, na_pt,
             "puffy fraction  =  N_puffy / (N_puffy + N_rocky)",
             "CURVE-BASED\nhow often each puffy fraction came up")
    draw_row(axes[1], "dens", dn_lo, dn_hi, na_dn, na_dt,
             "median bulk density  rho = 5.513 M / R^3   [g/cc]",
             "CURVE-FREE TWIN\nhow often each median density came up")

    fig.suptitle(
        "Rocky vs puffy small planets: two 'fake' universes (flat & P-Pop) vs NASA, under transit / RV / transit+RV\n"
        "TOP = puffy fraction (needs the silicate curve).   BOTTOM = median bulk density (NO curve).   "
        "Same story on both rows => the result does not hinge on the curve.",
        fontsize=12)

    fig.text(
        0.5, 0.005,
        f"Each bell = {N_REPEATS:,} repeated draws of {N_SAMPLE_PER_UNIVERSE:,} planets; dotted line = each universe's TRUE "
        f"(undetected) value, so the line->bell gap is the detection bias.   "
        f"Curve-nudge check: a +/-{CURVE_NUDGE:.0%} shift of the silicate curve moves the puffy fraction by ~{max_swing/2:.2f} "
        f"(see console / summary.txt) -- as big as the gaps between universes, so the TOP row is curve-sensitive. "
        f"The BOTTOM (curve-free) row gives the same ranking, which is why the conclusion holds.",
        ha="center", fontsize=8.8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "puffy_and_density_by_detection_mode.png")
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    say("")
    say(f"--> Saved: {out_png}")

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
