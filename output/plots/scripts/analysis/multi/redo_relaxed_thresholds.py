"""48_redo_relaxed_thresholds.py — re-run the paper's threshold-dependent analyses under the
relaxed Cold Rocky Desert definition: precision cuts 30% mass / 10% radius, radius floor
R >= 1.35 R_earth (S < 50 and the silicate ceiling unchanged).

Every v2 draft number traces to one of the parts below. Originals (15/17/20/21) are
loaded by path and patched in memory; nothing is written to paper/figures/ — v2 figures go
to paper/figures_v2/ and printed numbers to stdout.

Run (from repo root, PYTHONPATH set):
    python scripts/statistical_analysis/48_redo_relaxed_thresholds.py census
    python scripts/statistical_analysis/48_redo_relaxed_thresholds.py fvol
    python scripts/statistical_analysis/48_redo_relaxed_thresholds.py maps
    python scripts/statistical_analysis/48_redo_relaxed_thresholds.py pvol
    python scripts/statistical_analysis/48_redo_relaxed_thresholds.py specs
    python scripts/statistical_analysis/48_redo_relaxed_thresholds.py xcold
    python scripts/statistical_analysis/48_redo_relaxed_thresholds.py design
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

NEW_MASS_PREC = 0.30
NEW_RAD_PREC = 0.10
NEW_CORNER_RADIUS = 1.35
N_REPEATS = 4000
COLD_INSOL = 50.0

OUT_DIR = ROOT / "output/plots" / "48_relaxed_redo"
FIGS_V2 = ROOT / "paper" / "figures_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_V2.mkdir(parents=True, exist_ok=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _corner_sub(s15, m_ref, r_ref, shift, mass_prec, rad_prec, r_floor, strict_gt,
                s_max=COLD_INSOL):
    s15.MAX_MASS_REL_UNCERTAINTY = mass_prec
    s15.MAX_RADIUS_REL_UNCERTAINTY = rad_prec
    nasa_all, _ = s15.load_and_filter_nasa(m_ref, r_ref, shift)
    win = s15.restrict_to_window(nasa_all)
    cold = win[win["flux_p"].to_numpy() < s_max].copy()
    r_sil = np.interp(cold["mass_p"], m_ref, r_ref, left=np.nan, right=np.nan)
    above = cold["radius_p"] > r_floor if strict_gt else cold["radius_p"] >= r_floor
    sub = cold[above & (cold["radius_p"] <= r_sil)].copy()
    sub["r_sil"] = np.interp(sub["mass_p"], m_ref, r_ref)
    return win, sub.sort_values("flux_p")


def _occupant_table(sub):
    hdr = (f"{'planet':<14}{'R':>7}{'+dR':>7}{'-dR':>7}{'dR%':>7}"
           f"{'M':>8}{'+dM':>7}{'-dM':>7}{'dM%':>7}{'S':>8}{'R_sil':>8}{'R-Rsil':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i in sub.index:
        rp_, mp_ = sub.at[i, "radius_p"], sub.at[i, "mass_p"]
        rep, rem = abs(sub.at[i, "radius_err_plus"]), abs(sub.at[i, "radius_err_minus"])
        mep, mem = abs(sub.at[i, "mass_err_plus"]), abs(sub.at[i, "mass_err_minus"])
        print(f"{sub.at[i, 'planet_name']:<14}{rp_:7.3f}{rep:7.3f}{rem:7.3f}"
              f"{max(rep, rem) / rp_ * 100:6.1f}%"
              f"{mp_:8.2f}{mep:7.2f}{mem:7.2f}{max(mep, mem) / mp_ * 100:6.1f}%"
              f"{sub.at[i, 'flux_p']:8.2f}{sub.at[i, 'r_sil']:8.3f}"
              f"{rp_ - sub.at[i, 'r_sil']:8.3f}")


def census():
    s15 = _load("s15_census", ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "rocky_scatter_gaia60pc.py")
    m_ref, r_ref = s15.load_rocky_reference_curve()
    shift = s15.compute_rocky_threshold_shift(m_ref, r_ref)

    print("\n" + "=" * 78)
    print("OCCUPANT LADDER (S<50, below silicate curve; window + funnel per script 15)")
    print("=" * 78)
    ladder = [
        ("paper cuts: 25%/8%, R>1.4", 0.25, 0.08, 1.4, True),
        ("relaxed:    30%/10%, R>=1.35", NEW_MASS_PREC, NEW_RAD_PREC, NEW_CORNER_RADIUS, False),
        ("no precision cut (two-sided errors still required), R>=1.35",
         10.0, 10.0, NEW_CORNER_RADIUS, False),
    ]
    for label, mp, rp, rf, strict in ladder:
        win, sub = _corner_sub(s15, m_ref, r_ref, shift, mp, rp, rf, strict)
        print(f"\n[{label}]  window N={len(win)}  occupants={len(sub)}: "
              f"{list(sub['planet_name'])}")

    win, sub = _corner_sub(s15, m_ref, r_ref, shift,
                           NEW_MASS_PREC, NEW_RAD_PREC, NEW_CORNER_RADIUS, False)
    print("\n" + "=" * 78)
    print("SIX-OCCUPANT TABLE (relaxed cuts)")
    print("=" * 78)
    _occupant_table(sub)

    print("\nPanel counts under relaxed cuts (window per script 15):")
    flux = win["flux_p"].to_numpy()
    for lab, sel in [("I<10", flux < 10), ("I<50", flux < 50), ("I>50", flux >= 50)]:
        print(f"  {lab}: N={int(sel.sum())}")
    print(f"  total window N={len(win)}")

    from run.ppop.uniform_generator import generate_flat_catalog
    from run.ppop.flat_detect import TESSData
    cat = generate_flat_catalog(1_000_000, seed=0, mass_model="powerlaw",
                                mr_C=1.03, mr_beta=0.29, mass_scatter_dex=0.15)
    m = cat["mass_p"].to_numpy()
    r = cat["radius_p"].to_numpy()
    f = cat["flux_p"].to_numpy()
    r_sil_flat = np.interp(m, m_ref, r_ref, left=np.nan, right=np.nan)
    cold_rocky = (f < COLD_INSOL) & (r >= NEW_CORNER_RADIUS) & (r <= r_sil_flat)
    rs_au = cat["radius_s"].to_numpy() * TESSData.R_SUN_AU
    rp_au = cat["radius_p"].to_numpy() * TESSData.R_EARTH_AU
    b = cat["semimajor_p"].to_numpy() * np.abs(np.cos(cat["inc_p"].to_numpy())) / rs_au
    transits = b <= 1.0 + rp_au / rs_au
    n_cold = int(cold_rocky.sum())
    p_geom = transits[cold_rocky].mean() if n_cold else np.nan
    print(f"\nGeometric transit probability, cold desert band "
          f"(S<{COLD_INSOL:g}, {NEW_CORNER_RADIUS} <= R <= R_sil), flat universe: "
          f"{p_geom:.3%}  (N_cold={n_cold:,} of 1e6)")
    stype = cat["stype"].astype(str).to_numpy()
    for st in ["A", "F", "G", "K", "M"]:
        sel = cold_rocky & (stype == st)
        if sel.sum() > 20:
            print(f"    {st}: P_geom={transits[sel].mean():.3%}  (N={int(sel.sum()):,})")
    gkm = cold_rocky & np.isin(stype, ["G", "K", "M"])
    print(f"    GKM pooled: P_geom={transits[gkm].mean():.3%}  (N={int(gkm.sum()):,})")

    pp = pd.read_csv(ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv",
                     usecols=lambda c: c in ["radius_p", "mass_p", "inc_p", "semimajor_p",
                                             "radius_s", "flux_p"], low_memory=False)
    pm = pp["mass_p"].to_numpy(float)
    pr = pp["radius_p"].to_numpy(float)
    pf = pp["flux_p"].to_numpy(float)
    p_band = ((pf < COLD_INSOL) & (pr >= NEW_CORNER_RADIUS)
              & (pr <= np.interp(pm, m_ref, r_ref, left=np.nan, right=np.nan)))
    p_rs_au = pp["radius_s"].to_numpy(float) * TESSData.R_SUN_AU
    p_b = pp["semimajor_p"].to_numpy(float) * np.abs(np.cos(pp["inc_p"].to_numpy(float))) / p_rs_au
    p_tr = p_b <= 1.0 + pr * TESSData.R_EARTH_AU / p_rs_au
    print(f"    P-Pop universe 0, same band, pooled (Gaia 60 pc mix): "
          f"P_geom={p_tr[p_band].mean():.3%}  (N={int(p_band.sum()):,})")

    s47 = _load("s47_v2", HERE / "47_fig1_relaxed_cuts.py")
    s47.main()
    src = ROOT / "output/plots" / "47_fig1_relaxed_cuts" / "rocky_mr_insolation_3panel_r135_prec10_30.png"
    shutil.copy(src, FIGS_V2 / "rocky_mr_insolation_3panel_v2.png")
    print(f"\nCopied fig 1 v2 -> {FIGS_V2 / 'rocky_mr_insolation_3panel_v2.png'}")


def fvol():
    S72 = _load("s72_v2", HERE / "puffy_cuts_flat.py")
    S72.NASA_MASS_PREC = NEW_MASS_PREC
    S72.NASA_RAD_PREC = NEW_RAD_PREC
    S72.N_REPEATS = N_REPEATS
    S72.OUT_DIR = str(OUT_DIR / "17_fvol")
    os.makedirs(S72.OUT_DIR, exist_ok=True)
    nasa = S72.load_nasa()
    n_box = len(nasa["m"])
    med_m = np.median(np.maximum(nasa["me1"], nasa["me2"]) / nasa["m"])
    med_r = np.median(np.maximum(nasa["re1"], nasa["re2"]) / nasa["r"])
    n_cold = int((nasa["ins"] < COLD_INSOL).sum())
    n_cold_m2 = int(((nasa["ins"] < COLD_INSOL) & (nasa["m"] > 2.0)).sum())
    print(f"\nNASA precision sample under {NEW_MASS_PREC:.0%}/{NEW_RAD_PREC:.0%} cuts: "
          f"N={n_box} in box (was 125 at 25%/8%)")
    print(f"  S<50: N={n_cold}   cold super-Earth (S<50, M>2): N={n_cold_m2}")
    print(f"  median fractional errors of this sample: mass {med_m:.3f}, radius {med_r:.3f} "
          f"(model noise constants remain {S72.MASS_FRAC_ERR}/{S72.RAD_FRAC_ERR})")
    print("\n--> script 17 table (flat + P-Pop, four cuts), relaxed cuts:")
    S72.main()

    s20 = _load("s20_v2", ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "flat_rocky_mr_vs_nasa.py")
    s20.S72.NASA_MASS_PREC = NEW_MASS_PREC
    s20.S72.NASA_RAD_PREC = NEW_RAD_PREC
    s20.S72.N_REPEATS = N_REPEATS
    s20.PAPER_FIG_DIR = FIGS_V2
    s20.OUT_DIR = str(OUT_DIR / "20_figs")
    os.makedirs(s20.OUT_DIR, exist_ok=True)
    print("\n--> script 20 figures + four-relation tensions, relaxed cuts:")
    s20.main()


def nasabells():
    S72 = _load("s72_nb", HERE / "puffy_cuts_flat.py")
    S72.NASA_MASS_PREC = NEW_MASS_PREC
    S72.NASA_RAD_PREC = NEW_RAD_PREC
    S72.N_REPEATS = N_REPEATS
    m_sil, r_sil = S72.load_silicate()
    rng = np.random.default_rng(S72.RNG_SEED)
    nasa = S72.load_nasa()
    print(f"\nObserved (NASA) volatile-fraction bells under "
          f"{NEW_MASS_PREC:.0%}/{NEW_RAD_PREC:.0%} cuts, {N_REPEATS} draws:")
    for cut_label, cut in S72.CUTS:
        nv, n_eff = S72.mc_nasa(nasa, cut, m_sil, r_sil, rng)
        print(f"  {cut_label:<20} mu={nv.mean():.3f}  sd={nv.std():.3f}  N_eff={n_eff}")


def maps(mission="TESS"):
    s21 = _load("s21_v2", ROOT / "important_plots" / "flat_transit_rv_3x3.py")
    s21.S44.MAX_MASS_REL_UNCERTAINTY = NEW_MASS_PREC
    s21.S44.MAX_RADIUS_REL_UNCERTAINTY = NEW_RAD_PREC
    s21.S44.COLD_CORNER_RADIUS = NEW_CORNER_RADIUS
    s21.TRANSIT_MISSION = mission
    s21.PAPER_FIG_DIR = FIGS_V2
    s21.OUT_DIR = OUT_DIR / "21_maps"
    s21.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--> script 21 selection maps ({mission} transit), desert R>={NEW_CORNER_RADIUS}, "
          f"overlay cuts {NEW_MASS_PREC:.0%}/{NEW_RAD_PREC:.0%}:")
    s21.main()


def _nasa_with_names(S72):
    df = pd.read_csv(S72.NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs()
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs()
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = (prov.str.contains("Mass|Msini", case=False, na=False)
            & ~prov.str.contains("Calc", case=False, na=False))
    prec = ((np.maximum(me1, me2) / m <= S72.NASA_MASS_PREC)
            & (np.maximum(re1, re2) / r <= S72.NASA_RAD_PREC))
    keep = (meas & prec & r.between(S72.BOX["r_lo"], S72.BOX["r_hi"])
            & m.between(S72.BOX["m_lo"], S72.BOX["m_hi"]))
    d = dict(name=df["pl_name"].astype(str)[keep].to_numpy(),
             m=m[keep].to_numpy(), r=r[keep].to_numpy(), ins=ins[keep].to_numpy(),
             me1=me1[keep].to_numpy(), me2=me2[keep].to_numpy(),
             re1=re1[keep].to_numpy(), re2=re2[keep].to_numpy())
    d["me1"] = np.where(np.isfinite(d["me1"]), d["me1"], S72.MASS_FRAC_ERR * d["m"])
    d["me2"] = np.where(np.isfinite(d["me2"]), d["me2"], S72.MASS_FRAC_ERR * d["m"])
    d["re1"] = np.where(np.isfinite(d["re1"]), d["re1"], S72.RAD_FRAC_ERR * d["r"])
    d["re2"] = np.where(np.isfinite(d["re2"]), d["re2"], S72.RAD_FRAC_ERR * d["r"])
    return d


def pvol(s_max=COLD_INSOL):
    S72 = _load("s72_pvol", HERE / "puffy_cuts_flat.py")
    S72.NASA_MASS_PREC = NEW_MASS_PREC
    S72.NASA_RAD_PREC = NEW_RAD_PREC
    S72.N_REPEATS = N_REPEATS
    m_sil, r_sil = S72.load_silicate()

    rng = np.random.default_rng(S72.RNG_SEED)
    nasa = S72.load_nasa()
    if s_max == COLD_INSOL:
        print(f"\nNASA bells ({NEW_MASS_PREC:.0%}/{NEW_RAD_PREC:.0%} cuts, {N_REPEATS} draws, "
              f"nasabells rng sequence):")
        for cut_label, cut in S72.CUTS:
            nv, n_eff = S72.mc_nasa(nasa, cut, m_sil, r_sil, rng)
            print(f"  {cut_label:<20} mu={nv.mean():.3f}  sd={nv.std():.3f}  N_eff={n_eff}")

    named = _nasa_with_names(S72)
    print(f"\nBox check: load_nasa N={len(nasa['m'])}, named replica N={len(named['m'])}")

    n = len(named["m"])
    rng2 = np.random.default_rng(S72.RNG_SEED + 1)
    in_cnt = np.zeros(n)
    vol_cnt = np.zeros(n)
    fv = []
    for _ in range(N_REPEATS):
        zm, zr = rng2.normal(size=n), rng2.normal(size=n)
        mb = np.clip(named["m"] + np.where(zm >= 0, zm * named["me1"], zm * named["me2"]), 1e-3, None)
        rb = np.clip(named["r"] + np.where(zr >= 0, zr * named["re1"], zr * named["re2"]), 1e-3, None)
        inm = (named["ins"] < s_max) & (mb > 2.0)
        vol = rb > np.interp(mb, m_sil, r_sil)
        in_cnt += inm
        vol_cnt += inm & vol
        if inm.sum() >= 5:
            fv.append(vol[inm].mean())
    p_in = in_cnt / N_REPEATS
    p_vol = np.where(in_cnt > 0, vol_cnt / np.maximum(in_cnt, 1), np.nan)
    fv = np.asarray(fv)
    print(f"per-draw cold super-Earth f_vol (fresh rng, same per-draw M>2 cut as mc_nasa): "
          f"mu={fv.mean():.3f}  sd={fv.std():.3f}")

    S72.NASA_MASS_PREC, S72.NASA_RAD_PREC = 0.25, 0.08
    old = _nasa_with_names(S72)
    S72.NASA_MASS_PREC, S72.NASA_RAD_PREC = NEW_MASS_PREC, NEW_RAD_PREC
    old_members = set(old["name"][(old["ins"] < s_max) & (old["m"] > 2.0)])

    member = (named["ins"] < s_max) & (named["m"] > 2.0)
    point_vol = named["r"] > np.interp(named["m"], m_sil, r_sil)
    rows = [i for i in np.argsort(named["ins"]) if member[i]]
    print(f"\nPoint classification of the N={int(member.sum())} cold super-Earths "
          f"(S<{s_max:g}, M>2): "
          f"{int(point_vol[member].sum())} volatile / {int((~point_vol[member]).sum())} rocky")
    print("Membership held at catalog insolation (S errors not propagated; mass/radius only).\n")
    hdr = f"{'planet':<16}{'M':>7}{'R':>7}{'S':>8}{'class':>10}{'P(vol|in)':>11}{'P(in)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i in rows:
        newflag = "" if named["name"][i] in old_members else "   <-- new at 30%/10%"
        print(f"{named['name'][i]:<16}{named['m'][i]:7.2f}{named['r'][i]:7.2f}"
              f"{named['ins'][i]:8.1f}{'volatile' if point_vol[i] else 'ROCKY':>10}"
              f"{p_vol[i]:11.2f}{p_in[i]:8.2f}{newflag}")

    secure_r = [i for i in rows if p_vol[i] <= 0.05 and not point_vol[i]]
    coin = [i for i in rows if 0.2 <= p_vol[i] <= 0.8]
    secure_v = [i for i in rows if p_vol[i] >= 0.95]
    print(f"\nsecure rocky (P(vol|in)<=0.05): {[named['name'][i] for i in secure_r]}")
    print(f"secure volatile (P(vol|in)>=0.95): N={len(secure_v)}")
    print(f"coin flips (0.2<=P<=0.8): {[named['name'][i] for i in coin]}")
    if secure_r:
        s_min = min(named["ins"][i] for i in secure_r)
        print(f"coldest SECURE rocky planet sits at S={s_min:.1f}; every rocky candidate "
              f"colder than that is a coin flip under its own published errors")

    out = pd.DataFrame({
        "planet": named["name"][rows], "mass": named["m"][rows], "radius": named["r"][rows],
        "insol": named["ins"][rows],
        "point_class": np.where(point_vol[rows], "volatile", "rocky"),
        "p_vol_given_in": p_vol[rows], "p_in": p_in[rows],
        "new_at_relaxed": [named["name"][i] not in old_members for i in rows]})
    path = OUT_DIR / (f"pvol_cold_superearths.csv" if s_max == COLD_INSOL
                      else f"pvol_cold_superearths_s{s_max:g}.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}")


def _spec_fig(df, dims, ycol, ysd, anchors, ylabel, path):
    import matplotlib.pyplot as plt
    order = np.argsort(df[ycol].to_numpy(), kind="stable")
    d = df.iloc[order].reset_index(drop=True)
    x = np.arange(len(d))
    nrow = sum(len(v) for v in dims.values())
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.0, nrow / 6.0]))
    if ysd:
        ax0.errorbar(x, d[ycol], yerr=d[ysd], fmt="none", ecolor="0.85", elinewidth=0.6, zorder=1)
    ax0.plot(x, d[ycol], ".", ms=3, color="tab:blue", zorder=2)
    for lab, spec in anchors.items():
        mask = np.ones(len(d), bool)
        for k, v in spec.items():
            mask &= np.isclose(d[k].to_numpy(), v)
        if mask.any():
            xi = int(np.flatnonzero(mask)[0])
            ax0.axvline(xi, color="tab:red", lw=0.8)
            ax0.annotate(f" {lab}: {d.at[xi, ycol]:g}", (xi, float(d[ycol].max())),
                         rotation=90, fontsize=7, ha="right", va="top", color="tab:red")
    ax0.set_ylabel(ylabel)
    colors = {"mass_prec": "tab:blue", "rad_prec": "tab:orange", "r_floor": "tab:green",
              "s_cut": "tab:purple", "mass_min": "tab:brown"}
    y, yticks, ylabels = 0, [], []
    for dim, vals in dims.items():
        for v in vals:
            sel = np.isclose(d[dim].to_numpy(), v)
            ax1.scatter(x[sel], np.full(int(sel.sum()), y), s=1.5, color=colors.get(dim, "k"))
            yticks.append(y)
            ylabels.append(f"{dim}: none" if dim.endswith("prec") and v >= 1 else f"{dim}={v:g}")
            y += 1
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels, fontsize=6)
    ax1.set_xlabel("specification rank (sorted by outcome)")
    ax1.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def specs():
    import io
    from contextlib import redirect_stdout
    import matplotlib
    matplotlib.use("Agg")

    mass_precs = [0.20, 0.25, 0.30, 0.40, 10.0]
    rad_precs = [0.05, 0.08, 0.10, 0.15, 10.0]
    r_floors = [1.30, 1.35, 1.40, 1.45, 1.50]
    s_cuts = [10.0, 30.0, 50.0, 80.0, 100.0]
    mass_mins = [1.5, 2.0, 2.5]
    spec_draws = 1000

    s15 = _load("s15_specs", ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "rocky_scatter_gaia60pc.py")
    m_ref, r_ref = s15.load_rocky_reference_curve()
    shift = s15.compute_rocky_threshold_shift(m_ref, r_ref)

    print(f"\nOccupant grid: {len(mass_precs)}x{len(rad_precs)}x{len(r_floors)}x{len(s_cuts)} "
          f"= {len(mass_precs) * len(rad_precs) * len(r_floors) * len(s_cuts)} specifications")
    occ_rows = []
    for mp in mass_precs:
        for rp in rad_precs:
            with redirect_stdout(io.StringIO()):
                s15.MAX_MASS_REL_UNCERTAINTY = mp
                s15.MAX_RADIUS_REL_UNCERTAINTY = rp
                nasa_all, _ = s15.load_and_filter_nasa(m_ref, r_ref, shift)
                win = s15.restrict_to_window(nasa_all)
            flux = win["flux_p"].to_numpy()
            rad = win["radius_p"].to_numpy()
            rsil = np.interp(win["mass_p"].to_numpy(), m_ref, r_ref, left=np.nan, right=np.nan)
            names = win["planet_name"].to_numpy()
            for rf in r_floors:
                for sc in s_cuts:
                    sel = (flux < sc) & (rad >= rf) & (rad <= rsil)
                    occ_rows.append(dict(mass_prec=mp, rad_prec=rp, r_floor=rf, s_cut=sc,
                                         n_window=len(win), n_occupants=int(sel.sum()),
                                         occupants=";".join(sorted(names[sel]))))
    occ = pd.DataFrame(occ_rows)
    occ.to_csv(OUT_DIR / "spec_curve_occupants.csv", index=False)
    union = sorted({p for s in occ["occupants"] if s for p in s.split(";")})
    print(f"occupants across all specs: min={occ['n_occupants'].min()}  "
          f"median={occ['n_occupants'].median():.0f}  max={occ['n_occupants'].max()}  "
          f"(specs with zero occupants: {(occ['n_occupants'] == 0).mean():.0%})")
    print(f"union of every planet ever counted as occupant ({len(union)}): {union}")
    for lab, spec in [("strict paper spec (25/8, R>=1.40, S<50)",
                       dict(mass_prec=0.25, rad_prec=0.08, r_floor=1.40, s_cut=50.0)),
                      ("relaxed paper spec (30/10, R>=1.35, S<50)",
                       dict(mass_prec=0.30, rad_prec=0.10, r_floor=1.35, s_cut=50.0))]:
        m = np.ones(len(occ), bool)
        for k, v in spec.items():
            m &= np.isclose(occ[k].to_numpy(), v)
        row = occ[m].iloc[0]
        pct = (occ["n_occupants"] < row["n_occupants"]).mean()
        print(f"  {lab}: N={row['n_occupants']}  (percentile {pct:.0%} of specs)")

    S72 = _load("s72_specs", HERE / "puffy_cuts_flat.py")
    S72.N_REPEATS = spec_draws
    m_sil, r_sil = S72.load_silicate()
    rng = np.random.default_rng(S72.RNG_SEED)
    print(f"\nf_vol grid: {len(mass_precs)}x{len(rad_precs)}x{len(mass_mins)}x{len(s_cuts)} "
          f"= {len(mass_precs) * len(rad_precs) * len(mass_mins) * len(s_cuts)} specifications, "
          f"{spec_draws} draws each (production nasabells uses {N_REPEATS})")
    fv_rows = []
    for mp in mass_precs:
        for rp in rad_precs:
            S72.NASA_MASS_PREC, S72.NASA_RAD_PREC = mp, rp
            nasa = S72.load_nasa()
            for mm in mass_mins:
                for sc in s_cuts:
                    nv, n_eff = S72.mc_nasa(nasa, dict(mass_min=mm, insol_max=sc),
                                            m_sil, r_sil, rng)
                    fv_rows.append(dict(mass_prec=mp, rad_prec=rp, mass_min=mm, s_cut=sc,
                                        mu=float(nv.mean()) if nv.size else np.nan,
                                        sd=float(nv.std()) if nv.size else np.nan,
                                        n_eff=n_eff))
    fv = pd.DataFrame(fv_rows)
    fv.to_csv(OUT_DIR / "spec_curve_fvol.csv", index=False)
    ok = fv["mu"].notna()
    rng_span = fv.loc[ok, "mu"].max() - fv.loc[ok, "mu"].min()
    med_sd = fv.loc[ok, "sd"].median()
    print(f"cold f_vol across specs: min={fv.loc[ok, 'mu'].min():.3f}  "
          f"median={fv.loc[ok, 'mu'].median():.3f}  max={fv.loc[ok, 'mu'].max():.3f}")
    print(f"spec-to-spec span {rng_span:.3f} vs median within-spec sd {med_sd:.3f} "
          f"(ratio {rng_span / med_sd:.1f}): thresholds move f_vol by "
          f"{'MORE' if rng_span > med_sd else 'less'} than measurement error does")
    m = (np.isclose(fv["mass_prec"], 0.30) & np.isclose(fv["rad_prec"], 0.10)
         & np.isclose(fv["mass_min"], 2.0) & np.isclose(fv["s_cut"], 50.0))
    row = fv[m].iloc[0]
    print(f"  relaxed paper spec (30/10, M>2, S<50): mu={row['mu']:.3f} sd={row['sd']:.3f} "
          f"N_eff={row['n_eff']}")

    _spec_fig(occ, dict(mass_prec=mass_precs, rad_prec=rad_precs,
                        r_floor=r_floors, s_cut=s_cuts),
              "n_occupants", None,
              {"strict": dict(mass_prec=0.25, rad_prec=0.08, r_floor=1.40, s_cut=50.0),
               "relaxed": dict(mass_prec=0.30, rad_prec=0.10, r_floor=1.35, s_cut=50.0)},
              "N cold-corner occupants", OUT_DIR / "spec_curve_occupants.png")
    _spec_fig(fv, dict(mass_prec=mass_precs, rad_prec=rad_precs,
                       mass_min=mass_mins, s_cut=s_cuts),
              "mu", "sd",
              {"relaxed": dict(mass_prec=0.30, rad_prec=0.10, mass_min=2.0, s_cut=50.0)},
              "cold super-Earth f_vol", OUT_DIR / "spec_curve_fvol.png")


def xcold(s_max=10.0):
    s15 = _load("s15_xcold", ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "rocky_scatter_gaia60pc.py")
    m_ref, r_ref = s15.load_rocky_reference_curve()
    shift = s15.compute_rocky_threshold_shift(m_ref, r_ref)

    print("\n" + "=" * 78)
    print(f"EXTRA-COLD ROCKY REGION: same band ({NEW_CORNER_RADIUS} <= R <= R_sil), S<{s_max:g}")
    print("=" * 78)
    ladder = [
        ("paper cuts: 25%/8%, R>1.4", 0.25, 0.08, 1.4, True),
        ("relaxed:    30%/10%, R>=1.35", NEW_MASS_PREC, NEW_RAD_PREC, NEW_CORNER_RADIUS, False),
        ("no precision cut (two-sided errors still required), R>=1.35",
         10.0, 10.0, NEW_CORNER_RADIUS, False),
    ]
    for label, mp, rp, rf, strict in ladder:
        win, sub = _corner_sub(s15, m_ref, r_ref, shift, mp, rp, rf, strict, s_max=s_max)
        print(f"\n[{label}]  window N={len(win)}  occupants={len(sub)}: "
              f"{list(sub['planet_name'])}")

    win, sub = _corner_sub(s15, m_ref, r_ref, shift, NEW_MASS_PREC, NEW_RAD_PREC,
                           NEW_CORNER_RADIUS, False, s_max=s_max)
    print(f"\nOCCUPANT TABLE (relaxed cuts, S<{s_max:g})")
    _occupant_table(sub)
    if "discovery_facility" in sub.columns:
        for i in sub.index:
            print(f"    {sub.at[i, 'planet_name']:<14} discovered by "
                  f"{sub.at[i, 'discovery_facility']}")
    keep_cols = [c for c in ["planet_name", "radius_p", "radius_err_plus", "radius_err_minus",
                             "mass_p", "mass_err_plus", "mass_err_minus", "flux_p", "r_sil",
                             "discovery_facility"] if c in sub.columns]
    path = OUT_DIR / f"xcold_occupants_s{s_max:g}.csv"
    sub[keep_cols].to_csv(path, index=False)
    print(f"Saved: {path}")

    if "discovery_facility" in win.columns:
        flux = win["flux_p"].to_numpy()
        mass = win["mass_p"].to_numpy()
        for lab, msk in [(f"window S<{s_max:g}", flux < s_max),
                         ("window cold super-Earths (S<50, M>2)", (flux < 50) & (mass > 2.0))]:
            vc = win.loc[msk, "discovery_facility"].fillna("Unknown").value_counts()
            print(f"\nDiscovery facilities, {lab} (N={int(msk.sum())}):")
            for k, v in vc.items():
                print(f"    {k}: {v}")

    S72 = _load("s72_xc", HERE / "puffy_cuts_flat.py")
    S72.NASA_MASS_PREC = NEW_MASS_PREC
    S72.NASA_RAD_PREC = NEW_RAD_PREC
    S72.N_REPEATS = N_REPEATS
    m_sil, r_sil = S72.load_silicate()
    rng = np.random.default_rng(S72.RNG_SEED)
    nasa = S72.load_nasa()
    cuts = [(f"insol<{s_max:g}", dict(insol_max=float(s_max))),
            (f"mass>2 & insol<{s_max:g}", dict(mass_min=2.0, insol_max=float(s_max)))]
    bells = {}
    print(f"\nObserved (NASA) bells, {N_REPEATS} draws:")
    for label, cut in cuts:
        nv, n_eff = S72.mc_nasa(nasa, cut, m_sil, r_sil, rng)
        bells[label] = (float(nv.mean()), float(nv.std()))
        print(f"  {label:<22} mu={nv.mean():.3f}  sd={nv.std():.3f}  N_eff={n_eff}")

    print("\nScenario comparison, same machinery as Table 4 "
          "(P-Pop pool = script 17; flat-Otegi pool = script 20 constants):")
    print("  building P-Pop pool...")
    pools = {"P-Pop": S72.build_pool("ppop", m_sil, r_sil)}
    print("  building flat-Otegi pool...")
    s20 = _load("s20_xc", ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "flat_rocky_mr_vs_nasa.py")
    pools["Flat-Otegi"] = s20.build_arrays(dict(mr_C=1.03, mr_beta=0.29), m_sil, r_sil)
    for label, cut in cuts:
        n_mu, n_sd = bells[label]
        print(f"\n  [{label}]  observed {n_mu:.3f} +/- {n_sd:.3f}")
        for uni, arr in pools.items():
            for drop, scen in [(False, "primordial"), (True, "escape-only")]:
                s, neff = S72.mc_universe(arr, drop, cut, m_sil, r_sil, rng)
                if s.size == 0:
                    print(f"    {uni:<11}{scen:<13} no draws with >=5 planets after the cut")
                    continue
                tens = abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
                print(f"    {uni:<11}{scen:<13} mu={s.mean():.3f} sd={s.std():.3f} "
                      f"({tens:.1f}sigma)  N_eff={neff:.0f}  draws={s.size}")

    pvol(s_max=s_max)


def design(n_draw=2_000_000):
    from run.ppop.uniform_generator import generate_flat_catalog
    from run.ppop.flat_detect import run_kepler, run_tess, run_rv_best, TESSData

    s15 = _load("s15_design", ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "rocky_scatter_gaia60pc.py")
    m_ref, r_ref = s15.load_rocky_reference_curve()

    print(f"\nFlat-Otegi universe, N={n_draw:,} draws, GKM hosts only (teff 2300-6000 K).")
    print(f"Desert band {NEW_CORNER_RADIUS} <= R <= R_sil(M); regions hot S>=50 / cold S<50 / "
          f"xcold S<10 (S follows from the log-uniform period draw, not a flat S prior).")
    cat = generate_flat_catalog(n_draw, seed=0, mass_model="powerlaw",
                                mr_C=1.03, mr_beta=0.29, mass_scatter_dex=0.15,
                                teff_lims=(2300.0, 6000.0))
    m = cat["mass_p"].to_numpy()
    r = cat["radius_p"].to_numpy()
    rsil = np.interp(m, m_ref, r_ref, left=np.nan, right=np.nan)
    band = (r >= NEW_CORNER_RADIUS) & (r <= rsil)
    sub = cat[band].reset_index(drop=True)
    print(f"band members: {len(sub):,} of {n_draw:,}")

    rs_au = sub["radius_s"].to_numpy() * TESSData.R_SUN_AU
    rp_au = sub["radius_p"].to_numpy() * TESSData.R_EARTH_AU
    b = sub["semimajor_p"].to_numpy() * np.abs(np.cos(sub["inc_p"].to_numpy())) / rs_au
    tr = b <= 1.0 + rp_au / rs_au
    idx_tr = np.flatnonzero(tr)
    trans = sub.iloc[idx_tr].reset_index(drop=True)
    print(f"transiting band members: {len(trans):,}; running Kepler + TESS on those, RV on all")

    kep = np.zeros(len(sub), bool)
    kep[idx_tr] = run_kepler(trans)["detected"].to_numpy(bool)
    tes = np.zeros(len(sub), bool)
    tes[idx_tr] = run_tess(trans)["detected"].to_numpy(bool)
    rv_out = run_rv_best(sub, mag_target=12.0)
    rv = (rv_out["detected"].to_numpy(bool) & rv_out["rv_is_target"].to_numpy(bool))

    fs = sub["flux_p"].to_numpy()
    stype = sub["stype"].astype(str).to_numpy()
    regions = [("hot S>=50", fs >= 50.0), ("cold S<50", fs < 50.0), ("xcold S<10", fs < 10.0)]
    comp = {}
    hdr = (f"{'region':<12}{'stars':<7}{'N':>9}{'P_geom':>8}{'Kep|tr':>8}{'TESS|tr':>8}"
           f"{'RV':>8}{'Kep&RV':>9}{'TESS&RV':>9}")
    print("\nFractions among band members; X|tr = among transiting; "
          "Kep&RV / TESS&RV = absolute completeness (geometry included)")
    print(hdr)
    print("-" * len(hdr))
    for reg_label, reg in regions:
        for st_label, st_sel in [("GKM", np.isin(stype, ["G", "K", "M"])),
                                 ("G", stype == "G"), ("K", stype == "K"), ("M", stype == "M")]:
            sel = reg & st_sel
            n = int(sel.sum())
            ntr = int((sel & tr).sum())
            if n < 50 or ntr < 20:
                continue
            row = dict(n=n,
                       p_geom=tr[sel].mean(),
                       kep_tr=(kep & sel).sum() / ntr,
                       tes_tr=(tes & sel).sum() / ntr,
                       rv=rv[sel].mean(),
                       kep_rv=(kep & rv & sel).sum() / n,
                       tes_rv=(tes & rv & sel).sum() / n)
            comp[(reg_label, st_label)] = row
            print(f"{reg_label:<12}{st_label:<7}{n:>9,}{row['p_geom']:>8.2%}{row['kep_tr']:>8.1%}"
                  f"{row['tes_tr']:>8.1%}{row['rv']:>8.1%}{row['kep_rv']:>9.3%}{row['tes_rv']:>9.3%}")
        print()

    print("Selection suppression, GKM pooled (absolute completeness ratios):")
    for mission, key in [("Kepler+RV", "kep_rv"), ("TESS+RV", "tes_rv")]:
        hot = comp.get(("hot S>=50", "GKM"))
        cold = comp.get(("cold S<50", "GKM"))
        xc = comp.get(("xcold S<10", "GKM"))
        if not (hot and cold):
            continue
        sup = cold[key] / hot[key] if hot[key] > 0 else np.nan
        pred = (cold["n"] * cold[key]) / (hot["n"] * hot[key]) if hot[key] > 0 else np.nan
        xtxt = f"xcold {xc[key]:.3%}, " if xc else ""
        print(f"  {mission}: {xtxt}cold {cold[key]:.3%}, hot {hot[key]:.3%}; "
              f"cold/hot completeness ratio {sup:.2f}; predicted detected cold:hot band count "
              f"ratio {pred:.3f} under the flat-period prior (observed 6/48 = {6 / 48:.3f})")
    rows = [dict(region=reg, stars=st, **v) for (reg, st), v in comp.items()]
    path = OUT_DIR / "design_mission_completeness.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "census"
    {"census": census, "fvol": fvol, "maps": maps,
     "mapskep": lambda: maps("Kepler"), "nasabells": nasabells,
     "pvol": pvol, "specs": specs, "xcold": xcold, "design": design}[part]()
