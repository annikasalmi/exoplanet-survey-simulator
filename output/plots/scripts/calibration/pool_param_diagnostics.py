"""
09_pool_param_diagnostics.py — WHY does transit favour the flat universe while RV
favours P-Pop? Compare the parameter spaces of the two pools (box-restricted) and the
detection-relevant drivers, side by side.

Run:
    python scripts/09_pool_param_diagnostics.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from tools.paths import LIFESIM_OUTER_DIR
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
FLAT_N_POOL = 300000
R_EARTH_OVER_R_SUN = 1.0 / 109.2


def load(population):
    if population == "flat":
        df = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=0)
    else:
        df = pd.read_csv(PPOP_CATALOG, low_memory=False)
    for c in ["radius_p", "mass_p", "p_orb", "radius_s", "teff_s", "temp_s", "distance_s", "flux_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    r = df["radius_p"]; m = df["mass_p"]
    keep = r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    if "flux_p" in df.columns:
        f = df["flux_p"]
        keep = keep & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"]))
    return df[keep].copy()


def teff_of(df):
    for c in ["teff_s", "temp_s"]:
        if c in df.columns and df[c].notna().any():
            return df[c].to_numpy(float)
    return np.full(len(df), np.nan)


def pct(x, q):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else np.percentile(x, q)


def frac(mask):
    mask = np.asarray(mask)
    return float(np.mean(mask)) if mask.size else np.nan


def main():
    rows = {}
    for pop in ["flat", "ppop"]:
        df = load(pop)
        rp = df["radius_p"].to_numpy(float)
        mp = df["mass_p"].to_numpy(float)
        per = df["p_orb"].to_numpy(float)
        rs = df["radius_s"].to_numpy(float) if "radius_s" in df.columns else np.full(len(df), np.nan)
        teff = teff_of(df)
        dist = df["distance_s"].to_numpy(float) if "distance_s" in df.columns else np.full(len(df), np.nan)

        depth_ppm = (rp * R_EARTH_OVER_R_SUN / rs) ** 2 * 1e6           # transit depth driver
        mstar_proxy = np.where(np.isfinite(rs) & (rs > 0), rs, np.nan)  # M*~R* (solar) rough
        k_proxy = mp * mstar_proxy ** (-2.0 / 3.0) * per ** (-1.0 / 3.0)  # RV K driver (arb units)

        kep = run_kepler(df)
        rv = run_rv_best(df, mag_target=12.0)
        det_t = kep["detected"].to_numpy(bool)
        det_r = rv["detected"].to_numpy(bool)
        trans_geo = kep["transiting_geometric"].to_numpy(bool) if "transiting_geometric" in kep.columns else np.zeros(len(df), bool)
        rv_target = rv["rv_is_target"].to_numpy(bool) if "rv_is_target" in rv.columns else np.zeros(len(df), bool)

        rows[pop] = dict(
            N=len(df),
            rp_med=np.nanmedian(rp), rp_gt18=frac(rp > 1.8), rp_gt15=frac(rp > 1.5),
            mp_med=np.nanmedian(mp), mp_lt05=frac(mp < 0.5), mp_lt1=frac(mp < 1.0),
            per_med=np.nanmedian(per), per_lt50=frac(per < 50),
            teff_med=np.nanmedian(teff), mdwarf=frac(teff < 3900),
            rs_med=np.nanmedian(rs), dist_med=np.nanmedian(dist),
            depth_med=np.nanmedian(depth_ppm), k_med=np.nanmedian(k_proxy),
            trans_geo=frac(trans_geo), rv_target=frac(rv_target),
            det_t_all=frac(det_t), det_t_obs=frac(det_t[trans_geo]) if trans_geo.any() else np.nan,
            det_r_all=frac(det_r), det_r_obs=frac(det_r[rv_target]) if rv_target.any() else np.nan,
        )

    f, p = rows["flat"], rows["ppop"]
    def line(label, kf, kp, fmt="{:.3g}"):
        print(f"  {label:<34}{fmt.format(kf):>14}{fmt.format(kp):>14}")

    print(f"\n{'':<34}{'FLAT':>14}{'P-Pop':>14}")
    print("  " + "-" * 60)
    line("N in box", f["N"], p["N"], "{:d}")
    print("  --- TRANSIT drivers (depth ~ Rp^2 / Rs^2) ---")
    line("median radius_p [Re]", f["rp_med"], p["rp_med"])
    line("frac radius_p > 1.8 Re", f["rp_gt18"], p["rp_gt18"])
    line("frac radius_p > 1.5 Re", f["rp_gt15"], p["rp_gt15"])
    line("median R_star [Rsun]", f["rs_med"], p["rs_med"])
    line("median transit depth [ppm]", f["depth_med"], p["depth_med"])
    line("frac geometrically transiting", f["trans_geo"], p["trans_geo"])
    line("TRANSIT detect rate (all in box)", f["det_t_all"], p["det_t_all"])
    line("TRANSIT detect rate (of transiting)", f["det_t_obs"], p["det_t_obs"])
    print("  --- RV drivers (K ~ Mp * M*^-2/3 * P^-1/3) ---")
    line("median mass_p [Me]", f["mp_med"], p["mp_med"])
    line("frac mass_p < 0.5 Me", f["mp_lt05"], p["mp_lt05"])
    line("frac mass_p < 1.0 Me", f["mp_lt1"], p["mp_lt1"])
    line("median period [d]", f["per_med"], p["per_med"])
    line("frac period < 50 d", f["per_lt50"], p["per_lt50"])
    line("median Teff_star [K]", f["teff_med"], p["teff_med"])
    line("frac M dwarf (Teff<3900)", f["mdwarf"], p["mdwarf"])
    line("median distance [pc]", f["dist_med"], p["dist_med"])
    line("median K proxy (arb)", f["k_med"], p["k_med"])
    line("frac RV target (mag<=12)", f["rv_target"], p["rv_target"])
    line("RV detect rate (all in box)", f["det_r_all"], p["det_r_all"])
    line("RV detect rate (of RV targets)", f["det_r_obs"], p["det_r_obs"])


if __name__ == "__main__":
    main()
