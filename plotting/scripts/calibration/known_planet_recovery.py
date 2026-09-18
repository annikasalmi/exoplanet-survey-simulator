"""Point the TESS and Kepler detectors at real planets and score how well they reproduce what each
mission actually detected. TESS: model SNR vs SPOC SNR on SPOC TOIs, and detected-or-not for every
known transiting planet against the TOI list (matched by position and period). Kepler: model MES vs
DR25 MES on KOIs. Run: python plotting/scripts/calibration/known_planet_recovery.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from tools.paths import PSCOMPPARS_CSV, EXOFOP_TOI_CSV
from science.catalogs import read_nasa_csv
from science.telescopes.tess.detection_model import TESSData
from plotting.scripts.calibration import kepler_calibration as kcal
from plotting.scripts.calibration import tess_calibration as tcal

THRESHOLD = 7.1
MATCH_ARCSEC = 20.0
MATCH_PERIOD_FRAC = 0.01
PERIOD_HARMONICS = (1 / 3, 1 / 2, 1, 2, 3)   # a TOI at a period alias still found the planet
TESS_FACILITY = "Transiting Exoplanet Survey Satellite (TESS)"


def _ratio_stats(model, official):
    ok = (official > 0) & np.isfinite(model) & np.isfinite(official)
    r = model[ok] / official[ok]
    found = official[ok] >= THRESHOLD
    return {"N": int(ok.sum()), "median ratio": float(np.median(r)),
            "16-84%": f"{np.percentile(r, 16):.2f}-{np.percentile(r, 84):.2f}",
            "recovered": float((model[ok][found] >= THRESHOLD).mean())}


def tess_vs_spoc():
    out = tcal.run_detector(tcal.prepare(tcal.load_or_download()))
    return _ratio_stats(pd.to_numeric(out["tess_snr"], errors="coerce").to_numpy(float),
                        pd.to_numeric(out["official_snr"], errors="coerce").to_numpy(float))


def kepler_vs_dr25():
    out = kcal.run_detector(kcal.prepare(kcal.load_or_download()))
    return _ratio_stats(pd.to_numeric(out["kepler_mes"], errors="coerce").to_numpy(float),
                        pd.to_numeric(out["koi_max_mult_ev"], errors="coerce").to_numpy(float))


def tess_vs_toi_list():
    """Every known transiting planet: did TESS detect it (a matching TOI, or a TESS discovery), and does
    the model say so? Scored with measured depths and with depths from radii, as simulations use."""
    d = read_nasa_csv(PSCOMPPARS_CSV)
    d = d[(d.tran_flag == 1) & d.pl_orbper.notna() & d.ra.notna()].reset_index(drop=True)
    t = read_nasa_csv(EXOFOP_TOI_CSV)
    t = t[t["TFOPWG Disposition"] != "FA"]
    tc = SkyCoord(t.RA.values, t.Dec.values, unit=(u.hourangle, u.deg))
    pc = SkyCoord(d.ra.values * u.deg, d.dec.values * u.deg)
    ip, it, _, _ = tc.search_around_sky(pc, MATCH_ARCSEC * u.arcsec)
    ratio = pd.to_numeric(t["Period (days)"], errors="coerce").values[it] / d.pl_orbper.values[ip]
    hit = np.zeros(len(ip), bool)
    for h in PERIOD_HARMONICS:
        hit |= np.abs(ratio / h - 1) < MATCH_PERIOD_FRAC
    truth = np.zeros(len(d), bool)
    truth[np.unique(ip[hit])] = True
    truth |= (d.disc_facility == TESS_FACILITY).to_numpy()

    r = pd.to_numeric(d.pl_rade, errors="coerce").to_numpy()
    p = d.pl_orbper.to_numpy()
    rows = []
    for path, frame in [("measured depth", d), ("depth from radii", d.drop(columns=["pl_trandep", "pl_trandur"]))]:
        out = TESSData(frame.copy(), source="pscomppars", use_tesspoint=True,
                       condition_on_observed=False).determine_detectable()
        pred = out["detected"].astype(bool).to_numpy()
        for label, m in [("all transiting", np.ones(len(d), bool)),
                         ("period > 10 d, R < 2", (p > 10) & (r < 2))]:
            rows.append({"inputs": path, "sample": label, "N": int(m.sum()),
                         "TESS found": int(truth[m].sum()), "model predicts": int(pred[m].sum()),
                         "recovered": float((pred & truth & m).sum() / truth[m].sum()),
                         "false detections": int((pred & ~truth & m).sum()),
                         "agree": float(((pred == truth) & m).sum() / m.sum())})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 160)
    print("\n=== TESS model SNR vs SPOC SNR (SPOC TOIs, their own sectors and CDPP) ===")
    print(tess_vs_spoc())
    print("\n=== Kepler model MES vs DR25 MES (KOIs, their own CDPP) ===")
    print(kepler_vs_dr25())
    print("\n=== TESS: known transiting planets vs the TOI list (real pointings) ===")
    print(tess_vs_toi_list().to_string(index=False, float_format=lambda x: f"{x:.1%}"))


if __name__ == "__main__":
    main()
