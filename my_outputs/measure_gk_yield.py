"""Measure per-universe transiting-in-window G and K yield in the existing
Gaia-60pc Kepler/TESS catalogs, using script-44's exact definitions:
  science window: flux_p in (0.1,1e4), radius_p in (0.6,2.2)
  Kepler transiting = transiting_geometric
  TESS   transiting = tess_observed & tess_transiting_geometric
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEP = ROOT / "run" / "kepler" / "data" / "Gaia"
TES = ROOT / "run" / "tess" / "data" / "Gaia_cdpp_v1"
FLUX = (0.1, 1e4)
RAD = (0.6, 2.2)


def as_bool(s):
    if s.dtype == bool:
        return s.fillna(False)
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def stype_clean(df):
    return df["stype"].astype(str).str.strip().str.upper().str[0]


def window(df):
    return ((df["flux_p"] > FLUX[0]) & (df["flux_p"] < FLUX[1])
            & (df["radius_p"] > RAD[0]) & (df["radius_p"] < RAD[1]))


def measure(directory, stem, transit_cols):
    print(f"\n=== {directory.name} / {stem} ===")
    print(f"{'uni':>4} {'G_tr':>7} {'K_tr':>7}")
    g_tot = k_tot = 0
    per_g, per_k = [], []
    for i in range(40):
        f = directory / f"{stem}_{i}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for c in ("flux_p", "radius_p"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["sc"] = stype_clean(d)
        tr = np.ones(len(d), dtype=bool)
        for c in transit_cols:
            tr &= as_bool(d[c]) if c in d.columns else True
        keep = window(d) & tr
        g = int((keep & (d["sc"] == "G")).sum())
        k = int((keep & (d["sc"] == "K")).sum())
        per_g.append(g); per_k.append(k)
        g_tot += g; k_tot += k
        print(f"{i:>4} {g:>7} {k:>7}")
    n = len(per_g)
    print(f"{'TOT':>4} {g_tot:>7} {k_tot:>7}   ({n} universes)")
    print(f"{'mean':>4} {np.mean(per_g):>7.1f} {np.mean(per_k):>7.1f}  per universe")
    return g_tot, k_tot, np.mean(per_g), np.mean(per_k)


kg, kk, kgm, kkm = measure(KEP, "kepler_catalog", ["transiting_geometric"])
tg, tk, tgm, tkm = measure(TES, "tess_catalog", ["tess_observed", "tess_transiting_geometric"])

TARGET = 8000
print("\n================ universes needed to reach 8000 ================")
for name, gtot, ktot, gm, km in (("Kepler", kg, kk, kgm, kkm),
                                  ("TESS", tg, tk, tgm, tkm)):
    ng = int(np.ceil((TARGET - gtot) / gm)) if gm > 0 else None
    nk = int(np.ceil((TARGET - ktot) / km)) if km > 0 else None
    print(f"{name}: G {gtot}->8000 needs ~{ng} more G-universes (@{gm:.0f}/uni); "
          f"K {ktot}->8000 needs ~{nk} more K-universes (@{km:.0f}/uni)")
