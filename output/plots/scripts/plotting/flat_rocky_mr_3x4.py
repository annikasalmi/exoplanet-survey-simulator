"""
62_flat_rocky_mr_3x4.py

Runs the script-75 3x4 transit/RV/joint detection map (FLAT-universe rocky selection function,
rows = Transit-test [TESS] / RV-test / Transit+RV joint, cols = F G K M) ONCE PER published rocky
mass-radius relation, so you can see how the rocky-super-Earth selection function depends on which
M-R relation you assume.

Four relations (post-2016; Bashi excluded as non-rocky, Chen's Terran branch used instead), each
imposed on the flat universe as mass_model="powerlaw" (R = C·M^β inverted to mass) + a common
log-normal mass scatter:
    Chen & Kipping 2017 · Otegi et al. 2020 · Edmondson et al. 2023 · Müller et al. 2024
=> 4 detection maps, each 3x4.

Only the RV / joint rows change between relations (transit is mass-independent); a higher-mass
relation makes cold rocky super-Earths easier to confirm by RV, brightening the joint row.

Run from repo root:
    python scripts/62_flat_rocky_mr_3x4.py
    python scripts/62_flat_rocky_mr_3x4.py --n-planets 1000000 --instrument HARPS
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.paths import LIFESIM_OUTER_DIR
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse script 75 (which pulls in 44/53) for the whole 3x4 pipeline + renderer.
S75 = _load_module("s75", ROOT / "plot" / "script plots" / "61_flat_transit_rv_3x4.py")
S44, S53 = S75.S44, S75.S53

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_tess

OUT_DIR = ROOT / "output/plots" / "62_flat_rocky_mr_3x4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MR_SCATTER_DEX = 0.15
N_PLANETS_DEFAULT = 500_000
TYPE_BOOST = 3                      # sample every FGKM type this many times denser (A stars omitted)
FGK_TEFF = (3700.0, 7500.0)        # F/G/K Teff span for the boost draw
M_TEFF = (2300.0, 3700.0)          # M Teff span for the boost draw

# (slug, name, equation, {mr_C, mr_beta})   R = C·M^β  (R in R⊕, M in M⊕)   [Chen dropped per request]
RELATIONS = [
    ("otegi2020",    "Otegi et al. 2020",     "R = 1.03 M^0.29  (rocky branch)",        dict(mr_C=1.03, mr_beta=0.29)),
    ("edmondson2023","Edmondson et al. 2023", "R = 0.99 M^0.34  (M ≲ 4-5 M⊕)",          dict(mr_C=0.99, mr_beta=0.34)),
    ("muller2024",   "Müller et al. 2024",    "R = 1.02 M^0.27  (M < 4.37 M⊕)",         dict(mr_C=1.02, mr_beta=0.27)),
]


def build_boosted_catalog(n_planets, seed, kw, scatter):
    """Flat powerlaw catalogue with every FGKM type sampled TYPE_BOOST× denser (A stars omitted — not in
    the FGKM maps). FGK and M each get separate top-up draws so each type is exactly TYPE_BOOST× its base
    count, filling the sparse cold windows for all four columns."""
    base = generate_flat_catalog(n_planets, seed=seed, mass_model="powerlaw",
                                 mass_scatter_dex=scatter, teff_lims=(2300.0, 7500.0), **kw)
    n_fgk = int(base["stype"].isin(["F", "G", "K"]).sum())
    n_m = int((base["stype"] == "M").sum())
    extra_fgk = generate_flat_catalog((TYPE_BOOST - 1) * n_fgk, seed=seed + 101, mass_model="powerlaw",
                                      mass_scatter_dex=scatter, teff_lims=FGK_TEFF, **kw)
    extra_m = generate_flat_catalog((TYPE_BOOST - 1) * n_m, seed=seed + 202, mass_model="powerlaw",
                                    mass_scatter_dex=scatter, teff_lims=M_TEFF, **kw)
    cat = pd.concat([base, extra_fgk, extra_m], ignore_index=True)
    cat["id"] = np.arange(len(cat)); cat["nstar"] = np.arange(len(cat))
    print(f"  base {len(base):,} (FGK {n_fgk:,}, M {n_m:,}) + FGK extra {len(extra_fgk):,} + M extra "
          f"{len(extra_m):,}  ->  FGK ×{TYPE_BOOST} = {TYPE_BOOST * n_fgk:,}, "
          f"M ×{TYPE_BOOST} = {TYPE_BOOST * n_m:,}, total {len(cat):,}")
    return cat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-planets", type=int, default=N_PLANETS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scatter", type=float, default=MR_SCATTER_DEX,
                    help="log-normal mass scatter (dex) around each rocky relation")
    ap.add_argument("--instrument", default="best", help="HARPS | NIRPS | best")
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--mag-target", type=float, default=12.0)
    args = ap.parse_args()

    print("=" * 70)
    print("78 — script-75 3x4 detection map for each of four rocky M-R relations")
    print("=" * 70)

    # NASA rocky overlay + threshold (identical to scripts 44/53/75), computed once.
    m_ref, r_ref = S44.load_rocky_reference_curve()
    shift = S44.compute_rocky_threshold_shift(m_ref, r_ref)
    _, rocky = S44.load_and_filter_nasa(m_ref, r_ref, shift)
    rocky_win = S44.restrict_to_window(rocky)
    color_map, major, counts = S53.build_warm_facility_styles(rocky_win)
    print(f"Rocky NASA planets in window: {len(rocky_win):,}   "
          f"(scatter = {args.scatter} dex, N_flat = {args.n_planets:,})\n")

    for slug, name, eq, kw in RELATIONS:
        print("-" * 70)
        print(f"Relation: {name}   {eq}")
        catalog = build_boosted_catalog(args.n_planets, args.seed, kw, args.scatter)
        ppop = S75.prepare_flat_tess(run_tess(catalog))
        ppop = S53.restrict_ppop_to_rocky(ppop, m_ref, r_ref, shift)
        ppop = S53.add_rv_ok(ppop, args.instrument, args.snr_threshold, args.n_obs, args.mag_target)
        out = OUT_DIR / f"flat_rocky_3x4_{slug}.png"
        S75.plot_flat_3x4(ppop, rocky_win, color_map, major, counts, args,
                          out_path=out, title_extra=f"Rocky M-R relation: {name}   ·   {eq}   ·   FGKM sampled ×{TYPE_BOOST}")

    print("\nDone. 4 maps in", OUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
