"""49_bayesian_relaxed_redo.py — re-run the two-universe binomial composition-odds test
(41_bayesian_cold_rocky_desert.py) and its occurrence-corrected companion
(43_occurrence_corrected_cold_rocky_desert.py) under the relaxed Cold Rocky Desert precision
cuts (30% mass, 10% radius), matching paper_v2 and driver 48. Originals are untouched; the
10M model pools are reused from my_outputs/41.../ by hardlink, so nothing is regenerated.

The composition test itself is unchanged from the old note: same binomial
    O_esc = L_esc / (L_esc + L_prim),
same flat-Otegi universes. Only the NASA precision gate is relaxed (25%/8% -> 30%/10%), which
grows the cold super-Earth panel (S<50, M>2) from 25 to ~35 planets. The prior-dependence
headline (P-Pop vs flat-Otegi) is a sigma-tension result that lives in driver 48 / paper_v2
Table 4; this script produces the binomial (flat-Otegi) side only.

Outputs -> my_outputs/49_bayesian_relaxed/ (model_stats.csv + figures), copied to
paper/figures_v2/ for bayesian_analysis_v2.tex.

Run (from repo root, PYTHONPATH set):
    python scripts/Statistical_Analysis/49_bayesian_relaxed_redo.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import importlib.util
import shutil
import sys
from pathlib import Path

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

STRICT_DIR = ROOT / "my_outputs" / "41_bayesian_cold_rocky_desert"
V2_DIR = ROOT / "my_outputs" / "49_bayesian_relaxed"
FIGS_V2 = ROOT / "paper" / "figures_v2"
POOLS = ["pool_powerlaw_mass_scatter_dex0.15_mr_C1.03_mr_beta0.29_N10000000_s0.npz",
         "pool_independent_N10000000_s0.npz"]
COPY_FIGS = ["model_odds.png", "model_stats.png", "likelihood_detection_maps.png",
             "posterior_predictive_maps.png", "occurrence_corrected_maps.png"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _link_pools():
    """Reuse the strict-run 10M pools without regenerating: hardlink (no extra disk), copy if
    hardlink is unavailable. The pools are the model universes and do not depend on the NASA gate."""
    V2_DIR.mkdir(parents=True, exist_ok=True)
    for name in POOLS:
        src, dst = STRICT_DIR / name, V2_DIR / name
        if dst.exists() or not src.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main():
    _link_pools()
    FIGS_V2.mkdir(parents=True, exist_ok=True)

    s41 = _load("s41_relaxed", HERE / "41_bayesian_cold_rocky_desert.py")
    s41.NASA_MASS_PREC = NEW_MASS_PREC
    s41.NASA_RAD_PREC = NEW_RAD_PREC
    s41._out_dir = lambda: str(V2_DIR)          # pool cache (hardlinked) + saves both land here
    print(f"=== binomial composition odds, relaxed cuts {NEW_MASS_PREC:.0%}/{NEW_RAD_PREC:.0%} ===")
    s41.main("kepler")

    s43 = _load("s43_relaxed", HERE / "43_occurrence_corrected_cold_rocky_desert.py")
    s43._bayes.NASA_MASS_PREC = NEW_MASS_PREC
    s43._bayes.NASA_RAD_PREC = NEW_RAD_PREC
    s43._bayes._out_dir = lambda: str(V2_DIR)   # shared universes read the hardlinked pools here
    s43._out_dir = lambda: str(V2_DIR)          # corrected-map figure + recovery report
    print(f"\n=== occurrence-corrected maps, relaxed overlay {NEW_MASS_PREC:.0%}/{NEW_RAD_PREC:.0%} ===")
    s43.main("kepler")

    for name in COPY_FIGS:
        src = V2_DIR / name
        if src.exists():
            shutil.copy2(src, FIGS_V2 / name)
            print(f"--> figures_v2/{name}")
    print(f"\n--> model_stats.csv -> {V2_DIR / 'model_stats.csv'}")


if __name__ == "__main__":
    main()
