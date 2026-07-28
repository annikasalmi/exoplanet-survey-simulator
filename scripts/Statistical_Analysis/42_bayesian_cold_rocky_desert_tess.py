"""
42_bayesian_cold_rocky_desert_tess.py — TESS counterpart of 41_bayesian_cold_rocky_desert.py.

Identical three-universe Bayesian model comparison in the cold rocky desert, but the transit
leg of the joint detector is TESS (run_tess) instead of Kepler (run_kepler). All method,
box, universe, and likelihood definitions are shared with script 88 — only the transit
mission differs. Outputs land in my_outputs/42_bayesian_cold_rocky_desert_tess/ so the
Kepler (script 88) and TESS results never overwrite each other.

Compare against script 88 (Kepler): a difference in the cold panel's detectability map or
model odds between the two missions is a mission/selection effect, not a change in the
underlying universes. TESS has a much shorter baseline than Kepler, so its cold (long-period)
completeness is lower; expect the cold rocky corner to be more selection-limited here.

Run:
    python scripts/Statistical_Analysis/42_bayesian_cold_rocky_desert_tess.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "bayes_cold_rocky_desert", _HERE / "41_bayesian_cold_rocky_desert.py")
_bayes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bayes)


if __name__ == "__main__":
    _bayes.main(mission="tess")
