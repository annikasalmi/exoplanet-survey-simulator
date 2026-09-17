#!/usr/bin/env python
"""Make every paper figure from a fresh clone, into results/paper/. None of them needs a P-Pop
universe: the flat catalogues are drawn inside each script, the rest is data/ or a download
(NASA Archive, MAST) cached under results/catalogs/. Fails if a script errors or leaves a listed
figure unwritten.
Usage: python run/make_paper_figures.py [--list] [script_name ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "results" / "paper"

# (script, extra args, figures it writes to results/paper/)
STEPS = [
    ("plotting/scripts/calibration/kepler_calibration.py", [],
     ["kepler_3in1_calibration.png"]),
    ("science/telescopes/tess/download_cdpp.py", [], []),  # SPOC CDPP tables for tess_calibration
    ("plotting/scripts/calibration/tess_calibration.py", [],
     ["tess_3in1_calibration.png"]),
    ("plotting/scripts/calibration/rv_detector_check.py", ["--fig1-only"],
     ["rv_k_vs_published_rvamp.png", "rv_sigmaK_3in1.png"]),
    ("plotting/scripts/analysis/flat_transit_rv_3x3.py", [],
     ["flat_transit_rv_3x3_otegi.png"]),
    ("plotting/scripts/analysis/rocky_scatter_gaia60pc.py", [],
     ["rocky_mr_insolation_3panel.png", "rocky_scatter_standalone.png"]),
    ("plotting/scripts/analysis/flat_rocky_mr_vs_nasa.py", [],
     ["flat_rocky_mr_relations_2x4_cold_corner.png", "flat_otegi_2x2_before_after.png",
      "flat_otegi_2x1_cold_cut.png", "flat_rocky_mr_2col_chen_otegi_cold.png",
      "flat_otegi_1x2_cold_cut.png"]),
    ("plotting/scripts/analysis/mc_comparison_statistic.py", [],
     ["mc_comparison_statistic_5000.png"]),
]


def main(argv):
    if "--list" in argv:
        for script, args, figures in STEPS:
            print(" ".join([script] + args))
            for f in figures:
                print(f"    results/paper/{f}")
        return

    wanted = [a for a in argv if not a.startswith("-")]
    steps = [s for s in STEPS if not wanted or Path(s[0]).stem in wanted]
    unknown = set(wanted) - {Path(s[0]).stem for s in STEPS}
    if unknown:
        sys.exit(f"Unknown script(s): {', '.join(sorted(unknown))}. See --list.")

    # Children import tools/, run/ and plotting/ from this checkout even when another
    # checkout is pip-installed in the environment.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.pop("N_DRAWS", None)  # mc_comparison_statistic's default, 5000, is the paper's
    t_all = time.time()
    for script, args, figures in steps:
        print(f"\n{'=' * 70}\n  {script} {' '.join(args)}\n{'=' * 70}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, script, *args], cwd=ROOT, env=env)
        if result.returncode != 0:
            sys.exit(f"ERROR: {script} exited with code {result.returncode}")
        # A copy left over from an earlier run must not count as this run's output.
        stale = [f for f in figures
                 if not (PAPER_DIR / f).exists() or (PAPER_DIR / f).stat().st_mtime < t0]
        if stale:
            sys.exit(f"ERROR: {script} finished but did not write: {', '.join(stale)}")
        print(f"  ({time.time() - t0:.0f} s)")

    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min. Figures in {PAPER_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
