#!/usr/bin/env python
"""Make every paper figure; goes into into results/paper/
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from tools.paths import REPO_ROOT, PAPER_FIGURES_DIR

ROOT = REPO_ROOT
PAPER_DIR = PAPER_FIGURES_DIR

# (script, extra args, figures it writes to results/paper/)
STEPS = [
    ("science/telescopes/tess/download_cdpp.py", [], []),  # SPOC CDPP tables for the TESS panel
    ("plotting/scripts/calibration/recovery_3x1.py", [],
     ["recovery_3x1.png"]),
    ("plotting/scripts/analysis/flat_transit_rv_3x3.py", [],
     ["flat_transit_rv_3x3_otegi.png"]),
    ("plotting/scripts/analysis/rocky_scatter_gaia60pc.py", [],
     ["rocky_mr_insolation_3panel.png", "rocky_scatter_standalone.png"]),
    ("plotting/scripts/analysis/flat_rocky_mr_vs_nasa.py", [],
     ["flat_rocky_mr_relations_2x4_low-insolation_corner.png",
      "flat_otegi_1x2_low-insolation_selection.png"]),
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

    # Children import tools/, plotting/ and science/ from this checkout even when another
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
