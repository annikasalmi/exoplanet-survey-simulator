#!/usr/bin/env python
"""Run flat_rocky_mr_vs_nasa and flat_transit_rv_3x3. Each script draws its own flat catalogues from
seeds and sizes set at its top, so there is nothing to generate beforehand.
Usage: python run/run_flat.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PLOTS = [
    ("plotting/scripts/analysis/multi/flat_rocky_mr_vs_nasa.py", "Flat rocky M-R vs NASA"),
    ("plotting/scripts/analysis/multi/flat_transit_rv_3x3.py", "Flat transit/RV selection map"),
]


def main():
    for script, label in PLOTS:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        result = subprocess.run([sys.executable, script], cwd=ROOT)
        if result.returncode != 0:
            print(f"ERROR: {script} failed with code {result.returncode}")
            sys.exit(1)

    print("\n" + "="*60)
    print("  Done. Outputs in results/figures/analysis/")
    print("="*60)


if __name__ == "__main__":
    main()
