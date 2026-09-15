#!/usr/bin/env python
"""Generate the flat-universe populations, then run flat_rocky_mr_vs_nasa and flat_transit_rv_3x3.
Usage: python run/run_flat.py [--rebuild | --plots-only | --skip-plots]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def run_cmd(cmd: list[str], label: str = ""):
    """Run a shell command and log output."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate flat universe populations and run analysis plots."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate flat universe configs even if cached",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip flat universe generation, run plots only",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Generate flat universes but skip plots",
    )

    args = parser.parse_args()

    if not args.plots_only:
        # Generate flat universe configs
        gen_cmd = ["python", "run/flat_universe/generate_flat_universes.py", "--all"]
        if args.rebuild:
            gen_cmd.append("--rebuild")
        run_cmd(gen_cmd, "Generating flat universe configurations")

    if not args.skip_plots:
        # Plot scripts that read the flat universes
        plots = [
            ("plotting/scripts/analysis/multi/flat_rocky_mr_vs_nasa.py", "Flat rocky M-R vs NASA"),
            ("plotting/scripts/analysis/multi/flat_transit_rv_3x3.py", "Flat transit/RV selection map"),
        ]

        for script, label in plots:
            run_cmd(["python", script], label)

    print("\n" + "="*60)
    print("  Done. Outputs in results/figures/analysis/")
    print("="*60)


if __name__ == "__main__":
    main()
