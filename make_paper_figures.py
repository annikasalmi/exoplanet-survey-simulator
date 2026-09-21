#!/usr/bin/env python
"""Make every paper figure; outputs go into results/paper/."""

import time

from science.telescopes.tess.download_cdpp import main as download_tess_cdpp
from plotting.paper_figures.flat_rocky_mr_vs_nasa import main as flat_rocky_mr_vs_nasa
from plotting.paper_figures.flat_transit_rv_3x3 import main as flat_transit_rv_3x3
from plotting.paper_figures.mc_comparison_statistic import main as mc_comparison_statistic
from plotting.paper_figures.recovery_3x1 import main as recovery_3x1
from plotting.paper_figures.rocky_scatter_gaia60pc import main as rocky_scatter_gaia60pc

from tools.paths import PAPER_FIGURES_DIR

def main():
    t_all = time.time()


    download_tess_cdpp()
    recovery_3x1()
    flat_transit_rv_3x3()
    rocky_scatter_gaia60pc()
    flat_rocky_mr_vs_nasa()
    mc_comparison_statistic()

    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min. Figures in {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
