#!/usr/bin/env python
"""Make every paper figure; outputs go into results/paper/."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from tools.paths import PAPER_FIGURES_DIR, REPO_ROOT

ROOT = REPO_ROOT
PAPER_DIR = PAPER_FIGURES_DIR


@dataclass(frozen=True)
class PaperFigureTask:
    name: str
    description: str
    figures: tuple[str, ...]
    run: Callable[[], None]


@contextmanager
def temporary_argv(program: str, args: tuple[str, ...] = ()):
    original = sys.argv[:]
    sys.argv = [program, *args]
    try:
        yield
    finally:
        sys.argv = original


def download_tess_cdpp() -> None:
    from science.telescopes.tess.download_cdpp import main

    main()


def make_recovery_3x1() -> None:
    from plotting.paper_figures.recovery_3x1 import main

    with temporary_argv("recovery_3x1.py"):
        main()


def make_flat_transit_rv_3x3() -> None:
    from plotting.paper_figures.flat_transit_rv_3x3 import main

    main()


def make_rocky_scatter_gaia60pc() -> None:
    from plotting.paper_figures.rocky_scatter_gaia60pc import main

    main()


def make_flat_rocky_mr_vs_nasa() -> None:
    from plotting.paper_figures.flat_rocky_mr_vs_nasa import main

    main()


def make_mc_comparison_statistic() -> None:
    os.environ.pop("N_DRAWS", None)
    from plotting.paper_figures.mc_comparison_statistic import main

    main()


TASKS = (
    PaperFigureTask(
        name="download_cdpp",
        description="Download SPOC CDPP tables for the TESS recovery panel",
        figures=(),
        run=download_tess_cdpp,
    ),
    PaperFigureTask(
        name="recovery_3x1",
        description="Kepler/TESS/RV recovery panel",
        figures=("recovery_3x1.png",),
        run=make_recovery_3x1,
    ),
    PaperFigureTask(
        name="flat_transit_rv_3x3",
        description="Flat-Otegi transit and RV selection maps",
        figures=("flat_transit_rv_3x3_otegi.png",),
        run=make_flat_transit_rv_3x3,
    ),
    PaperFigureTask(
        name="rocky_scatter_gaia60pc",
        description="Gaia 60 pc rocky scatter figures",
        figures=("rocky_mr_insolation_3panel.png", "rocky_scatter_standalone.png"),
        run=make_rocky_scatter_gaia60pc,
    ),
    PaperFigureTask(
        name="flat_rocky_mr_vs_nasa",
        description="Rocky mass-radius relation and Otegi comparison panels",
        figures=(
            "flat_rocky_mr_relations_2x4_low-insolation_corner.png",
            "flat_otegi_1x2_low-insolation_selection.png",
        ),
        run=make_flat_rocky_mr_vs_nasa,
    ),
    PaperFigureTask(
        name="mc_comparison_statistic",
        description="Monte Carlo comparison statistic",
        figures=("mc_comparison_statistic_5000.png",),
        run=make_mc_comparison_statistic,
    ),
)


def configure_environment() -> None:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig-paper"))


def selected_tasks(argv: list[str]) -> list[PaperFigureTask]:
    wanted = [arg for arg in argv if not arg.startswith("-")]
    by_name = {task.name: task for task in TASKS}
    unknown = set(wanted) - set(by_name)
    if unknown:
        choices = ", ".join(sorted(by_name))
        sys.exit(f"Unknown figure task(s): {', '.join(sorted(unknown))}. Choices: {choices}")
    return [task for task in TASKS if not wanted or task.name in wanted]


def list_tasks() -> None:
    for task in TASKS:
        print(f"{task.name}: {task.description}")
        for figure in task.figures:
            print(f"    results/paper/{figure}")


def assert_figures_written(task: PaperFigureTask, started_at: float) -> None:
    stale = [
        figure for figure in task.figures
        if not (PAPER_DIR / figure).exists()
        or (PAPER_DIR / figure).stat().st_mtime < started_at
    ]
    if stale:
        sys.exit(f"ERROR: {task.name} finished but did not write: {', '.join(stale)}")


def main(argv: list[str]) -> None:
    configure_environment()
    if "--list" in argv:
        list_tasks()
        return

    t_all = time.time()
    for task in selected_tasks(argv):
        print(f"\n{'=' * 70}\n  {task.name}: {task.description}\n{'=' * 70}", flush=True)
        t0 = time.time()
        task.run()
        assert_figures_written(task, t0)
        print(f"  ({time.time() - t0:.0f} s)")

    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min. Figures in {PAPER_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
