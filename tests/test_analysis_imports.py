"""Regression tests for import-safe analysis scripts."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import matplotlib

from tools.paths import RESULTS_DIR


ANALYSIS_MODULES = [
    "plotting.scripts.analysis.bayesian_cold_rocky_desert",
    "plotting.scripts.analysis.flat_rocky_mr_vs_nasa",
    "plotting.scripts.analysis.flat_transit_rv_3x3",
    "plotting.scripts.analysis.mc_comparison_statistic",
    "plotting.scripts.analysis.puffy_cuts_flat",
    "plotting.scripts.analysis.rocky_scatter_gaia60pc",
]

STYLE_KEYS = (
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "legend.fontsize",
    "xtick.labelsize",
    "ytick.labelsize",
)


def _forget_analysis_modules():
    for name in ANALYSIS_MODULES:
        sys.modules.pop(name, None)


def test_analysis_imports_have_no_process_or_filesystem_side_effects(monkeypatch):
    _forget_analysis_modules()
    mkdir_calls = []
    monkeypatch.setattr(Path, "mkdir", lambda self, *args, **kwargs: mkdir_calls.append(self))
    monkeypatch.setattr(
        os, "makedirs", lambda name, *args, **kwargs: mkdir_calls.append(Path(name)))

    path_before = list(sys.path)
    omp_before = os.environ.get("OMP_NUM_THREADS")
    style_before = {key: matplotlib.rcParams[key] for key in STYLE_KEYS}

    for name in ANALYSIS_MODULES:
        importlib.import_module(name)

    results_dir = Path(RESULTS_DIR)
    project_mkdir_calls = [
        path for path in mkdir_calls if path == results_dir or results_dir in path.parents
    ]
    assert project_mkdir_calls == []
    assert sys.path == path_before
    assert os.environ.get("OMP_NUM_THREADS") == omp_before
    assert {key: matplotlib.rcParams[key] for key in STYLE_KEYS} == style_before


def test_analysis_imports_do_not_override_dependency_configuration():
    _forget_analysis_modules()

    puffy = importlib.import_module("plotting.scripts.analysis.puffy_cuts_flat")
    repeats_before = puffy.N_REPEATS
    importlib.import_module("plotting.scripts.analysis.flat_rocky_mr_vs_nasa")
    assert puffy.N_REPEATS == repeats_before

    bayes = importlib.import_module("plotting.scripts.analysis.bayesian_cold_rocky_desert")
    bayes_before = (bayes.FLAT_N_POOL, bayes.CHUNK, bayes.OUT_DIRS.copy())
    importlib.import_module("plotting.scripts.analysis.mc_comparison_statistic")
    assert (bayes.FLAT_N_POOL, bayes.CHUNK, bayes.OUT_DIRS) == bayes_before
