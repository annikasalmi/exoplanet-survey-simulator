"""Regression tests for import-safe analysis scripts."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import sys

import matplotlib
import numpy as np

from science.physics import is_rocky, is_volatile
from science.statistics import (
    binned_fraction_2d, binomial_model_posterior, perturb_fractional,
)
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
    loaded = sys.modules
    for name in ANALYSIS_MODULES:
        loaded.pop(name, None)


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
    bayes_before = (bayes.FLAT_N_POOL, bayes.CHUNK, bayes._OUT_NAME.copy())
    importlib.import_module("plotting.scripts.analysis.mc_comparison_statistic")
    assert (bayes.FLAT_N_POOL, bayes.CHUNK, bayes._OUT_NAME) == bayes_before


def test_science_never_imports_presentation_code():
    offenders = []
    for path in (Path(__file__).resolve().parents[1] / "science").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(
                name == "plotting" or name.startswith("plotting.")
                or name == "matplotlib" or name.startswith("matplotlib.")
                for name in names
            ):
                offenders.append(str(path))
    assert offenders == []


def test_removed_modules_are_not_kept_as_import_shims():
    root = Path(__file__).resolve().parents[1]
    removed = [
        root / "tools" / "exoplanet_catalog.py",
        root / "plotting" / "exoplanet_data_utils.py",
        root / "science" / "cold_desert.py",
        root / "science" / "parameters.py",
        root / "science" / "uncertainty.py",
    ]
    assert not any(path.exists() for path in removed)


def test_curve_classification_has_one_boundary_definition():
    curve_mass = np.array([1.0, 2.0, 3.0])
    curve_radius = np.array([1.0, 1.2, 1.4])
    mass = np.array([1.0, 2.0, 3.0])
    radius = np.array([0.9, 1.2, 1.5])
    np.testing.assert_array_equal(
        is_rocky(mass, radius, curve_mass, curve_radius), [True, True, False]
    )
    np.testing.assert_array_equal(
        is_volatile(mass, radius, curve_mass, curve_radius), [False, False, True]
    )


def test_uncertainty_model_is_reproducible():
    error = {"mass": 0.25, "radius": 0.08}
    first = perturb_fractional(np.ones(4), np.ones(4), np.random.default_rng(7), error)
    second = perturb_fractional(np.ones(4), np.ones(4), np.random.default_rng(7), error)
    np.testing.assert_allclose(first, second)


def test_core_statistics_do_not_require_plotting():
    fraction, count = binned_fraction_2d(
        [1, 1, 2], [1, 1, 2], [True, False, True], [True, True, True],
        [0, 1.5, 3], [0, 1.5, 3], min_count=1,
    )
    np.testing.assert_allclose(count, [[2, 0], [0, 1]])
    assert fraction[0, 0] == 0.5
    log_likelihood, posterior = binomial_model_posterior({"a": 0.5, "b": 0.1}, 5, 10)
    assert set(log_likelihood) == {"a", "b"}
    assert np.isclose(sum(posterior.values()), 1.0)
