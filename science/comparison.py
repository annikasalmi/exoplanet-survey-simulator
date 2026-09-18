"""Observed-versus-simulated population comparison statistics."""

from __future__ import annotations

import numpy as np

from science.catalogs import COMPARISON_PARAMETER_BOX
from science.physics import is_super_earth, is_volatile
from science.statistics import (
    NASA_MEASUREMENT_ERROR,
    SIMULATED_MEASUREMENT_ERROR,
    binned_fraction_2d,
    perturb_asymmetric,
    perturb_fractional,
)


COLD_DESERT_MAX_INSOLATION = 50.0
INSOLATION_BINS = [
    ("I < 10", COMPARISON_PARAMETER_BOX["f_lo"], 10.0),
    ("I < 50", COMPARISON_PARAMETER_BOX["f_lo"], COLD_DESERT_MAX_INSOLATION),
    ("I > 50", COLD_DESERT_MAX_INSOLATION, COMPARISON_PARAMETER_BOX["f_hi"]),
]


def monte_carlo_population_fraction(
    population,
    cut,
    curve_mass,
    curve_radius,
    rng,
    *,
    drop_super_earths=False,
    sample_size=20_000,
    repeats=10_000,
    error=SIMULATED_MEASUREMENT_ERROR,
):
    """Volatile fractions in repeated detected synthetic surveys."""
    selected = population["joint_detected"].to_numpy(bool, copy=True)
    if drop_super_earths:
        selected &= ~is_super_earth(population["mass"], population["radius"])
    if cut.get("insol_max") is not None:
        selected &= population["insolation"].to_numpy() < cut["insol_max"]
    available = np.flatnonzero(selected)
    if available.size == 0:
        return np.empty(0), 0.0
    mass = population["mass"].to_numpy()
    radius = population["radius"].to_numpy()

    fractions = np.full(repeats, np.nan)
    counts = np.zeros(repeats, dtype=int)
    for index in range(repeats):
        chosen = rng.choice(available, sample_size, replace=True)
        observed_mass, observed_radius = perturb_fractional(
            mass[chosen], radius[chosen], rng, error
        )
        if cut.get("mass_min") is not None:
            keep = observed_mass > cut["mass_min"]
            observed_mass, observed_radius = observed_mass[keep], observed_radius[keep]
        if observed_mass.size < 5:
            continue
        fractions[index] = is_volatile(
            observed_mass, observed_radius, curve_mass, curve_radius
        ).mean()
        counts[index] = observed_mass.size
    valid = np.isfinite(fractions)
    return fractions[valid], float(counts[valid].mean()) if valid.any() else 0.0


def monte_carlo_observed_fraction(
    sample,
    cut,
    curve_mass,
    curve_radius,
    rng,
    *,
    repeats=10_000,
):
    """Volatile fractions from a fixed sample's asymmetric uncertainties."""
    selected = np.ones(len(sample), dtype=bool)
    if cut.get("insol_max") is not None:
        selected &= sample["insolation"].to_numpy() < cut["insol_max"]
    sample = sample.loc[selected]

    fractions = np.full(repeats, np.nan)
    counts = np.zeros(repeats, dtype=int)
    for index in range(repeats):
        mass = perturb_asymmetric(
            sample["mass"], sample["mass_error_plus"], sample["mass_error_minus"], rng
        )
        radius = perturb_asymmetric(
            sample["radius"],
            sample["radius_error_plus"],
            sample["radius_error_minus"],
            rng,
        )
        if cut.get("mass_min") is not None:
            keep = mass > cut["mass_min"]
            mass, radius = mass[keep], radius[keep]
        if mass.size < 5:
            continue
        fractions[index] = is_volatile(
            mass, radius, curve_mass, curve_radius
        ).mean()
        counts[index] = mass.size
    valid = np.isfinite(fractions)
    return fractions[valid], int(round(counts[valid].mean())) if valid.any() else 0


def observed_volatile_count(
    sample, lo, hi, curve_mass, curve_radius, *, mass_min=None,
):
    """Count volatile and total observed planets in an insolation interval."""
    selected = sample["insolation"].between(lo, hi, inclusive="left").to_numpy(copy=True)
    if mass_min is not None:
        selected &= sample["mass"].to_numpy() > mass_min
    volatile = is_volatile(
        sample.loc[selected, "mass"],
        sample.loc[selected, "radius"],
        curve_mass,
        curve_radius,
    )
    return int(volatile.sum()), int(selected.sum())


def observed_fraction_uncertainty(
    sample, lo, hi, curve_mass, curve_radius, rng, *, mass_min=None, repeats=4000,
):
    """Mean and standard deviation from published asymmetric uncertainties."""
    cut = {"insol_max": hi, "mass_min": mass_min}
    interval = sample[sample["insolation"] >= lo]
    fractions, _ = monte_carlo_observed_fraction(
        interval, cut, curve_mass, curve_radius, rng, repeats=repeats
    )
    if fractions.size == 0:
        return np.nan, np.nan
    return float(fractions.mean()), float(fractions.std())


def predicted_volatile_fraction(
    population,
    lo,
    hi,
    curve_mass,
    curve_radius,
    rng,
    *,
    mass_min=None,
    repeats=4000,
    error=SIMULATED_MEASUREMENT_ERROR,
):
    """Mean volatile fraction for a detected synthetic population interval."""
    selected = (
        population["joint_detected"]
        & population["insolation"].between(lo, hi, inclusive="left")
    ).to_numpy()
    mass = population.loc[selected, "mass"].to_numpy()
    radius = population.loc[selected, "radius"].to_numpy()

    fractions = np.full(repeats, np.nan)
    counts = np.zeros(repeats, dtype=int)
    for index in range(repeats):
        observed_mass, observed_radius = perturb_fractional(mass, radius, rng, error)
        if mass_min is not None:
            keep = observed_mass > mass_min
            observed_mass, observed_radius = observed_mass[keep], observed_radius[keep]
        if observed_mass.size < 5:
            continue
        fractions[index] = is_volatile(
            observed_mass, observed_radius, curve_mass, curve_radius
        ).mean()
        counts[index] = observed_mass.size
    valid = np.isfinite(fractions)
    if not valid.any():
        return np.nan, np.nan, 0
    return (
        float(fractions[valid].mean()),
        float(fractions[valid].std()),
        int(counts[valid].mean()),
    )


def detection_fraction_map(
    population, lo, hi, mass_edges, radius_edges, *, min_count=5,
):
    """Joint-detection fraction over a mass-radius grid and insolation interval."""
    in_bin = population["insolation"].between(lo, hi, inclusive="left").to_numpy()
    return binned_fraction_2d(
        population["mass"],
        population["radius"],
        population["joint_detected"].to_numpy() & in_bin,
        population["transit_eligible"].to_numpy() & in_bin,
        mass_edges,
        radius_edges,
        min_count=min_count,
    )


def mock_survey_volatile_fractions(
    population,
    lo,
    hi,
    curve_mass,
    curve_radius,
    rng,
    sample_size,
    *,
    repeats=5000,
    mass_min=2.0,
    error=NASA_MEASUREMENT_ERROR,
    exclude_super_earths=False,
):
    """Volatile fractions in NASA-sized synthetic surveys."""
    selected = (
        population["joint_detected"]
        & population["insolation"].between(lo, hi, inclusive="left")
    ).to_numpy(copy=True)
    if exclude_super_earths:
        selected &= ~is_super_earth(population["mass"], population["radius"])
    mass = population.loc[selected, "mass"].to_numpy()
    radius = population.loc[selected, "radius"].to_numpy()

    fractions = np.full(repeats, np.nan)
    for index in range(repeats):
        observed_mass, observed_radius = perturb_fractional(mass, radius, rng, error)
        eligible = np.flatnonzero(observed_mass > mass_min)
        if eligible.size < sample_size:
            continue
        chosen = rng.choice(eligible, sample_size, replace=False)
        fractions[index] = is_volatile(
            observed_mass[chosen], observed_radius[chosen], curve_mass, curve_radius
        ).mean()
    return fractions[np.isfinite(fractions)]


def observed_volatile_fraction_draws(
    sample,
    lo,
    hi,
    curve_mass,
    curve_radius,
    rng,
    *,
    repeats=5000,
    mass_min=2.0,
    error=NASA_MEASUREMENT_ERROR,
):
    """Volatile fractions after a declared fractional observed-data error model."""
    selected = sample["insolation"].between(lo, hi, inclusive="left").to_numpy()
    mass = sample.loc[selected, "mass"].to_numpy()
    radius = sample.loc[selected, "radius"].to_numpy()

    fractions = np.full(repeats, np.nan)
    for index in range(repeats):
        observed_mass, observed_radius = perturb_fractional(mass, radius, rng, error)
        keep = observed_mass > mass_min
        observed_mass = observed_mass[keep]
        if observed_mass.size == 0:
            continue
        fractions[index] = is_volatile(
            observed_mass, observed_radius[keep], curve_mass, curve_radius
        ).mean()
    return fractions[np.isfinite(fractions)]


__all__ = [
    "COLD_DESERT_MAX_INSOLATION", "INSOLATION_BINS", "detection_fraction_map",
    "mock_survey_volatile_fractions", "monte_carlo_observed_fraction",
    "monte_carlo_population_fraction", "observed_fraction_uncertainty",
    "observed_volatile_count", "observed_volatile_fraction_draws",
    "predicted_volatile_fraction",
]
