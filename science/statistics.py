"""Reusable statistics and uncertainty models with no presentation dependencies."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom


def binned_fraction_2d(x, y, numerator, denominator, x_bins, y_bins, *, min_count=1):
    """Return numerator/denominator and denominator counts on a 2-D grid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    numerator = np.asarray(numerator, dtype=bool)
    denominator = np.asarray(denominator, dtype=bool)
    valid = np.isfinite(x) & np.isfinite(y)
    total, _, _ = np.histogram2d(
        x[valid & denominator], y[valid & denominator], bins=[x_bins, y_bins]
    )
    selected, _, _ = np.histogram2d(
        x[valid & numerator & denominator], y[valid & numerator & denominator],
        bins=[x_bins, y_bins],
    )
    fraction = np.divide(
        selected, total, out=np.full_like(selected, np.nan, dtype=float), where=total > 0
    )
    fraction[total < min_count] = np.nan
    return fraction, total


def fit_quantile_power_law(x, y, *, quantile=0.90):
    """Fit ``log10(y) = slope*log10(x) + intercept`` at a quantile."""
    log_x = np.log10(np.asarray(x, dtype=float))
    log_y = np.log10(np.asarray(y, dtype=float))
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    log_x, log_y = log_x[valid], log_y[valid]
    if log_x.size < 4:
        return None

    slope, intercept = np.polyfit(log_x, log_y, 1)

    def loss(params):
        prediction = params[0] * log_x + params[1]
        residual = log_y - prediction
        weighted = np.where(
            residual >= 0, quantile * residual, (quantile - 1) * residual
        )
        return np.sum(weighted)

    fallback = (float(slope), float(np.quantile(log_y - slope * log_x, quantile)))
    result = minimize(
        loss, x0=[slope, intercept], method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000},
    )
    if result.success or result.fun < loss(fallback):
        return float(result.x[0]), float(result.x[1])
    return fallback


def binomial_model_posterior(probabilities, successes, trials):
    """Equal-prior model log likelihoods and normalized posterior weights."""
    keys = list(probabilities)
    log_likelihood = np.array([
        binom.logpmf(
            successes, trials, float(np.clip(probabilities[key], 1e-4, 1 - 1e-4))
        )
        for key in keys
    ])
    weights = np.exp(log_likelihood - log_likelihood.max())
    weights /= weights.sum()
    return dict(zip(keys, log_likelihood)), dict(zip(keys, weights))


def gaussian_density(x, mean, standard_deviation):
    """Normal density, or zeros for a degenerate distribution."""
    x = np.asarray(x, dtype=float)
    if standard_deviation <= 0:
        return np.zeros_like(x)
    return (
        np.exp(-0.5 * ((x - mean) / standard_deviation) ** 2)
        / (standard_deviation * np.sqrt(2 * np.pi))
    )


SIMULATED_MEASUREMENT_ERROR = {"mass": 0.20, "radius": 0.046}
NASA_MEASUREMENT_ERROR = {"mass": 0.25, "radius": 0.08}


def perturb_fractional(mass, radius, rng, error=SIMULATED_MEASUREMENT_ERROR):
    """Apply a named log-normal fractional error mapping."""
    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)
    return (
        mass * np.exp(rng.normal(0.0, error["mass"], mass.shape)),
        radius * np.exp(rng.normal(0.0, error["radius"], radius.shape)),
    )


def perturb_asymmetric(values, error_plus, error_minus, rng, *, floor=1e-3):
    """Draw a split-normal measurement using published positive/negative errors."""
    values = np.asarray(values, dtype=float)
    error_plus = np.asarray(error_plus, dtype=float)
    error_minus = np.asarray(error_minus, dtype=float)
    z = rng.normal(size=values.shape)
    result = values + np.where(z >= 0, z * error_plus, z * error_minus)
    return np.clip(result, floor, None)


def fractional_error_to_log10_sigma(
    error_plus, error_minus, value, *, floor=0.0, missing=np.nan
):
    """Convert asymmetric linear errors to a floored log10-space sigma."""
    error = np.maximum(
        np.abs(np.asarray(error_plus, dtype=float)),
        np.abs(np.asarray(error_minus, dtype=float)),
    )
    sigma = (error / np.asarray(value, dtype=float)) / np.log(10.0)
    sigma = np.where(np.isfinite(sigma), sigma, missing)
    return np.maximum(sigma, floor)


__all__ = [
    "NASA_MEASUREMENT_ERROR", "SIMULATED_MEASUREMENT_ERROR", "binned_fraction_2d",
    "binomial_model_posterior", "fit_quantile_power_law",
    "fractional_error_to_log10_sigma", "gaussian_density", "perturb_asymmetric",
    "perturb_fractional",
]
