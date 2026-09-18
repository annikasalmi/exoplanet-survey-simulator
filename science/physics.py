"""Physical relations and classifications shared across the project."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from tools import physics_constants as const
from tools.paths import SILICON_CURVE


SUPER_EARTH_MIN_MASS = 2.0


def blackbody_spectral_radiance(wavelength_m, temperature):
    """Planck spectral radiance in W sr⁻¹ m⁻³ with stable large exponents."""
    wavelength = np.asarray(wavelength_m, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    exponent = const.h * const.c / (wavelength * const.k * temperature)
    denominator = np.expm1(np.minimum(exponent, 700.0))
    radiance = 2 * const.h * const.c**2 / wavelength**5 / denominator
    return np.where(exponent > 700.0, 0.0, radiance)


@lru_cache(maxsize=None)
def load_mass_radius_curve(path=SILICON_CURVE) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column mass-radius curve, sorted and stripped of bad rows."""
    data = np.loadtxt(Path(path), comments="#")
    mass = np.asarray(data[:, 0], dtype=float)
    radius = np.asarray(data[:, 1], dtype=float)
    valid = np.isfinite(mass) & np.isfinite(radius) & (mass > 0) & (radius > 0)
    order = np.argsort(mass[valid])
    return mass[valid][order], radius[valid][order]


def radius_on_curve(mass, curve_mass, curve_radius, *, shift=0.0, outside="nan"):
    """Interpolate a threshold curve with an explicit out-of-domain policy."""
    if outside == "nan":
        left = right = np.nan
    elif outside == "edge":
        left = right = None
    else:
        raise ValueError("outside must be 'nan' or 'edge'")
    radius = np.asarray(curve_radius, dtype=float) + shift
    return np.interp(
        np.asarray(mass, dtype=float),
        np.asarray(curve_mass, dtype=float),
        radius,
        left=radius[0] if left is None else left,
        right=radius[-1] if right is None else right,
    )


def is_rocky(mass, radius, curve_mass=None, curve_radius=None, *, shift=0.0):
    """Classify planets on or below the supplied rocky mass-radius threshold."""
    if curve_mass is None or curve_radius is None:
        curve_mass, curve_radius = load_mass_radius_curve()
    threshold = radius_on_curve(
        mass, curve_mass, curve_radius, shift=shift, outside="edge"
    )
    return np.asarray(radius, dtype=float) <= threshold


def is_volatile(mass, radius, curve_mass=None, curve_radius=None, *, shift=0.0):
    """Classify planets above the supplied rocky mass-radius threshold."""
    if curve_mass is None or curve_radius is None:
        curve_mass, curve_radius = load_mass_radius_curve()
    threshold = radius_on_curve(
        mass, curve_mass, curve_radius, shift=shift, outside="edge"
    )
    return np.asarray(radius, dtype=float) > threshold


def is_super_earth(mass, radius, *, minimum_mass=SUPER_EARTH_MIN_MASS):
    """The project definition: rocky and more massive than ``minimum_mass``."""
    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)
    rocky = is_rocky(mass, radius)
    return rocky & (mass > minimum_mass)


def infer_stellar_type(teff) -> pd.Series | np.ndarray:
    """Classify effective temperatures using the project's A/F/G/K/M bounds."""
    is_series = isinstance(teff, pd.Series)
    values = pd.to_numeric(teff, errors="coerce") if is_series else np.asarray(teff, dtype=float)
    index = teff.index if is_series else None
    out = np.full(np.shape(values), "Unknown", dtype=object)
    out[(values > 0) & (values < 3700)] = "M"
    out[(values >= 3700) & (values < 5200)] = "K"
    out[(values >= 5200) & (values < 6000)] = "G"
    out[(values >= 6000) & (values < 7500)] = "F"
    out[values >= 7500] = "A"
    return pd.Series(out, index=index, dtype=object) if is_series else out


def add_stellar_type(
    frame: pd.DataFrame,
    *,
    target="stype_clean",
    candidates=(
        "teff_s", "temp_s", "st_teff", "effective_temperature",
        "stellar_eff_temp", "koi_steff",
    ),
) -> pd.DataFrame:
    """Return a copy with a normalized stellar-type column."""
    result = frame.copy()
    if "stype" in result.columns:
        raw = result["stype"].astype(str).str.strip().str.upper()
        result[target] = raw.str[0].where(raw.str.len() > 0, "Unknown")
        return result
    source = next((name for name in candidates if name in result.columns), None)
    result[target] = infer_stellar_type(result[source]) if source else "Unknown"
    return result


__all__ = [
    "SUPER_EARTH_MIN_MASS", "add_stellar_type", "blackbody_spectral_radiance",
    "infer_stellar_type", "is_rocky", "is_super_earth", "is_volatile",
    "load_mass_radius_curve", "radius_on_curve",
]
