"""Curve-based flat universe: silicate/volatile planets (B), or no super-Earths (A)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from science.populations.universes.flat_baseline import DEFAULTS, flat_nonphysical
from tools.paths import SILICON_CURVE

OTEGI_VOLATILE = dict(mr_C=0.70, mr_beta=0.63)   # R = 0.70 M^0.63 (Otegi et al. 2020)
SUPER_EARTH_MIN_MASS = 2.0        # M_earth
SUPER_EARTH_FRAC_SD = 0.20        # fractional radius width around the silicate line
SUB_NEPTUNE_FRAC_SD = 0.20        # ... and around the volatile-rich relation

@lru_cache(maxsize=1)
def load_silicate():
    """Pure-silicate mass-radius curve, sorted by mass (read once)."""
    d = np.loadtxt(SILICON_CURVE, comments="#")
    o = np.argsort(d[:, 0])
    return d[o, 0].astype(float), d[o, 1].astype(float)


def is_super_earth(mass, radius) -> np.ndarray:
    """On or below the silicate line and above SUPER_EARTH_MIN_MASS."""
    m_sil, r_sil = load_silicate()
    mass = np.asarray(mass, float)
    return (np.asarray(radius, float) <= np.interp(mass, m_sil, r_sil)) & (mass > SUPER_EARTH_MIN_MASS)

def flat_radii_curves(n_planets: int = 150_000, seed: int = 0, *,
                variant: str = "superearths_supneptunes", radius_lims=DEFAULTS["radius_lims"],
                mass_lims=DEFAULTS["mass_lims"], **kwargs) -> pd.DataFrame:
    """Redraw radii around the silicate/volatile curves with an explicit population variant."""
    if variant not in ("superearths_supneptunes", "only_subneptunes"):
        raise ValueError(f"Unknown curve variant: {variant}")
    cat = flat_nonphysical(n_planets, seed=seed, radius_lims=radius_lims,
                        mass_lims=mass_lims, **kwargs)
    m_sil, r_sil = load_silicate()
    mass = cat["mass_p"].to_numpy(float)
    se = is_super_earth(mass, cat["radius_p"])
    volatile_radius = OTEGI_VOLATILE["mr_C"] * mass ** OTEGI_VOLATILE["mr_beta"]
    mu = np.where(se, np.interp(mass, m_sil, r_sil), volatile_radius)
    frac_sd = np.where(se, SUPER_EARTH_FRAC_SD, SUB_NEPTUNE_FRAC_SD)
    # Own seed stream, so chunked pools seeded 0, 1, 2... never reuse it.
    radius = mu * (1.0 + frac_sd * np.random.default_rng([seed, 1]).standard_normal(mass.size))
    r_lo, r_hi = radius_lims
    cat["radius_p"] = radius
    keep = (radius >= r_lo) & (radius <= r_hi)
    if variant == "only_subneptunes":
        keep &= ~is_super_earth(mass, radius)
    return cat[keep].reset_index(drop=True)
