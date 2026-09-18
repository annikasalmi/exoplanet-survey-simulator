"""The flat universes: flat_nonphysical (uniform radius, Otegi rocky mass), universe B
(super-Earths and sub-Neptunes) and universe A (B without its super-Earths).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from science.populations.flat import generate_flat_catalog, DEFAULTS
from tools.paths import SILICON_CURVE

OTEGI_ROCKY = dict(mr_C=1.03, mr_beta=0.29)      # R = 1.03 M^0.29 (Otegi et al. 2020)
OTEGI_VOLATILE = dict(mr_C=0.70, mr_beta=0.63)   # R = 0.70 M^0.63 (Otegi et al. 2020)
MR_SCATTER_DEX = 0.15             # log-normal mass scatter around the rocky relation
SUPER_EARTH_MIN_MASS = 2.0        # M_earth
SUPER_EARTH_FRAC_SD = 0.20        # fractional radius width around the silicate line
SUB_NEPTUNE_FRAC_SD = 0.20        # ... and around the volatile-rich relation


@lru_cache(maxsize=1)
def load_silicate():
    """Pure-silicate mass-radius curve, sorted by mass (read once)."""
    d = np.loadtxt(SILICON_CURVE, comments="#")
    o = np.argsort(d[:, 0])
    return d[o, 0].astype(float), d[o, 1].astype(float)


def otegi_volatile_radius(mass):
    return OTEGI_VOLATILE["mr_C"] * mass ** OTEGI_VOLATILE["mr_beta"]


def is_super_earth(mass, radius) -> np.ndarray:
    """On or below the silicate line and above SUPER_EARTH_MIN_MASS."""
    m_sil, r_sil = load_silicate()
    mass = np.asarray(mass, float)
    return (np.asarray(radius, float) <= np.interp(mass, m_sil, r_sil)) & (mass > SUPER_EARTH_MIN_MASS)


def flat_nonphysical(n_planets: int = 150_000, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Uniform radius, Otegi rocky mass. kwargs go to generate_flat_catalog (box, or mr_C/mr_beta)."""
    kw = dict(mass_model="powerlaw", mass_scatter_dex=MR_SCATTER_DEX, **OTEGI_ROCKY)
    kw.update(kwargs)
    return generate_flat_catalog(n_planets, seed=seed, **kw)


def flat_superearths_subneptunes(n_planets: int = 150_000, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Universe B: flat_nonphysical with radii redrawn around the silicate line (super-Earths) or
    Otegi's volatile-rich relation (the rest). Redraws outside the radius box drop."""
    cat = flat_nonphysical(n_planets, seed=seed, **kwargs)
    m_sil, r_sil = load_silicate()
    mass = cat["mass_p"].to_numpy(float)
    se = is_super_earth(mass, cat["radius_p"])
    mu = np.where(se, np.interp(mass, m_sil, r_sil), otegi_volatile_radius(mass))
    frac_sd = np.where(se, SUPER_EARTH_FRAC_SD, SUB_NEPTUNE_FRAC_SD)
    # Own seed stream, so chunked pools seeded 0, 1, 2... never reuse it.
    radius = mu * (1.0 + frac_sd * np.random.default_rng([seed, 1]).standard_normal(mass.size))
    r_lo, r_hi = kwargs.get("radius_lims", DEFAULTS["radius_lims"])
    cat["radius_p"] = radius
    return cat[(radius >= r_lo) & (radius <= r_hi)].reset_index(drop=True)


def flat_subneptunes_only(n_planets: int = 150_000, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Universe A: universe B without its super-Earths."""
    b = flat_superearths_subneptunes(n_planets, seed=seed, **kwargs)
    return b[~is_super_earth(b["mass_p"], b["radius_p"])].reset_index(drop=True)
