"""
A universe where all planet characteristics have a uniform prior, as defined in DEFAULTS, except for mass and radius.
(ie distance from earth, period, eccentricity, and stellar type are all uniform in the ranges given in DEFAULTS).
Only exception is radii follow Otegi volatile and/or silicate mass-radius relations with scatter after the mass is drawn.
."""

from __future__ import annotations

import numpy as np
import pandas as pd

from science.physics import is_super_earth, load_mass_radius_curve, radius_on_curve
from science.universes.flat_baseline import DEFAULTS, flat_nonphysical

OTEGI_VOLATILE = dict(mr_C=0.70, mr_beta=0.63)   # R = 0.70 M^0.63 (Otegi et al. 2020)
SUPER_EARTH_FRAC_SD = 0.20        # fractional radius width around the silicate line
SUB_NEPTUNE_FRAC_SD = 0.20        # ... and around the volatile-rich relation


def flat_radii_curves(n_planets: int = 150_000, seed: int = 0, *,
                variant: str = "superearths_supneptunes", radius_lims=DEFAULTS["radius_lims"],
                mass_lims=DEFAULTS["mass_lims"], **kwargs) -> pd.DataFrame:
    """Redraw radii around the silicate/volatile curves with an explicit population variant."""
    if variant not in ("superearths_supneptunes", "only_subneptunes"):
        raise ValueError(f"Unknown curve variant: {variant}")
    cat = flat_nonphysical(n_planets, seed=seed, radius_lims=radius_lims,
                        mass_lims=mass_lims, **kwargs)
    m_sil, r_sil = load_mass_radius_curve()
    mass = cat["mass_p"].to_numpy(float)
    se = is_super_earth(mass, cat["radius_p"])
    volatile_radius = OTEGI_VOLATILE["mr_C"] * mass ** OTEGI_VOLATILE["mr_beta"]
    silicate_radius = radius_on_curve(mass, m_sil, r_sil, outside="edge")
    mu = np.where(se, silicate_radius, volatile_radius)
    frac_sd = np.where(se, SUPER_EARTH_FRAC_SD, SUB_NEPTUNE_FRAC_SD)
    # Own seed stream, so chunked pools seeded 0, 1, 2... never reuse it.
    radius = mu * (1.0 + frac_sd * np.random.default_rng([seed, 1]).standard_normal(mass.size))
    r_lo, r_hi = radius_lims
    cat["radius_p"] = radius
    keep = (radius >= r_lo) & (radius <= r_hi)
    if variant == "only_subneptunes":
        keep &= ~is_super_earth(mass, radius)
    return cat[keep].reset_index(drop=True)
