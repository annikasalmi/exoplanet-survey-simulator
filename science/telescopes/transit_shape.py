"""Transit signal from geometry instead of a flat box: an opaque planet crossing a quadratically
limb-darkened star on a circular orbit. Detectors use the rms of the dip over T14; divided by the
noise at T14 that is the matched-filter SNR of one transit. Grazing transits need no special case.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Quadratic coefficients (a, b) for a solar-type star: PHOENIX, Teff 5800 K, log g 4.5, [M/H] 0,
# least-squares fit.
LIMB_DARKENING = {
    "TESS": (0.3495, 0.2247),    # Claret 2017, A&A 600, A30, table 15
    "Kepler": (0.4729, 0.1871),  # Claret & Bloemen 2011, A&A 529, A75, tableab, Kp band
}

_LOG_K = np.linspace(np.log(1e-3), np.log(0.5), 80)   # k = Rp / R*
_B_FRAC = np.linspace(0.0, 1.0, 101)                   # b / (1 + k); 1 = last contact, no transit
_N_POINTS = 801                                        # points across the chord


def _overlap(z, k):
    """Fraction of the planet disk (radius k) inside the unit stellar disk, centres z apart."""
    z = np.asarray(z, float)
    f = np.where(z <= 1 - k, 1.0, 0.0)
    part = (z > 1 - k) & (z < 1 + k)
    zz = np.where(part, z, 1.0)
    k0 = np.arccos(np.clip((k**2 + zz**2 - 1) / (2 * k * zz), -1, 1))
    k1 = np.arccos(np.clip((1 - k**2 + zz**2) / (2 * zz), -1, 1))
    lens = k**2 * k0 + k1 - 0.5 * np.sqrt(np.clip(4 * zz**2 - (1 + zz**2 - k**2) ** 2, 0, None))
    return np.where(part, lens / (np.pi * k**2), f)


@lru_cache(maxsize=None)
def _tables(u1: float, u2: float):
    """rms and peak of the dip over T14, both in units of k^2, on the (log k, b/(1+k)) grid."""
    mean_intensity = 1 - u1 / 3 - u2 / 6
    rms = np.zeros((_LOG_K.size, _B_FRAC.size))
    peak = np.zeros_like(rms)
    t = np.linspace(-1, 1, _N_POINTS)
    for i, k in enumerate(np.exp(_LOG_K)):
        for j, bf in enumerate(_B_FRAC[:-1]):
            b = bf * (1 + k)
            half = np.sqrt((1 + k) ** 2 - b**2)
            z = np.sqrt(b**2 + (t * half) ** 2)
            mu = np.sqrt(np.clip(1 - np.minimum(z, 1) ** 2, 0, 1))
            s = _overlap(z, k) * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) / mean_intensity
            rms[i, j] = np.sqrt(np.mean(s**2))
            peak[i, j] = s.max()
    return rms, peak


def _interp(table, k, b):
    bad = ~(np.isfinite(k) & np.isfinite(b))
    k, b = np.where(bad, 0.01, k), np.where(bad, 0.0, b)
    lk = np.clip(np.log(np.clip(k, 1e-12, None)), _LOG_K[0], _LOG_K[-1])
    bf = np.clip(b / (1 + k), 0.0, 1.0)
    x = np.interp(lk, _LOG_K, np.arange(_LOG_K.size))
    y = bf * (_B_FRAC.size - 1)
    i0 = np.clip(np.floor(x).astype(int), 0, _LOG_K.size - 2)
    j0 = np.clip(np.floor(y).astype(int), 0, _B_FRAC.size - 2)
    fx, fy = x - i0, y - j0
    value = ((1 - fx) * (1 - fy) * table[i0, j0] + fx * (1 - fy) * table[i0 + 1, j0]
             + (1 - fx) * fy * table[i0, j0 + 1] + fx * fy * table[i0 + 1, j0 + 1])
    return np.where(bad, np.nan, value)


def signal_rms_ppm(k, b, band: str, observed_depth_ppm=None):
    """rms of the transit dip over T14 (ppm). With a measured mid-transit depth, the model shape is
    scaled to it; otherwise the dip follows from k. Zero for b >= 1 + k."""
    k = np.asarray(k, float)
    b = np.abs(np.asarray(b, float))
    rms, peak = (_interp(t, k, b) for t in _tables(*LIMB_DARKENING[band]))
    model = k**2 * rms * 1e6
    if observed_depth_ppm is None:
        return model
    observed = np.asarray(observed_depth_ppm, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(peak > 0, observed * rms / peak, np.where(np.isfinite(peak), 0.0, np.nan))
    return np.where(np.isfinite(observed), scaled, model)


def t14_hours(p_days, a_over_rstar, k, b):
    """First-to-fourth contact duration for a circular orbit (hours); 0 when the planet misses."""
    a = np.asarray(a_over_rstar, float)
    chord = np.sqrt(np.clip((1 + np.asarray(k, float)) ** 2 - np.asarray(b, float) ** 2, 0, None))
    sin_i = np.sqrt(np.clip(1 - (np.asarray(b, float) / a) ** 2, 1e-12, None))
    return np.asarray(p_days, float) * 24 / np.pi * np.arcsin(np.clip(chord / (a * sin_i), 0, 1))


def impact_from_t14(t14_hr, p_days, a_over_rstar, k):
    """Impact parameter implied by a measured T14 on a circular orbit (clipped to [0, 1 + k))."""
    k = np.asarray(k, float)
    s = np.asarray(a_over_rstar, float) * np.sin(np.pi * np.clip(np.asarray(t14_hr, float) / 24
                                                                 / np.asarray(p_days, float), 0, 0.5))
    return np.sqrt(np.clip((1 + k) ** 2 - s**2, 0, (1 + k) ** 2 * (1 - 1e-6)))


def loglog_interp(x, xp, fp):
    """Interpolate fp(xp) at x in log-log space, extrapolating linearly beyond the ends."""
    lx, lxp, lfp = np.log(np.asarray(x, float)), np.log(xp), np.log(fp)
    i = np.clip(np.searchsorted(lxp, lx) - 1, 0, len(xp) - 2)
    w = (lx - lxp[i]) / (lxp[i + 1] - lxp[i])
    return np.exp(lfp[i] + w * (lfp[i + 1] - lfp[i]))
