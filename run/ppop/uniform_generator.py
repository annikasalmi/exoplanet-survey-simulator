"""
uniform_generator.py — fully-FLAT synthetic planet catalogue for detector tests.

Every parameter is drawn FLAT and INDEPENDENT within P-Pop-style bounds. This is
NOT P-Pop: it does not use the Gaia star catalogue, the SAG13/Dressing/Bergsten
occurrence rates, the Chen2017 mass-from-radius relation, or any P-Pop orbit
model. The point is a control population with no astrophysical prior baked in, so
when you run KeplerData / TESSData / RVData on it you read off the detector's
sensitivity edge directly instead of the occurrence-rate distribution.

What is drawn flat (independently):
    radius_p   U[0.5, 2.2]   R_earth        (linear; small range)
    mass_p     logU[0.1, 12] M_earth        (independent of radius -> M-R scatter)
    p_orb      logU[0.5, 20000] d           (period; P-Pop SAG13 range)
    ecc_p      U[0, 0.9]
    inc        isotropic (cos i ~ U[0,1])   -> realistic transit geometry
    teff_s     U[2300, 9700] K              (flat -> equal A..M, unlike real Gaia)
    distance_s U[1, 60] pc                  (P-Pop distance range)

Derived (fundamental physics the detectors themselves assume — NOT P-Pop priors):
    radius_s, mass_s  from teff_s via a main-sequence interpolation (so the host
                      stars are physical rather than impossible R/T/M combos)
    l_sun  = radius_s^2 * (teff_s/5772)^4
    a (semimajor_p) = (mass_s * (p_orb/yr)^2)^(1/3)            [Kepler's third law]
    flux_p (insolation) = l_sun / a^2                          [coupled to orbit]

Insolation is COUPLED to the orbit (as requested): it is derived from period +
star, then the sample is rejection-filtered to the bound [1e-2, 1e4] I_earth, so
hotter/closer planets genuinely transit more — the radius-insolation and
mass-radius maps then show real detector structure rather than a flat field.

Apparent magnitudes are not drawn directly; each detector derives its band
magnitude from l_sun + distance_s + teff_s, so brightness is consistent across
Kepler / TESS V / RV V / RV J(NIRPS).

Usage:
    from run.ppop.uniform_generator import generate_flat_catalog, get_or_build_catalog
    df = generate_flat_catalog(n_planets=200000, seed=0)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# Default output location for cached flat catalogues.
UNIFORM_OUT_DIR = os.path.join(ROOT, "my_outputs", "uniform_box")

# Default parameter box (matches P-Pop bounds; small-planet focus per request).
DEFAULTS = dict(
    radius_lims=(0.5, 2.2),       # R_earth
    mass_lims=(0.1, 12.0),        # M_earth
    period_lims=(0.5, 20000.0),   # d
    ecc_lims=(0.0, 0.9),
    teff_lims=(2300.0, 9700.0),   # K
    distance_lims=(1.0, 60.0),    # pc
    insol_lims=(1e-2, 1e4),       # I_earth (rejection bound on the derived flux)
)

# Main-sequence Teff -> (radius, mass) anchors (Pecaut & Mamajek 2013, dwarfs),
# ascending in Teff for np.interp. Used only to give each flat-drawn Teff a
# physical stellar radius/mass; this is a stellar relation, not a planet prior.
_MS_TEFF = np.array([2300., 2500., 2850., 3000., 3150., 3400., 3550., 3870.,
                     4400., 5280., 5500., 5900., 6500., 7200., 9700.])
_MS_RAD = np.array([0.09, 0.11, 0.15, 0.20, 0.26, 0.36, 0.45, 0.59,
                    0.70, 0.85, 0.92, 1.10, 1.30, 1.50, 2.10])
_MS_MASS = np.array([0.08, 0.085, 0.10, 0.15, 0.21, 0.35, 0.44, 0.57,
                     0.68, 0.88, 0.92, 1.05, 1.30, 1.60, 2.10])

_YEAR_D = 365.25


def _stype_from_teff(teff):
    t = np.asarray(teff, dtype=float)
    out = np.full(t.shape, "M", dtype=object)
    out[t >= 3700] = "K"
    out[t >= 5200] = "G"
    out[t >= 6000] = "F"
    out[t >= 7500] = "A"
    return out


def _logU(rng, lo, hi, n):
    return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n)


# --- mass-radius relations (optional; the default flat universe keeps M independent of R) ---
# "independent": mass drawn log-uniform, decoupled from radius (the prior-free control).
# "mean":  mass from radius via the MEAN Chen2017/Forecaster broken power law (average of the
#          shipped posterior) + tunable log-normal scatter -> a physical R-M relation.
# "ppop":  mass from radius via P-Pop's full Chen2017/Forecaster (native probabilistic scatter).
_MEAN_MR_CURVE = None       # cached (R_grid, M_grid) for the mean-relation inverse R->M


def _mean_forecaster_curve():
    """Deterministic mean Forecaster M->R curve, returned as ascending (R_grid, M_grid)."""
    global _MEAN_MR_CURVE
    if _MEAN_MR_CURVE is None:
        import h5py
        hp = ROOT / "PPop" / "MassModels" / "Forecaster" / "fitting_parameters.h5"
        with h5py.File(hp, "r") as h5:
            mu = h5["hyper_posterior"][:].mean(0)          # mean of posterior hyperparameters
        c0, slope, trans = mu[0], mu[1:5], mu[9:12]        # [c0, slope(4), sigma(4), trans(3)]
        c = np.zeros(4); c[0] = c0
        for i in range(1, 4):
            c[i] = c[i - 1] + trans[i - 1] * (slope[i - 1] - slope[i])
        ts = np.insert(np.insert(trans, 3, np.inf), 0, -np.inf)
        logM = np.linspace(-3.0, 3.0, 4000)
        logR = np.zeros_like(logM)
        for i in range(4):
            ind = (logM >= ts[i]) & (logM < ts[i + 1])
            logR[ind] = c[i] + logM[ind] * slope[i]
        R, M = 10.0 ** logR, 10.0 ** logM
        o = np.argsort(R)
        _MEAN_MR_CURVE = (R[o], M[o])
    return _MEAN_MR_CURVE


def _mass_from_radius(radius, mass_model, rng, mass_lims, scatter_dex, mr_C=1.0, mr_beta=0.28):
    if mass_model == "mean":
        Rg, Mg = _mean_forecaster_curve()
        logm = np.log10(np.interp(radius, Rg, Mg))
        if scatter_dex > 0:
            logm = logm + rng.normal(0.0, scatter_dex, len(radius))
        mass = 10.0 ** logm
    elif mass_model == "powerlaw":
        # published rocky/small-planet fit R = mr_C * M^mr_beta, inverted to M = (R/mr_C)^(1/mr_beta)
        logm = np.log10((np.asarray(radius, float) / mr_C) ** (1.0 / mr_beta))
        if scatter_dex > 0:
            logm = logm + rng.normal(0.0, scatter_dex, len(radius))
        mass = 10.0 ** logm
    elif mass_model == "ppop":
        from PPop.MassModels.Chen2017 import MassModel
        mass = MassModel(rng).RadiusToMass(np.asarray(radius, float))
    else:
        raise ValueError(f"unknown mass_model {mass_model!r}")
    return np.clip(mass, mass_lims[0], mass_lims[1])


def generate_flat_catalog(
    n_planets: int = 200000,
    seed: int = 0,
    *,
    radius_lims=DEFAULTS["radius_lims"],
    mass_lims=DEFAULTS["mass_lims"],
    period_lims=DEFAULTS["period_lims"],
    ecc_lims=DEFAULTS["ecc_lims"],
    teff_lims=DEFAULTS["teff_lims"],
    distance_lims=DEFAULTS["distance_lims"],
    insol_lims=DEFAULTS["insol_lims"],
    mass_model: str = "independent",
    mass_scatter_dex: float = 0.05,
    mr_C: float = 1.0,
    mr_beta: float = 0.28,
) -> pd.DataFrame:
    """
    Build a fully-flat synthetic planet catalogue (see module docstring).

    Returns a DataFrame with the detector-ready columns:
        radius_p, mass_p, p_orb, ecc_p, inc_p, semimajor_p, flux_p,
        radius_s, mass_s, teff_s, temp_s, l_sun, distance_s, ra, dec, stype, nstar, id
    """
    rng = np.random.default_rng(seed)
    keep_R, keep_M, keep_P, keep_E, keep_I = [], [], [], [], []
    keep_inc, keep_T, keep_Rs, keep_Ms, keep_D, keep_L, keep_a, keep_F = ([] for _ in range(8))

    have = 0
    insol_lo, insol_hi = insol_lims
    # Oversample and reject on the insolation bound; loop until we have n_planets.
    while have < n_planets:
        batch = max(int((n_planets - have) * 1.8), 20000)

        radius_p = rng.uniform(radius_lims[0], radius_lims[1], batch)
        mass_p = _logU(rng, mass_lims[0], mass_lims[1], batch)
        p_orb = _logU(rng, period_lims[0], period_lims[1], batch)
        ecc_p = rng.uniform(ecc_lims[0], ecc_lims[1], batch)
        inc_p = np.arccos(rng.uniform(0.0, 1.0, batch))            # isotropic, [0, pi/2]
        teff_s = rng.uniform(teff_lims[0], teff_lims[1], batch)
        distance_s = rng.uniform(distance_lims[0], distance_lims[1], batch)

        radius_s = np.interp(teff_s, _MS_TEFF, _MS_RAD)
        mass_s = np.interp(teff_s, _MS_TEFF, _MS_MASS)
        l_sun = radius_s ** 2 * (teff_s / 5772.0) ** 4
        semimajor_p = (mass_s * (p_orb / _YEAR_D) ** 2) ** (1.0 / 3.0)   # AU
        flux_p = l_sun / semimajor_p ** 2                               # I_earth

        m = (flux_p >= insol_lo) & (flux_p <= insol_hi)
        if not m.any():
            continue
        keep_R.append(radius_p[m]); keep_M.append(mass_p[m]); keep_P.append(p_orb[m])
        keep_E.append(ecc_p[m]); keep_inc.append(inc_p[m]); keep_T.append(teff_s[m])
        keep_Rs.append(radius_s[m]); keep_Ms.append(mass_s[m]); keep_D.append(distance_s[m])
        keep_L.append(l_sun[m]); keep_a.append(semimajor_p[m]); keep_F.append(flux_p[m])
        have += int(m.sum())

    def cat(parts):
        return np.concatenate(parts)[:n_planets]

    teff = cat(keep_T)
    df = pd.DataFrame({
        "radius_p": cat(keep_R),
        "mass_p": cat(keep_M),
        "p_orb": cat(keep_P),
        "ecc_p": cat(keep_E),
        "inc_p": cat(keep_inc),
        "semimajor_p": cat(keep_a),
        "flux_p": cat(keep_F),
        "radius_s": cat(keep_Rs),
        "mass_s": cat(keep_Ms),
        "teff_s": teff,
        "temp_s": teff,
        "l_sun": cat(keep_L),
        "distance_s": cat(keep_D),
    })
    # Sky position (flat) for any visibility model; detectors here use it loosely.
    df["ra"] = rng.uniform(0.0, 360.0, len(df))
    df["dec"] = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, len(df))))  # isotropic sky
    df["stype"] = _stype_from_teff(teff)
    df["nstar"] = np.arange(len(df))   # one synthetic star per planet (independent draws)
    df["id"] = np.arange(len(df))
    # Optionally replace the independent (log-uniform) mass by a mass-radius relation.
    # Insolation/orbit/star do not depend on mass_p, so radius/flux/star are IDENTICAL across
    # mass models (a controlled comparison), and mass_model="independent" is byte-for-byte the
    # original catalogue (shared cache stays valid).
    if mass_model != "independent":
        df["mass_p"] = _mass_from_radius(df["radius_p"].to_numpy(), mass_model, rng,
                                         mass_lims, mass_scatter_dex, mr_C, mr_beta)
    return df


def get_or_build_catalog(cache_path: str, rebuild: bool = False, **kwargs) -> pd.DataFrame:
    """Load a cached flat catalogue, or build (and cache) it. Shared by the scripts."""
    if (not rebuild) and os.path.exists(cache_path):
        print(f"--> Loading cached flat catalogue: {cache_path}")
        return pd.read_csv(cache_path)
    print(f"--> Building flat catalogue: {kwargs}")
    catalog = generate_flat_catalog(**kwargs)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    catalog.to_csv(cache_path, index=False)
    print(f"--> Cached flat catalogue ({len(catalog)} planets) to: {cache_path}")
    return catalog


if __name__ == "__main__":
    out_csv = os.path.join(UNIFORM_OUT_DIR, "flat_catalog.csv")
    df = get_or_build_catalog(out_csv, rebuild=True, n_planets=200000, seed=0)
    print("Rows:", len(df))
    print(df[["radius_p", "mass_p", "p_orb", "flux_p", "semimajor_p",
              "teff_s", "radius_s", "distance_s"]].describe().round(3).to_string())
    print("stype counts:\n", df["stype"].value_counts().to_string())
