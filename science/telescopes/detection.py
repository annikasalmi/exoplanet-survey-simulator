"""Survey normalization, detection, and population selection."""

from __future__ import annotations

import hashlib
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

from science import physics_constants as const
from science.catalogs import parameter_box_mask, restrict_science_window
from science.physics import add_stellar_type, is_super_earth, radius_on_curve
from science.universes.flat_baseline import flat_nonphysical
from science.universes.flat_curves import flat_radii_curves
from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.rv.detection_model import RVData
from science.telescopes.tess.detection_model import TESSData


DEFAULT_RV_MAG_TARGET = None
POPULATION_GENERATORS = {
    "superearths_supneptunes": partial(
        flat_radii_curves, variant="superearths_supneptunes"
    ),
    "flat_nonphysical": flat_nonphysical,
}


def run_rv_best(
    catalog: pd.DataFrame,
    mag_target: float | None = None,
    source: str = "ppop",
) -> pd.DataFrame:
    """Combine each planet's HARPS and NIRPS detections."""
    harps = RVData(
        catalog.copy(), source=source, instrument="HARPS"
    ).determine_detectable()
    nirps = RVData(
        catalog.copy(), source=source, instrument="NIRPS"
    ).determine_detectable()
    best = harps.copy()
    best["detected"] = harps["detected"].astype(bool) | nirps["detected"].astype(bool)
    best["detected_best"] = best["detected"]
    best["detected_worst"] = best["detected"]
    harps_magnitude = pd.to_numeric(harps["rv_mag"], errors="coerce")
    nirps_magnitude = pd.to_numeric(nirps["rv_mag"], errors="coerce")
    if mag_target is not None:
        best["rv_is_target"] = (
            (harps_magnitude <= mag_target) | (nirps_magnitude <= mag_target)
        )
    best["rv_best_band"] = np.where(
        nirps["detected"].astype(bool) & ~harps["detected"].astype(bool),
        "NIRPS",
        "HARPS",
    )
    return best


def as_boolean(series):
    """Normalize common serialized boolean representations."""
    if series.dtype == bool:
        return series.fillna(False)
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y", "t"])


def _standardize_flux_radius(frame, *, kepler=False):
    frame = frame.copy()
    flux = next(
        (name for name in ("pl_insol", "insolation", "insolation_flux", "koi_insol")
         if name in frame.columns),
        None,
    )
    radius = next(
        (name for name in ("pl_rade", "koi_prad", "planet_radius")
         if name in frame.columns),
        None,
    )
    rename = {}
    if "flux_p" not in frame and flux:
        rename[flux] = "flux_p"
    if "radius_p" not in frame and radius:
        rename[radius] = "radius_p"
    frame = frame.rename(columns=rename)
    numeric = ["flux_p", "radius_p"]
    if kepler:
        numeric += ["kepler_mes", "n_transits_keplerish"]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def prepare_kepler_catalog(
    frame,
    *,
    mes_threshold=7.1,
    noise_scale=1.0,
    insolation=(0.1, 1e4),
    radius=(0.6, 2.2),
    stellar_types=("F", "G", "K", "M"),
):
    """Normalize a saved Kepler simulation and apply its detection case."""
    frame = add_stellar_type(_standardize_flux_radius(frame, kepler=True))
    if "detected" not in frame and "detected_best" in frame:
        frame["detected"] = frame["detected_best"]
    if "bright_enough_kepler" not in frame:
        frame["bright_enough_kepler"] = True
    if "kepler_enough_transits" not in frame:
        frame["kepler_enough_transits"] = (
            frame["n_transits_keplerish"] >= 3
            if "n_transits_keplerish" in frame else True
        )
    required = [
        "flux_p", "radius_p", "kepler_mes", "transiting_geometric",
        "bright_enough_kepler", "kepler_enough_transits",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Kepler P-Pop missing required columns: {missing}")
    for column in (
        "transiting_geometric", "bright_enough_kepler", "kepler_enough_transits",
    ):
        frame[column] = as_boolean(frame[column])
    frame = frame.dropna(subset=["flux_p", "radius_p", "kepler_mes"]).copy()
    frame = restrict_science_window(
        frame, insolation=insolation, radius=radius, stellar_types=stellar_types
    )
    mes = pd.to_numeric(frame["kepler_mes"], errors="coerce").fillna(0.0) / noise_scale
    frame["detected_case"] = (
        frame["transiting_geometric"] & frame["bright_enough_kepler"]
        & frame["kepler_enough_transits"] & (mes >= mes_threshold)
    )
    frame["denominator_case"] = frame["transiting_geometric"]
    return frame


def prepare_tess_catalog(
    frame,
    *,
    snr_threshold=7.1,
    noise_scale=1.0,
    insolation=(0.1, 1e4),
    radius=(0.6, 2.2),
    stellar_types=("F", "G", "K", "M"),
):
    """Normalize a saved TESS simulation and apply its detection case."""
    frame = add_stellar_type(_standardize_flux_radius(frame))
    for column in ("tess_star_bright_enough", "tess_observed"):
        if column not in frame:
            frame[column] = True
    if "tess_enough_transits" not in frame:
        frame["tess_enough_transits"] = (
            pd.to_numeric(frame["tess_n_transits"], errors="coerce") >= 3
            if "tess_n_transits" in frame else True
        )
    required = [
        "radius_p", "flux_p", "tess_snr", "tess_observed",
        "tess_transiting_geometric", "tess_star_bright_enough",
        "tess_enough_transits",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"TESS P-Pop missing required columns: {missing}")
    for column in ("flux_p", "radius_p", "tess_snr"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "tess_observed", "tess_transiting_geometric", "tess_star_bright_enough",
        "tess_enough_transits",
    ):
        frame[column] = as_boolean(frame[column])
    frame = frame.dropna(subset=["flux_p", "radius_p", "tess_snr"]).copy()
    frame = restrict_science_window(
        frame, insolation=insolation, radius=radius, stellar_types=stellar_types
    )
    snr = pd.to_numeric(frame["tess_snr"], errors="coerce").fillna(0.0) / noise_scale
    frame["detected_case"] = (
        frame["tess_observed"] & frame["tess_transiting_geometric"]
        & frame["tess_star_bright_enough"] & frame["tess_enough_transits"]
        & (snr >= snr_threshold)
    )
    frame["denominator_case"] = (
        frame["tess_observed"] & frame["tess_transiting_geometric"]
    )
    return frame


def run_transit_rv_selection(
    catalog, *, transit_mission="kepler", rv_mag_target=DEFAULT_RV_MAG_TARGET,
):
    """Return one explicit, detector-independent survey-selection table."""
    mission = transit_mission.lower()
    if mission == "kepler":
        transit = KeplerData(catalog.copy(), source="ppop").determine_detectable()
        transit_eligible = transit["transiting_geometric"].astype(bool).to_numpy()
    elif mission == "tess":
        transit = TESSData(
            catalog.copy(), source="ppop"
        ).determine_detectable()
        transit_eligible = (
            transit["tess_observed"].astype(bool)
            & transit["tess_transiting_geometric"].astype(bool)
        ).to_numpy()
    else:
        raise ValueError("transit_mission must be 'Kepler' or 'TESS'")
    transit_detected = transit["detected"].astype(bool).to_numpy() & transit_eligible
    rv = run_rv_best(catalog, mag_target=rv_mag_target)
    rv_detected = rv["detected"].astype(bool).to_numpy() & transit_eligible
    values = {
        "mass": pd.to_numeric(catalog["mass_p"], errors="coerce").to_numpy(float),
        "radius": pd.to_numeric(catalog["radius_p"], errors="coerce").to_numpy(float),
        "insolation": pd.to_numeric(catalog["flux_p"], errors="coerce").to_numpy(float),
        "transit_eligible": transit_eligible,
        "transit_detected": transit_detected,
        "rv_detected": rv_detected,
        "joint_detected": transit_detected & rv_detected,
    }
    values["effective_temperature"] = (
        pd.to_numeric(catalog["teff_s"], errors="coerce").to_numpy(float)
        if "teff_s" in catalog else np.full(len(catalog), np.nan)
    )
    return pd.DataFrame(values)


def build_detected_population(
    population,
    *,
    ppop_catalog=None,
    flat_size=300_000,
    seed=0,
    rv_mag_target=DEFAULT_RV_MAG_TARGET,
    box=None,
):
    """Build a box-filtered flat or saved P-Pop selection table."""
    if population == "flat":
        pool = flat_radii_curves(
            flat_size, seed=seed, variant="superearths_supneptunes"
        )
    elif population == "ppop":
        path = Path(ppop_catalog) if ppop_catalog is not None else None
        if path is None or not path.exists():
            raise FileNotFoundError(f"P-Pop catalog not found: {path}")
        pool = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError("population must be 'flat' or 'ppop'")

    if box is not None:
        keep = parameter_box_mask(box, pool["mass_p"], pool["radius_p"], pool["flux_p"])
        pool = pool[keep].copy()
    if population == "flat":
        return run_transit_rv_selection(
            pool, transit_mission="kepler", rv_mag_target=rv_mag_target
        )

    transit_detected = as_boolean(pool["detected"]).to_numpy()
    rv = run_rv_best(pool, mag_target=rv_mag_target)
    rv_detected = rv["detected"].astype(bool).to_numpy()
    effective_temperature = (
        pd.to_numeric(pool["teff_s"], errors="coerce").to_numpy(float)
        if "teff_s" in pool else np.full(len(pool), np.nan)
    )
    return pd.DataFrame({
        "mass": pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float),
        "radius": pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float),
        "insolation": pd.to_numeric(pool["flux_p"], errors="coerce").to_numpy(float),
        "effective_temperature": effective_temperature,
        "transit_eligible": transit_detected,
        "transit_detected": transit_detected,
        "rv_detected": rv_detected,
        "joint_detected": transit_detected & rv_detected,
    })


@lru_cache(maxsize=1)
def _science_code_fingerprint() -> str:
    """Hash of every science/ module, so a cached pool is rebuilt when detectors or generators change."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha1()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def make_detected_pool(
    pool_name,
    *,
    pool_size,
    chunk_size,
    cache_dir,
    mission="kepler",
    seed=0,
    box,
    rv_mag_target=DEFAULT_RV_MAG_TARGET,
):
    """Generate, detect, and cache a population in bounded-memory chunks."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"selection_{mission}_{pool_name}_N{pool_size}_s{seed}_{_science_code_fingerprint()}.npz"
    if cache.exists():
        with np.load(cache) as archive:
            return pd.DataFrame({field: archive[field] for field in archive.files})
    if pool_name not in POPULATION_GENERATORS:
        choices = ", ".join(sorted(POPULATION_GENERATORS))
        raise ValueError(f"unknown population {pool_name!r}; choose {choices}")

    parts = []
    completed = 0
    chunk_index = 0
    while completed < pool_size:
        size = min(chunk_size, pool_size - completed)
        catalog = POPULATION_GENERATORS[pool_name](size, seed=seed + chunk_index)
        keep = parameter_box_mask(
            box, catalog["mass_p"], catalog["radius_p"], catalog["flux_p"]
        )
        parts.append(run_transit_rv_selection(
            catalog[keep].copy(),
            transit_mission=mission,
            rv_mag_target=rv_mag_target,
        ))
        completed += size
        chunk_index += 1
    result = pd.concat(parts, ignore_index=True)
    np.savez(cache, **{column: result[column].to_numpy() for column in result})
    return result


def split_universes(superearths_supneptunes):
    """Return superearths_supneptunes and only_subneptunes, which excludes rocky super-Earths."""
    keep = ~is_super_earth(superearths_supneptunes["mass"], superearths_supneptunes["radius"])
    return {
        "rocky_formation": superearths_supneptunes,
        "escape_only": superearths_supneptunes.loc[keep].reset_index(drop=True),
    }


def select_rocky_transiting(catalog, curve_mass, curve_radius, *, minimum_radius):
    """Keep rocky planets above a radius floor with transiting geometry."""
    threshold = radius_on_curve(catalog["mass_p"], curve_mass, curve_radius)
    rocky = (catalog["radius_p"].to_numpy() <= threshold) & np.isfinite(threshold)
    selected = catalog[
        rocky & (catalog["radius_p"].to_numpy() > minimum_radius)
    ].copy()
    stellar_radius_au = selected["radius_s"].to_numpy() * const.R_SUN_IN_AU
    planet_radius_au = selected["radius_p"].to_numpy() * const.R_EARTH_IN_AU
    impact = (
        selected["semimajor_p"].to_numpy()
        * np.abs(np.cos(selected["inc_p"].to_numpy()))
        / stellar_radius_au
    )
    return selected[
        impact <= 1.0 + planet_radius_au / stellar_radius_au
    ].copy().reset_index(drop=True)


def missed_fraction_in_window(
    panel, test, *, max_insolation=50.0, min_radius=1.4, main_only=True,
):
    """Return missed fraction, denominator, and misses in a declared window."""
    if main_only and "main" in panel:
        panel = panel[panel["main"]]
    region = (
        (panel["insolation"] < max_insolation)
        & (panel["radius"] > min_radius)
        & panel["transit_eligible"]
    ).to_numpy()
    denominator = int(region.sum())
    if denominator == 0:
        return None, 0, 0
    passed = int((region & panel[test].to_numpy()).sum())
    missed = denominator - passed
    return missed / denominator, denominator, missed


def build_stellar_selection_panel(
    config,
    curve_mass,
    curve_radius,
    *,
    radius_limits=(0.5, 2.2),
    period_limits=(0.2, 20_000.0),
    minimum_radius=0.6,
    transit_mission="TESS",
    rv_mag_target=DEFAULT_RV_MAG_TARGET,
):
    """Generate the main and optional map-only selection samples for one host type."""
    samples = []
    catalogs = []
    sizes = [(config["n"], config["seed"], True)]
    if config.get("map_extra_n"):
        sizes.append((config["map_extra_n"], config["map_extra_seed"], False))
    for size, seed, main in sizes:
        catalog = flat_nonphysical(
            size, seed=seed, radius_lims=radius_limits, period_lims=period_limits,
            teff_lims=config["teff"], insol_lims=config["insol"],
        )
        catalog = select_rocky_transiting(
            catalog, curve_mass, curve_radius, minimum_radius=minimum_radius
        )
        selected = run_transit_rv_selection(
            catalog, transit_mission=transit_mission, rv_mag_target=rv_mag_target
        )
        samples.append(selected.assign(main=main))
        catalogs.append(catalog)
    extra_catalog = catalogs[1] if len(catalogs) == 2 else None
    return pd.concat(samples, ignore_index=True), catalogs[0], extra_catalog


__all__ = [
    "DEFAULT_RV_MAG_TARGET", "as_boolean", "build_detected_population",
    "build_stellar_selection_panel", "make_detected_pool",
    "missed_fraction_in_window", "prepare_kepler_catalog", "prepare_tess_catalog",
    "run_rv_best", "run_transit_rv_selection", "select_rocky_transiting",
    "split_universes",
]
