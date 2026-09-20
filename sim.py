"""Edit the settings below, then run `python sim.py`.

Planet distributions live in science/populations/universes/. The telescope
models that check detection of those distributions live in science/telescopes/. 
"""

from __future__ import annotations

import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from plotting.plot_population import plot_population
from science.catalogs import read_nasa_csv
from science.populations.universes.flat_baseline import flat_nonphysical
from science.populations.universes.flat_curves import flat_radii_curves, is_super_earth
from science.populations.universes.ppop import PPop
from science.telescopes.detection import run_rv_best
from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.tess.detection_model import TESSData
from tools.paths import CATALOGS_DIR, LIFESIM_DATA_DIR, PSCOMPPARS_CSV


# ---------------------------------------------------------------------------
# Edit these settings for normal use.
# ---------------------------------------------------------------------------
UNIVERSE_OPTIONS = (
    "flat_nonphysical",
    "flat_radii_curves",
    "nasa_exoplanets",
    "ppop",
)
FLAT_RADII_VARIANT_OPTIONS = ("superearths_supneptunes", "only_subneptunes")
TELESCOPE_OPTIONS = ("kepler", "tess", "rv", "hwo", "lifesim")

### SET UP UNIVERSE
UNIVERSE = "flat_nonphysical"
FLAT_RADII_VARIANT = "superearths_supneptunes"
RUN_BOTH_FLAT_RADII_VARIANTS = False
NASA_CONFIRMED_TRANSITING_ONLY = True
NASA_REQUIRE_MEASURED_MASS = True

SEED = 0
N_PLANETS = 20_000
STAR_CATALOG = "Gaia"

### SET UP TELESCOPE BEING USED
TELESCOPES = ("kepler", "tess", "rv")
RV_MAG_TARGET = 12.0
TESS_DEFAULTS = {
    "use_cdpp_tables": False,
    "min_transits": 2,
    "snr_threshold": 7.1,
    "phase_mode": "random",
    "tmag_limit": 16.0,
}

### RUN MULTIPLE TELESCOPES / UNIVERSES
NRUNS = 1
PARALLEL = False
MAX_WORKERS = mp.cpu_count()
RUN_ANEW = True
PLOT = True

def generate_universe(universe, seed, output_stem=None, output_dir=None):
    """Create one source population."""
    if universe == "flat_nonphysical":
        return flat_nonphysical(N_PLANETS, seed=seed)
    if universe == "flat_radii_curves":
        if FLAT_RADII_VARIANT not in FLAT_RADII_VARIANT_OPTIONS:
            raise ValueError(
                f"Unknown FLAT_RADII_VARIANT {FLAT_RADII_VARIANT!r}. "
                f"Options: {FLAT_RADII_VARIANT_OPTIONS}"
            )
        return flat_radii_curves(N_PLANETS, seed=seed, variant=FLAT_RADII_VARIANT)
    if universe == "nasa_exoplanets":
        catalog = read_nasa_csv(Path(PSCOMPPARS_CSV))
        if NASA_CONFIRMED_TRANSITING_ONLY and "tran_flag" in catalog.columns:
            catalog = catalog[catalog["tran_flag"] == 1].copy()
        if NASA_REQUIRE_MEASURED_MASS and "pl_bmassprov" in catalog.columns:
            provenance = catalog["pl_bmassprov"].astype(str)
            measured = provenance.str.contains("Mass|Msini", case=False, na=False)
            measured &= ~provenance.str.contains("Calc", case=False, na=False)
            catalog = catalog[measured].copy()
        if "pl_insol" in catalog.columns:
            catalog = catalog[catalog["pl_insol"].notna()].copy()
        return catalog.reset_index(drop=True)
    if universe == "ppop":
        out_dir = Path(output_dir or CATALOGS_DIR) / "ppop_raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = output_stem or f"ppop_{STAR_CATALOG}_seed{seed}"
        population = PPop(rng=np.random.default_rng(seed), star_catalog=STAR_CATALOG)
        raw = population.run_ppop(data_path=str(out_dir / stem))
        population.catalog_from_ppop(df=raw)
        population.catalog_remove_distance(stype="A", mode="larger", dist=0.0)
        return population.catalog
    raise ValueError(f"Unknown universe {universe!r}. Options: {UNIVERSE_OPTIONS}")


def run_telescope(catalog, telescope, seed):
    """Apply one telescope model."""
    source = "nasa" if UNIVERSE == "nasa_exoplanets" else "ppop"
    if telescope == "kepler":
        out = KeplerData(catalog.copy(), source=source).determine_detectable()
        catalog["kepler_detected"] = out["detected"].astype(bool).to_numpy()
        return catalog
    if telescope == "tess":
        out = TESSData(
            catalog.copy(), source=source, random_seed=seed, **TESS_DEFAULTS
        ).determine_detectable()
        catalog["tess_detected"] = out["detected"].astype(bool).to_numpy()
        return catalog
    if telescope == "rv":
        out = run_rv_best(catalog, mag_target=RV_MAG_TARGET, source=source)
        catalog["rv_detected"] = out["detected"].astype(bool).to_numpy()
        for col in ("rv_is_target", "rv_best_band"):
            if col in out:
                catalog[col] = out[col].to_numpy()
        return catalog
    if telescope == "hwo":
        from science.telescopes.hwo.detection_model import HWOData

        out = HWOData(catalog.copy()).determine_detectable()
        catalog["hwo_detected"] = out["detected_best"].astype(bool).to_numpy()
        for col in ("detected_best", "detected_worst"):
            if col in out:
                catalog[f"hwo_{col}"] = out[col].astype(bool).to_numpy()
        return catalog
    raise ValueError(f"Unknown telescope {telescope!r}. Options: {TELESCOPE_OPTIONS}")


def run_lifesim(seed, run_id):
    """Run the LIFE instrument workflow."""
    if UNIVERSE != "ppop":
        raise ValueError("LIFEsim currently runs on the P-Pop universe only.")

    import lifesim

    rng = np.random.default_rng(seed)
    raw_dir = Path(CATALOGS_DIR) / "ppop_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    population = PPop(rng=rng, star_catalog=STAR_CATALOG)
    raw = population.run_ppop(data_path=str(raw_dir / f"lifesim_ppop_seed{seed}"))

    bus = lifesim.Bus()
    bus.data.options.set_scenario("baseline")
    bus.data.options.set_manual(diameter=4.0)
    out_dir = Path(LIFESIM_DATA_DIR) / STAR_CATALOG
    out_dir.mkdir(parents=True, exist_ok=True)
    bus.data.options.set_manual(output_path=str(out_dir))
    bus.data.options.set_manual(output_filename=f"/test_runs_{run_id}")
    bus.data.catalog_from_ppop(str(raw_dir / f"lifesim_ppop_seed{seed}"), df=raw)

    instrument = lifesim.Instrument(name="inst", rng=rng)
    bus.add_module(instrument)
    bus.add_module(lifesim.TransmissionMap(name="transm"))
    bus.add_module(lifesim.PhotonNoiseExozodi(name="exo"))
    bus.add_module(lifesim.PhotonNoiseLocalzodi(name="local"))
    bus.add_module(lifesim.PhotonNoiseStar(name="star"))
    bus.connect(("inst", "transm"))
    bus.connect(("inst", "exo"))
    bus.connect(("inst", "local"))
    bus.connect(("inst", "star"))
    bus.connect(("star", "transm"))

    opt = lifesim.Optimizer(name="opt")
    ahgs = lifesim.AhgsModule(name="ahgs")
    bus.add_module(opt)
    bus.add_module(ahgs)
    bus.connect(("transm", "opt"))
    bus.connect(("inst", "opt"))
    bus.connect(("opt", "ahgs"))

    instrument.get_snr()
    opt.ahgs()
    bus.save()
    return bus.data.catalog


def run_both_flat_radii_variants(run_id, run_anew):
    """Run both flat_radii_curves variants."""
    telescopes = tuple(TELESCOPES)
    if "lifesim" in telescopes:
        raise ValueError("RUN_BOTH_FLAT_RADII_VARIANTS does not run through LIFEsim.")

    seed = SEED + run_id
    variants = ("only_subneptunes", "superearths_supneptunes")
    paths = {}
    for variant in variants:
        out_dir = Path(CATALOGS_DIR) / "flat_radii_curves" / variant / "_".join(telescopes)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths[variant] = out_dir / (
            f"flat_radii_curves_{variant}_{'_'.join(telescopes)}"
            f"_seed{seed}_n{N_PLANETS}.csv"
        )
    if not run_anew and all(path.exists() for path in paths.values()):
        frames = []
        for variant in variants:
            df = pd.read_csv(paths[variant])
            df["run"] = run_id
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    full = flat_radii_curves(
        N_PLANETS, seed=seed, variant="superearths_supneptunes"
    )
    for telescope in telescopes:
        full = run_telescope(full, telescope, seed)
    without = full[~is_super_earth(full["mass_p"], full["radius_p"])].reset_index(drop=True)
    frames = {
        "only_subneptunes": without.assign(universe_type="only_subneptunes"),
        "superearths_supneptunes": full.assign(universe_type="superearths_supneptunes"),
    }
    for variant, df in frames.items():
        df.to_csv(paths[variant], index=False)
    return pd.concat([frames[variant].assign(run=run_id) for variant in variants], ignore_index=True)


def run_single(run_id=0, run_anew=RUN_ANEW):
    """Run one seeded simulation."""
    telescopes = tuple(TELESCOPES)
    seed = SEED + run_id

    if RUN_BOTH_FLAT_RADII_VARIANTS:
        return run_both_flat_radii_variants(run_id, run_anew)

    variant = FLAT_RADII_VARIANT if UNIVERSE == "flat_radii_curves" else None
    out_dir = Path(CATALOGS_DIR) / UNIVERSE
    if variant:
        out_dir = out_dir / variant
    out_dir = out_dir / "_".join(telescopes)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_part = f"_{variant}" if variant else ""
    if UNIVERSE.startswith("flat"):
        filename = f"{UNIVERSE}{variant_part}_{'_'.join(telescopes)}_seed{seed}_n{N_PLANETS}.csv"
    else:
        filename = f"{UNIVERSE}{variant_part}_{'_'.join(telescopes)}_{STAR_CATALOG}_seed{seed}.csv"
    path = out_dir / filename
    if not run_anew and path.exists():
        df = pd.read_csv(path)
        df["run"] = run_id
        return df

    if telescopes == ("lifesim",):
        catalog = run_lifesim(seed, run_id)
    else:
        if "lifesim" in telescopes:
            raise ValueError("Use TELESCOPES = ('lifesim',) because LIFEsim writes its own catalog.")
        catalog = generate_universe(UNIVERSE, seed, output_stem=path.stem, output_dir=path.parent)
        for telescope in telescopes:
            catalog = run_telescope(catalog, telescope, seed)

    catalog.to_csv(path, index=False)
    catalog["run"] = run_id
    print(f"Saved {len(catalog):,} planets: {path}")
    return catalog


def run_all():
    """Run configured simulations."""
    if UNIVERSE not in UNIVERSE_OPTIONS:
        raise ValueError(f"Unknown UNIVERSE {UNIVERSE!r}. Options: {UNIVERSE_OPTIONS}")
    unknown = sorted(set(TELESCOPES) - set(TELESCOPE_OPTIONS))
    if unknown:
        raise ValueError(f"Unknown telescope(s): {unknown}. Options: {TELESCOPE_OPTIONS}")

    start = time.time()
    if isinstance(NRUNS, int):
        if NRUNS < 0:
            raise ValueError("NRUNS must be non-negative")
        run_ids = list(range(NRUNS))
    else:
        run_ids = list(NRUNS)
    if PARALLEL and len(run_ids) > 1:
        with mp.get_context("spawn").Pool(processes=min(len(run_ids), MAX_WORKERS)) as pool:
            frames = pool.map(partial(run_single, run_anew=RUN_ANEW), run_ids)
    else:
        frames = [run_single(run_id, run_anew=RUN_ANEW) for run_id in run_ids]

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"Total time: {time.time() - start:.2f} seconds")

    if PLOT and len(run_ids) == 1 and TELESCOPES != ("lifesim",):
        variant = FLAT_RADII_VARIANT if UNIVERSE == "flat_radii_curves" else None
        variant_part = f"_{variant}" if variant else ""
        plot_population(out, name=f"{UNIVERSE}{variant_part}_{'_'.join(TELESCOPES)}_seed{SEED + run_ids[0]}")
    return out

if __name__ == "__main__":
    run_all()
