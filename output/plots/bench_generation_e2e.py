"""END-TO-END P-Pop generation benchmark, as run by run_sim.py.

Measures the REAL per-star cost of SystemGenerator.SimulateUniverses (planet
draw + Chen2017 mass + He2019 stability rejection + the getmaxrpproj fsolve
geometry loop + per-star write_df reflection), on the actual Gaia-60pc catalog,
for the two FGK planet distributions:

    case A: Bergsten2022  (ours)
    case B: SAG13         (Annika / Kopparapu-2018 FGK baseline)

Only the FGK->distribution mapping differs; M dwarfs use Dressing2015 in both,
and the rest of the pipeline (mass/ecc/orbit/stability/geometry/IO) is identical.
A fixed random subset of stars is used (same indices for both cases) so the two
runs see the same hosts; we then extrapolate to 1,000,000 planets.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.paths import LIFESIM_OUTER_DIR
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import PPop.SystemGenerator as SystemGenerator
from PPop.StarCatalogs import gaia
from PPop.PlanetDistributions import Bergsten2022, SAG13, Dressing2015
from PPop.ScalingModels import BinarySuppression
from PPop.MassModels import Chen2017
from PPop.EccentricityModels import Circular
from PPop.StabilityModels import He2019
from PPop.OrbitModels import Random
from PPop.AlbedoModels import Uniform
from PPop.ExozodiModels import Ertel2020

STYPES = ["A", "F", "G", "K", "M"]
DIST_RANGE = [0.0, 60.0]
DEC_RANGE = [-90.0, 90.0]
N_STARS_SAMPLE = 1500      # subset of the real catalog timed per case
SUBSET_SEED = 12345        # picks WHICH stars (same for both cases)
DRAW_SEED = 0              # RNG for the planet draws (same for both cases)
TARGET_PLANETS = 1_000_000


def build_sysgen(fgk_model, draw_rng):
    stype_to_model = {"A": SAG13, "F": fgk_model, "G": fgk_model,
                      "K": fgk_model, "M": Dressing2015}
    return SystemGenerator.SystemGenerator(
        gaia, stype_to_model, BinarySuppression, Chen2017, Circular,
        He2019, Random, Uniform, Ertel2020, STYPES, DIST_RANGE, DEC_RANGE,
        "baseline", False, 1, None, True, rng=draw_rng)


def subset_indices(n_full, n_take, seed):
    rng = np.random.default_rng(seed)
    n_take = min(n_take, n_full)
    idx = rng.choice(n_full, size=n_take, replace=False)
    idx.sort()
    return idx


def bench(name, fgk_model, star_idx):
    sysgen = build_sysgen(fgk_model, np.random.default_rng(DRAW_SEED))
    # Restrict to the SAME stars for every case.
    sysgen.StarCatalog.SC = sysgen.StarCatalog.SC[star_idx]
    n_stars = len(sysgen.StarCatalog.SC)

    t0 = time.perf_counter()
    df = sysgen.SimulateUniverses("bench", Nuniverses=1, write_to_file=False)
    dt = time.perf_counter() - t0

    n_planets = len(df)
    return dict(name=name, n_stars=n_stars, n_planets=n_planets, secs=dt,
                planets_per_s=n_planets / dt, ms_per_star=dt / n_stars * 1e3,
                planets_per_star=n_planets / n_stars,
                stab_fails=getattr(sysgen.StabilityModel, "Nfails", float("nan")))


def main():
    # Build one sysgen just to learn the full catalog size, then pick a subset.
    probe = build_sysgen(Bergsten2022, np.random.default_rng(DRAW_SEED))
    n_full = len(probe.StarCatalog.SC)
    idx = subset_indices(n_full, N_STARS_SAMPLE, SUBSET_SEED)
    print(f"Full Gaia catalog: {n_full} stars; timing a fixed random "
          f"subset of {len(idx)} stars (same stars for both cases).\n")

    rows = [bench("Bergsten2022 (ours)", Bergsten2022, idx),
            bench("SAG13 (Annika FGK)", SAG13, idx)]

    w = 24
    print(f"{'metric':32s}" + "".join(f"{r['name']:>{w}s}" for r in rows))
    print("-" * (32 + w * len(rows)))

    def line(label, key, fmt):
        print(f"{label:32s}" + "".join(f"{fmt.format(r[key]):>{w}s}" for r in rows))

    line("stars timed", "n_stars", "{:d}")
    line("planets generated", "n_planets", "{:d}")
    line("planets / star", "planets_per_star", "{:.3f}")
    line("wall time (s)", "secs", "{:.1f}")
    line("ms per star", "ms_per_star", "{:.2f}")
    line("planets / second", "planets_per_s", "{:,.0f}")
    line("He2019 stability re-draws", "stab_fails", "{:,.0f}")
    print("-" * (32 + w * len(rows)))

    for r in rows:
        secs = TARGET_PLANETS / r["planets_per_s"]
        h = int(secs // 3600); m = int((secs % 3600) // 60); s = int(secs % 60)
        print(f"{r['name']:>24s}: 1,000,000 planets (1 thread) ~ "
              f"{secs:8.0f} s = {h}h{m:02d}m{s:02d}s")


if __name__ == "__main__":
    main()
