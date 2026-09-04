"""cProfile the real SimulateUniverses path to attribute per-star cost
(draw vs He2019 stability vs getmaxrpproj fsolve vs write_df reflection)."""
import cProfile
import io
import pstats
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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

N = 300
SEED_SUBSET = 12345


def build(fgk):
    s2m = {"A": SAG13, "F": fgk, "G": fgk, "K": fgk, "M": Dressing2015}
    return SystemGenerator.SystemGenerator(
        gaia, s2m, BinarySuppression, Chen2017, Circular, He2019, Random,
        Uniform, Ertel2020, ["A", "F", "G", "K", "M"], [0.0, 60.0],
        [-90.0, 90.0], "baseline", False, 1, None, True,
        rng=np.random.default_rng(0))


def profile(name, fgk):
    sg = build(fgk)
    full = len(sg.StarCatalog.SC)
    idx = np.random.default_rng(SEED_SUBSET).choice(full, size=N, replace=False)
    idx.sort()
    sg.StarCatalog.SC = sg.StarCatalog.SC[idx]

    pr = cProfile.Profile()
    pr.enable()
    df = sg.SimulateUniverses("p", Nuniverses=1, write_to_file=False)
    pr.disable()

    print(f"\n================ {name}: {len(df)} planets / {N} stars ================")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(18)
    # Trim the noisy header path lines.
    for ln in s.getvalue().splitlines():
        if "function calls" in ln or "Ordered by" in ln or ln.strip().startswith("ncalls") or ln.strip():
            print(ln)


if __name__ == "__main__":
    profile("Bergsten2022", Bergsten2022)
    profile("SAG13", SAG13)
