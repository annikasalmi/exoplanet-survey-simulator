"""Throughput benchmark: P-Pop planet draw, Bergsten2022 (ours, FGK) vs SAG13
(Annika / Kopparapu-2018 / Quanz-2022 baseline for FGK).  Isolates the planet
distribution: same per-star Poisson draw loop, no stability/mass/orbit/IO."""
import importlib.util
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "PPop" / "PlanetDistributions"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PD / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlanetDistribution


class Star:
    __slots__ = ("Mass",)
    def __init__(self, mass):
        self.Mass = mass


def bench(name, N_STARS=100_000, seed=0):
    rng = np.random.default_rng(seed)
    Cls = load(name)

    t0 = time.perf_counter()
    dist = Cls("baseline", rng=rng)
    t_init = time.perf_counter() - t0

    # FGK host masses (Msun) spanning Bergsten's mass bins; SAG13 ignores Star.
    masses = rng.uniform(0.56, 1.63, N_STARS)
    stars = [Star(m) for m in masses]

    # Warm up (first call JITs np.interp grids etc.).
    dist.draw(Scale=1.0, Star=stars[0])

    t0 = time.perf_counter()
    n_planets = 0
    for s in stars:
        Rp, _ = dist.draw(Scale=1.0, Star=s)
        n_planets += len(Rp)
    t_draw = time.perf_counter() - t0

    per_star_us = t_draw / N_STARS * 1e6
    planets_per_star = n_planets / N_STARS
    planets_per_sec = n_planets / t_draw if t_draw > 0 else float("nan")
    return dict(name=name, t_init=t_init, N_STARS=N_STARS, n_planets=n_planets,
                per_star_us=per_star_us, planets_per_star=planets_per_star,
                planets_per_sec=planets_per_sec)


def main():
    rows = [bench("Bergsten2022"), bench("SAG13")]
    TARGET = 1_000_000  # "enough" planets

    print(f"\n{'metric':38s}{'Bergsten2022 (ours)':>22s}{'SAG13 (Annika FGK)':>22s}")
    print("-" * 82)
    def line(label, key, fmt):
        a, b = rows[0][key], rows[1][key]
        print(f"{label:38s}{fmt.format(a):>22s}{fmt.format(b):>22s}")
    line("init / precompute time (s)",      "t_init",          "{:.3f}")
    line("draw time per star (us)",         "per_star_us",     "{:.2f}")
    line("planets per star (occurrence)",   "planets_per_star","{:.3f}")
    line("planets / second",                "planets_per_sec", "{:,.0f}")
    print("-" * 82)
    for r in rows:
        secs = TARGET / r["planets_per_sec"]
        stars_needed = TARGET / r["planets_per_star"]
        print(f"{r['name']:>20s}: to draw {TARGET:,} planets = "
              f"{secs:6.2f} s  (needs {stars_needed:,.0f} star-draws)")


if __name__ == "__main__":
    main()
