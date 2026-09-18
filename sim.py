"""Choose a universe, apply telescope detectors, and save the catalog and plots."""

import argparse
from pathlib import Path

import numpy as np

from science.populations.universes.ppop import PPop
from science.populations.universes.flat_baseline import flat_nonphysical
from science.populations.universes.flat_curves import flat_radii_curves
from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.tess.detection_model import TESSData
from science.telescopes.detection import run_rv_best
from plotting.plot_population import plot_population
from tools.paths import CATALOGS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("flat_nonphysical", "flat_radii_curves", "ppop"),
                        default="flat_nonphysical")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-planets", type=int, default=20_000, help="Flat-universe draw count")
    parser.add_argument(
        "--variant", choices=("superearths_supneptunes", "only_subneptunes"),
        default="superearths_supneptunes", help="Curve-based flat variant",
    )
    parser.add_argument("--star-catalog", default="Gaia", help="P-Pop star catalog")
    args = parser.parse_args()

    out_dir = Path(CATALOGS_DIR) / args.universe
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.universe == "flat_nonphysical":
        catalog = flat_nonphysical(args.n_planets, seed=args.seed)
        name = f"flat_nonphysical_seed{args.seed}_n{args.n_planets}"
    elif args.universe == "flat_radii_curves":
        catalog = flat_radii_curves(args.n_planets, seed=args.seed, variant=args.variant)
        name = f"flat_radii_curves_{args.variant}_seed{args.seed}_n{args.n_planets}"
    else:
        name = f"ppop_{args.star_catalog}_seed{args.seed}"
        population = PPop(rng=np.random.default_rng(args.seed), star_catalog=args.star_catalog)
        raw = population.run_ppop(data_path=str(out_dir / name))
        population.catalog_from_ppop(df=raw)
        population.catalog_remove_distance(stype="A", mode="larger", dist=0.0)
        catalog = population.catalog

    catalog["kepler_detected"] = KeplerData(catalog.copy(), source="ppop").determine_detectable()["detected"]
    catalog["tess_detected"] = TESSData(catalog.copy(), source="ppop", use_cdpp_tables=False).determine_detectable()["detected"]
    catalog["rv_detected"] = run_rv_best(catalog, mag_target=12.0)["detected"]

    output = out_dir / f"{name}.csv"
    catalog.to_csv(output, index=False)
    print(f"Saved {len(catalog):,} planets: {output}")
    plot_population(catalog, name=name)
