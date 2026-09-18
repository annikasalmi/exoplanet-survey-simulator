import os
from pathlib import Path

import numpy as np
import pandas as pd

from science.populations.ppop import PPop, set_star_catalog
from science.telescopes.kepler.detection_model import KeplerData
from run.multi_run import run_universes
from tools.paths import KEPLER_DATA_DIR, PSCOMPPARS_CSV
from science.catalogs import read_nasa_csv

# Each universe peaks at 2-2.5 GB, so more workers than this swaps on 16 GB.
MAX_WORKERS = 5

NASA_DATA_DIR = Path(KEPLER_DATA_DIR) / "NASA"
NASA_INPUT_CSV = Path(PSCOMPPARS_CSV)
NASA_OUTPUT_CSV = NASA_DATA_DIR / "kepler_catalog_nasa_pscomppars.csv"

# Detection settings for the NASA PSCompPars population.
FALLBACK_CDPP_PPM = KeplerData.CDPP_NONSTELLAR_KP12_PPM
MISSION_DURATION_DAYS = 4 * 365.25
MIN_TRANSITS = 3
MES_THRESHOLD = 7.1
KEPLER_MAG_LIMIT = 16.0


def run_single(i, star_catalog='Gaia'):
    '''
    Runs a single instance of the PPop simulation and Kepler data analysis.
    '''
    print(f"Running Kepler for run {i} with star catalog {star_catalog}")
    rng = np.random.default_rng(i)
    PPopObj = set_star_catalog(PPop(rng=rng), star_catalog)

    data_path = os.path.join(KEPLER_DATA_DIR, f'test_runs_kepler_{i}')
    df = PPopObj.run_ppop(data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)

    kepler_data = KeplerData(PPopObj.catalog)
    kepler_data.determine_detectable()

    df = kepler_data.catalog
    save_dir = os.path.join(KEPLER_DATA_DIR, star_catalog)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'kepler_catalog_{i}.csv'), index=False)

    return df

def run_nasa_pscomppars(input_csv=NASA_INPUT_CSV, output_csv=NASA_OUTPUT_CSV):
    """Run Kepler detection on NASA PSCompPars planets; returns a DataFrame of results."""
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find NASA input CSV:\n{input_csv}\n\n"
            "It ships in data/exoplanet_csv/; otherwise export the pscomppars table from the NASA Exoplanet Archive."
        )

    print(f"Loading NASA PSCompPars: {input_csv}")
    df = read_nasa_csv(input_csv)
    print(f"Raw NASA rows: {len(df):,}")

    # The pipeline wants transiting, confirmed planets with a measured (not
    # relation-derived) mass and a known insolation -- the cuts the old
    # pre-filtered export carried in its filename.
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    keep = (
        (df.get("tran_flag", 1) == 1)
        & df["pl_insol"].notna()
        & prov.str.contains("Mass", case=False, na=False)
        & ~prov.str.contains("Calc", case=False, na=False)
    )
    if "soltype" in df.columns:
        keep &= df["soltype"].astype(str).str.contains("Conf", case=False, na=False)
    df = df[keep].copy()
    print(f"After transiting/confirmed/measured-mass/insolation cuts: {len(df):,}")

    model = KeplerData(
        df,
        source="pscomppars",
        validate_for_detection=True,
        fallback_cdpp_ppm=FALLBACK_CDPP_PPM,
        mission_duration_days=MISSION_DURATION_DAYS,
        min_transits=MIN_TRANSITS,
        mes_threshold=MES_THRESHOLD,
        kepler_mag_limit=KEPLER_MAG_LIMIT,
        use_observed_transit_flag_for_nasa=True,
        use_observed_transit_depth_for_nasa=True,
        assume_bright_if_kepmag_missing_for_nasa=True,
    )

    out = model.determine_detectable()
    out["run"] = 0
    out["model_input_file"] = input_csv.name
    out["model_population"] = "NASA_PSCompPars_transiting_confirmed_RM_insolation"

    # Radius bins for older plotting code
    out["radius_bin"] = pd.cut(
        out["radius_p"],
        bins=[0, 1.5, 3.0, 6.0, float("inf")],
        labels=["<1.5", "1.5–3.0", "3.0–6.0", ">6.0"],
        include_lowest=True,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    return out


def main(parallel=False, nruns=np.arange(1), star_catalog='Gaia', run_anew=True, population='ppop'):
    '''
    population: 'ppop' simulates universes; 'nasa_pscomppars' runs the detector on
    the real archive instead and ignores the other arguments.
    '''
    if population == 'nasa_pscomppars':
        return run_nasa_pscomppars()
    if population != 'ppop':
        raise ValueError(f"Unknown population: {population}")
    return run_universes(
        run_single, nruns, star_catalog, run_anew, parallel, MAX_WORKERS,
        catalog_dir=KEPLER_DATA_DIR, catalog_stem="kepler_catalog",
    )


if __name__ == '__main__':
    main()
