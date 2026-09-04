"""
run_TESS.py — fixed runner for CDPP-first TESSData

Fixes:
- no duplicate TESSData keyword arguments
- all TESS detector settings live in one dict, then user kwargs override defaults
- compatible with run_sim(..., use_combined_sample=True, plot=True)

Put this file at:
    run/tess/run_TESS.py
"""

from __future__ import annotations

import os
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from PPop.StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3, gaia
from run.ppop.ppop_generator import PPop
from telescopes.tess.detection_model import TESSData

try:
    from tools.paths import TESS_DATA_DIR
except Exception:
    ROOT = Path(__file__).resolve().parents[2]
    TESS_DATA_DIR = str(ROOT / "run" / "tess" / "data")


TESS_DEFAULTS = {
    "use_tesspoint": True,
    "use_mast_tic": False,
    "cdpp_dir": str(Path("run") / "tess" / "data" / "CDPP"),
    "use_cdpp_tables": True,
    # Fallback when tess-point is unavailable.  tess-point is active for the 400
    # Gaia catalogs (tess_sector_source == "tess-point"), so this default is never
    # used in practice.  Value 5 = median observed across the Gaia population
    # (mean=7, p25=3, p75=8; CVZ |elat|>78° fraction = 4.3% → 13 sectors).
    "default_n_sectors": 5,
    "min_transits": 2,
    "snr_threshold": 7.1,
    "phase_mode": "random",
    "tmag_limit": 16.0,
}


def set_star_catalog(ppop_obj, star_catalog: str):
    if star_catalog == "CrossfieldBrightSample":
        ppop_obj.StarCatalog = CrossfieldBrightSample
    elif star_catalog == "ExoCat_1":
        ppop_obj.StarCatalog = ExoCat_1
    elif star_catalog == "LTC_3":
        ppop_obj.StarCatalog = LTC_3
    elif star_catalog == "LTC_2":
        ppop_obj.StarCatalog = LTC_2
    elif star_catalog == "Gaia":
        ppop_obj.StarCatalog = gaia
    else:
        raise ValueError(f"Unknown star catalog: {star_catalog}")
    return ppop_obj


def _save_catalog_dir(star_catalog: str, use_combined_sample: bool, tess_experiment: str | None = None) -> str:
    base = star_catalog
    if tess_experiment:
        base = f"{base}_{tess_experiment}"
    save_dir = os.path.join(TESS_DATA_DIR, base)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _build_tess_kwargs(user_kwargs: dict | None, run_seed: int) -> dict:
    """Merge defaults safely. User values override defaults; no duplicate keywords."""
    tess_kwargs = dict(TESS_DEFAULTS)
    if user_kwargs:
        tess_kwargs.update(user_kwargs)

    # `source` is passed explicitly to TESSData below, so remove it if user supplied it.
    tess_kwargs.pop("source", None)

    # Stable per-run random phase.
    tess_kwargs.setdefault("random_seed", int(run_seed))
    return tess_kwargs


def run_single(
    i,
    star_catalog="Gaia",
    use_combined_sample=False,
    tess_experiment: str | None = "cdpp_v1",
    tess_kwargs: dict | None = None,
):
    print(f"Running TESS for run {i} with star catalog {star_catalog}")

    rng = np.random.default_rng(i)
    ppop_obj = PPop(rng=rng, use_combined_sample=use_combined_sample)
    ppop_obj = set_star_catalog(ppop_obj, star_catalog)

    filename = f"test_runs_tess_{i}"
    data_path = os.path.join(TESS_DATA_DIR, filename)
    save_dir = _save_catalog_dir(star_catalog, use_combined_sample, tess_experiment)

    df = ppop_obj.run_ppop(data_path=data_path)
    ppop_obj.catalog_from_ppop(data_path, df=df)

    # IMPORTANT: do not also pass use_tesspoint/use_cdpp_tables explicitly here.
    # They are already inside `_build_tess_kwargs`, and can be overridden there.
    detector_kwargs = _build_tess_kwargs(tess_kwargs, run_seed=i)
    print(f"  running TESS detection on {len(ppop_obj.catalog)} planets...")
    tess_data = TESSData(ppop_obj.catalog, source="ppop", **detector_kwargs)
    tess_data.determine_detectable()

    out = tess_data.catalog
    out_path = os.path.join(save_dir, f"tess_catalog_{i}.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved TESS catalog to: {out_path}")
    return out


def run_tess_import_catalog(i, star_catalog, use_combined_sample=False, tess_experiment: str | None = "cdpp_v1"):
    save_dir = _save_catalog_dir(star_catalog, use_combined_sample, tess_experiment)
    path = os.path.join(save_dir, f"tess_catalog_{i}.csv")
    return pd.read_csv(path)


def main(
    parallel=False,
    nruns=np.arange(1),
    star_catalog="Gaia",
    run_anew=True,
    use_combined_sample=False,
    tess_experiment: str | None = "cdpp_v1",
    tess_kwargs: dict | None = None,
    **unused_kwargs,
):
    if unused_kwargs:
        print(f"Ignoring unused run_TESS kwargs: {sorted(unused_kwargs)}")

    start = time.time()

    if run_anew:
        runner = partial(
            run_single,
            star_catalog=star_catalog,
            use_combined_sample=use_combined_sample,
            tess_experiment=tess_experiment,
            tess_kwargs=tess_kwargs,
        )
    else:
        runner = partial(
            run_tess_import_catalog,
            star_catalog=star_catalog,
            use_combined_sample=use_combined_sample,
            tess_experiment=tess_experiment,
        )

    if parallel:
        # Cap workers at 2 for TESS. Each TESS worker is much heavier than Kepler:
        # ~216 MB CDPP tables + tess-point + the full universe (~0.5 GB) + a
        # 300k-sample exozodi KDE + detection structures ~= 2.5-4 GB EACH. On a
        # 16 GB box, 4+ workers exhaust RAM (seen: ArrayMemoryError in the exozodi
        # KDE resample). 2 workers (~5-8 GB) leaves headroom.
        with mp.Pool(processes=min(2, mp.cpu_count())) as pool:
            results = pool.map(runner, nruns)
    else:
        results = [runner(i=i) for i in nruns]

    df_concat = (
        pd.concat(results, keys=nruns)
        .reset_index(level=0)
        .rename(columns={"level_0": "run"})
        .reset_index(drop=True)
    )

    print(f"Total time: {time.time() - start:.2f} seconds")
    return df_concat


if __name__ == "__main__":
    df = main(parallel=False, nruns=np.arange(1), star_catalog="Gaia", run_anew=True)
    print("TESS run finished.")
    print("Rows:", len(df))
    if "detected" in df.columns:
        print(df["detected"].value_counts(dropna=False))
    if "tess_reason_category" in df.columns:
        print(df["tess_reason_category"].value_counts(dropna=False))
