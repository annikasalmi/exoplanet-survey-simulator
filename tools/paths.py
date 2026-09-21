from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIFESIM_DIR = REPO_ROOT / "lifesim"
PPOP_DIR = REPO_ROOT / "PPop"
PPOP_DATA_DIR = PPOP_DIR / "data"
PPOP_STAR_DIR = PPOP_DIR / "StarCatalogs"
PPOP_TEST_PLANET_POP = PPOP_DATA_DIR / "test_planet_pop.txt"
LTC_3_CATALOG = PPOP_STAR_DIR / "LTC_3.csv"
DATA_DIR = REPO_ROOT / "data"

RESULTS_DIR = REPO_ROOT / "results"

CATALOGS_DIR = RESULTS_DIR / "catalogs"
KEPLER_DATA_DIR = CATALOGS_DIR / "kepler"
TESS_DATA_DIR = CATALOGS_DIR / "tess"
RV_DATA_DIR = CATALOGS_DIR / "rv"
HWO_DATA_DIR = CATALOGS_DIR / "hwo"
LIFESIM_DATA_DIR = CATALOGS_DIR / "lifesim"
FLAT_UNIVERSE_DATA_DIR = CATALOGS_DIR / "flat_universe"
TESS_REFERENCE_DATA_DIR = REPO_ROOT / "science" / "telescopes" / "tess" / "data"

FIGURES_DIR = RESULTS_DIR / "figures"
PLOTS_DIR = FIGURES_DIR / "simulation"
ANALYSIS_DIR = FIGURES_DIR / "analysis"
CALIBRATION_DIR = FIGURES_DIR / "calibration"
OTHER_FIGURES_DIR = FIGURES_DIR / "other"
PAPER_FIGURES_DIR = RESULTS_DIR / "paper"

SILICON_CURVE = DATA_DIR / "silicon_curve.ddat"
_EXOPLANET_CSV_DIR = DATA_DIR / "exoplanet_csv"
EXOPLANETS_2026_CSV = _EXOPLANET_CSV_DIR / "exoplanets_2026.csv"
KOI_CUMULATIVE_CSV = _EXOPLANET_CSV_DIR / "koi_cumulative_stellar.csv"
PSCOMPPARS_CSV = _EXOPLANET_CSV_DIR / "pscomppars_2026.csv"
CDPP_DIR = DATA_DIR / "CDPP"
