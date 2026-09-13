import os

LIFESIM_OUTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIFESIM_INNER_DIR = os.path.join(LIFESIM_OUTER_DIR, "lifesim")
PPOP_DIR = os.path.join(LIFESIM_OUTER_DIR, "PPop")
PPOP_DATA_DIR = os.path.join(PPOP_DIR, "data")
PPOP_STAR_DIR = os.path.join(PPOP_DIR, "StarCatalogs")
#Hongyi added for kepler data 5/18
DATA_DIR = os.path.join(LIFESIM_OUTER_DIR, "data")

# ---------------------------------------------------------------------------
# Everything the pipelines generate lives under results/. Keeping it out of the
# source packages means .gitignore needs one rule instead of a per-pipeline one,
# and a new pipeline cannot leak a multi-hundred-MB catalogue into a commit by
# someone forgetting to add another.
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(LIFESIM_OUTER_DIR, "results")

CATALOGS_DIR = os.path.join(RESULTS_DIR, "catalogs")
KEPLER_DATA_DIR = os.path.join(CATALOGS_DIR, "kepler")
TESS_DATA_DIR = os.path.join(CATALOGS_DIR, "tess")
HWO_DATA_DIR = os.path.join(CATALOGS_DIR, "hwo")
LIFESIM_DATA_DIR = os.path.join(CATALOGS_DIR, "lifesim")
FLAT_UNIVERSE_DATA_DIR = os.path.join(CATALOGS_DIR, "flat_universe")

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PLOTS_DIR = os.path.join(FIGURES_DIR, "simulation")      # plot_all
ANALYSIS_DIR = os.path.join(FIGURES_DIR, "analysis")     # the analysis scripts
CALIBRATION_DIR = os.path.join(FIGURES_DIR, "calibration")
PAPER_FIGURES_DIR = os.path.join(RESULTS_DIR, "paper")

LOGGING = os.path.join(RESULTS_DIR, "logs")

# Older name for the analysis figures; kept so existing scripts keep working.
MY_OUTPUTS_DIR = ANALYSIS_DIR

SILICON_CURVE = os.path.join(DATA_DIR, "silicon_curve.ddat")
EXOPLANET_CSV_DIR = os.path.join(DATA_DIR, "exoplanet_csv")
KOI_CUMULATIVE_CSV = os.path.join(EXOPLANET_CSV_DIR, "koi_cumulative_stellar.csv")
EXOFOP_TOI_CSV = os.path.join(EXOPLANET_CSV_DIR, "exofop_toi.csv")
PSCOMPPARS_CSV = os.path.join(EXOPLANET_CSV_DIR, "pscomppars_2026.csv")
KEPLER_REF_CURVE = os.path.join(LIFESIM_OUTER_DIR, "telescopes", "kepler", "reference_curves", "ref.ddat")
