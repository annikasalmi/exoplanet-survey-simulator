import os

LIFESIM_OUTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIFESIM_INNER_DIR = os.path.join(LIFESIM_OUTER_DIR, "lifesim")
PPOP_DIR = os.path.join(LIFESIM_OUTER_DIR, "PPop")
PPOP_DATA_DIR = os.path.join(PPOP_DIR, "data")
PPOP_STAR_DIR = os.path.join(PPOP_DIR, "StarCatalogs")
LIFESIM_DATA_DIR = os.path.join(LIFESIM_INNER_DIR, "data")