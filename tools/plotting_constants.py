"""
Constants for plotting routines in the project.
"""
from typing import List, Optional
import numpy as np

STAR_ORDER: List[str] = ['F', 'G', 'K', 'M']
BIN_LABELS: List[str] = ['<1.5', '1.5–3.0', '3.0–6.0', 'Rocky HZ']
TEMP_ZONES: List[str] = ['hot', 'habitable', 'cold']
DISTANCE_LABELS: List[str] = ['< 3', '3 - 5', '5 - 7', '7 - 9', '9 - 11', '11 - 13', '13 - 15', '15 - 20']
STAR_COLORS: List[str] = ['lightblue', 'deepskyblue', 'midnightblue', 'forestgreen']
STAR_HATCHES: List[Optional[str]] = ['...', 'ooo', 'OO', None]
TEMP_COLORS: List[str] = ['red', 'gold', 'blue']
BAR_WIDTH_STAR: float = 0.2
BAR_WIDTH_TEMP: float = 0.2
BAR_WIDTH_DIST: float = 0.6
ERROR_LABEL_OFFSET: float = 5
HATCHES: List[Optional[str]] = ['//', '---', '\\\\']

# Plot configuration for temperature and distance
PLOT_CONFIGS = {
    'temp': {'col': 'temp_p', 'label': 'Temperature [K]', 'range': (125, 305)},
    'distance': {'col': 'distance_s', 'label': 'Distance [pc]', 'range': (0, 15)}
}

# Planet type filters as lambdas
PLANET_TYPE_FILTERS = {
    'Rocky HZ': lambda df: (df['habitable'] == True) & (df['radius_p'] < 1.5),
    'HZ Rocky around G-type (Sun-like) stars': lambda df: (df['habitable'] == True) &
                                                         (df['radius_p'] <= 1.5) &
                                                         (df['stype'].str.contains('G')),
    'HZ around M dwarfs': lambda df: (df['habitable'] == True) &
                                    (df['stype'].str.contains('M'))
}

# Panel configuration for 3D plots
PANEL_CONFIGS = [
    {
        'x': 'temp_p', 'y': 'radius_p',
        'xbins': np.linspace(125, 305, 40), 'ybins': np.linspace(0, 8, 30),
        'xlabel': 'Temperature [K]', 'ylabel': 'Radius [Rearth]', 'xscale': 'linear',
        'title': 'Temp vs Radius'
    },
    {
        'x': 'distance_s', 'y': 'radius_p',
        'xbins': np.linspace(0, 15, 40), 'ybins': np.linspace(0, 8, 30),
        'xlabel': 'Distance [pc]', 'ylabel': 'Radius [Rearth]', 'xscale': 'linear',
        'title': 'Radius vs Distance'
    },
    {
        'x': 'p_orb', 'y': 'mass_p',
        'xbins': np.linspace(0.5, 500, 40), 'ybins': np.linspace(0, 10, 30),
        'xlabel': 'Orbital Period [days]', 'ylabel': 'Mass [Mearth]', 'xscale': 'linear',
        'title': 'Mass vs Period'
    },
]

# Rejection plot constants
REJECTION_COLUMN_MAPPING = {
    '# photons hitting detector': 'photon_rate_value_best',
    'Flux Ratio': 'flux_ratio_value_best',
    'IWA': 'maxangsep',
    'Exozodi': 'z',
}
REJECTION_LABELS = ['Maximum angular separation', 'Flux Ratio', 'IWA', 'Exozodi']
REJECTION_COLORS = {
    '# photons hitting detector': 'black',
    'Flux Ratio': 'red',
    'IWA': 'blue',
    'Exozodi': 'gold',
}
REJECTION_SCENARIO_LABELS = {'best': 'Best Case Scenario'}

DETECTION_COLORS = {
    'Radial Velocity': 'blue',
    'Transit': 'black', 
    'Imaging': 'red',
    'Microlensing': 'orange',
    'Astrometry': 'purple',
    'Timing': 'brown',
    'Other': 'gray'
}