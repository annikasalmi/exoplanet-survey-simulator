"""
Constants for plotting routines in the project.
"""
from typing import List, Optional

STAR_ORDER: List[str] = ['F', 'G', 'K', 'M']
BIN_LABELS: List[str] = ['<1.5', '1.5–3.0', '3.0–6.0', 'Rocky HZ']
TEMP_ZONES: List[str] = ['hot', 'habitable', 'cold']
DISTANCE_LABELS: List[str] = ['< 3', '3 - 5', '5 - 7', '7 - 9', '9 - 11', '11 - 13', '13 - 15', '> 15']
STAR_COLORS: List[str] = ['lightblue', 'deepskyblue', 'midnightblue', 'forestgreen']
STAR_HATCHES: List[Optional[str]] = ['...', 'ooo', 'OO', None]
TEMP_COLORS: List[str] = ['red', 'gold', 'blue']
BAR_WIDTH_STAR: float = 0.2
BAR_WIDTH_TEMP: float = 0.2
BAR_WIDTH_DIST: float = 0.6
