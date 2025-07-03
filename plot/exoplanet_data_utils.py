import os
import pandas as pd
from tools.paths import LIFESIM_OUTER_DIR

def load_exoplanet_luminosity_distance(region_lum=(0.01, 0.4), region_dist=(4, 14), return_names=False):
    """Load exoplanets_2025.csv and return DataFrame with 'Luminosity', 'Distance', and optionally 'Planet Name' in the specified region.
    Uses columns: 'pl_name' (planet name), 'st_lum' (luminosity), 'sy_dist' (distance)."""
    exo_path = os.path.join(LIFESIM_OUTER_DIR, 'exoplanets_2025.csv')
    exo_df = pd.read_csv(exo_path)
    if not isinstance(exo_df, pd.DataFrame):
        exo_df = pd.DataFrame(exo_df)

    # Filter for valid luminosity and distance
    exo_df = exo_df[(exo_df['st_lum'].notnull()) & (exo_df['sy_dist'].notnull())]
    if not isinstance(exo_df, pd.DataFrame) or exo_df.empty:
        return None

    # Rename columns for consistency
    exo_df = exo_df.rename(columns={'st_lum': 'Luminosity', 'sy_dist': 'Distance', 'pl_name': 'Planet Name'})
    # Only keep those in the specified region
    mask = (
        (exo_df['Luminosity'] >= region_lum[0]) & (exo_df['Luminosity'] <= region_lum[1]) &
        (exo_df['Distance'] >= region_dist[0]) & (exo_df['Distance'] <= region_dist[1])
    )
    exo_df = exo_df[mask]
    if not isinstance(exo_df, pd.DataFrame) or exo_df.empty:
        return None
    cols = ['Luminosity', 'Distance']
    if return_names and 'Planet Name' in exo_df.columns:
        cols.append('Planet Name')
    return exo_df[cols].copy() 