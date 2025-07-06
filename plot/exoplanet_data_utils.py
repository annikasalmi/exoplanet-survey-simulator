import os
import pandas as pd
from tools.paths import LIFESIM_OUTER_DIR
from tools import physics_constants as const

def load_exoplanet_luminosity_distance(region_lum=(0.001, 10), region_dist=(4, 35), return_names=False):
    """Load exoplanets_2025.csv and return DataFrame with 'Luminosity', 'Distance', and optionally 'Planet Name' in the specified region.
    Uses columns: 'pl_name' (planet name), 'st_lum' (luminosity), 'sy_dist' (distance), 'pl_rade' (radius)."""
    exo_path = os.path.join(LIFESIM_OUTER_DIR, 'exoplanets_2025.csv')
    exo_df = pd.read_csv(exo_path)
    if not isinstance(exo_df, pd.DataFrame):
        exo_df = pd.DataFrame(exo_df)

    # Filter for valid luminosity, distance, and radius
    exo_df = exo_df[(exo_df['st_lum'].notnull()) & (exo_df['sy_dist'].notnull()) & (exo_df['pl_rade'].notnull())]
    if not isinstance(exo_df, pd.DataFrame) or exo_df.empty:
        print("No planets with valid luminosity, distance, and radius data")
        return None

    # Filter for radius < 2.6 Earth radii
    exo_df = exo_df[exo_df['pl_rade'] < const.R_earth_max_habitable]
    if not isinstance(exo_df, pd.DataFrame) or exo_df.empty:
        print("No planets with radius < 2.6 Earth radii")
        return None

    # Rename columns for consistency
    exo_df = exo_df.rename(columns={'st_lum': 'Luminosity', 'sy_dist': 'Distance', 'pl_name': 'Planet Name', 
                                   'discoverymethod': 'Detection Method', 'pl_rade': 'Radius (R⊕)'})
    
    # Convert log10(luminosity) to linear luminosity for filtering
    exo_df['Luminosity_linear'] = 10**exo_df['Luminosity']
    
    # Only keep those in the specified region
    mask = (
        (exo_df['Luminosity_linear'] >= region_lum[0]) & (exo_df['Luminosity_linear'] <= region_lum[1]) &
        (exo_df['Distance'] >= region_dist[0]) & (exo_df['Distance'] <= region_dist[1])
    )
    exo_df = exo_df[mask]
    if not isinstance(exo_df, pd.DataFrame) or exo_df.empty:
        print(f"No planets in specified region: L={region_lum}, D={region_dist}")
        return None
    
    # Use linear luminosity for output
    exo_df['Luminosity'] = exo_df['Luminosity_linear']
    cols = ['Luminosity', 'Distance', 'Radius (R⊕)']
    if return_names and 'Planet Name' in exo_df.columns:
        cols.append('Planet Name')
        if 'Detection Method' in exo_df.columns:
            cols.append('Detection Method')
    return exo_df[cols].copy()

def filter_exoplanets(df: pd.DataFrame, 
                     min_lum: float = 0.002, 
                     max_lum: float = 10.0,
                     min_dist: float = 4.0, 
                     max_dist: float = 15.0) -> pd.DataFrame:
    """
    Filter exoplanets based on stellar luminosity and distance.
    
    Args:
        df: DataFrame with exoplanet data
        min_lum: Minimum stellar luminosity in L☉ (default: 0.002)
        max_lum: Maximum stellar luminosity in L☉ (default: 10.0)
        min_dist: Minimum distance in parsecs (default: 4.0)
        max_dist: Maximum distance in parsecs (default: 15.0)
    
    Returns:
        Filtered DataFrame
    """
    # Convert log10 luminosity to linear if needed
    if 'st_lum' in df.columns:
        # Check if luminosity values are in log10 format
        max_lum_val = df['st_lum'].max()
        if isinstance(max_lum_val, (int, float)) and max_lum_val < 10:  # Likely log10 values
            df = df.copy()
            df['st_lum'] = 10**df['st_lum']
    
    # Filter by luminosity and distance
    mask = (
        (df['st_lum'] >= min_lum) & 
        (df['st_lum'] <= max_lum) &
        (df['sy_dist'] >= min_dist) & 
        (df['sy_dist'] <= max_dist)
    )
    
    filtered_df = df[mask].copy()
    return filtered_df 