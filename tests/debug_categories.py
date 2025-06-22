#!/usr/bin/env python3
"""
Debug script to investigate category assignment.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from plot.helpers import assign_category


def debug_category_assignment(df):
    """Debug the category assignment process."""
    print("=== Category Assignment Debug ===")
    
    # Check data ranges
    print(f"\nData ranges:")
    print(f"Radius range: {df['radius_p'].min():.2f} - {df['radius_p'].max():.2f}")
    print(f"Temperature range: {df['temp_p'].min():.2f} - {df['temp_p'].max():.2f}")
    print(f"Star types: {sorted(df['stype'].unique())}")
    
    # Check habitable planets
    habitable_planets = df[df['habitable'] == True]
    print(f"\nHabitable planets: {len(habitable_planets)} out of {len(df)} total")
    
    # Check radius bins for habitable planets
    print(f"\nHabitable planets by radius:")
    hab_radius_bins = pd.cut(habitable_planets['radius_p'], 
                            bins=[0, 1.5, 1.8, 4.0, 8.0, np.inf], 
                            labels=['<1.5', '1.5-1.8', '1.8-4.0', '4.0-8.0', '>8.0'])
    radius_counts = hab_radius_bins.value_counts().sort_index()
    for bin_name, count in radius_counts.items():
        print(f"  {bin_name}: {count} planets")
    
    # Check star types for habitable planets
    print(f"\nHabitable planets by star type:")
    star_counts = habitable_planets['stype'].value_counts()
    for star_type, count in star_counts.items():
        print(f"  {star_type}: {count} planets")
    
    # Apply category assignment
    df['category'] = df.apply(assign_category, axis=1)
    
    # Check all categories
    print(f"\nAll categories found:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} planets")
    
    # Check for None categories
    none_categories = df[df['category'].isna()]
    if len(none_categories) > 0:
        print(f"\nPlanets with None category: {len(none_categories)}")
        print("Sample of None categories:")
        sample_none = none_categories[['radius_p', 'habitable', 'stype', 'temp_p']].head(5)
        print(sample_none)
    
    # Check specific conditions for Habitable Sub-Neptunes
    print(f"\nChecking conditions for Habitable Sub-Neptunes:")
    sub_neptune_range = df[(df['radius_p'] >= 1.8) & (df['radius_p'] < 4.0)]
    print(f"Planets in 1.8 ≤ r < 4.0 range: {len(sub_neptune_range)}")
    
    hab_sub_neptune_range = sub_neptune_range[sub_neptune_range['habitable'] == True]
    print(f"Habitable planets in 1.8 ≤ r < 4.0 range: {len(hab_sub_neptune_range)}")
    
    if len(hab_sub_neptune_range) > 0:
        print("Sample habitable sub-Neptune candidates:")
        sample = hab_sub_neptune_range[['radius_p', 'habitable', 'stype', 'temp_p']].head(5)
        print(sample)
        
        # Apply category assignment to these specifically
        sample['category'] = sample.apply(assign_category, axis=1)
        print("Categories assigned to these planets:")
        print(sample[['radius_p', 'habitable', 'stype', 'category']])
    
    return df


def test_category_logic():
    """Test the category assignment logic with sample data."""
    print("\n=== Testing Category Logic ===")
    
    # Create test cases
    test_cases = [
        {'radius_p': 1.0, 'habitable': True, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 1.0, 'habitable': False, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 1.2, 'habitable': True, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 1.6, 'habitable': False, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 1.6, 'habitable': True, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 2.0, 'habitable': False, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 2.0, 'habitable': True, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 5.0, 'habitable': False, 'stype': 'G', 'temp_p': 300},
        {'radius_p': 10.0, 'habitable': False, 'stype': 'G', 'temp_p': 300},
    ]
    
    for i, case in enumerate(test_cases):
        category = assign_category(pd.Series(case))
        print(f"Test {i+1}: r={case['radius_p']}, hab={case['habitable']}, stype={case['stype']} → {category}")


if __name__ == "__main__":
    # Test the logic first
    test_category_logic()
    
    # If you have a DataFrame, you can debug it like this:
    # df = pd.read_csv('your_data.csv')  # Load your data
    # debug_category_assignment(df) 