"""
Example usage of performance optimizations for plotting.
"""
import pandas as pd
import numpy as np
from plot.plot_by_type import PlotPlanetType

def optimized_plotting_example():
    """Example of using optimized plotting functions."""
    
    # 1. Use the batch plotting method for better performance
    plotter = PlotPlanetType(df, nruns=10, star_catalog='Gaia', name='HWO')
    
    # Option 1: Use the optimized batch method
    plotter.plot_all_batch()  # If available
    
    # Option 2: Use individual methods with caching
    plotter.plot_by_planet()
    plotter.plot_by_star() 
    plotter.plot_distances()
    
    # 2. For very large datasets, use parallel processing
    if len(df) > 50000:
        plotter._precompute_values_parallel(n_chunks=8)
    
    # 3. Monitor performance (if using the performance monitor)
    try:
        from plot.performance_monitor import monitor_performance
        
        @monitor_performance
        def plot_with_monitoring():
            plotter.plot_all()
        
        plot_with_monitoring()
    except ImportError:
        # Fallback without monitoring
        plotter.plot_all()

def memory_optimization_tips():
    """Tips for memory optimization."""
    
    # 1. Use categorical dtypes for repeated values
    df['stype'] = df['stype'].astype('category')
    df['temp_zone'] = df['temp_zone'].astype('category')
    
    # 2. Use appropriate numeric dtypes
    df['radius_p'] = df['radius_p'].astype('float32')  # If precision allows
    df['temp_p'] = df['temp_p'].astype('float32')
    
    # 3. Drop unnecessary columns early
    needed_columns = ['radius_p', 'temp_p', 'stype', 'distance_s', 'run']
    df = df[needed_columns]
    
    # 4. Use chunked processing for very large datasets
    chunk_size = 10000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        # Process chunk
        del chunk  # Explicitly delete to free memory

if __name__ == "__main__":
    # Example usage
    print("Performance optimization example")
    print("1. Use batch plotting methods")
    print("2. Enable parallel processing for large datasets") 
    print("3. Monitor performance with decorators")
    print("4. Optimize memory usage with appropriate dtypes") 