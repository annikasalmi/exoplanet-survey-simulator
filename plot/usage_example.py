"""
Example usage of the multiprocessing plotting functionality.
"""
import pandas as pd
import numpy as np
from plot.plot import plot_all, plot_all_sequential


def example_usage():
    """Example of how to use the multiprocessing plotting."""
    
    # Create sample data (replace with your actual data)
    print("Creating sample data...")
    np.random.seed(42)
    
    # Generate sample planet data
    n_planets = 5000
    n_runs = 3
    
    data = []
    for run in range(n_runs):
        for i in range(n_planets):
            planet = {
                'run': run,
                'radius_p': np.random.uniform(0.5, 10.0),
                'temp_p': np.random.uniform(100, 500),
                'distance_s': np.random.uniform(1, 20),
                'stype': np.random.choice(['M', 'K', 'G'], p=[0.7, 0.2, 0.1]),
                'habitable': np.random.choice([True, False], p=[0.3, 0.7]),
                'detected_best': np.random.choice([True, False], p=[0.2, 0.8]),
                'detected_worst': np.random.choice([True, False], p=[0.1, 0.9]),
            }
            data.append(planet)
    
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} rows")
    
    # Example 1: Use multiprocessing (default)
    print("\n" + "="*50)
    print("EXAMPLE 1: Multiprocessing (default)")
    print("="*50)
    plot_all(df, nruns=n_runs, star_catalog='Test', sim_name='HWO')
    
    # Example 2: Force sequential execution
    print("\n" + "="*50)
    print("EXAMPLE 2: Sequential execution")
    print("="*50)
    plot_all(df, nruns=n_runs, star_catalog='Test', sim_name='HWO', 
             use_multiprocessing=False)
    
    # Example 3: Use backward compatibility function
    print("\n" + "="*50)
    print("EXAMPLE 3: Backward compatibility")
    print("="*50)
    plot_all_sequential(df, nruns=n_runs, star_catalog='Test', sim_name='HWO')
    
    # Example 4: LIFEsim (only 2 plotting classes)
    print("\n" + "="*50)
    print("EXAMPLE 4: LIFEsim simulation")
    print("="*50)
    plot_all(df, nruns=n_runs, star_catalog='Test', sim_name='LIFEsim')


def performance_tips():
    """Tips for optimal performance."""
    
    print("\n" + "="*50)
    print("PERFORMANCE TIPS")
    print("="*50)
    
    print("1. Multiprocessing works best when:")
    print("   - You have multiple CPU cores")
    print("   - Dataset is large (>1000 planets)")
    print("   - Multiple plotting classes are used")
    
    print("\n2. Sequential execution is better when:")
    print("   - Dataset is small (<1000 planets)")
    print("   - You're on Windows (multiprocessing issues)")
    print("   - You need to debug plotting issues")
    
    print("\n3. Expected speedup:")
    print("   - 2-3x faster with 3 plotting classes")
    print("   - Depends on CPU cores and dataset size")
    print("   - Larger datasets benefit more from multiprocessing")
    
    print("\n4. Memory usage:")
    print("   - Each process uses its own memory")
    print("   - Total memory usage = n_processes × single_process_memory")
    print("   - Monitor memory usage for very large datasets")


if __name__ == "__main__":
    print("Multiprocessing Plotting Usage Example")
    print("="*50)
    
    # Run examples
    example_usage()
    
    # Show performance tips
    performance_tips()
    
    print("\n" + "="*50)
    print("Example completed!")
    print("="*50) 