"""
Test script for multiprocessing plotting functionality.
"""
import time
import pandas as pd
import numpy as np
from plot.plot import plot_all, plot_all_sequential


def create_test_data(n_planets=10000, n_runs=5):
    """Create test data for benchmarking."""
    print(f"Creating test data with {n_planets} planets and {n_runs} runs...")
    
    # Generate random planet data
    np.random.seed(42)  # For reproducible results
    
    data = []
    for run in range(n_runs):
        for i in range(n_planets):
            planet = {
                'run': run,
                'radius_p': np.random.uniform(0.5, 10.0),
                'temp_p': np.random.uniform(100, 500),
                'distance_s': np.random.uniform(1, 20),
                'stype': np.random.choice(['M', 'K', 'G', 'F'], p=[0.7, 0.2, 0.08, 0.02]),
                'habitable': np.random.choice([True, False], p=[0.3, 0.7]),
                'detected_best': np.random.choice([True, False], p=[0.2, 0.8]),
                'detected_worst': np.random.choice([True, False], p=[0.1, 0.9]),
            }
            data.append(planet)
    
    df = pd.DataFrame(data)
    print(f"Created DataFrame with {len(df)} rows")
    return df


def benchmark_plotting(df, nruns=5, star_catalog='Test', sim_name='HWO'):
    """Benchmark sequential vs multiprocessing plotting."""
    
    print("\n" + "="*60)
    print("BENCHMARKING PLOTTING PERFORMANCE")
    print("="*60)
    
    # Test sequential plotting
    print("\n1. Testing sequential plotting...")
    start_time = time.time()
    try:
        plot_all_sequential(df, nruns, star_catalog, sim_name)
        sequential_time = time.time() - start_time
        print(f"Sequential plotting completed in {sequential_time:.2f} seconds")
    except Exception as e:
        print(f"Sequential plotting failed: {e}")
        sequential_time = None
    
    # Test multiprocessing plotting
    print("\n2. Testing multiprocessing plotting...")
    start_time = time.time()
    try:
        plot_all(df, nruns, star_catalog, sim_name, use_multiprocessing=True)
        multiprocessing_time = time.time() - start_time
        print(f"Multiprocessing plotting completed in {multiprocessing_time:.2f} seconds")
    except Exception as e:
        print(f"Multiprocessing plotting failed: {e}")
        multiprocessing_time = None
    
    # Compare results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    if sequential_time and multiprocessing_time:
        speedup = sequential_time / multiprocessing_time
        print(f"Sequential time: {sequential_time:.2f} seconds")
        print(f"Multiprocessing time: {multiprocessing_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("✅ Multiprocessing is faster!")
        else:
            print("⚠️  Sequential is faster (this can happen with small datasets)")
    else:
        print("❌ Could not complete benchmark due to errors")


def test_different_dataset_sizes():
    """Test multiprocessing with different dataset sizes."""
    
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {size} PLANETS")
        print(f"{'='*60}")
        
        df = create_test_data(size, n_runs=3)
        benchmark_plotting(df, nruns=3, star_catalog='Test', sim_name='HWO')


def test_error_handling():
    """Test error handling in multiprocessing."""
    
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    # Create a small test dataset
    df = create_test_data(100, n_runs=1)
    
    # Test with invalid parameters
    try:
        plot_all(df, nruns=1, star_catalog='Test', sim_name='Invalid')
        print("✅ Error handling test passed")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


if __name__ == "__main__":
    print("Multiprocessing Plotting Test Suite")
    print("="*60)
    
    # Test with different dataset sizes
    test_different_dataset_sizes()
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60) 