"""
Performance monitoring utilities for plotting functions.
"""
import time
import functools
import psutil
import os

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        print(f"{func.__name__}: {end_time - start_time:.2f}s, "
              f"Memory: {end_memory - start_memory:.1f}MB")
        
        return result
    return wrapper

def profile_dataframe_operations(df, operation_name):
    """Profile DataFrame operations."""
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    print(f"Starting {operation_name} on DataFrame with {len(df)} rows")
    
    return start_time, start_memory

def log_operation_completion(operation_name, start_time, start_memory):
    """Log completion of DataFrame operations."""
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    print(f"Completed {operation_name}: {end_time - start_time:.2f}s, "
          f"Memory: {end_memory - start_memory:.1f}MB") 