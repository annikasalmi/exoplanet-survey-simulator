import multiprocessing as mp
from functools import partial
import matplotlib
import os
from output.plots.plot_by_type import PlotPlanetType
from output.plots.plot_by_type_LTC3 import PlotPlanetTypeLTC3, PlanetDetectionPlotterLTC3
from output.plots.plot_detections import PlanetDetectionPlotter
from output.plots.plot_rejections import PlanetRejectionPlotter
from output.plots.plot_hz_limits import PlotHZLimits
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})


def _run_plotter_class(plotter_class, df, nruns, star_catalog, sim_name, **kwargs):
    """
    Helper function to run a plotting class in a separate process.
    This function needs to be at module level for multiprocessing.
    """
    try:
        # Set matplotlib backend to non-interactive for multiprocessing
        matplotlib.use('Agg')
        
        plotter = plotter_class(df=df, name=sim_name, nruns=nruns, star_catalog=star_catalog, **kwargs)
        plotter.plot_all()
        return f"{plotter_class.__name__} completed successfully"
    except Exception as e:
        return f"{plotter_class.__name__} failed: {str(e)}"


def plot_all(df, nruns=1, star_catalog='Gaia', sim_name='HWO', use_multiprocessing=True):
    """
    Generate all plots using multiprocessing for parallel execution.
    
    Args:
        df: DataFrame with planet data
        nruns: Number of simulation runs
        star_catalog: Name of the star catalog used
        sim_name: Name of the simulation/mission
        use_multiprocessing: Whether to use multiprocessing (default: True)
    """
    # Normalize simulation name
    if sim_name.lower() == 'hwo':
        sim_name = 'HWO'
    elif sim_name.lower() == 'lifesim':
        sim_name = 'LIFEsim'
    
    # Use the LTC3 plotters if star_catalog is LTC_3
    if star_catalog == 'LTC_3':
        main_plotter = PlotPlanetTypeLTC3
        detection_plotter = PlanetDetectionPlotterLTC3
    else:
        main_plotter = PlotPlanetType
        detection_plotter = PlanetDetectionPlotter
    
    # Define plotting tasks
    plotting_tasks = [
        (main_plotter, {}),
        (detection_plotter, {}),
    ]
    
    # Add rejection plotting only for HWO
    if sim_name == 'HWO' or sim_name == 'HWO_exoplanets':
        plotting_tasks.append((PlanetRejectionPlotter, {}))
        plotting_tasks.append((PlotHZLimits, {}))
        
    if use_multiprocessing and len(plotting_tasks) > 1:
        # Use multiprocessing for parallel execution
        print(f"Running {len(plotting_tasks)} plotting tasks in parallel...")
        
        # Check if we're on Windows (multiprocessing issues)
        if os.name == 'nt':
            print("Warning: Running on Windows. Multiprocessing may have issues.")
            print("Consider using use_multiprocessing=False if you encounter problems.")
        
        # Create partial function with fixed arguments
        plot_func = partial(_run_plotter_class, df=df, nruns=nruns, 
                           star_catalog=star_catalog, sim_name=sim_name)
        
        # Run plotting tasks in parallel
        try:
            with mp.Pool(processes=min(len(plotting_tasks), mp.cpu_count())) as pool:
                results = []
                for plotter_class, kwargs in plotting_tasks:
                    result = pool.apply_async(plot_func, args=(plotter_class,), kwds=kwargs)
                    results.append(result)
                
                # Wait for all tasks to complete and collect results
                for i, result in enumerate(results):
                    try:
                        status = result.get(timeout=300)  # 5 minute timeout per task
                        print(f"Task {i+1}: {status}")
                    except Exception as e:
                        print(f"Task {i+1} failed with timeout or error: {e}")
            
            print("All plotting tasks completed.")
            
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to sequential execution...")
            # Fallback to sequential execution
            _run_sequential(plotting_tasks, df, nruns, star_catalog, sim_name)
        
    else:
        # Fallback to sequential execution
        print("Running plotting tasks sequentially...")
        _run_sequential(plotting_tasks, df, nruns, star_catalog, sim_name)


def _run_sequential(plotting_tasks, df, nruns, star_catalog, sim_name):
    """Helper function for sequential execution."""
    for plotter_class, kwargs in plotting_tasks:

        print(f"Running {plotter_class.__name__}...")
        plotter = plotter_class(df=df, name=sim_name, nruns=nruns, 
                                star_catalog=star_catalog, **kwargs)
        plotter.plot_all()
        print(f"{plotter_class.__name__} completed successfully")


def plot_all_sequential(df, nruns=1, star_catalog='Gaia', sim_name='HWO'):
    """
    Original sequential plotting function for backward compatibility.
    """
    return plot_all(df, nruns, star_catalog, sim_name, use_multiprocessing=False)