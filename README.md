# mdwarf-habitability

## Project Overview

This project simulates and analyses the efficiency of detecting habitable exoplanets around M-dwarf stars with LIFE and HWO. It includes tools for running simulations, analysing detection statistics, and visualising results.

## Features
- Simulation of exoplanet populations multiple times
- Detection scenarios for LIFEsim (from https://github.com/fdannert/LIFEsim) and for HWO with simulated exoplanet populations
- Visualisation tools for detection efficiency, rejections, and parameter distributions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mdwarf-habitability.git
   cd mdwarf-habitability
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- To run simulations, use the scripts in the `run/` directory (e.g., `run_sim.py`, `hwo_run_multiple.py`).
- An example script to run the simulation 500 times and then analyse the results for HWO is included as simulation_demo.py.
- To generate plots and analyse results, use the scripts in the `plot/` directory.
- Example:
   ```bash
   python run/run_sim.py
   python plot/plot.py
   ```
- For integration tests:
   ```bash
   export PYTHONPATH=$(pwd)
   pytest
   ```

## Testing

- Unit and integration tests are located in the `tests/` directory.
- Run all tests with:
   ```bash
   pytest
   ```

## Project Structure

- `lifesim/` - Core simulation and instrument modules - from https://github.com/fdannert/LIFEsim
- `run/` - Scripts to run simulations
- `plot/` - Plotting and analysis scripts
- `tools/` - Utility scripts and constants
- `PPop/` - Population models and star catalogs - from https://github.com/kammerje/P-pop
- `tests/` - Unit and integration tests

## Contact
For questions or contributions, please contact Annika Salmi at annikaksalmi@gmail.com