# mdwarf-habitability

## Overview

Tools to simulate and analyze the detectability of potentially habitable exoplanets around M dwarfs for two mission concepts: LIFE (mid-IR interferometer) and HWO (Habitable Worlds Observatory). The repo provides repeatable simulation runs, result aggregation, and plotting utilities. Cursor was used to optimize and speed up plotting.

## Key features
- Simulation of exoplanet populations across many Monte Carlo runs (via a forked PPop, https://github.com/kammerje/P-pop)
- Detection modeling for LIFE (via a forked LIFEsim: https://github.com/fdannert/LIFEsim) and HWO
- HWO simulation code custom written
- Consolidated result catalogs and rich plotting for detection efficiency, rejections, and parameter distributions

## Installation

Prerequisites: Python 3.10+ recommended.

1) Clone the repository
```bash
git clone <repo-url>
cd mdwarf-habitability
```

2) (Recommended) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Make the repository importable during local runs
```bash
export PYTHONPATH=$(pwd)
```

## Quickstart

Run a full HWO simulation (500 Monte Carlo runs by default are demonstrated in the script):
```bash
python run/run_sim.py
```

This will:
- Execute the configured simulation (`run/hwo/hwo_run_multiple.py` or `run/lifesim/lifesim_run_multiple.py` via `run/run_sim.py`)
- Log progress under `logs/`
- Produce consolidated outputs and generate plots via `plot/plot.py`

To plot existing exoplanet data with HWO detectability overlays, see the helper in `run/run_sim.py` (function `run_exoplanet_plotting`).

## Usage notes
- Simulation drivers live in `run/` (e.g., `run_sim.py`). Adjust run parameters there (number of runs, catalogs, parallelization, etc.).
- Plotting utilities are in `plot/`. The primary entry is `plot/plot.py` (also used by `run/run_sim.py`).
- The LIFE and HWO instrument/core logic lives under `lifesim/`.

## Testing
```bash
pytest
```
Tests live in `tests/`. Ensure `PYTHONPATH` is set to the repo root before running tests.

## Project structure (high-level)
- `lifesim/`: Core simulation and instrument modules (forked/adapted from `fdannert/LIFEsim`)
- `run/`: Simulation entry points and orchestration
- `plot/`: Analysis and plotting scripts
- `tools/`: Utilities (paths, constants, catalog helpers)
- `PPop/`: Population models and star catalogs (from `kammerje/P-pop`)
- `tests/`: Unit and integration tests
- `docs/`: Sphinx documentation (work in progress)

## Acknowledgements
- LIFEsim: `https://github.com/fdannert/LIFEsim`
- P-pop: `https://github.com/kammerje/P-pop`

## License
GPL-3.0. See `LICENSE`.

## Contact
Questions, issues, or contributions: Annika Salmi — annikaksalmi@gmail.com