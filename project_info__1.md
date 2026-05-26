# mdwarf-habitability / LIFEsim — Codebase Overview

## Summary

This repository is a research-grade simulation and analysis workbench for exoplanet detectability around M dwarfs, with two major mission families: **LIFE/LIFEsim** and **HWO**. It also includes a **Kepler** toy-analysis path and a vendored **P-Pop** population generator that produces the synthetic planet catalogs feeding all of the simulations.

The key idea is that this is **not one app**, but a set of reusable simulation pipelines that all start from the same planet-population machinery and then diverge into mission-specific detectability models and plotting. A developer usually works in one of three layers: `PPop` for population synthesis, `run/` for experiment orchestration, and `lifesim/` for shared simulation/data logic.

## Architecture

### Primary architectural pattern

The repository is organized as a **scientific pipeline with mission-specific adapters**:

1. **Population generation**: `PPop` creates synthetic star–planet catalogs.
2. **Catalog normalization**: `lifesim.core.data.Data` / `PPop.PPop.catalog_from_ppop()` convert raw P-Pop output into a standardized `pandas.DataFrame`.
3. **Mission-specific detectability**:
   - `lifesim.core.hwo_data.HWOData` applies HWO visibility/detection criteria.
   - `lifesim.core.kepler_data.KeplerData` applies a Kepler-style transit toy model.
   - LIFEsim uses the package’s original `Bus`/`Module` architecture to connect instrument and noise modules.
4. **Aggregation and plotting**: `run/*` collects multiple Monte Carlo runs and `plot/plot.py` renders summary plots.

### Major subsystems and how they relate

#### 1) `PPop/` and `run/ppop/`
`PPop/` is a vendored copy of the population synthesis engine. The file `run/ppop/ppop_generator.py` wraps it in a callable class so the repo can launch many runs programmatically.

This matters because the rest of the repository assumes a consistent catalog schema. The P-Pop wrapper is where the raw generator output is standardized into columns like `radius_p`, `p_orb`, `semimajor_p`, `temp_s`, `distance_s`, `stype`, and `habitable`.

#### 2) `lifesim/`
This is the reusable core package. It contains:
- the **module bus architecture** (`lifesim.core.core`)
- the **shared catalog/data container** (`lifesim.core.data`)
- mission-specific postprocessors (`hwo_data.py`, `kepler_data.py`)
- instrument and noise modules (`lifesim.instrument.*`)
- optimization helpers (`lifesim.optimize.*`)
- GUI helpers (`lifesim.gui.*`)

LIFEsim itself is the strongest example of a modular plugin-like design in the repo: modules connect through sockets on a central `Bus`, rather than directly calling each other.

#### 3) `run/`
This is the experiment layer. It contains mission-specific entry points:
- `run/hwo/hwo_run_multiple.py`
- `run/kepler/run_Kepler.py`
- `run/lifesim/lifesim_run_multiple.py`
- `run/run_sim.py` as the orchestration front door

These scripts generate catalogs, run detection logic, save outputs, and optionally plot.

#### 4) `plot/`
The plotting layer is separate from simulation. It uses multiple plotter classes and can run some plots in parallel via multiprocessing.

### Technology stack

- **Language**: Python
- **Core libraries**: `numpy`, `pandas`, `astropy`, `matplotlib`, `tqdm`, `yaml`, `GitPython`
- **Parallelism**: Python `multiprocessing` and `threading`
- **Packaging**: `setuptools`
- **Runtime target**: Python 3.8+ in package metadata, with README recommending 3.10+

### How execution starts

The main orchestration entry point is `run/run_sim.py`. That file:
- imports the mission runners
- wraps them in progress/logging helpers
- chooses which mission to run
- performs plotting after simulation

Typical execution flows are:
- `python run/run_sim.py` for a top-level experiment
- `python run/hwo/hwo_run_multiple.py` for HWO-only Monte Carlo runs
- `python run/kepler/run_Kepler.py` for Kepler runs
- `python run/lifesim/lifesim_run_multiple.py` for LIFEsim runs

## Directory Structure

```text
project-root/
├── lifesim/
│   ├── core/                 — Shared simulation/data abstractions
│   │   ├── core.py           — Bus/Module architecture
│   │   ├── data.py           — Standard catalog storage and import/export
│   │   ├── hwo_data.py       — HWO detectability post-processing
│   │   ├── kepler_data.py    — Kepler transit detectability model
│   │   └── modules.py        — Abstract module interfaces
│   ├── instrument/           — LIFEsim instrument/noise modules
│   ├── optimize/             — Optimization/scheduling modules
│   ├── gui/                  — GUI helpers and assets
│   └── util/                 — Constants, options, physics utilities
├── PPop/                     — Vendored P-Pop population generator
│   ├── P-pop.py              — Original monolithic script
│   └── ...                   — Population, star catalog, and model definitions
├── run/
│   ├── run_sim.py            — Top-level simulation orchestrator
│   ├── hwo/                  — HWO runner and HWO outputs
│   ├── kepler/               — Kepler runner and Kepler outputs
│   ├── lifesim/              — LIFEsim runner and LIFEsim outputs
│   └── ppop/                 — Thin wrapper around P-Pop for repeatable runs
├── plot/                     — Plotting framework and plotter classes
├── scripts/                  — One-off analysis/plot scripts
├── tests/                    — Unit/integration checks
└── tools/                    — Paths, constants, catalog helpers
```

## Key Abstractions

### `Bus`
- **File**: `lifesim/core/core.py`
- **Responsibility**: Central coordination object for a LIFEsim simulation.
- **Interface**: `add_module()`, `connect()`, `disconnect()`, `save()`, `write_config()`
- **Lifecycle**: Created per simulation run; holds the shared `Data` object and the module graph.
- **Used by**: `run/lifesim/lifesim_run_multiple.py`, `run/lifesim/lifesim_run_exoplanets.py`, GUI/demo flows.

### `Module`
- **File**: `lifesim/core/core.py`
- **Responsibility**: Base class for LIFEsim components that communicate via sockets.
- **Interface**: `add_socket()`, `run_socket()`, `connect_module()`, `disconnect_module()`
- **Lifecycle**: Subclassed by instrument, transmission, noise, and optimization modules.
- **Used by**: `lifesim.instrument.*`, `lifesim.optimize.*`, `lifesim.core.modules`.

### `Data`
- **File**: `lifesim/core/data.py`
- **Responsibility**: Central storage for the standardized exoplanet catalog, simulation options, and output data.
- **Interface**: `catalog_from_ppop()`, `import_catalog()`, `export_catalog()`, `catalog_remove_distance()`, `catalog_safe_add()`
- **Lifecycle**: Owned by `Bus`; shared by all modules in LIFEsim.
- **Used by**: Nearly every LIFEsim module and runner.

### `HWOData`
- **File**: `lifesim/core/hwo_data.py`
- **Responsibility**: Mission-specific postprocessor that computes HWO detectability flags and diagnostic quantities.
- **Interface**: `determine_detectable()`, `calc_flux_ratio()`, `calc_photons()`, `calc_iwa_constraint()`
- **Lifecycle**: Created from a standardized catalog/DataFrame and used once per HWO run.
- **Used by**: `run/hwo/hwo_run_multiple.py`, `run/run_sim.py`, `run/hwo/hwo_demo.py`.

### `KeplerData`
- **File**: `lifesim/core/kepler_data.py`
- **Responsibility**: Toy Kepler transit-selection model.
- **Interface**: `determine_detectable()`, `calc_transit_depth_fraction()`, `calc_transiting_from_inclination()`, `calc_star_brightness_proxy()`
- **Lifecycle**: Created from the shared catalog after P-Pop generation.
- **Used by**: `run/kepler/run_Kepler.py`, `run/run_sim.py`.

### `PPop`
- **File**: `run/ppop/ppop_generator.py`
- **Responsibility**: Reusable wrapper around P-Pop that generates a catalog and normalizes it to the repo’s schema.
- **Interface**: `run_ppop()`, `catalog_from_ppop()`, `catalog_remove_distance()`
- **Lifecycle**: Instantiated per Monte Carlo run with a seeded RNG.
- **Used by**: HWO, Kepler, and LIFEsim runners.

### `plot_all`
- **File**: `plot/plot.py`
- **Responsibility**: Runs the plotter classes for the chosen dataset, optionally in parallel.
- **Interface**: `plot_all()`, `plot_all_sequential()`
- **Lifecycle**: Called after simulations complete.
- **Used by**: `run/run_sim.py`, some mission-specific runners.

### `tools.paths`
- **File**: `tools/paths.py`
- **Responsibility**: Centralizes filesystem layout for outputs and data.
- **Interface**: Constants like `HWO_DATA_DIR`, `LIFESIM_DATA_DIR`, `KEPLER_DATA_DIR`, `LOGGING`
- **Lifecycle**: Imported everywhere paths matter.
- **Used by**: Runners, plotting, and data import/export.

## Data Flow

### 1) Monte Carlo catalog generation
1. `run/run_sim.py` or a mission runner seeds a NumPy RNG per run.
2. `run/ppop/ppop_generator.PPop.run_ppop()` calls `PPop.SystemGenerator.SystemGenerator(...).SimulateUniverses(...)`.
3. The raw P-Pop output is normalized into a standard catalog schema via `catalog_from_ppop()`.
4. The standardized catalog is then filtered with utility methods like `catalog_remove_distance()`.

### 2) HWO detectability path
1. `run/hwo/hwo_run_multiple.py` generates a P-Pop catalog.
2. It wraps the catalog in `HWOData`.
3. `HWOData.determine_detectable()` computes:
   - IWA pass/fail
   - flux-ratio pass/fail
   - minimum photon-rate pass/fail
   - zodi pass/fail
   - combined `detected_best` / `detected_worst`
4. The result is saved to `run/hwo/data/<star_catalog>/hwo_catalog_<i>.csv`.
5. `plot/plot.py` summarizes the aggregated runs.

### 3) Kepler detectability path
1. `run/kepler/run_Kepler.py` generates a P-Pop catalog.
2. It wraps it in `KeplerData`.
3. `KeplerData.determine_detectable()` computes:
   - geometric transiting condition from inclination
   - transit depth
   - a star brightness proxy
   - best/worst detection flags
4. The result is saved to `run/kepler/data/<star_catalog>/kepler_catalog_<i>.csv`.

### 4) LIFEsim path
1. `run/lifesim/lifesim_run_multiple.py` generates a catalog.
2. It creates a `lifesim.Bus`.
3. It adds instrument, transmission, photon-noise, and optimization modules.
4. The bus connects modules via sockets and then the instrument/optimizer run.
5. Final catalogs and config are exported with `bus.save()`.

### 5) Plotting path
1. A combined DataFrame is passed to `plot.plot_all()`.
2. The plotter chooses plot classes based on the simulation name and star catalog.
3. If enabled, individual plot jobs run in separate processes using `multiprocessing.Pool`.

## Non-Obvious Behaviors & Design Decisions

### Why `hwo_data.py` is under `lifesim/core`
This is a good question, and the answer is architectural rather than organizational. `HWOData` is not an HWO runner; it is a **catalog transform and detectability model** that operates on the same standardized data contract used across the repo. It lives in `lifesim/core` because:
- it depends on the shared catalog schema defined by `lifesim.core.data.Data`
- it is a lightweight sibling of `KeplerData`, not a top-level experiment script
- it behaves like a core domain model, not like orchestration

So even though HWO itself has a runner in `run/hwo/`, the `HWOData` class is intentionally placed with other shared data logic. The confusing part is that the file name sounds like a mission folder, but the role is really “mission-specific data processor.”

### Why `lifesim`, `hwo`, and `kepler` are not all parallel at the package level
They are parallel in the **runtime/output sense**, not in the code architecture sense:
- `run/hwo/`, `run/kepler/`, and `run/lifesim/` are parallel runner directories.
- `lifesim/` is the actual reusable package.
- HWO and Kepler logic is mostly implemented as **data processors** (`HWOData`, `KeplerData`) plus runners, not as full packages comparable to `lifesim`.

That means your instinct is partly right: the *experiments* are parallel, but the *code organization* is split by abstraction level, not by mission name.

### What “parallel” means here
Yes — in this repository, “parallel” mostly means **parallel computing** via Python multiprocessing, not a mathematical parallel structure.

There are two main places where it appears:
1. **Run-level parallelism**
   - `run/hwo/hwo_run_multiple.py`
   - `run/kepler/run_Kepler.py`
   - `run/lifesim/lifesim_run_multiple.py`

   These optionally use `multiprocessing.Pool` to run many Monte Carlo realizations simultaneously across CPU cores.

2. **Plot-level parallelism**
   - `plot/plot.py`

   The code dispatches independent plotting tasks in separate processes. This is useful because each plot class is independent and matplotlib work is CPU- and I/O-heavy.

There is also a thread-based progress/timer wrapper in `run/run_sim.py`, but that is only for progress reporting, not computation.

### Hidden invariants
Several parts of the repo assume a fixed catalog schema:
- P-Pop output must be converted to fields like `radius_p`, `temp_s`, `distance_s`, `stype`, `maxangsep`, `habitable`
- `HWOData` requires columns like `temp_p`, `temp_s`, `radius_p`, `radius_s`, `distance_s`, `maxangsep`, `z`
- `KeplerData` requires at least `radius_p`, `radius_s`, `semimajor_p`, `p_orb`

If those columns are missing or renamed, downstream logic fails quickly.

### State management
State lives in a few clear places:
- `Bus.data` is the main mutable state for LIFEsim
- mission runners keep per-run DataFrames in memory, then concatenate them
- `tools.paths` centralizes output directories
- `run/run_sim.py` adds derived columns like `radius_bin` before plotting

### Error propagation
There is relatively little deep error handling. Most errors are allowed to surface, which is common in research code:
- missing columns raise `ValueError`
- incompatible module connections raise `ValueError`
- bad path formats often fail immediately during file I/O
- `run/run_sim.py` wraps the mission function to log and re-raise exceptions

### Performance-sensitive choices
- `multiprocessing.Pool` is used for independent Monte Carlo runs and plot tasks.
- `HWOData` and `KeplerData` use vectorized NumPy/Pandas operations rather than row-by-row loops.
- `run/run_sim.py` and `plot/plot.py` avoid interactive backends during batch runs.

### A small but important oddity
`run/run_sim.py` currently has some brittle details:
- it prints `len(nruns)` even though `nruns` is a NumPy array, which works there but reflects a run-count mindset rather than a generic integer
- it imports some symbols twice and mixes HWO/Kepler/LIFEsim logic in one file
- it assumes downstream catalog columns exist before plotting

That is not “bad” so much as typical scientific orchestration code that evolved organically.

## Module Reference

| File | Purpose |
|------|---------|
| `run/run_sim.py` | Top-level orchestration for mission runs and plotting |
| `run/hwo/hwo_run_multiple.py` | HWO Monte Carlo runner with optional multiprocessing |
| `run/kepler/run_Kepler.py` | Kepler Monte Carlo runner with optional multiprocessing |
| `run/lifesim/lifesim_run_multiple.py` | LIFEsim Monte Carlo runner wiring the Bus/modules together |
| `run/lifesim/lifesim_run_exoplanets.py` | LIFEsim path for observed exoplanet catalogs |
| `run/lifesim/lifesim_run_w_imports.py` | Experimental/incomplete runner scaffold |
| `run/hwo/hwo_demo.py` | Minimal demo showing HWO detectability workflow |
| `run/ppop/ppop_generator.py` | Programmatic wrapper around P-Pop generation and catalog normalization |
| `lifesim/core/core.py` | Bus/Module architecture and config serialization |
| `lifesim/core/data.py` | Shared catalog storage, import/export, and habitable-zone augmentation |
| `lifesim/core/hwo_data.py` | HWO-specific detectability calculations |
| `lifesim/core/kepler_data.py` | Kepler toy transit detectability calculations |
| `lifesim/core/modules.py` | Abstract module interfaces for instruments, transmission, optimization |
| `lifesim/instrument/instrument.py` | Concrete LIFE instrument module |
| `lifesim/instrument/transmission.py` | Transmission map module |
| `lifesim/instrument/pn_*.py` | Photon noise modules |
| `lifesim/optimize/optimizer.py` | Optimization orchestration |
| `lifesim/optimize/ahgs.py` | aHGS optimization implementation |
| `plot/plot.py` | Plot orchestration, including multiprocessing support |
| `tools/paths.py` | Central path constants for outputs and data |
| `tools/exoplanet_catalog.py` | External exoplanet catalog loading/filtering |
| `PPop/P-pop.py` | Original monolithic P-Pop script |
| `PPop/SystemGenerator.py` | Core P-Pop generator used by the wrapper |
| `scripts/*.py` | One-off analysis and figure generation scripts |
| `tests/*.py` | Unit/integration checks for data and pipeline behavior |

## Suggested Reading Order

1. `README.md` — high-level intent and how the repo is meant to be used.
2. `tools/paths.py` — shows the filesystem structure and where outputs land.
3. `run/run_sim.py` — the top-level orchestration flow.
4. `run/ppop/ppop_generator.py` — how catalogs are generated and normalized.
5. `lifesim/core/data.py` — the shared catalog contract and mutation points.
6. `lifesim/core/hwo_data.py` and `lifesim/core/kepler_data.py` — the mission-specific detection logic.

## Overall structure and “beauty” of the repository

The strength of this repository is that it splits the work into clean layers:
- **population synthesis** is isolated in `PPop`
- **shared catalog handling** is centralized in `lifesim.core.data`
- **mission-specific logic** is in small processors (`HWOData`, `KeplerData`) or in the LIFEsim module graph
- **batch experiments** live in `run/`
- **visualization** lives in `plot/`

That separation makes it easier to swap catalogs, compare missions, and run large batches without rewriting the core machinery. The design is a little uneven in places because it grew by adding missions over time, but the overall architecture is genuinely good: it keeps the science logic close to the data model and keeps orchestration separate from calculation.
