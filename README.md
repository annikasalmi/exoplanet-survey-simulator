# exoplanet-survey-simulator

Simulates whether planned missions can detect different types of simulated exoplanets.
These simulations include LIFE, forked from [LIFEsim](https://github.com/fdannert/LIFEsim).
Additional simulations written for this repo are HWO, TESS, Kepler, HARPS, and NIRPS.

Simulated planets are generated from a modified copy of [P-Pop](https://github.com/kammerje/P-pop) and [chenjj2/forecaster](https://github.com/chenjj2/forecaster).

## Install

Needs Python 3.9 or newer.

```bash
git clone https://github.com/annikasalmi/exoplanet-survey-simulator.git
cd exoplanet-survey-simulator
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

```bash
python simulation_demo.py
```

That builds one flat universe, runs it through the Kepler, TESS and RV detection
models, and writes plots under `results/figures/`. It takes under a minute. To try
another pipeline, change `SIM_NAME` at the top of the script; every option except
`flat_universe` builds a P-Pop universe and takes about 30 minutes.

`run/run_sim.py` runs the full study. As checked in, it runs 10 Kepler and 10 TESS
universes on the Gaia 60 pc catalog, which takes half an hour or longer, then plots with
`plotting/plot.py` into `results/figures/simulation/`. The HWO and LIFEsim runs
are in the same file but commented out.

## Layout

- `run/`: simulation entry points
- `telescopes/`: detection models for Kepler, TESS, HWO and RV
- `plotting/`: plotting and analysis code
- `tools/`: shared paths and constants
- `data/`: input data (tracked)
- `results/`: everything the pipelines write (git-ignored, see `results/README.md`)
- `lifesim/`, `PPop/`: the vendored forks

## Credit and licence

LIFEsim is by Felix Dannert, Maurice Ottiger and Sascha Quanz (ETH Zürich). P-pop is
by Jens Kammerer. Forecaster is by Jingjing Chen.

The project is GPL-3.0, carried over from LIFEsim. The MIT parts keep their own
licence files. Contact: Annika Salmi, annikaksalmi@gmail.com
