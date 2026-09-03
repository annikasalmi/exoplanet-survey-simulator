# exoplanet-survey-simulator

Simulates whether planned missions can detect different types of simulated exoplanets.
These simulations include LIFE, forked from [LIFEsim](https://github.com/fdannert/LIFEsim).
Additional simulations written for this repo are HWO, TESS, Kepler, HARPS, and NIRPS.

Simulated planets are generated from a modified copy of [P-Pop](https://github.com/fdannert/P-pop) and [chenjj2/forecaster](https://github.com/chenjj2/forecaster).

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
python run/run_sim.py
```

That runs the simulation set up in `run/hwo/` or `run/lifesim/` and draws default plots from `plot/plot.py`. 

## Credit and licence

LIFEsim is by Felix Dannert, Maurice Ottiger and Sascha Quanz (ETH Zürich). P-pop is
by Jens Kammerer. Forecaster is by Jingjing Chen.

The project is GPL-3.0, carried over from LIFEsim. The MIT parts keep their own
licence files. Contact: Annika Salmi, annikaksalmi@gmail.com
