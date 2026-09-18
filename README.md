# exoplanet-survey-simulator

**Primary purpose:** this is a reusable framework for simulating exoplanet survey and
detection pipelines. It also includes the configurations, data snapshots, and scripts
needed to reproduce the accompanying paper. It focuses on rocky to sub-Neptune sized planets.

The framework can generate synthetic planets, apply selection effects, propagate
measurement uncertainties, compare detections with catalogued planets, and produce
analysis figures. 

The framework supports different science questions by allowing population models,
instruments, selection cuts, and comparison statistics to be changed without rewriting the
full pipeline. Included detection models cover Kepler, TESS, HARPS/NIRPS-like
radial-velocity observations, HWO, and LIFE.

This table summarizes the detectors:
## Instrument model fidelity

| Instrument | Modeled | Approximation/calibration | Omitted |
|---|---|---|---|
| Kepler | Transit geometry, depth, duration, count, brightness, CDPP and 7.1 MES | Magnitude-scaled fallback CDPP; 0.84 factor calibrated to DR25 MES | DR25 depth/window maps, injection recovery, cadence gaps, dilution and vetting |
| TESS | Transit geometry, phase, sector windows, CDPP, optional dilution and 7.1 S/N | Five sectors and binned SPOC CDPP by default; 0.80 factor calibrated to SPOC TOIs | Target completeness, detailed gaps, injection recovery and vetting |
| HARPS/NIRPS-like RV | Keplerian amplitude, V/J brightness, noise/jitter, 100 epochs and 5-sigma threshold | Population-level magnitudes, masses and jitter; checked against published amplitudes | Real schedules, aliases, multi-planet fits, activity mitigation and target allocation |
| HWO | IWA, blackbody flux ratio, photon rate and exozodi best/worst cuts | Fixed wavelength endpoints and thresholds; uncalibrated | Phase completeness, reflected light, contrast curves, exposure S/N, systematics and scheduling |
| LIFE | Nulling transmission, planet signal, photon backgrounds, baseline and time optimization | LIFEsim baseline with 4 m diameter; no flown-instrument calibration | Empirical completeness and unconfigured hardware/systematic noise |

Treat the toy models as selection-effect experiments, not absolute mission-yield predictions.

## Workflow

1. Draw one of the two flat universes or the science-based P-Pop universe.
2. Apply the selected transit, radial-velocity, coronagraph, or interferometer model.
3. Add measurement uncertainties and analysis cuts.
4. Compare the recovered population with observed exoplanet catalogues.
5. Generate calibration, selection-function, and population-comparison figures.

The P-Pop "real" universe is vendored from [P-Pop](https://github.com/kammerje/P-pop) and
[Forecaster](https://github.com/chenjj2/forecaster). The LIFE planetary detction model is based on the included
[LIFEsim](https://github.com/fdannert/LIFEsim) fork. `UPSTREAM.md` files list the changes.

## Install

Python 3.9 or newer is required.

```bash
git clone https://github.com/annikasalmi/exoplanet-survey-simulator.git
cd exoplanet-survey-simulator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```
Data files in the repo are recorded in [`data/manifest.json`](data/manifest.json).

## Quickstart

```bash
python sim.py
```

`sim.py` draws the baseline flat universe, runs the Kepler, TESS and RV detectors,
and saves the catalog and plots under `results/`.

There are two main types of flat universes:
- baseline flat: radius is uniformly drawn from 0 to 12 R_Earth
- curve-based flat: planets are placed around the silicate curve or volatile curve

P-Pop is the separate science-based universe from the vendored P-Pop model.

The flat-universe masses are taken from radius and propagated forward with the
Otegi (2020) mass-radius relation with scatter.

The three implementations live in `science/populations/universes/flat_baseline.py`,
`flat_curves.py`, and `ppop.py`. Import each generator directly from its module.
`sim.py` selects one, generates the catalog, then applies the telescope detectors.
Analysis plots, including the
transit/RV 3x3 map, call these same generators with their paper-specific bounds.

The curve-based flat simulations support:
- universe B, where planets exist on either the silicate or volatile curve
- universe A, where the super-Earths on the silicate curve are removed

```bash
python sim.py --universe flat_baseline --n-planets 20000 --seed 0
python sim.py --universe flat_curves --variant B
python sim.py --universe flat_curves --variant A
python sim.py --universe ppop --star-catalog LTC_2
```

The batch pipelines under `run/` retain their catalog caching and multi-run
support, including the paired A/B comparison and the HWO and LIFEsim models.
Run the paired A/B likelihood analysis with `python -m plotting.likelihood_ratio_plotter`.

## Reproducing the paper's analysis figures

```bash
python run/make_paper_figures.py
```

## Credit and licence

LIFEsim is by Felix Dannert, Maurice Ottiger, and Sascha Quanz (ETH Zürich); P-Pop is by
Jens Kammerer; Forecaster is by Jingjing Chen. The project is GPL-3.0, carried over from
LIFEsim, while included MIT-licensed components retain their own licences.

Contact: Annika Salmi, annikaksalmi@gmail.com
