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

Edit the options at the top of `sim.py`, then run it. This generates the selected
universe, applies the selected telescope models, handles single or parallel multi-run
caches, and writes catalogs plus plots under `results/`.

Set `UNIVERSE` to one of:
- `flat_nonphysical`
- `flat_radii_curves`
- `nasa_exoplanets`
- `ppop`

For `flat_radii_curves`, set `FLAT_RADII_VARIANT` to either
`superearths_supneptunes` or `only_subneptunes`, or set
`RUN_BOTH_FLAT_RADII_VARIANTS = True` to run both. Set `TELESCOPES` to any of
`kepler`, `tess`, `rv`, or `hwo`. Use `TELESCOPES = ("lifesim",)` by itself for
LIFE. Set `NRUNS > 1` and `PARALLEL = True` for parallel batches.

Simulation runs should go through `sim.py`.

## Reproducing the paper's analysis figures

```bash
python make_paper_figures.py
```

This writes every paper figure to `results/paper/`, downloading and caching the catalogues
it needs; no P-Pop universe is required. `--list` prints each script and the figures it
makes, and naming a script runs only that one. To run them individually instead, the
scripts are under `plotting/scripts/`.
This table summarizes the detectors:
## Instrument model fidelity

| Instrument | Modeled | Approximation/calibration | Omitted |
|---|---|---|---|
| Kepler | Limb-darkened transit shape, impact parameter, T14, count, brightness, CDPP at T14 and 7.1 MES | Solar limb darkening; magnitude-scaled fallback CDPP with measured duration scaling; no correction factor (model/DR25 MES median 0.98) | DR25 depth/window maps, injection recovery, cadence gaps, dilution and vetting |
| TESS | Limb-darkened transit shape, impact parameter, T14, phase, sector windows, CDPP at T14, optional dilution and 7.1 S/N | Five sectors and binned SPOC CDPP by default; solar limb darkening; no correction factor (model/SPOC SNR median 0.98) | Target completeness, detailed gaps, injection recovery and vetting |
| HARPS/NIRPS-like RV | Keplerian amplitude, V/J brightness, noise/jitter, 100 epochs and 5-sigma threshold | Population-level magnitudes, masses and jitter; checked against published amplitudes | Real schedules, aliases, multi-planet fits, activity mitigation and target allocation |
| HWO | IWA, blackbody flux ratio, photon rate and exozodi best/worst cuts | Fixed wavelength endpoints and thresholds; uncalibrated | Phase completeness, reflected light, contrast curves, exposure S/N, systematics and scheduling |
| LIFE | Nulling transmission, planet signal, photon backgrounds, baseline and time optimization | LIFEsim baseline with 4 m diameter | Empirical completeness and unconfigured hardware/systematic noise |

Treat the toy models as selection-effect experiments, not absolute mission-yield predictions.

## Credit and licence

LIFEsim is by Felix Dannert, Maurice Ottiger, and Sascha Quanz (ETH Zürich); P-Pop is by
Jens Kammerer; Forecaster is by Jingjing Chen. The project is GPL-3.0, carried over from
LIFEsim, while included MIT-licensed components retain their own licences.

Contact: Annika Salmi, annikaksalmi@gmail.com
