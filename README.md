# exoplanet-survey-simulator

**Primary purpose:** this is a reusable framework for simulating exoplanet survey and
detection pipelines. It also includes the configurations, data snapshots, and scripts
needed to reproduce the accompanying paper.

The framework can generate synthetic planets, apply selection effects, propagate
measurement uncertainties, compare detections with catalogued planets, and produce
analysis figures.

The framework supports different science questions by allowing population models,
instruments, selection cuts, and comparison statistics to be changed without rewriting the
full pipeline. Included detection models cover Kepler, TESS, HARPS/NIRPS-like
radial-velocity observations, HWO, and LIFE.

The workflow is "end-to-end;" the instruments are imperfectly represented by the basic
physics rules they follow. This table summarizes the detectors:

## Instrument model fidelity

| Instrument | Fidelity | Modeled | Approximation/calibration | Omitted |
|---|---|---|---|---|
| Kepler | Toy | Transit geometry, depth, duration, count, brightness, CDPP and 7.1 MES | Magnitude-scaled fallback CDPP; 0.84 factor calibrated to DR25 MES | DR25 depth/window maps, injection recovery, cadence gaps, dilution and vetting |
| TESS | Toy | Transit geometry, phase, sector windows, CDPP, optional dilution and 7.1 S/N | Five sectors and binned SPOC CDPP by default; 0.80 factor calibrated to SPOC TOIs | Target completeness, detailed gaps, injection recovery and vetting |
| HARPS/NIRPS-like RV | Toy | Keplerian amplitude, V/J brightness, noise/jitter, 100 epochs and 5-sigma threshold | Population-level magnitudes, masses and jitter; checked against published amplitudes | Real schedules, aliases, multi-planet fits, activity mitigation and target allocation |
| HWO | Exploratory cuts | IWA, blackbody flux ratio, photon rate and exozodi best/worst cuts | Fixed wavelength endpoints and thresholds; uncalibrated | Phase completeness, reflected light, contrast curves, exposure S/N, systematics and scheduling |
| LIFE | Concept-yield model | Nulling transmission, planet signal, photon backgrounds, baseline and time optimization | LIFEsim baseline with 4 m diameter; no flown-instrument calibration | Empirical completeness and unconfigured hardware/systematic noise |

Treat the toy models as selection-effect experiments, not absolute mission-yield predictions.

## Workflow

1. Draw a controlled flat population or an occurrence-rate-based P-Pop universe.
2. Apply the selected transit, radial-velocity, coronagraph, or interferometer model.
3. Add measurement uncertainties and analysis cuts.
4. Compare the recovered population with observed exoplanet catalogues.
5. Generate calibration, selection-function, and population-comparison figures.

The P-Pop universes use Bergsten et al. (2022) occurrence rates for FGK stars and Dressing
& Charbonneau (2015) for M dwarfs. Planet masses and radii use the included modified copies
of [P-Pop](https://github.com/kammerje/P-pop) and
[Forecaster](https://github.com/chenjj2/forecaster). LIFE is based on the included
[LIFEsim](https://github.com/fdannert/LIFEsim) fork.

## Install

Python 3.9 or newer is required.

```bash
git clone https://github.com/annikasalmi/exoplanet-survey-simulator.git
cd exoplanet-survey-simulator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Use an editable install from a clone; the code expects the repository layout.
`setup.py` is the only dependency list. Extras: `pip install -e ".[test]"` adds pytest;
`".[tess]"` adds tess-point, needed only to rebuild the committed TESS sector grid.

Tracked input provenance is recorded in [`data/manifest.json`](data/manifest.json).

## Quickstart

```bash
python sim.py
```

`sim.py` draws seeded flat universes A and B, runs the Kepler, TESS and RV detectors through
`run_sim`, and saves the catalogs and plots under `results/`.

There are other simulations that can be run as well.

## Flat universes

The flat analyses use no occurrence rates. They draw planets from `science/populations/universes.py`:
Universe A is the no-super-Earth null population; Universe B is the mixed alternative that
tests how adding super-Earths changes survey outcomes.

| Function | Planets | Used by |
|---|---|---|
| `flat_nonphysical()` | Uniform radius, 0.5–2.2 R⊕; mass from Otegi's rocky relation (R = 1.03 M^0.29, 0.15 dex scatter) | 3x3 selection map, M-R relation grids, generator comparison |
| `flat_superearths_subneptunes()` (universe B) | `flat_nonphysical` with radii redrawn and masses kept. Super-Earths (on or below the silicate line, M > 2 M⊕) get a 20% normal around the silicate line; the rest get one around Otegi's volatile-rich relation (R = 0.70 M^0.63) | A/B pipeline, likelihood-ratio plot, Otegi 1x2, puffy cuts, Bayesian comparison, MC comparison statistic |
| `flat_subneptunes_only()` (universe A) | Universe B without its super-Earths | same as B |

Each script sets its own size, seed and parameter ranges. `bayesian_cold_rocky_desert.py` also
draws a pool with mass independent of radius, only to map detection probability.

## Reproducing the paper's analysis figures

```bash
python run/make_paper_figures.py
```

This writes the publication figures to `results/paper/`. Use `--list` to show the mapping
from figures to scripts, or pass a script name to run one analysis:

```bash
python run/make_paper_figures.py rocky_scatter_gaia60pc
```

Two inputs are downloaded on the first run and cached under `results/catalogs/`: the
published-K sample from the NASA Exoplanet Archive (`rv_detector_check`) and the TESS SPOC
CDPP tables from MAST (`tess_calibration`). The other tables ship in `data/exoplanet_csv/`;
set `DOWNLOAD_NASA_DATA = True` in `rocky_scatter_gaia60pc.py` to refresh the transiting one.

| Figure (`results/paper/`) | Script | Input |
|---|---|---|
| `kepler_3in1_calibration.png` | `plotting/scripts/calibration/kepler_calibration.py` | `data/exoplanet_csv/koi_cumulative_stellar.csv` |
| `tess_3in1_calibration.png` | `plotting/scripts/calibration/tess_calibration.py` | `data/exoplanet_csv/exofop_toi.csv`; SPOC CDPP tables for sectors 1-106 from MAST |
| `rv_k_vs_published_rvamp.png`, `rv_sigmaK_3in1.png` | `plotting/scripts/calibration/rv_detector_check.py --fig1-only` | NASA Archive query (planets with published K) |
| `flat_transit_rv_3x3_otegi.png` | `plotting/scripts/analysis/flat_transit_rv_3x3.py` | `flat_nonphysical`; `data/exoplanet_csv/pscomppars_transiting_mass_insol.csv` |
| `rocky_mr_insolation_3panel.png`, `rocky_scatter_standalone.png` | `plotting/scripts/analysis/rocky_scatter_gaia60pc.py` | `data/exoplanet_csv/pscomppars_transiting_mass_insol.csv`; `data/silicon_curve.ddat` |
| `flat_rocky_mr_relations_2x4_cold_corner.png`, `flat_otegi_2x2_before_after.png`, `flat_otegi_2x1_cold_cut.png`, `flat_otegi_1x2_cold_cut.png`, `flat_rocky_mr_2col_chen_otegi_cold.png` | `plotting/scripts/analysis/flat_rocky_mr_vs_nasa.py` | `flat_nonphysical` and universe B; `data/exoplanet_csv/pscomppars_2026.csv` |
| `mc_comparison_statistic_5000.png` | `plotting/scripts/analysis/mc_comparison_statistic.py` | universes A and B; `data/exoplanet_csv/pscomppars_2026.csv` |

## Terms

- **Universe A / B**: two flat universes (see "Flat universes"). B holds super-Earths and sub-Neptunes; A drops the super-Earths.
- **Otegi**: the Otegi et al. (2020) mass-radius relations, R = 1.03 M^0.29 (rocky) and R = 0.70 M^0.63 (volatile-rich).

## Repository layout

- `sim.py`: runs the simulations
- `run/`: one pipeline per telescope, plus the paper-figure runner
- `science/`: population generators and telescope detection models for Kepler, TESS, HWO and RV
- `plotting/`: plotting and analysis code
- `tools/`: shared paths and constants
- `data/`: input data (tracked)
- `results/`: everything the pipelines write (git-ignored, see `results/README.md`)
- `lifesim/`, `PPop/`: modified copies of outside code (see below)

## Vendored code

Only `PPop/` and `lifesim/` come from other projects; their `UPSTREAM.md` files list the changes.

- `PPop/`: [P-pop](https://github.com/kammerje/P-pop) with [Forecaster](https://github.com/chenjj2/forecaster).
  Draws the occurrence-rate universes, called through `science/populations/ppop.py`.
- `lifesim/`: [LIFEsim](https://github.com/fdannert/LIFEsim). The LIFE detection model, run by `run/lifesim/`.

## Tests

```bash
pytest
```

The tests run the detection pipelines on a small Gaia sample and keep generated files out
of `results/`. CI runs them on every push and pull request.

## Credit and licence

LIFEsim is by Felix Dannert, Maurice Ottiger, and Sascha Quanz (ETH Zürich); P-Pop is by
Jens Kammerer; Forecaster is by Jingjing Chen. The project is GPL-3.0, carried over from
LIFEsim, while included MIT-licensed components retain their own licences.

Contact: Annika Salmi, annikaksalmi@gmail.com
