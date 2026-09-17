# exoplanet-survey-simulator

Simulates whether planned missions can detect different types of simulated exoplanets.
These simulations include LIFE, forked from [LIFEsim](https://github.com/fdannert/LIFEsim).
Additional simulations written for this repo are HWO, TESS, Kepler, HARPS, and NIRPS.

Simulated planets are generated from a modified copy of [P-Pop](https://github.com/kammerje/P-pop) and [chenjj2/forecaster](https://github.com/chenjj2/forecaster).
Occurrence rates are Bergsten et al. (2022) for FGK stars and Dressing & Charbonneau (2015) for M dwarfs.

## Install

Needs Python 3.9 or newer.

```bash
git clone https://github.com/annikasalmi/exoplanet-survey-simulator.git
cd exoplanet-survey-simulator
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Only this editable install from a clone works; the code expects the repo layout.

## Quickstart

```bash
python simulation_demo.py
```

That draws flat populations of rocky planets around G, K and M stars, runs them
through the TESS transit and HARPS/NIRPS RV detection models, and plots the fraction
detected in a 3x3 insolation-radius map, with NASA's measured rocky planets on top.
The figure goes to `results/figures/analysis/flat_transit_rv_3x3/`. It takes under a minute. To try
another pipeline, change `SIM_NAME` at the top of the script; every option except
`flat_universe` builds a P-Pop universe and takes about 20 minutes.

## Paper figures

```bash
python run/make_paper_figures.py
```

This makes every paper figure into `results/paper/` in about a minute. The first run also
needs internet access: two NASA Exoplanet Archive queries, cached under
`results/catalogs/kepler/NASA/`, and the 106 per-sector SPOC CDPP tables (about 230 MB) from
MAST into `results/catalogs/tess/CDPP/`, which only the TESS calibration figure reads. The
script stops with an error if a step fails or does not write the figures listed for it.
`--list` prints the map below, and naming a script (e.g.
`python run/make_paper_figures.py rocky_scatter_gaia60pc`) runs only that one.

None of the paper figures needs a P-Pop universe. The flat populations are drawn inside
each script from the seeds set at its top.

| Figure (`results/paper/`) | Script | Input |
|---|---|---|
| `kepler_3in1_calibration.png` | `plotting/scripts/calibration/kepler_calibration.py` | `data/exoplanet_csv/koi_cumulative_stellar.csv` |
| `tess_3in1_calibration.png` | `plotting/scripts/calibration/tess_calibration.py` | `data/exoplanet_csv/exofop_toi.csv`; SPOC CDPP tables for sectors 1-106 from MAST |
| `rv_k_vs_published_rvamp.png`, `rv_sigmaK_3in1.png` | `plotting/scripts/calibration/rv_detector_check.py --fig1-only` | NASA Archive query (planets with published K) |
| `flat_transit_rv_3x3_otegi.png` | `plotting/scripts/analysis/multi/flat_transit_rv_3x3.py` | flat population; NASA Archive query (transiting planets with masses) |
| `rocky_mr_insolation_3panel.png`, `rocky_scatter_standalone.png` | `plotting/scripts/analysis/multi/rocky_scatter_gaia60pc.py` | NASA Archive query (same as above); `data/silicon_curve.ddat` |
| `flat_rocky_mr_relations_2x4_cold_corner.png`, `flat_otegi_2x2_before_after.png`, `flat_otegi_2x1_cold_cut.png`, `flat_otegi_1x2_cold_cut.png`, `flat_rocky_mr_2col_chen_otegi_cold.png` | `plotting/scripts/analysis/multi/flat_rocky_mr_vs_nasa.py` | flat population; `data/exoplanet_csv/pscomppars_2026.csv` |
| `mc_comparison_statistic_5000.png` | `plotting/scripts/analysis/multi/mc_comparison_statistic.py` | flat population; `data/exoplanet_csv/pscomppars_2026.csv` |

### What does need P-Pop universes

`run/run_sim.py` runs the full P-Pop study. As checked in, it builds 10 Kepler and 10 TESS
universes on the Gaia 60 pc catalog and plots each set with `plotting/plot.py` into
`results/figures/simulation/`. One universe takes about 20 minutes, almost all of it drawing
the planets (21 min for Kepler on an 11-core, 18 GB Mac). Kepler runs 5 universes at a time
and TESS 2 (memory limits, see `MAX_WORKERS` in `run/kepler/run_kepler.py` and
`run/tess/run_tess.py`), so the whole run takes roughly 3-4 hours. That total is extrapolated
from single universes, not timed end to end. The HWO and LIFEsim runs are in the same file
but commented out.

These read the Gaia universes `run_sim.py` writes. None of their output is in the paper:

- `rocky_scatter_gaia60pc.py --full`: the Kepler/TESS detection-fraction maps, from all 10 of each
- `puffy_cuts_flat.py`: Kepler universe 0
- `rv_detector_check.py` without `--fig1-only`: the RV pipeline stage check, from up to 60 TESS universes

## Terms

- **Universe A / B**: two flat universes, where A drops rocky planets above 2 Earth masses and B keeps them.
- **Otegi**: the Otegi et al. (2020) mass-radius relations, R = 1.03 M^0.29 (rocky) and R = 0.70 M^0.63 (volatile-rich).

## Layout

- `run/`: simulation entry points
- `telescopes/`: detection models for Kepler, TESS, HWO and RV
- `plotting/`: plotting and analysis code
- `tools/`: shared paths and constants
- `data/`: input data (tracked)
- `results/`: everything the pipelines write (git-ignored, see `results/README.md`)
- `lifesim/`, `PPop/`: the vendored forks

## Tests

```bash
pytest
```

This runs every pipeline end to end on the Gaia stars within 10 pc, in about three
minutes, and writes nothing under `results/`. CI runs it on every push and pull request.

## Credit and licence

LIFEsim is by Felix Dannert, Maurice Ottiger and Sascha Quanz (ETH Zürich). P-pop is
by Jens Kammerer. Forecaster is by Jingjing Chen.

The project is GPL-3.0, carried over from LIFEsim. The MIT parts keep their own
licence files. Contact: Annika Salmi, annikaksalmi@gmail.com
