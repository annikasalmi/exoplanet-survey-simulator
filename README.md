# exoplanet-survey-simulator

An end-to-end framework for studying how exoplanet detection pipelines transform an
underlying planet population into an observed sample. It can generate synthetic planets,
apply instrument and survey selection effects, propagate measurement uncertainties,
compare detections with catalogued planets, and produce analysis figures.

The framework supports different science questions by allowing population models,
instruments, selection cuts, and comparison statistics to be changed without rewriting the
full pipeline. Included detection models cover Kepler, TESS, HARPS/NIRPS-like
radial-velocity observations, HWO, and LIFE.

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

## Quickstart

```bash
python simulation_demo.py
```

The default demo draws a controlled planet population, applies the TESS transit and
HARPS/NIRPS-like RV models, and writes a detection-fraction map under
`results/figures/analysis/flat_transit_rv_3x3/`. It takes under a minute.

Set `SIM_NAME` in `simulation_demo.py` to run `kepler`, `tess`, `rv`, `hwo`, or `lifesim`.
These options build a P-Pop universe and take about 20 minutes per universe.

## Analysis figures

```bash
python run/make_paper_figures.py
```

This writes the publication figures to `results/paper/`. Use `--list` to show the mapping
from figures to scripts, or pass a script name to run one analysis:

```bash
python run/make_paper_figures.py rocky_scatter_gaia60pc
```

The first complete run requires internet access for NASA Exoplanet Archive queries and the
TESS SPOC CDPP tables from MAST. Downloads are cached under `results/catalogs/`.

| Figure (`results/paper/`) | Script | Input |
|---|---|---|
| `kepler_3in1_calibration.png` | `plotting/scripts/calibration/kepler_calibration.py` | `data/exoplanet_csv/koi_cumulative_stellar.csv` |
| `tess_3in1_calibration.png` | `plotting/scripts/calibration/tess_calibration.py` | `data/exoplanet_csv/exofop_toi.csv`; SPOC CDPP tables for sectors 1-106 from MAST |
| `rv_k_vs_published_rvamp.png`, `rv_sigmaK_3in1.png` | `plotting/scripts/calibration/rv_detector_check.py --fig1-only` | NASA Archive query (planets with published K) |
| `flat_transit_rv_3x3_otegi.png` | `plotting/scripts/analysis/flat_transit_rv_3x3.py` | flat population; NASA Archive query (transiting planets with masses) |
| `rocky_mr_insolation_3panel.png`, `rocky_scatter_standalone.png` | `plotting/scripts/analysis/rocky_scatter_gaia60pc.py` | NASA Archive query (same as above); `data/silicon_curve.ddat` |
| `flat_rocky_mr_relations_2x4_cold_corner.png`, `flat_otegi_2x2_before_after.png`, `flat_otegi_2x1_cold_cut.png`, `flat_otegi_1x2_cold_cut.png`, `flat_rocky_mr_2col_chen_otegi_cold.png` | `plotting/scripts/analysis/flat_rocky_mr_vs_nasa.py` | flat population; `data/exoplanet_csv/pscomppars_2026.csv` |
| `mc_comparison_statistic_5000.png` | `plotting/scripts/analysis/mc_comparison_statistic.py` | flat population; `data/exoplanet_csv/pscomppars_2026.csv` |

## Full simulation runs

`run/run_sim.py` orchestrates multiple P-Pop universes and plotting. As configured, it
builds 10 Kepler and 10 TESS universes on the Gaia 60 pc catalogue; LIFE and HWO runs can
be enabled in the same file. One universe takes about 20 minutes. Outputs go to `results/`.

## Repository layout

- `rocky_scatter_gaia60pc.py --full`: the Kepler/TESS detection-fraction maps, from all 10 of each
- `puffy_cuts_flat.py`: Kepler universe 0
- `rv_detector_check.py` without `--fig1-only`: the RV pipeline stage check, from up to 60 TESS universes

## Terms

- **Universe A / B**: two flat universes, where A drops rocky planets above 2 Earth masses and B keeps them.
- **Otegi**: the Otegi et al. (2020) mass-radius relations, R = 1.03 M^0.29 (rocky) and R = 0.70 M^0.63 (volatile-rich).

## Layout

- `run/`: simulation entry points
- `science/`: population generators and telescope detection models for Kepler, TESS, HWO and RV
- `plotting/`: plotting and analysis code
- `tools/`: shared paths and constants
- `data/`: input data (tracked)
- `results/`: everything the pipelines write (git-ignored, see `results/README.md`)
- `lifesim/`, `PPop/`: the vendored forks

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
