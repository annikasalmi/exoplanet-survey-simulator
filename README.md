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

## Full simulation runs

`run/run_sim.py` orchestrates multiple P-Pop universes and plotting. As configured, it
builds 10 Kepler and 10 TESS universes on the Gaia 60 pc catalogue; LIFE and HWO runs can
be enabled in the same file. One universe takes about 20 minutes. Outputs go to `results/`.

## Repository layout

- `run/`: simulation and figure-generation entry points
- `telescopes/`: Kepler, TESS, RV, and HWO detection models
- `lifesim/`: LIFE instrument simulation
- `plotting/`: calibration, visualization, and analysis code
- `PPop/`: modified population generator and Forecaster copy
- `tools/`, `data/`: shared code and tracked inputs
- `results/`: generated catalogues, logs, and figures; see `results/README.md`
- `docs/`: LIFEsim component documentation

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
