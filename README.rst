exoplanet-survey-simulator
==========================

Tools to simulate and analyze the detectability of potentially habitable exoplanets
around M dwarfs, for the LIFE and HWO mission concepts.

For usage instructions, see ``README.md`` at the project root.

Provenance
----------

This repository builds on two existing codebases, each documented in its own
``UPSTREAM.md``:

* ``lifesim/`` — fork of LIFEsim (Felix Dannert, Maurice Ottiger, Sascha Quanz,
  ETH Zürich) at ``a2b8eeb``, GPLv3: https://github.com/fdannert/LIFEsim
* ``PPop/`` — modified copy of P-pop (Jens Kammerer), MIT:
  https://github.com/kammerje/P-pop
* ``PPop/MassModels/Forecaster/`` — Forecaster (Jingjing Chen), MIT:
  https://github.com/chenjj2/forecaster

Everything under ``hwo/``, ``run/``, ``plot/``, ``tools/`` and ``tests/`` was written
for this project.

License
-------

GPL-3.0 (see ``LICENSE``). Vendored MIT components retain their own licence files.
