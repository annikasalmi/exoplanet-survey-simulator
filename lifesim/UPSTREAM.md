# Forked LIFEsim

This is a fork of [LIFEsim](https://github.com/fdannert/LIFEsim) by Felix Dannert, Maurice
Ottiger and Sascha Quanz (ETH Zürich), under GPLv3. The fork point is `a2b8eeb` (2023-09-20).
`core/data.py` had catalog handling reworked to match DataFrame handling throughout the repo more smoothly.
Additional changes were to `core/core.py`, `instrument/instrument.py` and `util/habitable.py` to use paths tooling
and to use seeded random numbers as opposed to np.random.

To see exactly what changed, run `git diff a2b8eeb..HEAD -- lifesim`.