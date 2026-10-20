# Population-comparison audit

For the implemented standardization and current inference, see [Population comparison methods](POPULATION_COMPARISON_METHODS.md).

Historical audit: the selection findings below describe the pre-update M>2 analysis. The subsequent selection update uses M>2.5 and R>1.4; rerun outputs supersede the counts and posterior values below.

Audited 20 October 2026 against the current working tree and files directly in results/paper (excluding old/). This describes current source behavior; most PNGs have no provenance sidecar, so their complete generating configuration cannot be verified from the images alone. No analysis code or figures were changed. Existing edits to mc_comparison_statistic.py and the untracked calc_mc_comparison_likelihood.py were preserved.

## Current figure procedures

| Output | Generator | Generated pool | MC realizations | Mock survey size | Measurement errors |
|---|---|---:|---:|---|---|
| flat_otegi_1x2_low-insolation_selection.png | flat_rocky_mr_vs_nasa.py; flat_radii_curves | 150,000 | 4,000 | 50 for sub-Neptune model; 100 for mixed model | Published asymmetric NASA errors; synthetic lognormal widths 0.20 mass, 0.046 radius |
| flat_rocky_mr_relations_2x4_low-insolation_corner.png | Same script; four flat_nonphysical mass-radius prescriptions | 150,000 per relation | 4,000 per distribution | Same unequal 50/100 sizes | Same as above |
| mc_comparison_statistic_5000.png | mc_comparison_statistic.py; flat_radii_curves through make_detected_pool | 2,000,000 | 5,000 default; N_DRAWS environment override | Same NASA-sized survey for both models in each bin | Independent lognormal widths 0.25 mass, 0.08 radius on NASA and synthetic planets |
| flat_transit_rv_3x3_otegi.png | flat_transit_rv_3x3.py | G: 3M plus 6M map-only supplement; K: 2.5M; M: 1.5M | No repeated volatile-fraction survey MC | Not applicable | Detection maps and rocky overlays |
| rocky_mr_insolation_3panel.png; rocky_scatter_standalone.png | rocky_scatter_gaia60pc.py | Different purpose: catalogue/selection diagnostics | No repeated volatile-fraction survey MC | Not applicable | Catalogue errors and stricter overlay quality cuts |
| recovery_3x1.png | recovery_3x1.py | Detector calibration samples | No repeated volatile-fraction survey MC | Not applicable | Calibration procedure |

Pool size means generator input size, not retained or jointly detected planet count.

## Shared and differing selections

- Main comparison box: 0.1 <= mass <= 12 Earth masses; 0.5 <= radius <= 2.2 Earth radii; 0.01 <= insolation <= 10,000 Earth fluxes.
- Fraction figures and discrepancy figure both use low-insolation I < 50 and perturbed mass > 2; neither adds radius > 1.4.
- The mass-radius figure's NASA loader has no initial insolation bounds; the discrepancy figure applies the full comparison box. Synthetic default generators already constrain insolation.
- Rocky map/overlay windows instead use radius > 0.6, insolation > 0.1, and stellar-type selections. The radius > 1.4 desert outline serves a different selection.
- Fraction/discrepancy simulations default to Kepler transit + RV; the paper 3x3 detection map explicitly uses TESS. This may be intentional but needs clear captions and a mission sensitivity comparison.
- All composition fractions use the unshifted pure-silicate curve, with volatile defined as radius above the threshold at the measured/perturbed mass. This is distinct from a radius-only valley threshold.

## Verified current catalogue counts

Loading pscomppars_2026.csv with the comparison box, measured mass provenance, and 25%/8% precision cuts gives 129 planets before the mass > 2 cut. Removing the initial insolation bounds gives 130.

| Interval, with mass > 2 | Volatile | Total |
|---|---:|---:|
| 0.01 <= I < 10 | 6 | 7 |
| 0.01 <= I < 50 | 20 | 27 |
| 50 <= I < 10,000 | 24 | 75 |

Adding radius > 1.4 to the low-insolation selection gives 22 planets. These are nominal-value counts; MC mass selection can change the observed denominator.

The loader admits HD 219134 f (mass 7.3, radius 1.31, I=12.38), despite transit_flag=0 and both mass/radius limit flags=-1. It contributes a rocky object to the 27-planet low-insolation sample. The separate rocky overlay explicitly rejects such limits and requires transits through its source query. There are no missing mass/radius uncertainty sides among the 129 selected planets.

## Monte Carlo mechanics

- Both current paper fraction workflows use mock_survey_volatile_fractions: perturb the entire detected pool, apply observed mass > 2, then select a fixed-size sample without replacement. Repeat this process; skip realizations if too few eligible planets remain.
- Model membership is defined using true mass/radius before observational noise; noise may move a planet across the composition boundary.
- NASA draws keep the catalogue objects fixed and perturb their measurements; they do not bootstrap catalogue membership. Reapplying the mass cut can change the denominator.
- The older monte_carlo_population_fraction routine instead samples with replacement before noise and the mass cut. Its 20,000 objects / 10,000 repeats belong to extra_plots/puffy_cuts_flat.py, not the current paper figure route.
- Seed 0 is common to the fraction scripts, but they share one advancing RNG across panels and figures. Earlier plotting operations alter later draws. Matching seed numbers does not establish matching realizations.
- The main comparison clips noised values neither back into the full radius/mass box nor into the original population; the box is selected before noise, and only the lower mass threshold is reapplied. This should be explicitly chosen and documented.

## Statistic and likelihood discrepancies

- Volatile fraction is k/n, where k counts planets above the silicate curve.
- Fraction plots show distributions of f and fitted Gaussian curves. Their printed tension uses abs(mean_model - mean_observed) / sqrt(sd_model^2 + sd_observed^2).
- The discrepancy plot displays z^4, with z=(f_model-f_observed)/sigma_observed. Its posterior uses z^2 evaluated at each model's MC mean, not the plotted fourth-power statistic.
- Main posterior: equal model priors; independent Gaussian bin errors; only I<50 and 50<=I<10,000 contribute. I<10 is diagnostic because it is nested within I<50.
- The standalone calc_mc_comparison_likelihood.py sums all three bins, double-counting the nested I<10 subset, so its total differs from the main posterior by construction.
- The current posterior uses model MC means and observed uncertainty only. Mock-survey spread is displayed but not integrated into the likelihood. This is a conditional Gaussian approximation, not a full posterior predictive likelihood.
- The saved posterior sidecar reports 5,000 draws, seed 0, chi-square 30.462667 and 102.896620, and escape-only posterior approximately 1.87e-16. These values belong to the current assumptions and require recomputation after selection/error changes.
- extra_plots/bayesian_low_insolation_rocky_desert.py instead uses binomial likelihoods at point-estimate predicted fractions, 4,000 noise realizations, and 10M generated pools. Its reported uncertainty estimates do not enter that binomial likelihood. A printed suggestion mentions beta-binomial improvement, but that is not the implemented likelihood.

## Proposed standardization

1. Share a single configuration and result computation between the population-fraction figure, discrepancy figure, and numerical likelihood report. Plotters should consume the same saved fraction draws and summary rather than independently recompute them.
2. Use the same NASA sample definition everywhere: transiting, measured mass, no mass/radius limits, valid two-sided uncertainties, declared precision cuts, and common comparison box. Save selected planet names and rejection counts. Recompute all counts after this change.
3. Keep mass > 2 and I < 50 as the primary fraction selection if retaining the current scientific question. Treat radius > 1.4 as a separately named sensitivity/diagnostic selection.
4. Use equal NASA-matched sample sizes for both competing models. Move N=50 and N=100 to an explicit forecast, with both models evaluated at each N.
5. Use 5,000 realizations consistently as an initial baseline, then check stability at higher draws and across independent seeds. Separate finite-pool convergence from draw-count convergence; more repeated draws cannot repair a sparse pool.
6. Select one primary uncertainty model. Published asymmetric NASA errors preserve the available measurement information. Synthetic errors can be assigned from the empirical uncertainty distribution. Keep the existing matched 25%/8% lognormal prescription as an explicit sensitivity case. Do not silently equate catalogue precision limits with every planet's measurement uncertainty.
7. Keep disjoint posterior bins [0.01,50) and [50,10,000); display [0.01,10) diagnostically. Rename the high bin I>=50 and document the exclusive upper bound.
8. Prefer showing the volatile-fraction distributions directly. If retaining a discrepancy axis, use a clearly defined z or z^2 consistently with the likelihood. Obtain the posterior from a shared, explicitly declared likelihood; a count-based or predictive treatment incorporating measurement ambiguity needs methodological implementation, not just renamed plot labels.
9. Give each model/bin/stage a reproducible independent RNG stream, so plot order does not affect inference. Record configuration, seed identifiers, valid/rejected draw counts, generator/detector settings, catalogue/code hashes, and sample membership beside every figure.
10. Extend the detected-pool cache key: it currently omits box bounds, RV magnitude target, and chunk size. Changing these can load an incompatible cached pool; chunk size also changes the generated random sequence.

## Sensitivity tests

Existing: four mass-radius prescriptions in the paper 2x4 figure; full versus precision-cut NASA samples in the extra Bayesian script. The latter is not run by make_paper_figures.py.

Recommended: published versus matched uncertainty models; strict catalogue eligibility; optional radius > 1.4 selection; Kepler versus TESS detection; pool size / independent pool seeds; MC draw count / independent resampling seeds; equal survey-size forecasts; classification-curve endpoint handling; and sensitivity of posterior weights to the likelihood approximation. A radius-valley threshold would define a different classification and should be labeled as such.
