# Shared population comparison

The current paper comparison is implemented in `science/population_comparison.py`. It supersedes the procedures described in the historical `population_comparison_audit.md`.

## Data, cuts, and conditioning

- Source: the checked-in NASA PSCompPars snapshot, `data/exoplanet_csv/pscomppars_2026.csv`.
- Require measured mass provenance (Mass or Msini; reject Calc), transiting flag 1, mass and radius limit flags exactly 0, and finite positive two-sided mass/radius uncertainties. Require maximum absolute one-sided fractional errors <=25% for mass and <=8% for radius. Missing uncertainties are excluded, not imputed.
- Common parent box: 0.1 <= M <= 12 Earth masses; 0.5 <= R <= 2.2 Earth radii; 0.01 <= I <= 10,000 Earth fluxes.
- Final comparison box: M > 2.5 and R > 1.4, with upper bounds retained from the parent box.
- Insolation intervals are half-open: [0.01,10), [0.01,50), [50,10,000). The first is diagnostic. Only the two disjoint latter intervals enter combined model weights.
- Select observed membership once using catalogue values. Each synthetic survey contains exactly the number N observed in that zone, for both models. This conditions on N and does not test absolute occurrence rates, survey yield, or the predicted number of detections.
- The catalogue identities and full configuration are saved with every shared result. Counts on the current snapshot are 6/7 volatile below I=10, 20/22 below I=50, and 22/66 at I>=50.

## Measurement uncertainty and mock surveys

- Primary observed errors: independent asymmetric normal perturbations, with upper/lower published errors used according to the sign of a standard normal deviate. Values are floored at 0.001 in their respective Earth units.
- Keep observed sample membership fixed in these classification draws. The observed denominator is always N. This explicitly replaces the old changing-denominator procedure.
- Primary simulated errors: draw a joint four-component fractional-error vector from the observed planets in the same zone (upper/lower mass and radius errors), then apply independent asymmetric perturbations to synthetic mass and radius. This uses the empirical distribution of measurement precision rather than treating the catalogue precision cut as every object's error. Pairing of the four error magnitudes is retained; correlated measurement deviations are not modeled.
- Select synthetic planets by joint transit+RV detection, insolation interval, and noisy mass/radius inside the final box. Sample N distinct eligible objects without replacement in each realization. Raise an error if a pool is too small; never silently discard failed draws.
- Nominal population membership (mixed population versus exclusion of true rocky super-Earths) is established before observational noise.
- Default generation: 4,000,000 requested planets, chunks of 2,000,000, Kepler transit selection, RV magnitude target 12, population seed 0.
- Default Monte Carlo: 5,000 realizations. Every model/bin/stage has a deterministic named RNG stream, independent of plot traversal order.
- Rocky: R <= the unshifted pure-silicate curve at M; volatile: R above it. Insolation errors and measurement covariances are not propagated. Synthetic pool generation occurs within the common parent box, so scattering from true values outside that parent box is not modeled.

## Shared likelihood

Let q_b(k) be the distribution of volatile counts produced by propagating the N observed planets' measurement errors in bin b. Estimate each planet's volatile-class probability from the measurement draws, then convolve its Bernoulli distribution with the others (a Poisson-binomial distribution). This preserves rare count combinations rather than truncating them to those appearing in a 5,000-draw histogram. In each model realization, estimate its selected volatile probability p from the noise-perturbed eligible pool. A Jeffreys Beta(V+1/2, T-V+1/2) draw represents finite-pool uncertainty and avoids hard p=0 or p=1 values.

The shared count score is

L_b = sum_k q_b(k) mean_j Binomial(k | N_b, p_bj).

Compute in log space. Multiply only the two disjoint bin scores; normalize across models under equal model priors. The I<10 score is reported separately. This is a measurement-marginalized binomial approximation: q is induced by the adopted local measurement-error model, rather than obtained from a full hierarchical fit to individual mass/radius measurements and selection. Model weights must be interpreted conditional on this approximation and the adopted populations.

The Otegi and four-relation plots show volatile-fraction distributions with the original smooth Gaussian display curves. The three-zone MC figure retains the original fourth-power discrepancy display, evaluated from the same shared draws. These display statistics and Gaussian curves do not determine model weights: the count likelihood and posterior are saved separately in the JSON outputs, without extra likelihood annotations on the figures. Observed uncertainty and finite model probability uncertainty both enter the count score; N-sized mock distributions also display finite-survey variation. The binomial approximation ignores the small finite-pool correction associated with sampling without replacement.

## Outputs and reproduction

- `python -m plotting.paper_figures.mc_comparison_statistic` generates the three-zone comparison and posterior JSON.
- `python -m plotting.paper_figures.flat_rocky_mr_vs_nasa` generates the Otegi panel and four-relation sensitivity figure using the same engine and 5,000 realizations. Otegi's primary density panel reads precisely the same saved draws as the three-zone figure; each alternate mass-radius population has its own saved draws but identical data/error/survey-size rules.
- `python -m plotting.paper_figures.calc_mc_comparison_likelihood` prints that same shared summary. Its optional arguments select a different explicitly recorded configuration, not a separate likelihood.
- `python -m plotting.paper_figures.check_population_comparison` checks 10,000 realizations, independent measurement and pool seeds, an 8M generated pool, and the matched 25%/8% lognormal-error sensitivity.
- `make_paper_figures.py` generates the primary figure suite. The numerical convergence/sensitivity checks are a separate command.
- Cached arrays and metadata live in `results/figures/analysis/population_comparison/`. Keys cover configuration, bins, catalogue/curve/science source hashes, and custom pool content. Detected-pool caches additionally include the box, RV target, and chunk size.
- `results/paper/population_comparison_checks.json` records numerical-check results, including deviations in fraction means and model weights. The 0.02 primary-bin mean-fraction tolerance is an explicit numerical diagnostic, not a scientific acceptance criterion or proof of likelihood-tail convergence. Inspect log likelihood changes as well.

The detector maps and recovery calibration have separate sampling requirements and are not N-sized population comparisons. Legacy scripts under `plotting/extra_plots/` retain exploratory methods and are not sources for the standardized paper likelihood; their old odds must not be mixed with the new paper outputs.

## Validation on 20 October 2026

The catalogue now includes the approved LHS 1478 b and TOI-2427 b updates. TOI-2427 b adds one nominal rocky planet to the higher-insolation sample (N=66). The convergence checks below use the updated snapshot.

Eleven unit tests cover strict boundaries, fixed observed membership, exact N denominators, independent RNG streams, count-likelihood identities, rare count combinations, and cache separation/reuse for scenario names containing periods. Run `python -m unittest discover -s tests -v`.

With the 4M primary pool, doubling to 10,000 draws changes primary-bin mean fractions by at most 0.00090. Independent resampling changes primary-bin mean fractions by at most 0.00184; an independent generated pool changes them by at most 0.00518; doubling the pool to 8M changes them by at most 0.00548. The favored combined model is unchanged. The seven-object I<10 diagnostic is less stable across generated pools and is excluded from the combined score. Full numeric values and the separate error-model sensitivity are saved in the checks JSON.
