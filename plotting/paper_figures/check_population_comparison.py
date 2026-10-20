"""Run and save numerical convergence checks plus the matched-error sensitivity."""
from dataclasses import replace
import json
from science.population_comparison import ComparisonConfig, run_comparison
from tools.paths import PAPER_FIGURES_DIR


def main():
    base = ComparisonConfig()
    configs = {
        'baseline': base,
        'double_draws': replace(base, draws=10000),
        'independent_resampling_seed': replace(base, seed=1),
        'independent_pool_seed': replace(base, pool_seed=10),
        'double_pool': replace(base, pool_size=base.pool_size*2),
        'matched_error_sensitivity': replace(base, error_model='matched_lognormal'),
    }
    results = {}
    for name, config in configs.items():
        print(f'Running {name}', flush=True)
        summary, _ = run_comparison(config)
        results[name] = {'config': summary['provenance']['config'], 'posterior':summary['posterior'],
                         'total_log_likelihood':summary['total_log_likelihood'], 'bins':summary['bins']}
    baseline = results['baseline']
    diagnostics = {}
    for name, item in results.items():
        if name in ('baseline', 'matched_error_sensitivity'):
            continue
        mean_deltas = [abs(item['bins'][b]['models'][m]['mean_fraction']-baseline['bins'][b]['models'][m]['mean_fraction'])
                       for b in baseline['bins'] for m in baseline['posterior']]
        max_delta = max(mean_deltas)
        primary_delta = max(abs(item['bins'][b]['models'][m]['mean_fraction']-baseline['bins'][b]['models'][m]['mean_fraction']) for b in ('I < 50','I ≥ 50') for m in baseline['posterior'])
        winner_stable = max(item['posterior'],key=item['posterior'].get) == max(baseline['posterior'],key=baseline['posterior'].get)
        diagnostics[name] = {'maximum_fraction_mean_change': max_delta,
                             'winning_model_unchanged': winner_stable,
                             'diagnostic_I_lt_10_fraction_change': max(abs(item['bins']['I < 10']['models'][m]['mean_fraction']-baseline['bins']['I < 10']['models'][m]['mean_fraction']) for m in baseline['posterior']),
                             'maximum_primary_fraction_mean_change': primary_delta,
                             'primary_within_0.02_fraction_tolerance': primary_delta <= .02}
    report = {'description':'Numerical checks, not additional independent data. Tolerance: primary-bin absolute fraction mean changes <=0.02; I<10 is diagnostic only; also inspect model log likelihood variation. Matched errors are a methodological sensitivity, not a convergence test.',
              'checks':diagnostics, 'runs':results}
    out = PAPER_FIGURES_DIR / 'population_comparison_checks.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(diagnostics,indent=2)); print(f'Saved {out}')


if __name__ == '__main__':
    main()
