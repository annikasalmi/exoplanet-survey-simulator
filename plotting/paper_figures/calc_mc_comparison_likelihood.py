"""Print the exact shared likelihood used by the paper plots; never combine nested bins."""
import argparse
import json
from science.population_comparison import ComparisonConfig, run_comparison


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--repeats', type=int, default=ComparisonConfig().draws)
    parser.add_argument('--seed', type=int, default=ComparisonConfig().seed)
    parser.add_argument('--pool-size', type=int, default=ComparisonConfig().pool_size)
    parser.add_argument('--chunk-size', type=int, default=ComparisonConfig().chunk_size)
    args = parser.parse_args()
    summary, _ = run_comparison(ComparisonConfig(draws=args.repeats, seed=args.seed,
                                                pool_size=args.pool_size, chunk_size=args.chunk_size))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
