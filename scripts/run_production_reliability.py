"""Command-line entry point for resumable production reliability."""

from __future__ import annotations

import argparse

from production_reliability import (
    DEFAULT_OBSERVED_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_CACHE,
    run_league,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', required=True, choices=('mlv', 'lovb', 'au'))
    parser.add_argument('--n-boot', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--observed-root', default=str(DEFAULT_OBSERVED_ROOT))
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--source-cache', default=str(DEFAULT_SOURCE_CACHE))
    args = parser.parse_args()
    if args.n_boot < 2:
        parser.error('--n-boot must be at least 2 for bootstrap variance')
    run_league(
        args.league,
        n_boot=args.n_boot,
        seed=args.seed,
        observed_root=args.observed_root,
        output_root=args.output_root,
        source_cache=args.source_cache,
    )


if __name__ == '__main__':
    main()
