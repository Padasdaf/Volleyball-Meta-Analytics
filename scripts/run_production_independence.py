"""Run registry-driven production independence without publication figures."""

from __future__ import annotations

import argparse

from production_independence import (
    DEFAULT_OBSERVED_ROOT,
    DEFAULT_OUTPUT_ROOT,
    run_league_independence,
    run_scope,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', choices=('mlv', 'lovb', 'au'))
    parser.add_argument('--all-leagues', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--observed-root', default=str(DEFAULT_OBSERVED_ROOT))
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    if args.all_leagues == (args.league is not None):
        parser.error('choose exactly one of --league or --all-leagues')
    if args.all_leagues:
        results = run_scope(
            observed_root=args.observed_root,
            output_root=args.output_root,
            prepare_only=args.prepare_only,
        )
    else:
        results = [run_league_independence(
            args.league,
            observed_root=args.observed_root,
            output_root=args.output_root,
            prepare_only=args.prepare_only,
        )]
    for result in results:
        print(
            f'{result.league}: {result.player_seasons} player-seasons, '
            f'{result.metric_count} metrics -> {result.output_directory}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
