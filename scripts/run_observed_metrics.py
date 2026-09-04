"""Command-line entry point for resumable observed metric production."""

from __future__ import annotations

import argparse

from metric_registry import METRIC_METADATA, metrics_for_league
from observed_metrics import DEFAULT_OUTPUT_ROOT, run_observed_metrics, run_scope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', choices=('mlv', 'lovb', 'au'))
    parser.add_argument('--season', type=int)
    parser.add_argument('--family', action='append', choices=(
        'conventional', 'advanced', 'evollve',
    ))
    parser.add_argument('--metric', action='append')
    parser.add_argument('--all-supported', action='store_true')
    parser.add_argument('--all-scope', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.all_scope:
        if args.league or args.season or args.family or args.metric:
            raise SystemExit('--all-scope cannot be combined with selectors')
        results = run_scope(
            output_root=args.output_root,
            force=args.force,
        )
    else:
        if args.league is None or args.season is None:
            raise SystemExit('--league and --season are required')
        supported = metrics_for_league(args.league)
        selected = []
        if args.all_supported or (not args.family and not args.metric):
            selected.extend(supported)
        for family in args.family or ():
            selected.extend(
                metric for metric in supported
                if METRIC_METADATA[metric].family == family
            )
        selected.extend(args.metric or ())
        results = [run_observed_metrics(
            args.league, args.season, metrics=tuple(dict.fromkeys(selected)),
            output_root=args.output_root,
            force=args.force,
        )]
    for result in results:
        print(
            f'{result.league} {result.season}: '
            f'{len(result.completed_metrics)} completed, '
            f'{len(result.pending_metrics)} pending -> '
            f'{result.output_directory}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
