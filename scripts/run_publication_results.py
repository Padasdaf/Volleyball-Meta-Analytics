"""Generate Phase 9 publication figures, tables, and results memo."""

from __future__ import annotations

import argparse

from publication_results import generate_publication_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render publication outputs from frozen Phase 6-8 artifacts"
    )
    parser.add_argument("--output-root", default="generated/publication")
    parser.add_argument("--memo", default="PUBLICATION_RESULTS.md")
    parser.add_argument(
        "--export-tracked", action="store_true",
        help="Export the reviewed durable bundle to figures/, tables/, and results/",
    )
    parser.add_argument("--tracked-root", default=".")
    args = parser.parse_args()
    result = generate_publication_results(
        output_root=args.output_root, memo_path=args.memo,
        tracked_root=(args.tracked_root if args.export_tracked else None),
    )
    for row in result["validation"].itertuples(index=False):
        print(f"{row.league}: {row.observed_player_seasons} player-seasons, {row.metrics} metrics")


if __name__ == "__main__":
    main()
