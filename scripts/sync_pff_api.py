"""Stage PFF API reports without overwriting the model's legacy exports.

Examples:
    python -m scripts.sync_pff_api --seasons 2025
    python -m scripts.sync_pff_api --seasons 2020-2025 --team-stats
    python -m scripts.sync_pff_api --seasons 2025 --validate-legacy

Set ``PFF_API_KEY`` in the environment.  Files are written beneath
``PFF_API_DIR`` (``source-data/pff_api`` by default); use ``PFF_DIR`` to point a
model run at the staged files only after validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import PFF_API_DIR, PFF_DIR
from src.data.pff_api import (FACET_REPORTS, POSITION_REPORTS,
                              TEAM_STAT_CATEGORIES, PffClient)


def _years(value: str) -> list[int]:
    years: set[int] = set()
    for piece in value.split(","):
        piece = piece.strip()
        if "-" in piece:
            first, last = (int(x) for x in piece.split("-", 1))
            years.update(range(min(first, last), max(first, last) + 1))
        else:
            years.add(int(piece))
    if not years:
        raise argparse.ArgumentTypeError("provide at least one season")
    return sorted(years)


def _reports(value: str) -> list[str]:
    values = sorted(FACET_REPORTS) if value == "all" else value.split(",")
    unknown = set(values) - set(FACET_REPORTS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown reports: {', '.join(sorted(unknown))}")
    return values


def _position_reports(value: str) -> list[str]:
    values = list(POSITION_REPORTS) if value == "all" else value.split(",")
    unknown = set(values) - set(POSITION_REPORTS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown position reports: {', '.join(sorted(unknown))}")
    return values


def _legacy_comparison(new_path: Path, legacy_dir: Path) -> str:
    old_path = legacy_dir / new_path.name
    if not old_path.exists():
        return "no legacy file"
    new = pd.read_csv(new_path, low_memory=False)
    old = pd.read_csv(old_path, low_memory=False)
    common_ids = 0
    if "player_id" in new and "player_id" in old:
        common_ids = len(set(new.player_id.dropna().astype(str)) &
                         set(old.player_id.dropna().astype(str)))
    missing_columns = sorted(set(old.columns) - set(new.columns))
    return (f"legacy rows={len(old)}, shared player ids={common_ids}, "
            f"missing legacy columns={missing_columns or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=_years, required=True,
                        help="comma/range list, e.g. 2024-2025")
    parser.add_argument("--reports", type=_reports, default=sorted(FACET_REPORTS),
                        help="comma list or 'all' (default: all)")
    parser.add_argument("--output-dir", type=Path, default=PFF_API_DIR)
    parser.add_argument("--division", default="fbs", choices=("fbs", "fcs", "lower"))
    parser.add_argument("--team-stats", action="store_true",
                        help="also stage all seven v2 team-stat tables")
    parser.add_argument("--team-stats-only", action="store_true",
                        help="stage team tables without the five player facets")
    parser.add_argument("--position-reports", type=_position_reports,
                        help="comma list or 'all' of the 19 v2 position reports")
    parser.add_argument("--position-reports-only", action="store_true",
                        help="skip legacy facets (requires --position-reports)")
    parser.add_argument("--validate-legacy", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="atomically replace existing staged files")
    args = parser.parse_args()
    if args.position_reports_only and not args.position_reports:
        parser.error("--position-reports-only requires --position-reports")

    client = PffClient()
    identity = client.whoami()
    if not identity.get("entitled"):
        raise SystemExit("PFF credential is valid but is not entitled to data")
    print(f"PFF authenticated: tier={identity.get('tier')}")

    for season in args.seasons:
        if not args.team_stats_only and not args.position_reports_only:
            for report in args.reports:
                path = client.sync_facet_report(
                    report, season, args.output_dir, division=args.division,
                    force=args.force)
                rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
                note = (_legacy_comparison(path, PFF_DIR)
                        if args.validate_legacy else "")
                print(f"{path.name}: {rows} rows" + (f"; {note}" if note else ""))
        if args.team_stats or args.team_stats_only:
            directory = client.sync_team_directory(
                season, args.output_dir, force=args.force)
            rows = sum(1 for _ in directory.open(encoding="utf-8")) - 1
            print(f"team_stats/{directory.name}: {rows} rows")
            for category in TEAM_STAT_CATEGORIES:
                path = client.sync_team_stats(
                    season, category, args.output_dir, force=args.force)
                rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
                print(f"team_stats/{path.name}: {rows} rows")
        if args.position_reports:
            for report in args.position_reports:
                path = client.sync_position_report(
                    season, report, args.output_dir, division=args.division,
                    force=args.force)
                rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
                print(f"position_reports/{path.name}: {rows} rows")

    rl = client.rate_limit
    if rl.remaining is not None:
        print(f"PFF read budget remaining: {rl.remaining}/{rl.limit}")


if __name__ == "__main__":
    main()
