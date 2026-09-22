"""Stage PFF API reports without overwriting the model's legacy exports.

Examples:
    python -m scripts.sync_pff_api --seasons 2025
    python -m scripts.sync_pff_api --seasons 2020-2025 --team-stats
    python -m scripts.sync_pff_api --seasons 2025 --validate-legacy
    python -m scripts.sync_pff_api --seasons 2021-2025 --team-stats-weekly

Set ``PFF_API_KEY`` in the environment.  Files are written beneath
``PFF_API_DIR`` (``source-data/pff_api`` by default); use ``PFF_DIR`` to point a
model run at the staged files only after validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from config import PFF_API_DIR, PFF_DIR
from src.data.pff_api import (FACET_REPORTS, POSITION_REPORTS,
                              TEAM_STAT_CATEGORIES, PffApiError, PffClient,
                              _atomic_csv, _snake_case)


# --- Pregame, season-to-date team tables -------------------------------------
# PFF's NCAA calendar has week ids 0-16 (regular season), 17 (conference
# championships), 18-20 (bowls, playoff, title game).  Matching every PFF game
# for 2021-2025 against CFBD games.csv showed PFF week N == CFBD week N for N >= 1,
# and PFF week 0 == the early games CFBD files under week 1.  So the table a
# model may use before CFBD week W is PFF weeks 0..W-1.
#
# The exception is the conference championships.  PFF always files them as week
# 17, but CFBD files them in regular week 14 for 2021-2023 and regular week 15
# from 2024 (the 12-team-playoff calendar).  A game
# PFF calls week 17 has therefore already been played before CFBD week
# CONF_CHAMP_CFBD_WEEK + 1 kicks off.
CONF_CHAMP_PFF_WEEK = 17
CONF_CHAMP_CFBD_WEEK = {2021: 14, 2022: 14, 2023: 14, 2024: 15, 2025: 15}
DEFAULT_CONF_CHAMP_CFBD_WEEK = 15

# If the budget does not stretch to every table, these land first (then the
# rest), newest season first.
WEEKLY_PRIORITY = ("offense-passing", "offense-rushing",
                   "defense-opponent-tendencies", "defense-passing",
                   "defense-rushing", "offense-overall-success",
                   "defense-overall-success")


def pregame_week_ids(season: int, through_week: int) -> list[int]:
    """PFF week ids covering CFBD weeks 1..through_week of ``season``."""
    weeks = list(range(0, int(through_week) + 1))
    ccg = CONF_CHAMP_CFBD_WEEK.get(int(season), DEFAULT_CONF_CHAMP_CFBD_WEEK)
    if through_week >= ccg:
        weeks.append(CONF_CHAMP_PFF_WEEK)
    return weeks


def weekly_team_stats_path(output_dir: Path, category: str, season: int,
                           through_week: int) -> Path:
    return (Path(output_dir) / "team_stats_weekly" /
            f"{category.replace('-', '_')}_{int(season)}_thru_w{int(through_week):02d}.csv")


def _throttle(client: PffClient, floor: float = 0.2, pause: float = 0.6) -> None:
    """Pause between reads; wait out the window when under ``floor`` of the budget."""
    rl = client.rate_limit
    if rl.limit and rl.remaining is not None and rl.remaining < floor * rl.limit:
        wait = (rl.reset - time.time() + 1.0) if rl.reset else 60.0
        wait = min(max(wait, 1.0), 120.0)
        print(f"  budget {rl.remaining}/{rl.limit}: waiting {wait:.0f}s for reset",
              flush=True)
        time.sleep(wait)
    elif pause:
        time.sleep(pause)


def sync_team_stats_weekly(client: PffClient, seasons, output_dir: Path,
                           categories=WEEKLY_PRIORITY, first_week: int = 2,
                           last_week: int = 15, force: bool = False,
                           pause: float = 0.6) -> dict:
    """Stage one pregame table per (category, season, week W in first..last).

    The file ``..._thru_wNN.csv`` holds CFBD weeks 1..NN (PFF weeks 0..NN, plus
    the conference championships once they have been played) and is the table a
    model may use before week NN+1.  Existing files are skipped, so a run that
    stops early resumes where it left off.
    """
    written, skipped, failed = [], [], []
    for category in categories:
        if category not in TEAM_STAT_CATEGORIES:
            raise ValueError(f"unknown PFF team-stat category: {category}")
        for season in sorted({int(s) for s in seasons}, reverse=True):
            for week in range(int(first_week), int(last_week) + 1):
                through = week - 1
                path = weekly_team_stats_path(output_dir, category, season, through)
                if path.exists() and not force:
                    skipped.append(path)
                    continue
                week_ids = pregame_week_ids(season, through)
                params = {"season": season, "category": category,
                          "weekIds": ",".join(str(w) for w in week_ids)}
                try:
                    payload = client._request("/v2/ncaa/teams/stats", params)
                except PffApiError as exc:
                    failed.append((path, str(exc)))
                    print(f"  FAILED {path.name}: {exc}", flush=True)
                    _throttle(client, pause=pause)
                    continue
                echoed = payload.get("weekIds")
                if echoed is not None:
                    echoed_ids = ([int(w) for w in str(echoed).split(",")]
                                  if not isinstance(echoed, list)
                                  else [int(w) for w in echoed])
                    if sorted(echoed_ids) != sorted(week_ids):
                        failed.append((path, f"weekIds echoed as {echoed}"))
                        _throttle(client, pause=pause)
                        continue
                frame = pd.DataFrame(payload.get("rows", [])).rename(columns=_snake_case)
                if frame.empty or "team_id" not in frame:
                    failed.append((path, "no rows"))
                    print(f"  EMPTY {path.name}", flush=True)
                    _throttle(client, pause=pause)
                    continue
                _atomic_csv(frame, path)
                written.append(path)
                rl = client.rate_limit
                print(f"  {path.name}: {len(frame)} rows "
                      f"(weeks {params['weekIds']}; budget {rl.remaining}/{rl.limit})",
                      flush=True)
                _throttle(client, pause=pause)
    return {"written": written, "skipped": skipped, "failed": failed}


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
    parser.add_argument("--team-stats-weekly", action="store_true",
                        help="ONLY stage pregame season-to-date team tables "
                             "(team_stats_weekly/*_thru_wNN.csv); skips existing files")
    parser.add_argument("--weekly-categories", default=",".join(WEEKLY_PRIORITY),
                        help="comma list for --team-stats-weekly, in priority order")
    parser.add_argument("--first-week", type=int, default=2,
                        help="first pregame week for --team-stats-weekly (default 2)")
    parser.add_argument("--last-week", type=int, default=15,
                        help="last pregame week for --team-stats-weekly (default 15)")
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

    if args.team_stats_weekly:
        result = sync_team_stats_weekly(
            client, args.seasons, args.output_dir,
            categories=[c.strip() for c in args.weekly_categories.split(",") if c.strip()],
            first_week=args.first_week, last_week=args.last_week, force=args.force)
        print(f"weekly team tables: {len(result['written'])} written, "
              f"{len(result['skipped'])} already present, "
              f"{len(result['failed'])} failed")
        for path, reason in result["failed"]:
            print(f"  missing {path.name}: {reason}")
        rl = client.rate_limit
        if rl.remaining is not None:
            print(f"PFF read budget remaining: {rl.remaining}/{rl.limit}")
        return

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
