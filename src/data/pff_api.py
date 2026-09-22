"""Small, testable client for the PFF Developer API.

The model historically consumed manually downloaded Premium Stats CSV files.
PFF's API exposes the same five league-wide reports under ``/v1`` and richer
team/position tables under ``/v2``.  This module keeps authentication, retries,
schema checks and atomic writes in one place so API refreshes cannot silently
change the production model.

Credentials are read only from ``PFF_API_KEY`` and are never written or logged.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Iterable

import pandas as pd

# requests uses certifi by default, which cannot see enterprise/local CAs held in
# the Windows certificate store.  Use native trust without weakening verification.
if os.name in {"nt", "posix"}:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:  # tests/minimal installs still get requests' normal CA path
        pass
import requests


BASE_URL = "https://api.pff.com"

# File names deliberately match the legacy exports consumed throughout the WAR
# build.  Response keys are the envelopes returned by the v1 endpoints.
FACET_REPORTS = {
    "blocking": ("/v1/facet/offense/blocking", "blocking_summary"),
    "defense": ("/v1/facet/defense/summary", "defense_summary"),
    "passing": ("/v1/facet/passing/summary", "passing_summary"),
    "receiving": ("/v1/facet/receiving/summary", "receiving_summary"),
    "rushing": ("/v1/facet/rushing/summary", "rushing_summary"),
}

TEAM_STAT_CATEGORIES = (
    "offense-overall-success",
    "offense-passing",
    "offense-rushing",
    "defense-overall-success",
    "defense-passing",
    "defense-rushing",
    "defense-opponent-tendencies",
)

POSITION_REPORTS = (
    "offense", "passing", "passing-depth", "passing-pressure",
    "receiving", "receiving-depth", "rushing", "blocking",
    "pass-blocking", "run-blocking", "defense", "run-defense",
    "pass-rush", "coverage", "special-teams", "kick-returns",
    "field-goals", "punting", "kickoffs",
)

REQUIRED_COLUMNS = {
    "blocking": {"player_id", "player", "position", "team_name",
                 "player_game_count", "grades_offense", "snap_counts_offense"},
    "defense": {"player_id", "player", "position", "team_name",
                "player_game_count", "grades_defense", "snap_counts_defense"},
    "passing": {"player_id", "player", "position", "team_name",
                "player_game_count", "grades_offense", "passing_snaps"},
    "receiving": {"player_id", "player", "position", "team_name",
                  "player_game_count", "grades_offense", "routes"},
    "rushing": {"player_id", "player", "position", "team_name",
                "player_game_count", "grades_offense", "attempts"},
}


class PffApiError(RuntimeError):
    """A stable, credential-safe representation of a PFF API failure."""


@dataclass(frozen=True)
class RateLimit:
    limit: int | None
    remaining: int | None
    reset: int | None


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snake_case(name: str) -> str:
    """Convert v2 camelCase keys to the v1/legacy snake_case convention."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower().replace("-", "_")


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv",
            prefix=f".{destination.stem}-", dir=destination.parent,
            delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class PffClient:
    def __init__(self, api_key: str | None = None,
                 session: requests.Session | None = None,
                 base_url: str = BASE_URL, timeout: float = 120.0,
                 max_retries: int = 3):
        self.api_key = api_key or os.environ.get("PFF_API_KEY")
        if not self.api_key:
            raise PffApiError(
                "PFF_API_KEY is not set. Put the credential in the environment; "
                "do not pass it on the command line or commit it to .env.")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = RateLimit(None, None, None)
        self._team_directory_cache: dict[int, pd.DataFrame] = {}

    def _request(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Accept": "application/json"}
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, headers=headers, params=params,
                                            timeout=self.timeout)
            except requests.RequestException as exc:
                raise PffApiError(f"PFF request failed: {type(exc).__name__}") from exc

            self.rate_limit = RateLimit(
                _as_int(response.headers.get("x-ratelimit-limit")),
                _as_int(response.headers.get("x-ratelimit-remaining")),
                _as_int(response.headers.get("x-ratelimit-reset")),
            )
            if response.status_code in {429, 503} and attempt < self.max_retries:
                retry_after = _as_int(response.headers.get("Retry-After"))
                delay = retry_after if retry_after is not None else 2 ** attempt
                time.sleep(max(0.0, float(delay)) + random.random() * 0.25)
                continue
            if response.ok:
                try:
                    return response.json()
                except ValueError as exc:
                    raise PffApiError("PFF returned a non-JSON success response") from exc

            request_id = response.headers.get("x-request-id")
            code = None
            message = None
            try:
                error = response.json().get("error", {})
                code, message = error.get("code"), error.get("message")
                request_id = error.get("request_id") or request_id
            except ValueError:
                pass
            detail = code or f"HTTP {response.status_code}"
            if message:
                # Do not trust an upstream error body not to echo request data.
                detail += f": {str(message).replace(self.api_key, '[redacted]')}"
            if request_id:
                detail += f" (request {request_id})"
            raise PffApiError(detail)
        raise AssertionError("unreachable")

    def whoami(self) -> dict:
        return self._request("/v1/auth/whoami")

    def facet_report(self, report: str, season: int, division: str = "fbs",
                     week: int | str | None = None,
                     game_id: int | None = None) -> pd.DataFrame:
        if report not in FACET_REPORTS:
            raise ValueError(f"unknown PFF facet report: {report}")
        path, envelope = FACET_REPORTS[report]
        if game_id is not None:
            params = {"game_id": int(game_id)}
        else:
            params = {"league": "ncaa", "season": int(season),
                      "division": division}
            if week is not None:
                params["week"] = str(week)
        payload = self._request(path, params)
        frame = pd.DataFrame(payload.get(envelope, []))
        missing = REQUIRED_COLUMNS[report] - set(frame.columns)
        if missing:
            raise PffApiError(
                f"{report} {season} schema is missing required columns: "
                f"{', '.join(sorted(missing))}")
        return frame

    def team_stats(self, season: int, category: str,
                   week_group: str = "REGPO",
                   week_ids: Iterable[int] | None = None) -> pd.DataFrame:
        if category not in TEAM_STAT_CATEGORIES:
            raise ValueError(f"unknown PFF team-stat category: {category}")
        params: dict[str, object] = {"season": int(season), "category": category}
        if week_ids is None:
            params["weekGroup"] = week_group
        else:
            params["weekIds"] = ",".join(str(int(w)) for w in week_ids)
        payload = self._request("/v2/ncaa/teams/stats", params)
        rows = payload.get("rows", [])
        frame = pd.DataFrame(rows).rename(columns=_snake_case)
        if "team_id" not in frame:
            raise PffApiError(
                f"{category} {season} schema is missing required column: team_id")
        return frame

    def position_report(self, season: int, report: str,
                        week_group: str = "REGPO", week: int | None = None,
                        week_to: int | None = None,
                        division: str | None = None) -> pd.DataFrame:
        if report not in POSITION_REPORTS:
            raise ValueError(f"unknown PFF position report: {report}")
        params: dict[str, object] = {"season": int(season),
                                    "weekGroup": week_group}
        if week is not None:
            params["week"] = int(week)
        if week_to is not None:
            params["weekTo"] = int(week_to)
        payload = self._request(f"/v2/ncaa/positions/reports/{report}", params)
        frame = pd.DataFrame(payload.get("rows", [])).rename(columns=_snake_case)
        if "games_played" in frame and "player_game_count" not in frame:
            frame = frame.rename(columns={"games_played": "player_game_count"})
        if division is not None:
            if division != "fbs":
                raise ValueError("local position-report filtering currently supports fbs only")
            directory = self.team_directory(season)
            groups = directory.group_ids.fillna("").astype(str).str.split(";")
            fbs = directory.loc[groups.map(lambda values: "11" in values)].copy()
            # PFF's directory city is nearly the CFBD canonical name.  Reuse the
            # explicit aliases maintained by the model's team-table loader.
            from src.data.pff_team import TEAM_ALIASES
            fbs["canonical_team"] = fbs.city.replace(TEAM_ALIASES)
            id_to_team = fbs.drop_duplicates("franchise_id").set_index(
                "franchise_id").canonical_team
            frame = frame[frame.franchise_id.isin(id_to_team.index)].copy()
            frame["team_name"] = frame.franchise_id.map(id_to_team)
        return frame

    def team_directory(self, season: int) -> pd.DataFrame:
        season = int(season)
        if season in self._team_directory_cache:
            return self._team_directory_cache[season].copy()
        payload = self._request("/v2/ncaa/teams", {"season": int(season)})
        frame = pd.DataFrame(payload.get("rows", [])).rename(columns=_snake_case)
        missing = {"franchise_id", "city", "group_ids"} - set(frame.columns)
        if missing:
            raise PffApiError(
                f"team directory {season} schema is missing required columns: "
                f"{', '.join(sorted(missing))}")
        self._team_directory_cache[season] = frame
        return frame.copy()

    def sync_facet_report(self, report: str, season: int, output_dir: Path,
                          division: str = "fbs", force: bool = False) -> Path:
        destination = Path(output_dir) / f"{report}_{int(season)}.csv"
        if destination.exists() and not force:
            raise FileExistsError(
                f"{destination} exists; pass force=True to replace it atomically")
        frame = self.facet_report(report, season, division=division)
        _atomic_csv(frame, destination)
        return destination

    def sync_team_stats(self, season: int, category: str, output_dir: Path,
                        week_group: str = "REGPO", force: bool = False) -> Path:
        safe_category = category.replace("-", "_")
        destination = (Path(output_dir) / "team_stats" /
                       f"{safe_category}_{int(season)}.csv")
        if destination.exists() and not force:
            raise FileExistsError(
                f"{destination} exists; pass force=True to replace it atomically")
        frame = self.team_stats(season, category, week_group=week_group)
        _atomic_csv(frame, destination)
        return destination

    def sync_team_directory(self, season: int, output_dir: Path,
                            force: bool = False) -> Path:
        destination = (Path(output_dir) / "team_stats" /
                       f"team_directory_{int(season)}.csv")
        if destination.exists() and not force:
            raise FileExistsError(
                f"{destination} exists; pass force=True to replace it atomically")
        _atomic_csv(self.team_directory(season), destination)
        return destination

    def sync_position_report(self, season: int, report: str, output_dir: Path,
                             division: str = "fbs", force: bool = False) -> Path:
        destination = (Path(output_dir) / "position_reports" /
                       f"{report.replace('-', '_')}_{int(season)}.csv")
        if destination.exists() and not force:
            raise FileExistsError(
                f"{destination} exists; pass force=True to replace it atomically")
        frame = self.position_report(season, report, division=division)
        required = {"player_id", "player", "position", "franchise_id", "team_name"}
        missing = required - set(frame.columns)
        if missing:
            raise PffApiError(
                f"{report} {season} schema is missing required columns: "
                f"{', '.join(sorted(missing))}")
        _atomic_csv(frame, destination)
        return destination
