"""Capture timestamped, append-only 2026 multi-book line history.

CFBD exposes provider/open/current prices but does not timestamp each quote.  The only
timestamp asserted here is ours: when this process successfully retrieved the payload.
Every successful check is recorded; quote events are appended only when a provider's
price changes.  A line becomes a qualified "close" only when a successful check was
captured within six hours before kickoff.

The script is stdlib-only so GitHub Actions can run it without installing the model.
It also publishes the latest board and a forward-validation payload for the website.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YEAR = 2026
BASE = "https://api.collegefootballdata.com"
LEDGER_DIR = ROOT / "data" / "market_snapshots"
QUOTES = LEDGER_DIR / f"lines_{YEAR}.jsonl"
CHECKS = LEDGER_DIR / f"checks_{YEAR}.jsonl"
ENTRIES = LEDGER_DIR / f"watchlist_{YEAR}.jsonl"
SETTLEMENTS = LEDGER_DIR / f"settlements_{YEAR}.jsonl"
STATUS = LEDGER_DIR / "status.json"
ODDS = ROOT / "viz" / "data" / "odds.json"
TRACKING = ROOT / "viz" / "data" / "market_tracking.json"
SCHEDULE = ROOT / "viz" / "data" / "schedule.json"
MODEL = ROOT / "viz" / "data" / "model_v4.json"
RATINGS = ROOT / "viz" / "data" / "ratings.json"
TEAMS = ROOT / "viz" / "data" / "teams.json"
PLAYOFF = ROOT / "viz" / "data" / "playoff.json"
HISTORICAL = ROOT / "audit" / "book_shopping_backtest.json"
AVAILABILITY = ROOT / "war_model" / "availability_events_2026.csv"

PROVIDER_ALIAS = {"Draft Kings": "DraftKings"}
QUOTE_FIELDS = ("spread", "spreadOpen", "overUnder", "overUnderOpen",
                "homeMoneyline", "awayMoneyline")

# These lines were not available to the model as a ready, forward-looking Week 0
# board. Both games also involved a first-year FBS team with only the newcomer
# fallback prior, so they are retained as results but excluded from every bet output.
BET_EXCLUDED_GAME_IDS = {401864577, 401866408}


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _fetch_cfbd_once(key: str, url: str) -> list[dict]:
    if os.name == "nt":
        config = (f'url = "{url}"\nheader = "Authorization: Bearer {key}"\n'
                  'header = "Accept: application/json"\nsilent\nshow-error\n'
                  'max-time = 45\nwrite-out = "\\n%{http_code}"\n')
        proc = subprocess.run(["curl.exe", "--config", "-"], input=config, text=True,
                              encoding="utf-8", capture_output=True, timeout=55)
        if proc.returncode or "\n" not in proc.stdout:
            raise RuntimeError(f"CFBD line request failed: {proc.stderr.strip()}")
        body, status = proc.stdout.rsplit("\n", 1)
        if not status.startswith("2"):
            raise RuntimeError(f"CFBD line request returned HTTP {status}")
        return json.loads(body)
    request = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json",
        "User-Agent": "cfb-model-v3-market-capture/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def fetch_cfbd(key: str, endpoint: str, params: dict) -> list[dict]:
    """Fetch CFBD data, retrying only transient network and server failures."""
    url = f"{BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    attempts = 3
    for attempt in range(attempts):
        try:
            return _fetch_cfbd_once(key, url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                subprocess.TimeoutExpired, RuntimeError) as exc:
            status = getattr(exc, "code", None)
            retryable = status is None or 500 <= int(status) < 600
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("CFBD request exhausted retries")


def fetch_lines(key: str) -> list[dict]:
    return fetch_cfbd(key, "/lines", {"year": YEAR, "seasonType": "regular"})


def fetch_games(key: str) -> list[dict]:
    """Fetch the authoritative completed flag and final scores.

    The lines endpoint is intentionally about prices. Its score fields are not a
    reliable completion signal, so final-score publishing uses /games instead.
    """
    return fetch_cfbd(key, "/games", {
        "year": YEAR, "seasonType": "regular", "division": "fbs"})


def publish_finals(games: list[dict], schedule_path: Path = SCHEDULE) -> int:
    """Merge completed CFBD scores into the compact schedule used by the site."""
    if not schedule_path.exists():
        return 0
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    by_id = {int(g["id"]): g for g in games if g.get("id") is not None}
    changed = 0
    for row in schedule:
        game = by_id.get(int(row["id"])) if row.get("id") is not None else None
        if not game or not game.get("completed"):
            continue
        hp, ap = game.get("homePoints"), game.get("awayPoints")
        if hp is None or ap is None:
            continue
        final = {"f": 1, "hp": int(hp), "ap": int(ap)}
        if any(row.get(key) != value for key, value in final.items()):
            row.update(final)
            changed += 1
    if changed:
        schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    return changed


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _static_win_probability(model: dict, home: str, away: str,
                            is_home: float = 0.0) -> float:
    a, b = model["teams"][home], model["teams"][away]
    diff = [x - y for x, y in zip(a, b)]
    logistic, margin = model["logistic"], model["margin"]
    z = (logistic.get("intercept", 0.0)
         + sum(c * x for c, x in zip(logistic["coef"], diff))
         + is_home * logistic["hfa"])
    m = (margin.get("intercept", 0.0)
         + sum(c * x for c, x in zip(margin["coef"], diff))
         + is_home * margin["hfa"])
    p = (model["ens_w"] * _sigmoid(z)
         + (1.0 - model["ens_w"])
         * 0.5 * (1.0 + math.erf(m / margin["sigma"] / math.sqrt(2.0))))
    p = max(1e-8, min(1.0 - 1e-8, p))
    scale = model.get("probability_scale", 1.0)
    return _sigmoid(scale * math.log(p / (1.0 - p)))


def _power_snapshot(model: dict, names: list[str], dynamic_ratings: dict) -> list[dict]:
    """Exact compact-data port of src.dynamic.current_power_ratings."""
    blend = float(model.get("dynamic", {}).get("blend", 0.75))
    mean_dynamic = sum(dynamic_ratings[t] for t in names) / len(names)
    rows = []
    for team in names:
        opponents = [opp for opp in names if opp != team]
        static_power = sum(_static_win_probability(model, team, opp)
                           for opp in opponents) / len(opponents)
        dynamic_power = sum(_sigmoid(dynamic_ratings[team] - dynamic_ratings[opp])
                            for opp in opponents) / len(opponents)
        # The published frame is standardized, so the average opponent is its mean
        # vector. Build a temporary mean row to use the exact frozen model math.
        mean_name = "__rating_field_average__"
        mean_vec = [sum(model["teams"][t][i] for t in names) / len(names)
                    for i in range(len(model["features"]))]
        model["teams"][mean_name] = mean_vec
        static_average = _static_win_probability(model, team, mean_name)
        model["teams"].pop(mean_name, None)
        dynamic_average = _sigmoid(dynamic_ratings[team] - mean_dynamic)
        rows.append({
            "team": team,
            "power": (1.0 - blend) * static_power + blend * dynamic_power,
            "vs_average": ((1.0 - blend) * static_average
                           + blend * dynamic_average),
        })
    rows.sort(key=lambda row: (-row["power"], row["team"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["power"] = round(row["power"], 6)
        row["vs_average"] = round(row["vs_average"], 6)
    return rows


def _read_optional_json(path: Path | None, fallback):
    if path is None or not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], q: float) -> float | None:
    """Linear percentile matching pandas/numpy's default interpolation."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _schedule_strength(model: dict, names: list[str], schedule: list[dict]) -> dict:
    """Mean standardized opponent strength using the exporter's exact definition."""
    features = model.get("features") or []
    try:
        oi, di = features.index("O"), features.index("D")
    except ValueError:
        return {team: (None, 0) for team in names}
    raw = {team: (float(model["teams"][team][oi])
                  + float(model["teams"][team][di])) / 2.0 for team in names}
    mean = statistics.mean(raw.values())
    sd = statistics.stdev(raw.values()) if len(raw) > 1 else 0.0
    standardized = {team: ((value - mean) / sd if sd else 0.0)
                    for team, value in raw.items()}
    total, count = ({team: 0.0 for team in names}, {team: 0 for team in names})
    field = set(names)
    for game in schedule:
        home, away = game.get("h"), game.get("a")
        home_in, away_in = home in field, away in field
        if home_in and away_in:
            total[home] += standardized[away]; count[home] += 1
            total[away] += standardized[home]; count[away] += 1
        elif home_in or away_in:
            team = home if home_in else away
            total[team] += -2.0; count[team] += 1
    return {team: ((total[team] / count[team]) if count[team] else None,
                   count[team]) for team in names}


def _rating_shell(team: str, model: dict, old_rows: list[dict],
                  team_meta: dict, playoff: dict, schedule_strength: dict) -> dict:
    """Build a complete dashboard row for a model team absent from ratings.json."""
    features = model.get("features") or []
    vector = model["teams"][team]
    values = dict(zip(features, vector))
    po = playoff.get(team, {})
    sos, games = schedule_strength.get(team, (None, 0))
    talent = _percentile([row["talent"] for row in old_rows
                          if row.get("talent") is not None], 0.05)
    row = {
        "team": team,
        "conference": team_meta.get(team, {}).get("conference")
                      or po.get("conference") or "—",
        "O": round(float(values["O"]), 3) if values.get("O") is not None else None,
        "D": round(float(values["D"]), 3) if values.get("D") is not None else None,
        # Newcomers receive the same fifth-percentile frame fallback everywhere.
        # Raw talent is not in the compact model vector, so recover that percentile
        # from the already-published rows until the full exporter next runs.
        "talent": round(float(talent), 3) if talent is not None else None,
        "returning": (round(float(values["returning"]), 3)
                      if values.get("returning") is not None else None),
        "sos": round(float(sos), 3) if sos is not None else None,
        "games": games,
        "avg_wins": po.get("avg_wins"),
        "avg_losses": po.get("avg_losses"),
    }
    for key in ("conf_champ", "playoff", "bye", "sf", "final", "champ"):
        row[key] = po.get(key, 0.0)
    return row


def replay_published_results(schedule_path: Path = SCHEDULE, model_path: Path = MODEL,
                             ratings_path: Path = RATINGS,
                             teams_path: Path | None = TEAMS,
                             playoff_path: Path | None = PLAYOFF) -> int:
    """Replay every published final from the preseason baseline, grouped by week.

    Rebuilding from the baseline on every capture makes the operation idempotent and
    preserves the validated no-within-week-lookahead rule even while a week is only
    partially complete.
    """
    if not (schedule_path.exists() and model_path.exists() and ratings_path.exists()):
        return 0
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    dynamic = model.get("dynamic") or {}
    current = dynamic.get("ratings") or {}
    # The compact model is the authoritative FBS universe. This deliberately does
    # not inherit the old 136-row ratings truncation: 2026 newcomers have fallback
    # vectors and are fully rated members of the schedule and simulation.
    names = [team for team in model.get("teams", {}) if team in current]
    if len(names) < 2:
        return 0

    baseline = dict(dynamic.get("preseason_ratings") or current)
    state = dict(baseline)
    finals = [g for g in schedule if g.get("f") and g.get("hp") is not None
              and g.get("ap") is not None and g.get("h") in names and g.get("a") in names]
    by_week: dict[int, list[dict]] = {}
    for game in finals:
        by_week.setdefault(int(game.get("w") or 0), []).append(game)

    preseason = _power_snapshot(model, names, state)
    history = [{"week": 0, "label": "Preseason", "completed_games": 0,
                "teams": preseason}]
    completed = 0
    k = float(dynamic.get("k", 0.20))
    margin_sigma = float(model["margin"]["sigma"])
    for week in sorted(by_week):
        changes: dict[str, float] = {}
        for game in by_week[week]:
            home, away = game["h"], game["a"]
            home_field = 0.0 if game.get("n") else 1.0
            gap = state[home] - state[away] + model["logistic"]["hfa"] * home_field
            expected = _sigmoid(gap)
            margin = float(game["hp"]) - float(game["ap"])
            expected_margin = margin_sigma * statistics.NormalDist().inv_cdf(
                min(max(expected, 0.01), 0.99))
            margin_score = min(max(
                (margin - expected_margin) / margin_sigma, -2.5), 2.5)
            delta = k * margin_score
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
        for team, delta in changes.items():
            state[team] += delta
        completed += len(by_week[week])
        unfinished = any(int(g.get("w") or 0) == week and not g.get("f") for g in schedule)
        history.append({"week": week,
                        "label": f"Week {week}{' to date' if unfinished else ''}",
                        "completed_games": completed,
                        "teams": _power_snapshot(model, names, state)})

    current_rows = history[-1]["teams"]
    by_team = {row["team"]: row for row in current_rows}
    old_rows = ratings.get("teams", [])
    old_by_team = {row["team"]: row for row in old_rows}
    team_meta = _read_optional_json(teams_path, {})
    playoff_rows = _read_optional_json(playoff_path, {}).get("teams", [])
    playoff = {row["team"]: row for row in playoff_rows}
    strengths = _schedule_strength(model, names, schedule)
    updated_rows = []
    for team in names:
        old = old_by_team.get(team) or _rating_shell(
            team, model, old_rows, team_meta, playoff, strengths)
        new = by_team[team]
        updated_rows.append({**old, "rank": new["rank"], "power": new["power"],
                             "vs_average": new["vs_average"]})
    updated_rows.sort(key=lambda row: (row.get("rank", 999), row["team"]))
    ratings["teams"] = updated_rows
    ratings["history"] = history
    ratings["updated_through"] = {
        "completed_fbs_games": len(finals),
        "week": max(by_week) if by_week else 0,
    }
    dynamic["preseason_ratings"] = baseline
    dynamic["ratings"] = {**current, **state}
    dynamic["k"] = k
    dynamic["update_rule"] = "robust_margin_residual_v1"
    dynamic["margin_score_cap"] = 2.5
    dynamic["completed_games"] = len(finals)
    dynamic["updated_through_week"] = max(by_week) if by_week else 0
    model["dynamic"] = dynamic
    model_path.write_text(json.dumps(model, indent=1, allow_nan=False), encoding="utf-8")
    ratings_path.write_text(json.dumps(ratings, indent=1, allow_nan=False), encoding="utf-8")
    return len(finals)


def flatten(raw: list[dict], captured_at: str) -> list[dict]:
    # CFBD can return both an old and new spelling for the same provider/game.
    # Collapse those aliases before downstream identity and hashing logic sees them.
    # Prefer the most complete quote, then the canonical provider spelling, then a
    # deterministic serialized value when two equally complete rows disagree.
    rows: dict[tuple[int, str], tuple[tuple, dict]] = {}
    for game in raw:
        for line in game.get("lines") or []:
            raw_provider = line.get("provider")
            provider = PROVIDER_ALIAS.get(raw_provider, raw_provider)
            if not provider:
                continue
            row = {"captured_at": captured_at, "season": YEAR,
                   "game_id": int(game["id"]), "week": game.get("week"),
                   "start": game.get("startDate"), "home": game.get("homeTeam"),
                   "away": game.get("awayTeam"), "neutral": bool(game.get("neutralSite")),
                   "provider": provider}
            values = {field: line.get(field) for field in QUOTE_FIELDS}
            # CFBD/Bovada uses -100000 as an unavailable-moneyline sentinel on some
            # large favorites and underdogs. It is not a tradable price and, if read
            # literally, makes both sides look 99.9% likely after de-vigging.
            for field in ("homeMoneyline", "awayMoneyline"):
                value = values[field]
                if value is not None and (float(value) == 0 or abs(float(value)) >= 100000):
                    values[field] = None
            if all(values[field] is None for field in QUOTE_FIELDS):
                continue
            row.update(values)
            priority = (
                sum(value is not None for value in values.values()),
                raw_provider == provider,
                json.dumps(values, sort_keys=True, separators=(",", ":"), default=str),
            )
            key = quote_key(row)
            if key not in rows or priority > rows[key][0]:
                rows[key] = (priority, row)
    return [rows[key][1] for key in sorted(rows)]


def quote_key(row: dict) -> tuple:
    return int(row["game_id"]), row["provider"]


def quote_value(row: dict) -> tuple:
    return tuple(row.get(field) for field in QUOTE_FIELDS)


def quote_payload_hash(rows: list[dict]) -> str:
    """Hash quote identities without comparing nullable numeric values."""
    serialized = [json.dumps((quote_key(row), quote_value(row)),
                             separators=(",", ":"), default=str)
                  for row in rows]
    return hashlib.sha256("\n".join(sorted(serialized)).encode()).hexdigest()


def latest_quotes(events: list[dict], before: datetime | None = None,
                  include_removed=False) -> dict[tuple, dict]:
    latest = {}
    for row in events:
        if before is not None and parse_time(row["captured_at"]) > before:
            continue
        latest[quote_key(row)] = row
    return latest if include_removed else {k: v for k, v in latest.items()
                                           if not v.get("removed")}


def implied(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def moneyline_research_candidate(model_side: float, market_side: float,
                                 minimum_gap: float = .15) -> bool:
    return model_side > .50 and model_side - market_side >= minimum_gap


def no_vig_home(line: dict) -> float | None:
    h, a = line.get("homeMoneyline"), line.get("awayMoneyline")
    if h is None or a is None:
        return None
    ih, ia = implied(float(h)), implied(float(a))
    return ih / (ih + ia)


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def model_probability(model: dict, home: str, away: str, neutral=False) -> float | None:
    a, b = model.get("teams", {}).get(home), model.get("teams", {}).get(away)
    if a is None or b is None:
        return None
    x = [u - v for u, v in zip(a, b)]
    home_flag = 0.0 if neutral else 1.0
    logistic, margin = model["logistic"], model["margin"]
    z = logistic.get("intercept", 0.0) + sum(c*v for c, v in zip(logistic["coef"], x)) \
        + home_flag * logistic["hfa"]
    m = margin.get("intercept", 0.0) + sum(c*v for c, v in zip(margin["coef"], x)) \
        + home_flag * margin["hfa"]
    p = model["ens_w"] * sigmoid(z) + (1-model["ens_w"]) * .5 * (
        1 + math.erf((m / margin["sigma"]) / math.sqrt(2)))
    p = max(1e-8, min(1-1e-8, p))
    p = sigmoid(model.get("probability_scale", 1.0) * math.log(p / (1-p)))
    dynamic = model.get("dynamic") or {}
    ratings = dynamic.get("ratings") or {}
    if home in ratings and away in ratings:
        pdyn = sigmoid(ratings[home] - ratings[away] + home_flag * logistic["hfa"])
        p = (1-dynamic.get("blend", 0.0))*p + dynamic.get("blend", 0.0)*pdyn
    return p


def model_fingerprint() -> str:
    return hashlib.sha256(MODEL.read_bytes()).hexdigest()[:16]


def reliability_bucket(edge: float, underdog: bool, historical: dict) -> dict | None:
    label = "underdog" if underdog else "favorite"
    for row in historical.get("calibration_buckets", []):
        if row["side_type"] == label and row["edge_low"] <= edge < row["edge_high"]:
            return row
    return None


def group_games(quotes: dict[tuple, dict]) -> dict[int, list[dict]]:
    games: dict[int, list[dict]] = {}
    for row in quotes.values():
        games.setdefault(int(row["game_id"]), []).append(row)
    return games


def weekly_payload(quotes: dict[tuple, dict]) -> list[dict]:
    out = []
    for _, lines in group_games(quotes).items():
        first = lines[0]
        books = {row["provider"]: {field: row.get(field) for field in QUOTE_FIELDS}
                 for row in lines}
        row = {"id": first["game_id"], "week": first.get("week"),
               "start": first.get("start"), "home": first.get("home"),
               "away": first.get("away"), "books": books}
        if int(first["game_id"]) in BET_EXCLUDED_GAME_IDS:
            row["bettingExcluded"] = True
        out.append(row)
    return sorted(out, key=lambda r: (r.get("week") or 99, r.get("start") or ""))


def update_weekly_board(odds: dict, quotes: dict[tuple, dict], captured_at: str,
                        lock_weekly_board: bool) -> bool:
    """Replace the actionable board only at its weekly lock.

    Raw quote observations continue to accumulate between locks. Keeping this write
    separate prevents the public picks and the lines used to grade them from drifting
    as sportsbooks move during the week.
    """
    if not lock_weekly_board and "weekly" in odds:
        existing_source = (odds.get("sources") or {}).get("cfbd_lines") or {}
        odds.setdefault("weekly_lock", {
            "locked_at": existing_source.get("as_of"),
            "cadence": "Monday 12:30 PM ET",
            "timezone": "America/New_York",
            "reason": "pre-lock baseline",
        })
        return False
    odds["weekly"] = weekly_payload(quotes)
    odds["weekly_lock"] = {
        "locked_at": captured_at,
        "cadence": "Monday 12:30 PM ET",
        "timezone": "America/New_York",
        "reason": "scheduled" if lock_weekly_board else "initial bootstrap",
    }
    odds.setdefault("sources", {})["cfbd_lines"] = {
        "book": "Multiple", "as_of": captured_at, "url": BASE + "/",
        "timestamp_semantics": "weekly board lock; Monday 12:30 PM ET"}
    return True


def valid_close(events: list[dict], checks: list[dict], game_id: int,
                kickoff: datetime) -> tuple[list[dict], dict | None]:
    eligible = [c for c in checks if parse_time(c["checked_at"]) <= kickoff]
    if not eligible:
        return [], None
    check = max(eligible, key=lambda c: c["checked_at"])
    age = (kickoff - parse_time(check["checked_at"])).total_seconds() / 3600
    if age > 6:
        return [], {"checked_at": check["checked_at"], "age_hours": age, "qualified": False}
    quotes = latest_quotes(events, parse_time(check["checked_at"]))
    rows = [r for (gid, _), r in quotes.items() if gid == game_id and no_vig_home(r) is not None]
    return rows, {"checked_at": check["checked_at"], "age_hours": age,
                  "qualified": len(rows) >= 2}


def availability_summary() -> dict:
    if not AVAILABILITY.exists():
        return {"events": 0, "last_event_at": None}
    with AVAILABILITY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {"events": len(rows),
            "last_event_at": max((r.get("observed_at") or "" for r in rows), default=None)}


def run(raw: list[dict], now: datetime, games: list[dict] | None = None,
        lock_weekly_board: bool = False) -> dict:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    captured_at = iso(now)
    current = flatten(raw, captured_at)
    prior_events = read_jsonl(QUOTES)
    prior = latest_quotes(prior_events, include_removed=True)
    changed = [row for row in current
               if quote_key(row) not in prior or prior[quote_key(row)].get("removed")
               or quote_value(row) != quote_value(prior[quote_key(row)])]
    current_keys = {quote_key(row) for row in current}
    for key, old in prior.items():
        if key in current_keys or old.get("removed"):
            continue
        tombstone = dict(old)
        tombstone.update({field: None for field in QUOTE_FIELDS})
        tombstone.update({"captured_at": captured_at, "removed": True})
        changed.append(tombstone)
    append_jsonl(QUOTES, changed)
    events = prior_events + changed
    payload_hash = quote_payload_hash(current)
    check = {"checked_at": captured_at, "source": "CFBD /lines",
             "timestamp_semantics": "retrieval time, not sportsbook quote time",
             "games": len({r["game_id"] for r in current}), "quotes": len(current),
             "changed_quotes": len(changed), "payload_hash": payload_hash}
    append_jsonl(CHECKS, [check])
    checks = read_jsonl(CHECKS)

    model = json.loads(MODEL.read_text())
    historical = json.loads(HISTORICAL.read_text()) if HISTORICAL.exists() else {}
    current_quotes = {quote_key(r): r for r in current}
    existing_entries = read_jsonl(ENTRIES)
    entered = {(int(e["game_id"]), e["side"]) for e in existing_entries}
    candidates, new_entries = [], []
    for game_id, lines in group_games(current_quotes).items():
        if game_id in BET_EXCLUDED_GAME_IDS:
            continue
        ml = [r for r in lines if no_vig_home(r) is not None]
        # One book is a quote, not a consensus. The researched rule was explicitly
        # multi-book and is not allowed to trigger until at least two providers carry
        # valid prices on both sides.
        if len(ml) < 2:
            continue
        first = ml[0]
        start = parse_time(first["start"])
        if start <= now:
            continue
        market_home = statistics.median(no_vig_home(r) for r in ml)
        model_home = model_probability(model, first["home"], first["away"], first.get("neutral"))
        if model_home is None:
            continue
        gap = model_home - market_home
        side = "home" if gap >= 0 else "away"
        team = first["home"] if side == "home" else first["away"]
        market_side = market_home if side == "home" else 1-market_home
        model_side = model_home if side == "home" else 1-model_home
        # A price disagreement is not an outright moneyline pick when the model
        # still makes that selected team more likely to lose than win.
        if not moneyline_research_candidate(model_side, market_side):
            continue
        price = max(float(r[f"{side}Moneyline"]) for r in ml)
        breakeven = implied(price)
        bucket = reliability_bucket(model_side-market_side, price > 0, historical)
        lower = bucket.get("posterior_lower_80") if bucket else None
        cleared = bool(len(ml) >= 2 and lower is not None and lower > breakeven + .01)
        row = {"game_id": game_id, "week": first.get("week"), "start": first["start"],
               "home": first["home"], "away": first["away"], "side": side, "team": team,
               "model_side_p": model_side, "consensus_side_p": market_side,
               "gap": model_side-market_side, "best_price": price,
               "break_even_p": breakeven, "books": len(ml),
               "historical_lower_80": lower, "uncertainty_cleared": cleared,
               "gate": "80% Jeffreys lower bound > best-price break-even + 1pp",
               "model_fingerprint": model_fingerprint()}
        candidates.append(row)
        if lock_weekly_board and (game_id, side) not in entered:
            entry = {"entered_at": captured_at, **row}
            new_entries.append(entry)
            entered.add((game_id, side))
    append_jsonl(ENTRIES, new_entries)
    all_entries = [entry for entry in existing_entries + new_entries
                   if int(entry["game_id"]) not in BET_EXCLUDED_GAME_IDS]

    existing_settlements = {(int(s["game_id"]), s["side"])
                            for s in read_jsonl(SETTLEMENTS)}
    result_rows = games if games is not None else raw
    raw_by_id = {int(g["id"]): g for g in result_rows}
    new_settlements = []
    for entry in all_entries:
        key = (int(entry["game_id"]), entry["side"])
        game = raw_by_id.get(key[0], {})
        hp = game.get("homePoints", game.get("homeScore"))
        ap = game.get("awayPoints", game.get("awayScore"))
        completed = game.get("completed")
        if (key in existing_settlements or hp is None or ap is None
                or completed is False):
            continue
        close_lines, close_meta = valid_close(events, checks, key[0], parse_time(entry["start"]))
        close_home = (statistics.median(no_vig_home(r) for r in close_lines)
                      if close_lines and close_meta and close_meta["qualified"] else None)
        close_side = (close_home if entry["side"] == "home" else 1-close_home) \
            if close_home is not None else None
        won = hp > ap if entry["side"] == "home" else ap > hp
        new_settlements.append({"settled_at": captured_at, "game_id": key[0],
            "side": key[1], "team": entry["team"], "home_score": hp, "away_score": ap,
            "won": won, "entry_price": entry["best_price"],
            "entry_consensus_p": entry["consensus_side_p"], "close_consensus_p": close_side,
            "consensus_clv": (close_side-entry["consensus_side_p"]
                              if close_side is not None else None),
            "close_capture": close_meta})
    append_jsonl(SETTLEMENTS, new_settlements)
    settlements = read_jsonl(SETTLEMENTS)
    qualified_clv = [s["consensus_clv"] for s in settlements if s.get("consensus_clv") is not None]

    odds = json.loads(ODDS.read_text()) if ODDS.exists() else {"markets": {}, "sources": {}}
    board_updated = update_weekly_board(
        odds, current_quotes, captured_at, lock_weekly_board)
    ODDS.write_text(json.dumps(odds, indent=1, allow_nan=False))

    finals_published = publish_finals(games) if games is not None else 0
    ratings_replayed = replay_published_results()
    previous_tracking = (json.loads(TRACKING.read_text())
                         if TRACKING.exists() else {})
    displayed_candidates = (sorted(candidates, key=lambda r: (-r["gap"], r["start"]))
                            if board_updated
                            else previous_tracking.get("current_candidates", []))
    tracking = {"checked_at": captured_at, "source": "CFBD /lines",
        "timestamp_semantics": "retrieval time, not sportsbook quote time",
        "quote_events": len(events), "successful_checks": len(checks),
        "changed_quotes_this_check": len(changed), "games_with_quotes": check["games"],
        "books_with_quotes": sorted({r["provider"] for r in current}),
        "watchlist_rule": {"minimum_gap": .15, "minimum_books": 2,
            "uncertainty_gate": "80% Jeffreys lower bound > best-price break-even + 1pp",
            "status": "forward research; never an automatic bet"},
        "weekly_board_updated_this_check": board_updated,
        "weekly_board_lock": odds.get("weekly_lock"),
        "current_candidates": displayed_candidates,
        "entries": len(all_entries), "new_entries_this_check": len(new_entries),
        "settlements": len(settlements), "qualified_clv_observations": len(qualified_clv),
        "mean_consensus_clv": statistics.mean(qualified_clv) if qualified_clv else None,
        "final_scores_published_this_check": finals_published,
        "completed_rating_games_replayed": ratings_replayed,
        "availability": availability_summary()}
    TRACKING.write_text(json.dumps(tracking, indent=2, allow_nan=False))
    STATUS.write_text(json.dumps(check, indent=2))
    return tracking


def latest_weekly() -> list[dict]:
    events = read_jsonl(QUOTES)
    return weekly_payload(latest_quotes(events)) if events else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-key", action="store_true")
    parser.add_argument("--lock-weekly-board", action="store_true",
                        help="publish the Monday 12:30 PM ET actionable board")
    args = parser.parse_args()
    load_env()
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        if args.allow_missing_key:
            print("CFBD_API_KEY is not configured; market capture skipped.")
            return
        raise RuntimeError("CFBD_API_KEY is not configured")
    now = utcnow()
    tracking = run(fetch_lines(key), now, fetch_games(key), args.lock_weekly_board)
    print(f"Market check {tracking['checked_at']}: {tracking['games_with_quotes']} games, "
          f"{tracking['changed_quotes_this_check']} changed quotes, "
          f"{len(tracking['current_candidates'])} research candidates, "
          f"{tracking['final_scores_published_this_check']} new final scores, "
          f"{tracking['completed_rating_games_replayed']} rating results replayed.")
    print("Weekly board " + ("locked." if tracking["weekly_board_updated_this_check"]
                             else "unchanged; background capture only."))
    print(f"-> {QUOTES}\n-> {TRACKING}")


if __name__ == "__main__":
    main()
