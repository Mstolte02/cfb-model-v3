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
HISTORICAL = ROOT / "audit" / "book_shopping_backtest.json"
AVAILABILITY = ROOT / "war_model" / "availability_events_2026.csv"

PROVIDER_ALIAS = {"Draft Kings": "DraftKings"}
QUOTE_FIELDS = ("spread", "spreadOpen", "overUnder", "overUnderOpen",
                "homeMoneyline", "awayMoneyline")


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


def fetch_cfbd(key: str, endpoint: str, params: dict) -> list[dict]:
    url = f"{BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    if os.name == "nt":
        config = (f'url = "{url}"\nheader = "Authorization: Bearer {key}"\n'
                  'header = "Accept: application/json"\nsilent\nshow-error\n'
                  'max-time = 45\nwrite-out = "\\n%{http_code}"\n')
        proc = subprocess.run(["curl.exe", "--config", "-"], input=config, text=True,
                              capture_output=True, timeout=55)
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


def flatten(raw: list[dict], captured_at: str) -> list[dict]:
    rows = []
    for game in raw:
        for line in game.get("lines") or []:
            provider = PROVIDER_ALIAS.get(line.get("provider"), line.get("provider"))
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
            rows.append(row)
    return rows


def quote_key(row: dict) -> tuple:
    return int(row["game_id"]), row["provider"]


def quote_value(row: dict) -> tuple:
    return tuple(row.get(field) for field in QUOTE_FIELDS)


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
        out.append({"id": first["game_id"], "week": first.get("week"),
                    "start": first.get("start"), "home": first.get("home"),
                    "away": first.get("away"), "books": books})
    return sorted(out, key=lambda r: (r.get("week") or 99, r.get("start") or ""))


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


def run(raw: list[dict], now: datetime, games: list[dict] | None = None) -> dict:
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
    payload_hash = hashlib.sha256(json.dumps(
        sorted((quote_key(r), quote_value(r)) for r in current), default=str).encode()).hexdigest()
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
        if abs(gap) < .15:
            continue
        side = "home" if gap >= 0 else "away"
        team = first["home"] if side == "home" else first["away"]
        market_side = market_home if side == "home" else 1-market_home
        model_side = model_home if side == "home" else 1-model_home
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
        if (game_id, side) not in entered:
            entry = {"entered_at": captured_at, **row}
            new_entries.append(entry)
            entered.add((game_id, side))
    append_jsonl(ENTRIES, new_entries)
    all_entries = existing_entries + new_entries

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
    odds["weekly"] = weekly_payload(current_quotes)
    odds.setdefault("sources", {})["cfbd_lines"] = {
        "book": "Multiple", "as_of": captured_at, "url": BASE + "/",
        "timestamp_semantics": "our successful retrieval time; not a sportsbook quote timestamp"}
    ODDS.write_text(json.dumps(odds, indent=1, allow_nan=False))

    finals_published = publish_finals(games) if games is not None else 0
    tracking = {"checked_at": captured_at, "source": "CFBD /lines",
        "timestamp_semantics": "retrieval time, not sportsbook quote time",
        "quote_events": len(events), "successful_checks": len(checks),
        "changed_quotes_this_check": len(changed), "games_with_quotes": check["games"],
        "books_with_quotes": sorted({r["provider"] for r in current}),
        "watchlist_rule": {"minimum_gap": .15, "minimum_books": 2,
            "uncertainty_gate": "80% Jeffreys lower bound > best-price break-even + 1pp",
            "status": "forward research; never an automatic bet"},
        "current_candidates": sorted(candidates, key=lambda r: (-r["gap"], r["start"])),
        "entries": len(all_entries), "new_entries_this_check": len(new_entries),
        "settlements": len(settlements), "qualified_clv_observations": len(qualified_clv),
        "mean_consensus_clv": statistics.mean(qualified_clv) if qualified_clv else None,
        "final_scores_published_this_check": finals_published,
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
    args = parser.parse_args()
    load_env()
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        if args.allow_missing_key:
            print("CFBD_API_KEY is not configured; market capture skipped.")
            return
        raise RuntimeError("CFBD_API_KEY is not configured")
    now = utcnow()
    tracking = run(fetch_lines(key), now, fetch_games(key))
    print(f"Market check {tracking['checked_at']}: {tracking['games_with_quotes']} games, "
          f"{tracking['changed_quotes_this_check']} changed quotes, "
          f"{len(tracking['current_candidates'])} research candidates, "
          f"{tracking['final_scores_published_this_check']} new final scores.")
    print(f"-> {QUOTES}\n-> {TRACKING}")


if __name__ == "__main__":
    main()
