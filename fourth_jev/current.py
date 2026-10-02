"""Export market-blind current-week states from the published live ensemble.

The published block is a frozen snapshot. This module never replays completed
games or reads odds, and it refuses to label a date-only kickoff as a timestamp.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fourth_jev.state import build_game_state
from scripts.ensemble_replay import probability


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _projection_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["team"]: {key: float(value) for key, value in row.items()
                              if key not in ("team", "conference") and value}
                for row in csv.DictReader(fh)}


def _team_block(team: str, ensemble: dict, projections: dict) -> dict:
    live = ensemble["state"]
    members = {}
    for member in ensemble["members"]:
        name = member["name"]
        first = float(member["initial"][team])
        current = float(live["ratings"][name][team])
        form = (live.get("form", {}).get(name) or {}).get(team)
        members[name] = {
            "preseason_logit_strength": first,
            "current_logit_strength": current,
            "rating_change": current - first,
            "form_offense": float(form[0]) if form else None,
            "form_defense": float(form[1]) if form else None,
            "form_games": int(form[2]) if form else 0,
        }
    return {
        "preseason": {
            "projected_war": ensemble.get("war_projected", {}).get(team),
            "position_projection": projections.get(team),
        },
        "current": {
            "ensemble_members": members,
            "pff_team_offense": (live.get("pff") or {}).get(team, [None, None])[0],
            "pff_team_defense": (live.get("pff") or {}).get(team, [None, None])[1],
            "inseason_war_delta": (live.get("war") or {}).get(team),
        },
    }


def export_current(model_path: Path, schedule_path: Path, team_path: Path,
                   projection_path: Path, issue_time: datetime) -> list[dict]:
    if issue_time.tzinfo is None:
        raise ValueError("issue_time must include a timezone")
    issue_time = issue_time.astimezone(timezone.utc)
    snapshot = model_path.read_bytes()
    model = json.loads(snapshot)
    ensemble = model["ensemble"]
    live = ensemble["state"]
    through = int(live["updated_through_slate"])
    if through < 0:
        raise ValueError("invalid published slate cutoff")
    week = through + 1
    teams = _json(team_path)
    projections = _projection_rows(projection_path)
    rows = []
    for game in _json(schedule_path):
        if game.get("f") or int(game.get("w", -1)) != week:
            continue
        # The schedule contains dates but not precise kickoffs. Never issue on
        # the game date: we cannot prove the snapshot precedes kickoff.
        if game["d"] <= issue_time.date().isoformat():
            continue
        home, away = game["h"], game["a"]
        if home not in teams or away not in teams:
            continue
        p = probability(ensemble, live, home, away, 0.0 if game.get("n") else 1.0)
        if p is None:
            continue
        game_id = str(game["id"])
        as_of = issue_time.isoformat().replace("+00:00", "Z")
        state = build_game_state(
            game={"game_id": game_id, "season": 2026, "week": week,
                  "home_team": home, "away_team": away,
                  "neutral": bool(game.get("n")), "venue": game.get("v"),
                  "kickoff_date": game["d"], "kickoff_timestamp": None},
            home=_team_block(home, ensemble, projections),
            away=_team_block(away, ensemble, projections),
            model={"model_version": ensemble["model_version"],
                   "home_win_probability": p,
                   "training_seasons": ensemble["training_seasons"],
                   "architecture": ensemble["architecture"]},
            context={"state_cutoff_basis": f"through completed slate {through}",
                     "pff_completed_slate_cutoffs": live.get("pff_cutoffs", []),
                     "war_completed_slate_cutoffs": live.get("war_cutoffs", [])},
            sources={"published_model_sha256": hashlib.sha256(snapshot).hexdigest(),
                     "model_file": model_path.name,
                     "market_included": False},
            as_of=as_of,
        )
        rows.append({"game_id": game_id, "as_of": as_of, "state": state})
    return rows


def write_current(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {output}")
    with output.open("x", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
