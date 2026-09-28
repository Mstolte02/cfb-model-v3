"""Create a public-safe view of frozen Jev forecasts for the web desk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fourth_jev.ledger import state_hash


def export_view(states: Path, ledger: Path, teams: Path) -> dict:
    meta = json.loads(teams.read_text(encoding="utf-8"))
    records = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        # Only the direct win pilot is eligible for display. Independent tail
        # Nouls can violate event nesting, and the v0.2 margin-bucket trial
        # disagreed sharply with the direct win question.
        if row.get("stage") != "football" or row.get("question_version") != "0.1.0":
            continue
        records[(row["game_id"], row["as_of"], row["state_hash"])] = row
    games = []
    as_of = None
    for line in states.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        state, game = row["state"], row["state"]["game"]
        key = (row["game_id"], row["as_of"], state_hash(state))
        record = records.get(key)
        if record is None:
            continue
        if as_of is None:
            as_of = row["as_of"]
        if as_of != row["as_of"]:
            raise ValueError("mixed forecast issue times")
        answers = record["answers"]
        probs = {"home_win": float(answers["home_win"]["noul"])}
        games.append({
            "id": row["game_id"],
            "home": game["home_team"], "away": game["away_team"],
            "date": game["kickoff_date"], "venue": game.get("venue"),
            "neutral": game["neutral"],
            "homeAbbr": meta[game["home_team"]]["abbreviation"],
            "awayAbbr": meta[game["away_team"]]["abbreviation"],
            "homeColor": meta[game["home_team"]]["color"],
            "awayColor": meta[game["away_team"]]["color"],
            "baselineHome": state["existing_model"]["home_win_probability"],
            "jev": probs,
            "tailShape": answers["tail_shape"]["choice"],
            "varianceScore": answers["variance_level"]["score"],
            "upsetPathScore": answers["upset_path_strength"]["score"],
            "jevModel": record["metadata"]["resolved_model"],
        })
    if not games:
        raise ValueError("no matching frozen Jev forecasts")
    first = json.loads(states.read_text(encoding="utf-8").splitlines()[0])["state"]
    return {
        "season": first["game"]["season"], "week": first["game"]["week"],
        "issuedAt": as_of,
        "baselineVersion": first["existing_model"]["model_version"],
        "modelSnapshotSha256": first["source_manifest"]["published_model_sha256"],
        "cutoff": first["context"], "games": games,
        "status": "experimental_unvalidated",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--teams", type=Path, default=Path("viz/data/teams.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    view = export_view(args.states, args.ledger, args.teams)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(view, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"exported {len(view['games'])} matched forecasts -> {args.out}")


if __name__ == "__main__":
    main()
