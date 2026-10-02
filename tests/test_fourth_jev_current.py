import json
from datetime import datetime, timezone

import pytest

from fourth_jev.current import export_current, write_current
from fourth_jev.ledger import make_record, state_hash
from scripts.fourth_jev_export_site import export_view


def test_current_export_is_pregame_and_market_blind(tmp_path):
    model = {
        "ensemble": {
            "model_version": "5.3", "architecture": "ensemble",
            "training_seasons": [2021], "war_projected": {"A": 1, "B": 2},
            "members": [{"name": "one", "initial": {"A": 0.4, "B": 0.2},
                         "columns": ["prior_level"], "scale": [1], "coef": [1],
                         "hfa_coef": 0, "margin_sigma": 14}],
            "state": {"updated_through_slate": 4, "ratings": {"one": {"A": 0.4, "B": 0.2}},
                      "form": {"one": None}, "pff": None, "war": None},
        }
    }
    schedule = [{"id": 1, "h": "A", "a": "B", "w": 5, "d": "2026-10-02", "n": 0},
                {"id": 2, "h": "A", "a": "B", "w": 4, "d": "2026-09-25", "f": 1}]
    teams = {"A": {"abbreviation": "AA", "color": "#123456"},
             "B": {"abbreviation": "BB", "color": "#654321"}}
    for name, value in [("model.json", model), ("schedule.json", schedule), ("teams.json", teams)]:
        (tmp_path / name).write_text(json.dumps(value))
    issued = datetime(2026, 9, 28, tzinfo=timezone.utc)
    rows = export_current(tmp_path/"model.json", tmp_path/"schedule.json",
                          tmp_path/"teams.json", tmp_path/"missing.csv", issued)
    assert len(rows) == 1
    assert rows[0]["state"]["game"]["kickoff_timestamp"] is None
    assert rows[0]["state"]["as_of"] == "2026-09-28T00:00:00Z"
    assert rows[0]["state"]["contract"]["market_blind"]
    assert "market" not in rows[0]["state"]
    assert rows[0]["state"]["source_manifest"]["market_included"] is False
    with pytest.raises(FileExistsError):
        out = tmp_path/"snapshot.jsonl"
        write_current(rows, out)
        write_current(rows, out)

    answers = {name: {"noul": .5} for name in
               ("home_win", "home_by_7_plus", "home_by_14_plus",
                "home_by_21_plus", "away_by_7_plus", "within_3")}
    answers.update({"tail_shape": {"choice": "balanced"},
                    "variance_level": {"score": 2},
                    "upset_path_strength": {"score": 1},
                    "margin_bucket": {"probabilities": {
                        "home_1_3": .1, "home_4_6": .1, "home_7_13": .1,
                        "home_14_20": .1, "home_21_plus": .1,
                        "away_1_3": .1, "away_4_6": .1, "away_7_13": .1,
                        "away_14_20": .1, "away_21_plus": .1}}})
    record = make_record(game_id="1", as_of=rows[0]["as_of"],
                         state=rows[0]["state"], model="jev-latest", answers=answers,
                         metadata={"resolved_model": "jev-1.13.0"})
    record["question_version"] = "0.1.0"
    assert record["state_hash"] == state_hash(rows[0]["state"])
    ledger = tmp_path/"ledger.jsonl"
    ledger.write_text(json.dumps(record)+"\n")
    view = export_view(out, ledger, tmp_path/"teams.json")
    assert view["baselineVersion"] == "5.3"
    assert view["games"][0]["jev"]["home_win"] == .5
