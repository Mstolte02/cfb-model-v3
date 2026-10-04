import pandas as pd

from scripts.fourth_jev_calibrate import coherence, paired_bootstrap, paired_frame


def test_paired_evaluation_keeps_exact_game_cutoff_sample():
    base = pd.DataFrame([
        {"game_id": "1", "as_of": "2025-W01-pregame", "season": 2025,
         "week": 1, "target": "home_win", "p": .6, "y": 1, "source": "cfb"},
        {"game_id": "2", "as_of": "2025-W01-pregame", "season": 2025,
         "week": 1, "target": "home_win", "p": .4, "y": 0, "source": "cfb"},
    ])
    jev = base.drop(columns="week").copy()
    jev["p"] = [.7, .3]
    jev["source"] = "jev"
    paired = paired_frame(base, jev)
    result = paired_bootstrap(paired, reps=50, seed=1)
    assert result["n"] == 2
    assert result["jev_minus_cfb_brier"] < 0


def test_tail_coherence_detects_non_nested_answers():
    answers = {name: {"noul": value} for name, value in {
        "home_win": .55, "home_by_7_plus": .65, "home_by_14_plus": .30,
        "home_by_21_plus": .20, "away_by_7_plus": .20, "within_3": .15,
    }.items()}
    result = coherence([{"game_id": "bad", "answers": answers}])
    assert result["direct_sets_checked"] == 1
    assert result["nesting_violations"] == 1
