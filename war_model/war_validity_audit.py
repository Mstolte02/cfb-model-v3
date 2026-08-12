"""Publish the evidence for (and limitations of) the player WAR layer."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    projection = json.loads((HERE / "projection_metrics.json").read_text())
    external = json.loads((HERE / "external_validation.json").read_text())
    v4 = json.loads((ROOT / "artifacts" / "v4_backtest.json").read_text())
    deltas = []
    for fold in v4["folds"]:
        scores = fold.get("candidate_scores", {})
        clean, player = scores.get("clean_core"), scores.get("core_war_projected")
        if clean is not None and player is not None:
            deltas.append({"season": fold["season"], "brier_gain_vs_clean_core": clean-player,
                           "selected": fold["selected"] == "core_war_projected"})
    ea_all = next(x for x in external["hist"] if x["group"] == "ALL")
    report = {
        "decision": "retain measured PFF/CFBD WAR; use EA only as an ordering prior for low-snap players",
        "ea_replacement": {
            "accepted": False,
            "reason": ("The reboot supplies only CFB25 (2024), CFB26 (2025), and CFB27 "
                       "(2026). No stable public full-player archive for both completed "
                       "historical editions was found, so there is not enough pregame "
                       "player history to refit and validate position weights."),
            "clean_team_test_2025": {"teams": 134, "ea_r": .420,
                                     "current_talent_r": .476,
                                     "incremental_r2": .001,
                                     "ea_residual_r": -.013},
        },
        "projection_holdout_2025": {
            "players": projection["holdout_n"], "correlation": projection["holdout_r"],
            "mae_wins": projection["holdout_mae"],
            "carry_forward_correlation": projection["carry_r"],
            "no_prior_snap_players": projection["nohist_n"],
            "no_prior_snap_correlation": projection["nohist_r"],
            "ex_ante_only": projection["ex_ante_only"],
        },
        "team_model_ablation": deltas,
        "independent_ea_convergent_validity": {
            "matched_players": external["hist_matched"],
            "war_ea_correlation": ea_all["r_value"],
            "partial_after_playing_time": ea_all["partial"],
            "note": "EA is an independent opinion, not ground truth.",
        },
        "limitations": [
            "The no-prior-snap individual projection is weak (r=.216).",
            "EA's low-snap ordering is not historically player-level validated.",
            "WAR is a projection component, not a literal causal value for a roster move.",
        ],
    }
    (HERE / "war_validity_audit.json").write_text(json.dumps(report, indent=2))
    (ROOT / "viz" / "data" / "war_validity.json").write_text(
        json.dumps(report, separators=(",", ":")))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
