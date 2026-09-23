"""The v5 live ensemble: fitted manifest, and the block the site publishes.

Four independently fitted expanded-WAR preseason models are each updated in season
by a robust margin-residual walk and opponent-adjusted EWMA form, then combined by a
logistic stack; the published probability is the unweighted mean of the four.

There is one runtime, scripts/ensemble_replay.py (stdlib, so the scheduled capture
can run it), and one live state, the ``ensemble`` block of viz/data/model_v4.json.
This module fits nothing and replays nothing: it stores the manifest that
scripts/train_live_ensemble.py fits, and turns it into that published block.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ARTIFACTS, PROJECTION_YEAR
from src import v4 as V4


MANIFEST_PATH = ARTIFACTS / "live_ensemble.json"
FRAME_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_live_team_frame.csv"


def model_payload(model: V4.ReciprocalTeamModel) -> dict:
    return {
        "feature_names": model.feature_names, "coef": model.coef.tolist(),
        "hfa_coef": model.hfa_coef, "margin_coef": model.margin_coef.tolist(),
        "margin_hfa": model.margin_hfa, "margin_sigma": model.margin_sigma,
        "C": model.C, "alpha": model.alpha,
        "ensemble_weight": model.ensemble_weight,
        "probability_scale": model.probability_scale, "version": model.version,
    }


def model_from_payload(value: dict) -> V4.ReciprocalTeamModel:
    return V4.ReciprocalTeamModel(
        feature_names=value["feature_names"], coef=np.asarray(value["coef"]),
        hfa_coef=float(value["hfa_coef"]),
        margin_coef=np.asarray(value["margin_coef"]),
        margin_hfa=float(value["margin_hfa"]),
        margin_sigma=float(value["margin_sigma"]), C=float(value.get("C", .1)),
        alpha=float(value.get("alpha", 10)),
        ensemble_weight=float(value.get("ensemble_weight", .5)),
        probability_scale=float(value.get("probability_scale", 1)),
        version=value.get("version", "4.1"))


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(payload: dict, path: Path = MANIFEST_PATH) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def team_war(comp: pd.DataFrame) -> dict:
    """Each team's standardised ``war_projected``, keyed by team.

    This is the roster-WAR input every member's preseason model was fitted on, and it
    is fixed for the season. It changes no probability: ``initial`` already holds its
    effect. It ships so the Team tables page can show the live model's own WAR input
    rather than the frozen v4 frame's, which the ensemble does not read.
    """
    return {team: round(float(comp.loc[team, "war_projected"]), 6) for team in comp.index}


def ensemble_block(manifest: dict, frame: pd.DataFrame, comp: pd.DataFrame) -> dict:
    """The published, stateless half of the live ensemble.

    ``frame`` is the fitted team frame; ``comp`` adds the fifth-percentile newcomer
    rows.  A team already in ``frame`` keeps the strength computed against
    ``frame`` - the average-matchup vector depends on the whole field, so adding
    newcomer rows would otherwise shift every existing team's week-0 rating.
    """
    members = []
    for entry in manifest["members"]:
        model = model_from_payload(entry["model"])
        initial = {team: round(float(model.team_logit_strength(
                       frame if team in frame.index else comp, team)), 12)
                   for team in comp.index}
        members.append({
            "name": entry["name"], "weight": entry["weight"],
            "columns": entry["stack"]["columns"], "scale": entry["stack"]["scale"],
            "coef": entry["stack"]["coef"], "score_k": entry["score_k"],
            "current_halflife": entry["current_halflife"],
            "hfa_coef": float(model.hfa_coef),
            "margin_sigma": float(model.margin_sigma), "initial": initial,
            # v5.1: the same stack plus PFF's season-to-date offence/defence
            # composites; used whenever the week's PFF table is in play.
            "stack_pff": entry.get("stack_pff"),
            # v5.2: the PFF stack plus in-season player WAR; used when the PFF
            # table and the WAR payload are both in play.
            "stack_war": entry.get("stack_war"),
        })
    return {
        "model_version": manifest["model_version"],
        "architecture": manifest["architecture"],
        "training_seasons": manifest["training_seasons"],
        "temporal_contract": manifest["temporal_contract"],
        "combination": "unweighted mean of member probabilities",
        "form_table": f"data/live/game_advanced_{PROJECTION_YEAR}.json",
        "min_form_games": 2,
        "margin_sigma": float(np.mean([m["margin_sigma"] for m in members])),
        "pff_form_table": f"data/live/pff_form_{PROJECTION_YEAR}.json",
        "war_team_table": f"data/live/inseason_war_team_{PROJECTION_YEAR}.json",
        # Display only; see team_war(). The scheduled capture keeps it as it is.
        "war_projected": team_war(comp),
        "members": members,
    }
