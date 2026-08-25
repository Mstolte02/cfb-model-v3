"""Phase 6 descriptive work: are play-call decisions a repeatable coach trait, and
do they explain the apparent coach effect?

Three questions, cheapest first, because each one can end the phase:

1. Do the metrics repeat? A decision that does not recur next season is a season of
   noise, not a tendency, and cannot forecast anything.
2. Are they new? Pass rate over expected that merely reproduces the raw early-down
   pass rate is a rename, not a measurement.
3. Do they mediate the coach effect? Adding lagged decision tendencies to the two-way
   HDFE decomposition shows how much of the coach variance share is decision-shaped.

None of this authorizes a feature. The predictive test is
``scripts.decision_predictive_backtest`` and it is the only thing that can.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS
from scripts.coach_effects import decomposition_frame
from src import decisions as D
from src.data import plays as PLAYS


OUT_JSON = ARTIFACTS / "decision_profile.json"
OUT_CSV = ARTIFACTS / "decision_profiles.csv"
BASE_X = ("prior_offense", "prior_defense", "talent", "returning")
# Compact and near-orthogonal: passing tendency and fourth-down aggression measure
# different things (see the orthogonality table), so both earn a slot.
DECISION_X = ("proe_lag", "fourth_go_oe_lag")
OUTCOMES = ("rating_overall", "rating_offense", "rating_defense")


def _hdfe(frame: pd.DataFrame, outcome: str, controls) -> dict:
    """Two-way HDFE variance shares for one control set."""
    import pyfixest as pf

    formula = f"{outcome} ~ {' + '.join(controls)} | program_fe + coach_fe"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model = pf.feols(formula, data=frame)
    used = model._data.copy()
    effects = model.fixef()
    program = used.program_fe.map(effects["C(program_fe)"]).fillna(0).to_numpy(float)
    coach = used.coach_fe.map(effects["C(coach_fe)"]).fillna(0).to_numpy(float)
    y = used[outcome].to_numpy(float)
    residual = np.asarray(model.resid(), dtype=float)
    xb = np.asarray(model.predict(), dtype=float) - program - coach
    total = float(np.var(y)) or 1.0
    return {
        "coach_share": float(np.var(coach)) / total,
        "program_share": float(np.var(program)) / total,
        "xb_share": float(np.var(xb)) / total,
        "residual_share": float(np.var(residual)) / total,
        "coach_program_corr": float(np.corrcoef(coach, program)[0, 1]),
        "n_used": int(len(used)),
        "coefficients": {k: float(v) for k, v in model.coef().items()},
    }


def build_profiles():
    """Every team-season profile, plus the league expectation models behind them."""
    frame = PLAYS.load()
    audit = PLAYS.audit_play_types()
    expectations = D.fit_expectations(frame)
    profiles = D.team_season_profiles(frame, expectations)
    return frame, profiles, expectations, audit


def orthogonality(profiles: pd.DataFrame) -> dict:
    """How much of each metric is new information versus the shipping raw rate."""
    raw = profiles.early_down_pass_rate
    out = {}
    for column in D.PROFILE_COLUMNS:
        if column == "early_down_pass_rate":
            continue
        out[column] = {
            "corr_with_raw_early_down_pass_rate": float(profiles[column].corr(raw)),
            "corr_with_proe": float(profiles[column].corr(profiles.proe)),
        }
    # The adjustment's own size: how far the situation model moves a team away from
    # its centred raw rate. Small here would mean the whole exercise is cosmetic.
    centred = raw - raw.mean()
    out["situation_adjustment_sd"] = float((profiles.proe - centred).std())
    out["proe_sd"] = float(profiles.proe.std())
    return out


def mediation(profiles: pd.DataFrame) -> dict:
    """Does lagged decision tendency absorb the coach variance share?"""
    frame = decomposition_frame()
    lag = profiles.copy()
    lag["season"] = lag.season + 1          # season N carries N-1 tendencies
    lag = lag.rename(columns={"proe": "proe_lag",
                              "fourth_go_oe": "fourth_go_oe_lag"})
    merged = frame.merge(lag[["season", "team", "proe_lag", "fourth_go_oe_lag"]],
                         on=["season", "team"], how="inner")
    merged["coach_fe"] = merged.coach_id.astype(str)
    result = {"n_rows": int(len(merged)),
              "n_rows_before_lag_join": int(len(frame)),
              "seasons": sorted(int(s) for s in merged.season.unique())}
    for outcome in OUTCOMES:
        without = _hdfe(merged, outcome, BASE_X)
        with_decisions = _hdfe(merged, outcome, (*BASE_X, *DECISION_X))
        result[outcome] = {
            "without_decisions": without,
            "with_decisions": with_decisions,
            "coach_share_change": (with_decisions["coach_share"] -
                                   without["coach_share"]),
            "coach_share_relative_change": (
                (with_decisions["coach_share"] - without["coach_share"]) /
                without["coach_share"] if without["coach_share"] else float("nan")),
        }
    return result


def main():
    frame, profiles, expectations, audit = build_profiles()
    stability = D.stability(profiles)
    result = {
        "contract": (
            "expectation models and profiles here are descriptive and use all "
            "seasons; the predictive script refits both inside expanding folds"),
        "plays": {"rows": int(len(frame)),
                  "play_type_audit": audit.to_dict("records"),
                  "expectation_n_early": expectations["n_early"],
                  "expectation_n_fourth": expectations["n_fourth"]},
        "normal_course_fourth_down": {
            "max_abs_lead": D.NORMAL_COURSE_MAX_LEAD,
            "min_seconds_left": D.NORMAL_COURSE_MIN_SECONDS,
            "shrinkage_prior_plays": D.FOURTH_PRIOR_PLAYS,
        },
        "team_seasons": int(len(profiles)),
        "distribution": profiles[D.PROFILE_COLUMNS].describe().to_dict(),
        "stability_lag1": stability.to_dict("records"),
        "orthogonality": orthogonality(profiles),
        "mediation": mediation(profiles),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    profiles.to_csv(OUT_CSV, index=False)

    print("\n--- stability (lag-1 r) ---")
    print(stability.round(3).to_string(index=False))
    print("\n--- mediation: coach variance share ---")
    for outcome in OUTCOMES:
        block = result["mediation"][outcome]
        print(f"{outcome:16s} without {block['without_decisions']['coach_share']:.4f}"
              f"  with {block['with_decisions']['coach_share']:.4f}"
              f"  change {block['coach_share_change']:+.4f}")
    print(f"\n-> {OUT_JSON}\n-> {OUT_CSV}")
    return result


if __name__ == "__main__":
    main()
