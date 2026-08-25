"""Select, fit, and publish the temporally clean v4 production model.

All structural and numeric choices are made by forward validation on completed
seasons.  The final model is then refit on every completed game season and emits the
2026 preseason frame, power ratings, and an empty dynamic state ready for results.

Run: python -m scripts.train_v4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA, PROJECTION_YEAR,
                    PROJECTION_TALENT_FALLBACK_YEAR, ROOT)
from scripts.train import load_bundle, projection_returning_raw
from scripts.v4_backtest import (CANDIDATES, choose_candidate, stack, tune,
                                 tune_dynamic, SELECTION_MIN_GAIN)
from src import oppadj as OA
from src import projection as P
from src import talent_sources as TS
from src import v4 as V4
from src.data import load, pff, war
from src.dynamic import WeeklyRatingState


SELECTION_PATH = ARTIFACTS / "v4_selection.json"
FRAME_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_v4_team_frame.csv"
RATINGS_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_power_ratings.csv"
STATE_PATH = ARTIFACTS / f"{PROJECTION_YEAR}_dynamic_state.json"


def _projection_preseason_inputs(talent, returning):
    """Add only inputs that would be available before the projection season."""
    rp = projection_returning_raw()
    returning[PROJECTION_YEAR] = P._z(rp)

    tal_csv = ROOT / "data" / f"talent_{PROJECTION_YEAR}.csv"
    if tal_csv.exists():
        raw = pd.read_csv(tal_csv).set_index("team")["talent"]
        talent[PROJECTION_YEAR] = P._z(raw)
        talent_source = tal_csv.name
    else:
        talent[PROJECTION_YEAR] = talent[PROJECTION_TALENT_FALLBACK_YEAR].copy()
        talent_source = f"{PROJECTION_TALENT_FALLBACK_YEAR}_proxy"

    # Keep teams such as service academies in the production universe even when the
    # recruiting composite does not cover them. This is an explicit low-percentile
    # imputation, not a silent row drop.
    missing = returning[PROJECTION_YEAR].index.difference(
        talent[PROJECTION_YEAR].index)
    fallback_teams = sorted(missing)
    if len(missing):
        floor = float(talent[PROJECTION_YEAR].quantile(.10))
        talent[PROJECTION_YEAR] = pd.concat([
            talent[PROJECTION_YEAR], pd.Series(floor, index=missing, dtype=float)])
    returning_source = (f"returning_{PROJECTION_YEAR}_cfbd.csv"
                        if (ROOT / "data" /
                            f"returning_{PROJECTION_YEAR}_cfbd.csv").exists()
                        else f"returning_{PROJECTION_YEAR}.csv")
    return {"returning_source": returning_source,
            "talent_source": talent_source,
            "talent_floor_teams": fallback_teams}


def build_inputs(include_projection=True):
    std, talent, returning, games, _ = load_bundle()
    projection_meta = {}
    if include_projection:
        projection_meta = _projection_preseason_inputs(talent, returning)
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    war_lag = war.lagged_team_talent({y: s.index for y, s in talent.items()})
    return std, talent, returning, games, od, pff_lag, war_lag, projection_meta


def build_frames(std, talent, returning, od, pff_lag, war_lag, years):
    war_projected = war.projected_team_talent(
        {y: s.index for y, s in talent.items()})
    # Positional recruiting and the rated portal are preseason facts for every
    # year built here, projection season included; the 2026 portal class and
    # classes through 2026 are what season-N leakage rules allow it to see.
    portal = TS.portal_features(years)
    groups = TS.group_features(years)
    frames = {}
    for year in years:
        frame = V4.build_frame(year, std, talent, returning, od, pff_lag,
                               war_lag, granular=True)
        if frame is not None:
            wp = war_projected.get(year, pd.Series(dtype=float)).reindex(frame.index)
            frame["war_projected"] = wp.fillna(0.0)
            frame.attrs["war_projected_coverage"] = float(wp.notna().mean())
            TS.attach(frame, portal[year], groups[year])
            frame["strength"] = frame.O + frame.D
            frames[year] = frame
    return frames


def select_final(frames, games):
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, cols)
                 for name, cols in CANDIDATES.items()}
    candidate, candidate_scores = choose_candidate(all_parts, GAME_YEARS)
    names, parts = CANDIDATES[candidate], all_parts[candidate]
    knobs, tuning = tune(parts, GAME_YEARS, names)
    dynamic_k, dynamic_blend, dynamic_tuning = tune_dynamic(
        parts, frames, GAME_YEARS, names, knobs)
    X, y, home, margins = stack(parts, GAME_YEARS)
    model = V4.fit(X, y, home, margins, names, **knobs)
    return model, {"selected": candidate, "features": names,
                   "candidate_scores": candidate_scores, "knobs": knobs,
                   "selection_rule": {"reference": "clean_core",
                                      "minimum_extension_brier_gain":
                                      SELECTION_MIN_GAIN},
                   "dynamic_k": dynamic_k,
                   "dynamic_blend": dynamic_blend,
                   "tuning": tuning, "dynamic_tuning": dynamic_tuning,
                   "training_seasons": GAME_YEARS,
                   "n_training_games": int(len(y))}


def main():
    load.require_key()
    if STATE_PATH.exists():
        live = WeeklyRatingState.load(STATE_PATH)
        if live.processed_games:
            raise RuntimeError(
                f"refusing to retrain over a live {PROJECTION_YEAR} state with "
                f"{len(live.processed_games)} processed games; archive the state "
                "and predictions deliberately before starting a new model lineage")
    print("Loading cached CFBD and licensed player inputs ...")
    inputs = build_inputs(include_projection=True)
    std, talent, returning, games, od, pff_lag, war_lag, projection_meta = inputs
    frames = build_frames(std, talent, returning, od, pff_lag, war_lag,
                          [*GAME_YEARS, PROJECTION_YEAR])
    if PROJECTION_YEAR not in frames:
        raise RuntimeError(f"could not construct {PROJECTION_YEAR} v4 team frame")

    model, selection = select_final(frames, games)
    model.save()
    V4.assert_reciprocal(model, frames[PROJECTION_YEAR])

    projection = frames[PROJECTION_YEAR]
    projection.to_csv(FRAME_PATH, index_label="team")
    ratings = V4.power_ratings(model, projection)
    ratings.to_csv(RATINGS_PATH, index=False)
    state = WeeklyRatingState.initialize(
        model, projection, PROJECTION_YEAR, selection["dynamic_k"],
        selection["dynamic_blend"])
    state.save(STATE_PATH)

    selection.update({"model_version": model.version,
                      "temporal_contract":
                      "season N uses completed N-1 team stats/player summaries "
                      "plus preseason-N recruiting and returning production",
                      "projection": projection_meta,
                      "projection_teams": int(len(projection)),
                      "player_policy":
                      "lagged team PFF/WAR may be selected; current 2026 roster "
                      "projections are reporting/scenario inputs, not silently "
                      "added to an unvalidated historical design"})
    SELECTION_PATH.write_text(json.dumps(selection, indent=2))

    print(f"Selected {selection['selected']}: {selection['features']}")
    print(f"Parameters: {selection['knobs']}")
    print(f"Weekly update: k={selection['dynamic_k']:.2f}, "
          f"blend={selection['dynamic_blend']:.2f}")
    print(f"Training games: {selection['n_training_games']}")
    print(f"\n{PROJECTION_YEAR} top 25:")
    for r in ratings.head(25).itertuples():
        print(f"  {r.rank:>3}. {r.team:<24} power={r.power:.3f}  "
              f"vs_avg={r.vs_average:.3f}")
    print(f"\nSaved {ARTIFACTS / 'model_v4.json'}")
    print(f"Saved {SELECTION_PATH}")
    print(f"Saved {RATINGS_PATH}")
    print(f"Saved {STATE_PATH}")


if __name__ == "__main__":
    main()
