"""Fit and publish the validated four-member expanded-WAR live ensemble.

Every member is fitted exactly as scripts/prior_decay_backtest.py fits its outer
fold, with the training pool set to every completed season: model knobs, the score
step and the EWMA half-life are tuned forward inside the pool, and the logistic
stack is trained on cross-fitted earlier-season rows.

    python -m scripts.train_live_ensemble              # production fit -> artifacts/
    python -m scripts.train_live_ensemble --check 2025 # refit on seasons before 2025
                                                       # and replay 2025 through the
                                                       # published runtime
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, PROJECTION_YEAR
from scripts import ensemble_replay as ER
from scripts import prior_decay_backtest as PD
from scripts import v4_backtest as BT
from scripts.train_v4 import build_frames, build_inputs
from src import live_ensemble as LE
from src import v4 as V4
from src.data import player_production


SPECS = ["full_new_war", "full_new_war_curvature",
         "full_new_war_production", "full_new_war_curvature_production"]
ARM = "ewma_elo_nodecay"
# PFF season-to-date offence and defence composites (src/pff_form.py). Each member
# carries a second stack with them; the runtime uses it whenever the week's PFF table
# exists, and the base stack otherwise. See audit/V5_EXTENSION_EXPERIMENTS.md.
PFF_COLUMNS = ["pff_O_diff", "pff_D_diff"]
KEYS = ["season", "week", "home_team", "away_team"]


def pff_features(parts_by_spec) -> pd.DataFrame:
    from src import pff_form
    games = pd.concat([part[4].assign(season=year)[KEYS]
                       for year, part in parts_by_spec[SPECS[0]].items()],
                      ignore_index=True)
    return pff_form.build_pff_form(games)[KEYS + PFF_COLUMNS]


def attach_extra(frames: dict[int, pd.DataFrame]) -> None:
    """The curvature and production columns, built as the backtest builds them."""
    components = player_production.by_year({year: frame.index
                                            for year, frame in frames.items()})
    for year, frame in frames.items():
        extra = components.get(year, pd.DataFrame(index=frame.index)).reindex(frame.index)
        frame["player_prod_war"] = extra.get(
            "player_prod_war", pd.Series(index=frame.index, dtype=float)).fillna(0.0)
        frame["talent_sq"] = frame.talent ** 2
        frame["returning_sq"] = frame.returning ** 2
        frame["talent_x_returning"] = frame.talent * frame.returning


def build(include_projection: bool = True, backtest_scaling: bool = False):
    std, talent, returning, games, od, pff_lag, war_lag, projection_meta = \
        build_inputs(include_projection=include_projection)
    years = [*GAME_YEARS, PROJECTION_YEAR] if include_projection else list(GAME_YEARS)
    frames = build_frames(std, talent, returning, od, pff_lag, war_lag, years)
    if backtest_scaling:
        from config import PFF_API_DIR
        expanded = PD.load_expanded_war(PFF_API_DIR / "war_all_facets",
                                        {y: f.index for y, f in frames.items()})
        for year, frame in frames.items():
            if year in expanded:
                frame["war_projected"] = expanded[year].reindex(frame.index).fillna(0.0)
    attach_extra(frames)
    parts_by_spec = {spec: V4.assemble(GAME_YEARS, frames, games,
                                       PD.SPEC_FEATURES[spec]) for spec in SPECS}
    raw = {year: PD.raw_game_stats(year) for year in GAME_YEARS}
    return frames, parts_by_spec, raw, projection_meta


def fit_member(spec: str, frames, parts, raw, pool: list[int],
               pff: pd.DataFrame | None = None) -> tuple[V4.ReciprocalTeamModel, dict]:
    names = PD.SPEC_FEATURES[spec]
    knobs, knob_trace = BT.tune(parts, pool, names)
    score_k, k_trace = PD.tune_k(parts, frames, pool, names, knobs)
    contexts = {}
    for index in range(1, len(pool)):
        validation, train = pool[index], pool[:index]
        model, _, _ = BT.fit_predict(parts, train, validation, names, knobs)
        contexts[validation] = (model, frames[validation], parts[validation])
    (current_half, prior_half, c), decay_trace = PD.select_decay(
        contexts, raw, ARM, score_k)
    designs = [PD.season_design(model, old_frame, part, raw[year], current_half,
                                prior_half, score_k).assign(season=year)
               for year, (model, old_frame, part) in contexts.items()]
    pooled = pd.concat(designs, ignore_index=True)
    scaler, stack = PD.fit_stack(pooled, PD.ARM_COLUMNS[ARM], c)
    stack_pff = None
    if pff is not None:
        with_pff = pooled.merge(pff, on=KEYS, how="left").fillna(
            {col: 0.0 for col in PFF_COLUMNS})
        cols = [*PD.ARM_COLUMNS[ARM], *PFF_COLUMNS]
        s2, m2 = PD.fit_stack(with_pff, cols, c)
        stack_pff = {"columns": cols, "scale": s2.scale_.tolist(),
                     "coef": m2.coef_[0].tolist(), "C": float(c)}
    X, y, h, margins = BT.stack(parts, pool)
    final = V4.fit(X, y, h, margins, names, **knobs)
    entry = {
        "name": spec, "weight": 1.0 / len(SPECS), "features": names,
        "model": LE.model_payload(final), "score_k": float(score_k),
        "current_halflife": None if not np.isfinite(current_half) else float(current_half),
        "stack": {"columns": PD.ARM_COLUMNS[ARM], "scale": scaler.scale_.tolist(),
                  "coef": stack.coef_[0].tolist(), "C": float(c)},
        "stack_pff": stack_pff,
        "selection": {"training_seasons": list(pool), "model_tuning": knob_trace,
                      "score_k_tuning": k_trace, "current_form_tuning": decay_trace},
    }
    print(f"fitted {spec} on {pool[0]}-{pool[-1]}: k={score_k}, "
          f"EWMA half-life={current_half}, C={c}")
    return final, entry


def fit_manifest(frames, parts_by_spec, raw, pool, projection_meta=None) -> dict:
    pff = pff_features(parts_by_spec)
    members = [fit_member(spec, frames, parts_by_spec[spec], raw, pool, pff)[1]
               for spec in SPECS]
    return {"schema_version": 3, "model_version": "5.1",
            "architecture": "equal_ensemble_expanded_war_score_innovation_ewma",
            "training_seasons": list(pool),
            "temporal_contract": "pregame only; current-season transforms use prior weeks",
            "projection": projection_meta or {}, "members": members}


def check(season: int, backtest_scaling: bool = False) -> None:
    """Refit on the seasons before ``season`` and replay it through ER.

    The members are the same fits prior_decay_backtest makes for that outer fold,
    so the replayed probabilities must equal its ensemble column.  This is the
    end-to-end test of training, the published block, and the stdlib runtime.

    One input differs by construction.  build_frames standardises war_projected
    across every team with a CFBD talent score before narrowing to the FBS frame;
    the backtest swapped its expanded column in standardised over the frame alone.
    ``backtest_scaling`` reproduces the backtest's scaling, so the comparison is
    exact; without it the check reports what the production scaling changes.
    """
    frames, parts_by_spec, raw, _ = build(include_projection=False,
                                          backtest_scaling=backtest_scaling)
    pool = [year for year in GAME_YEARS if year < season]
    manifest = fit_manifest(frames, parts_by_spec, raw, pool)
    frame = frames[season]
    block = LE.ensemble_block(manifest, frame, frame)
    games = parts_by_spec[SPECS[0]][season][4]
    finals = [{"week": int(g.week), "season_type": "regular", "home": g.home_team,
               "away": g.away_team, "neutral": bool(g.neutral_site),
               "home_score": 0, "away_score": 0} for g in games.itertuples()]
    margins = parts_by_spec[SPECS[0]][season][3]
    for row, margin in zip(finals, margins):
        row["home_score"] = float(margin)
    stats = raw[season]
    rows = [dict(zip(ER.FORM_FIELDS, values)) for values in
            stats[list(ER.FORM_FIELDS)].itertuples(index=False, name=None)]
    events = ER.replay(block, finals, rows)["events"]
    got = pd.DataFrame([{"week": e["week"], "home_team": e["home"],
                         "away_team": e["away"], "p_runtime": e["p_home"]}
                        for e in events])
    want = pd.read_csv(PD.OUT_ENSEMBLE_CSV)
    want = want[want.season == season]
    joined = want.merge(got, on=["week", "home_team", "away_team"], how="inner")
    diff = (joined.expanded_equal_ensemble - joined.p_runtime).abs()
    print(f"{season}: {len(joined)} of {len(want)} backtest games matched; "
          f"max |p_backtest - p_runtime| = {diff.max():.3e}")
    y = joined.y.to_numpy(float)
    print(f"  Brier backtest={np.mean((joined.expanded_equal_ensemble - y) ** 2):.6f}"
          f"  runtime={np.mean((joined.p_runtime - y) ** 2):.6f}")

    # v5.1: the same replay with PFF, the table built by the CI's stdlib composite
    # from the staged weekly CSVs, against the harness's pandas-built predictions.
    pff = staged_pff_payload(season, list(frame.index))
    events = ER.replay(block, finals, rows, pff=pff)["events"]
    got = pd.DataFrame([{"week": e["week"], "home_team": e["home"],
                         "away_team": e["away"], "p_runtime": e["p_home"]}
                        for e in events])
    from config import ARTIFACTS
    harness = pd.read_csv(ARTIFACTS / "v5_extension_predictions.csv")
    harness = harness[(harness.variant == "pff_outcome_composite") &
                      (harness.season == season)]
    if harness.empty:
        print("  (run v5_extension_backtest --only pff_outcome_composite to compare PFF)")
        return
    joined = harness.merge(got, on=["week", "home_team", "away_team"], how="inner")
    diff = (joined.p - joined.p_runtime).abs()
    print(f"{season} with PFF: {len(joined)} of {len(harness)} harness games matched; "
          f"max |p_harness - p_runtime| = {diff.max():.3e}")
    y = joined.y.to_numpy(float)
    print(f"  Brier harness={np.mean((joined.p - y) ** 2):.6f}"
          f"  runtime={np.mean((joined.p_runtime - y) ** 2):.6f}")


def staged_pff_payload(season: int, teams) -> dict:
    """{cutoffs: {W: {team: [O, D]}}} from the staged weekly tables, via the CI code."""
    from src import pff_form
    directory = pd.read_csv(pff_form.SEASON / f"team_directory_{season}.csv",
                            low_memory=False)[["franchise_id", "city"]].to_dict("records")
    cutoffs = {}
    for thru in range(1, 16):
        paths = [pff_form.WEEKLY / f"{c.replace('-', '_')}_{season}_thru_w{thru:02d}.csv"
                 for c in ER.PFF_CATEGORIES]
        if not all(p.exists() for p in paths):
            continue
        tables = [pd.read_csv(p, low_memory=False).to_dict("records") for p in paths]
        cutoffs[str(thru)] = ER.pff_composite(directory, *tables, teams)
    return {"season": season, "cutoffs": cutoffs}


def main():
    frames, parts_by_spec, raw, projection_meta = build(include_projection=True)
    manifest = fit_manifest(frames, parts_by_spec, raw, list(GAME_YEARS),
                            projection_meta)
    frame = frames[PROJECTION_YEAR]
    frame.to_csv(LE.FRAME_PATH, index_label="team")
    LE.save_manifest(manifest)
    print(f"-> {LE.MANIFEST_PATH}\n-> {LE.FRAME_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=int, help="validate one completed season")
    parser.add_argument("--backtest-scaling", action="store_true",
                        help="with --check, standardise WAR as the backtest did")
    args = parser.parse_args()
    check(args.check, args.backtest_scaling) if args.check else main()
