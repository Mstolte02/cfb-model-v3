"""Historical, leakage-safe game-state materialization for Fourth & Jev.

This module reuses the validated v5/v5.1 training and replay machinery. For an outer
season N it fits only on seasons < N, reconstructs each week's state before that
week's games, and emits one market-blind Fourth & Jev state per matchup.

The historical model's temporal unit is a week. If a cached CFBD games payload has an
actual kickoff timestamp we attach it as metadata; otherwise the honest as-of label is
"SEASON-WEEK-pregame", not an invented timestamp.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterator

import pandas as pd

from config import DATA_RAW, GAME_YEARS
from fourth_jev.state import build_game_state
from scripts import ensemble_replay as ER
from scripts import train_live_ensemble as T
from src import live_ensemble as LE


_NORMAL = statistics.NormalDist()


def cfb_distribution(home_win_probability: float, margin_sigma: float) -> dict:
    """Coherent CFB-model margin baseline for the Jev questions.

    The live model publishes its margin by reading the win probability through the
    same probit link used by the rating update.  Tail events use half-point boundaries
    because realized college-football margins are integers.  This is deliberately a
    simple Normal benchmark, not a claim that the score distribution has Normal tails.
    """
    p = min(max(float(home_win_probability), 1e-8), 1 - 1e-8)
    sigma = float(margin_sigma)
    margin = sigma * _NORMAL.inv_cdf(p)

    def above(integer_margin: int) -> float:
        boundary = integer_margin - 0.5
        return 1.0 - _NORMAL.cdf((boundary - margin) / sigma)

    def below(integer_margin: int) -> float:
        boundary = integer_margin + 0.5
        return _NORMAL.cdf((boundary - margin) / sigma)

    return {
        "predicted_margin": margin,
        "margin_sigma": sigma,
        "probabilities": {
            "home_win": p,
            "home_by_7_plus": above(7),
            "home_by_14_plus": above(14),
            "home_by_21_plus": above(21),
            "away_by_7_plus": below(-7),
            "within_3": (_NORMAL.cdf((3.5 - margin) / sigma)
                         - _NORMAL.cdf((-3.5 - margin) / sigma)),
        },
        "tail_assumption": "Normal margin; integer events use half-point boundaries",
    }


def _json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def team_frame_block(frame: pd.DataFrame, team: str) -> dict:
    """All numeric/string preseason features already present in the production frame."""
    row = frame.loc[team]
    return {str(k): _json_scalar(v) for k, v in row.items()}


def _raw_kickoffs(season: int) -> dict[tuple[int, str, str], dict]:
    """Optional metadata from CFBD's cached raw game response."""
    path = DATA_RAW / f"games_{season}.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for game in raw:
        key = (int(game.get("week") or 0), game.get("homeTeam"), game.get("awayTeam"))
        out[key] = {
            "id": game.get("id"),
            "start_date": game.get("startDate"),
            "venue": game.get("venue"),
            "conference_game": game.get("conferenceGame"),
            "season_type": game.get("seasonType"),
        }
    return out


def _finals(part) -> list[dict]:
    meta, margins = part[4], part[3]
    rows = []
    for game, margin in zip(meta.itertuples(), margins):
        # The replay only needs margin. Encoding it as home_score with away_score=0
        # preserves the exact update used by the validated runtime.
        rows.append({
            "week": int(game.week),
            "season_type": "regular",
            "home": game.home_team,
            "away": game.away_team,
            "neutral": bool(game.neutral_site),
            "home_score": float(margin),
            "away_score": 0.0,
            "source_index": int(game.source_index),
        })
    return rows


def _form_rows(raw: pd.DataFrame) -> list[dict]:
    return [
        dict(zip(ER.FORM_FIELDS, values))
        for values in raw[list(ER.FORM_FIELDS)].itertuples(index=False, name=None)
    ]


def _member_dynamic_block(ensemble: dict, state: dict, team: str) -> dict:
    out = {}
    for member in ensemble["members"]:
        name = member["name"]
        initial = float(member["initial"][team])
        current = float(state["ratings"][name][team])
        form = (state.get("form", {}).get(name) or {}).get(team)
        out[name] = {
            "preseason_logit_strength": initial,
            "current_logit_strength": current,
            "rating_change": current - initial,
            "form_O": float(form[0]) if form else None,
            "form_D": float(form[1]) if form else None,
            "form_games": int(form[2]) if form else 0,
        }
    pff = (state.get("pff") or {}).get(team)
    return {
        "ensemble_members": out,
        "pff_current_offense": float(pff[0]) if pff else None,
        "pff_current_defense": float(pff[1]) if pff else None,
    }


def build_outer_context(season: int, *, with_pff: bool = True) -> dict:
    """Fit the exact historical outer-fold ensemble for one season."""
    if season not in GAME_YEARS or season < 2023:
        raise ValueError(f"historical v5/v5.1 outer fold needs 2023-{max(GAME_YEARS)}")
    frames, parts_by_spec, raw, _ = T.build(include_projection=False, backtest_scaling=False)
    pool = [year for year in GAME_YEARS if year < season]
    manifest = T.fit_manifest(frames, parts_by_spec, raw, pool)
    frame = frames[season]
    ensemble = LE.ensemble_block(manifest, frame, frame)
    pff = T.staged_pff_payload(season, list(frame.index)) if with_pff else None
    return {
        "season": season,
        "training_seasons": pool,
        "frames": frames,
        "parts": parts_by_spec,
        "raw": raw,
        "manifest": manifest,
        "ensemble": ensemble,
        "pff": pff,
    }


def iter_game_states(context: dict) -> Iterator[dict]:
    season = int(context["season"])
    frame = context["frames"][season]
    part = context["parts"][T.SPECS[0]][season]
    finals = _finals(part)
    form_rows = _form_rows(context["raw"][season])
    ensemble = context["ensemble"]
    pff = context["pff"]
    kickoff = _raw_kickoffs(season)

    by_week: dict[int, list[dict]] = {}
    for game in finals:
        by_week.setdefault(int(game["week"]), []).append(game)

    for week in sorted(by_week):
        # stop_before returns the state at the start of this slate, based only on
        # completed earlier slates. This is the same temporal contract as production.
        pre = ER.replay(ensemble, finals, form_rows, stop_before=week, pff=pff)["state"]
        for game in by_week[week]:
            home, away = game["home"], game["away"]
            hfa = 0.0 if game["neutral"] else 1.0
            model_p = ER.probability(ensemble, pre, home, away, hfa)
            cfb_baseline = cfb_distribution(model_p, ensemble["margin_sigma"])
            key = (week, home, away)
            raw_meta = kickoff.get(key, {})
            game_id = raw_meta.get("id") or f"{season}-{week}-{home}-vs-{away}"
            # The state is assembled at the start of the game week. A cached
            # kickoff timestamp is game metadata, not this feature cutoff.
            as_of = f"{season}-W{week:02d}-pregame"

            home_block = {
                "preseason": team_frame_block(frame, home),
                "current": _member_dynamic_block(ensemble, pre, home),
            }
            away_block = {
                "preseason": team_frame_block(frame, away),
                "current": _member_dynamic_block(ensemble, pre, away),
            }
            state = build_game_state(
                game={
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "neutral": bool(game["neutral"]),
                    "venue": raw_meta.get("venue"),
                    "conference_game": raw_meta.get("conference_game"),
                    "kickoff": raw_meta.get("start_date"),
                },
                home=home_block,
                away=away_block,
                model={
                    "architecture": context["manifest"].get("architecture"),
                    "model_version": context["manifest"].get("model_version"),
                    "home_win_probability": model_p,
                    "predicted_margin": cfb_baseline["predicted_margin"],
                    "margin_sigma": cfb_baseline["margin_sigma"],
                    "baseline_probabilities": cfb_baseline["probabilities"],
                    "tail_assumption": cfb_baseline["tail_assumption"],
                    "training_seasons": context["training_seasons"],
                },
                context={
                    "state_cutoff_basis": "start of game week",
                    "current_week_information_uses": "weeks strictly before game week",
                },
                sources={
                    "production_frame": "scripts.train_live_ensemble.build",
                    "dynamic_replay": "scripts.ensemble_replay",
                    "current_form": "CFBD game advanced stats, prior weeks only",
                    "pff_form": "PFF staged weekly composites" if pff is not None else None,
                },
                as_of=as_of,
            )
            yield {
                "game_id": str(game_id),
                "as_of": as_of,
                "state": state,
                "outcome": {
                    "home_margin": float(game["home_score"] - game["away_score"]),
                    "home_win": bool(game["home_score"] > game["away_score"]),
                    "home_by_7_plus": bool(game["home_score"] - game["away_score"] >= 7),
                    "home_by_14_plus": bool(game["home_score"] - game["away_score"] >= 14),
                    "home_by_21_plus": bool(game["home_score"] - game["away_score"] >= 21),
                    "away_by_7_plus": bool(game["away_score"] - game["home_score"] >= 7),
                    "within_3": bool(abs(game["home_score"] - game["away_score"]) <= 3),
                },
            }


def materialize_season(season: int, output: str | Path, *, with_pff: bool = True) -> int:
    context = build_outer_context(season, with_pff=with_pff)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as fh:
        for row in iter_game_states(context):
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count
