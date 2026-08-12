"""Leakage-safe pace, game-script, and quick-pass matchup features.

Drive summaries are converted to rolling pregame traits.  All games in a week see
the same start-of-week history, matching the live rating state's temporal contract.
Season N quarterback style comes only from N-1 PFF passing rows and N-1 TruMedia
team pressure/style summaries.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_RAW, GAME_YEARS, PFF_DIR
from src.data import cfbd_client, trumedia
from src.data.pff import TEAM_MAP


KEYS = ["season", "week", "home_team", "away_team"]
ROLLING_GAMES = 12
NEUTRAL_SECONDS_PER_PLAY = 26.5
FAST_SECONDS_PER_PLAY = 25.0
SLOW_SECONDS_PER_PLAY = 28.0

PACE_IDENTITY = ["pace_off_diff", "pace_def_diff", "plays_drive_diff"]
SCRIPT_WINDOWS = [
    "script_ypp_diff", "script_points_diff", "middle8_net_diff", "q4_net_diff",
]
STATE_CONTROL = [
    "trail_win_diff", "lead_hold_diff", "pace_control_diff",
]
PACE_MATCHUP = [
    "pace_style_fit_edge", "pace_control_clash_edge", "script_clash_edge",
]
QUICK_PRESSURE = [
    "qb_quickness_diff", "early_pass_diff", "quick_vs_pressure_edge",
    "quick_vs_blitz_edge", "pressure_exposure_edge",
    "deep_time_pressure_edge", "sack_conversion_edge", "quick_efficiency_edge",
]


def _clock_seconds(value) -> float:
    value = value or {}
    return float((value.get("minutes") or 0) * 60 + (value.get("seconds") or 0))


def _points(drive: dict) -> float:
    start = drive.get("startOffenseScore")
    end = drive.get("endOffenseScore")
    if start is None or end is None:
        return 0.0
    return float(np.clip(float(end) - float(start), 0.0, 8.0))


def _real_drives(drives: list[dict]) -> list[dict]:
    excluded = {"END OF GAME", "END OF HALF", "END OF 4TH QUARTER"}
    return [d for d in drives if int(d.get("plays") or 0) > 0
            and int(d.get("startPeriod") or 0) <= 4
            and str(d.get("driveResult") or "").upper() not in excluded]


def _pace_totals(drives: list[dict]) -> tuple[float, float]:
    seconds = plays = 0.0
    for d in drives:
        n = float(d.get("plays") or 0)
        elapsed = _clock_seconds(d.get("elapsed"))
        # Bad provider rows occasionally encode a quarter break as a 30-60 minute
        # drive.  A football drive's own clock consumption cannot exceed a quarter,
        # and extreme single-drive rates do not represent play-clock preference.
        spp = elapsed / n if n else np.nan
        if n >= 2 and 8.0 <= spp <= 60.0 and elapsed <= 900.0:
            seconds += elapsed
            plays += n
    return seconds, plays


def _drive_window(drives: list[dict], predicate) -> tuple[float, float, float]:
    selected = [d for d in drives if predicate(d)]
    plays = float(sum(int(d.get("plays") or 0) for d in selected))
    yards = float(sum(float(d.get("yards") or 0) for d in selected))
    points = float(sum(_points(d) for d in selected))
    return points, yards / plays if plays else np.nan, float(len(selected))


def _team_game(drives: list[dict], team: str, won: bool) -> dict:
    offense = sorted(_real_drives([d for d in drives if d.get("offense") == team]),
                     key=lambda d: int(d.get("driveNumber") or 0))
    seconds, pace_plays = _pace_totals(offense)
    plays = float(sum(int(d.get("plays") or 0) for d in offense))
    first = []
    first_plays = 0
    for d in offense:
        if first_plays >= 15:
            break
        first.append(d)
        first_plays += int(d.get("plays") or 0)
    script_points, script_ypp, _ = _drive_window(first, lambda _: True)

    def middle_eight(d):
        period, clock = int(d.get("startPeriod") or 0), _clock_seconds(d.get("startTime"))
        return (period == 2 and clock <= 240) or (period == 3 and clock >= 660)

    middle_points, middle_ypp, _ = _drive_window(offense, middle_eight)
    q4_points, q4_ypp, _ = _drive_window(
        offense, lambda d: int(d.get("startPeriod") or 0) == 4)
    trailing = [d for d in offense
                if float(d.get("startOffenseScore") or 0) <
                float(d.get("startDefenseScore") or 0)]
    leading = [d for d in offense
               if float(d.get("startOffenseScore") or 0) >
               float(d.get("startDefenseScore") or 0)]
    q4 = [d for d in drives if int(d.get("startPeriod") or 0) == 4]
    q4_state = 0
    if q4:
        first_q4 = min(q4, key=lambda d: int(d.get("driveNumber") or 0))
        off_margin = (float(first_q4.get("startOffenseScore") or 0) -
                      float(first_q4.get("startDefenseScore") or 0))
        q4_state = int(np.sign(off_margin if first_q4.get("offense") == team
                               else -off_margin))
    return {
        "off_spp": (float(np.clip(seconds / pace_plays, 18.0, 40.0))
                    if pace_plays >= 15 else np.nan),
        "plays_drive": plays / len(offense) if offense else np.nan,
        "script_ypp": script_ypp,
        "script_points": script_points,
        "middle8_points": middle_points,
        "middle8_ypp": middle_ypp,
        "q4_points": q4_points,
        "q4_ypp": q4_ypp,
        "trail_ppd": (sum(_points(d) for d in trailing) / len(trailing)
                      if trailing else np.nan),
        "lead_ppd": (sum(_points(d) for d in leading) / len(leading)
                     if leading else np.nan),
        "q4_trail_op": float(q4_state < 0),
        "q4_trail_win": float(won and q4_state < 0),
        "q4_lead_op": float(q4_state > 0),
        "q4_lead_win": float(won and q4_state > 0),
        "won": float(won),
    }


def build_team_game_metrics(years=GAME_YEARS) -> dict[tuple, dict[str, dict]]:
    """Return matchup-keyed home/away postgame metrics from cached raw drives."""
    out = {}
    for year in years:
        games = json.loads((DATA_RAW / f"games_{year}.json").read_text())
        game_by_id = {int(g["id"]): g for g in games if g.get("completed")
                      and g.get("homePoints") is not None}
        grouped = defaultdict(list)
        for d in cfbd_client.drives(year):
            grouped[int(d["gameId"])].append(d)
        for game_id, drives in grouped.items():
            game = game_by_id.get(game_id)
            if not game:
                continue
            home, away = game["homeTeam"], game["awayTeam"]
            home_row = _team_game(drives, home,
                                  float(game["homePoints"]) > float(game["awayPoints"]))
            away_row = _team_game(drives, away,
                                  float(game["awayPoints"]) > float(game["homePoints"]))
            seconds_h, plays_h = _pace_totals(
                _real_drives([d for d in drives if d.get("offense") == home]))
            seconds_a, plays_a = _pace_totals(
                _real_drives([d for d in drives if d.get("offense") == away]))
            total_plays = plays_h + plays_a
            game_spp = (float(np.clip((seconds_h + seconds_a) / total_plays,
                                      18.0, 40.0))
                        if total_plays >= 30 else np.nan)
            for own, opp in ((home_row, away_row), (away_row, home_row)):
                own["opp_off_spp"] = opp["off_spp"]
                own["middle8_net"] = own["middle8_points"] - opp["middle8_points"]
                own["q4_net"] = own["q4_points"] - opp["q4_points"]
                own["game_spp"] = game_spp
            key = (int(year), int(game.get("week") or 0), home, away)
            out[key] = {"home": home_row, "away": away_row}
    return out


def _mean(rows: list[dict], name: str, default=0.0) -> float:
    values = np.asarray([r.get(name, np.nan) for r in rows], float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float(default)


def _conditional_win(rows: list[dict], field: str, predicate) -> float:
    chosen = [r for r in rows if np.isfinite(r.get(field, np.nan)) and predicate(r)]
    wins = sum(float(r[field]) for r in chosen)
    return float((wins + 2.0) / (len(chosen) + 4.0))


def _summary(history: list[dict]) -> dict[str, float]:
    rows = history[-ROLLING_GAMES:]
    trail_op = sum(r.get("q4_trail_op", 0.0) for r in rows)
    lead_op = sum(r.get("q4_lead_op", 0.0) for r in rows)
    return {
        "off_spp": _mean(rows, "off_spp", NEUTRAL_SECONDS_PER_PLAY),
        "def_spp": _mean(rows, "opp_off_spp", NEUTRAL_SECONDS_PER_PLAY),
        "plays_drive": _mean(rows, "plays_drive", 6.0),
        "script_ypp": _mean(rows, "script_ypp", 5.0),
        "script_points": _mean(rows, "script_points", 3.0),
        "middle8_net": _mean(rows, "middle8_net"),
        "q4_net": _mean(rows, "q4_net"),
        "trail_win": ((sum(r.get("q4_trail_win", 0.0) for r in rows) + 2.0) /
                      (trail_op + 4.0)),
        "lead_hold": ((sum(r.get("q4_lead_win", 0.0) for r in rows) + 2.0) /
                      (lead_op + 4.0)),
        "fast_win": _conditional_win(
            rows, "won", lambda r: r.get("game_spp", 99) <= FAST_SECONDS_PER_PLAY),
        "slow_win": _conditional_win(
            rows, "won", lambda r: r.get("game_spp", 0) >= SLOW_SECONDS_PER_PLAY),
        "pace_control": _mean(rows, "pace_control"),
    }


def _matchup_row(key: tuple, home: dict, away: dict) -> dict:
    expected = np.mean([home["off_spp"], away["off_spp"],
                        home["def_spp"], away["def_spp"]])
    fast_weight = 1.0 / (1.0 + math.exp((expected - 26.5) / 1.5))
    home_fit = fast_weight * home["fast_win"] + (1-fast_weight) * home["slow_win"]
    away_fit = fast_weight * away["fast_win"] + (1-fast_weight) * away["slow_win"]
    clash = min(abs(home["off_spp"] - away["off_spp"]), 10.0)
    return {
        **dict(zip(KEYS, key)),
        "pace_off_diff": home["off_spp"] - away["off_spp"],
        "pace_def_diff": home["def_spp"] - away["def_spp"],
        "plays_drive_diff": home["plays_drive"] - away["plays_drive"],
        "script_ypp_diff": home["script_ypp"] - away["script_ypp"],
        "script_points_diff": home["script_points"] - away["script_points"],
        "middle8_net_diff": home["middle8_net"] - away["middle8_net"],
        "q4_net_diff": home["q4_net"] - away["q4_net"],
        "trail_win_diff": home["trail_win"] - away["trail_win"],
        "lead_hold_diff": home["lead_hold"] - away["lead_hold"],
        "pace_control_diff": home["pace_control"] - away["pace_control"],
        "pace_style_fit_edge": home_fit - away_fit,
        "pace_control_clash_edge": clash * (home["pace_control"] -
                                             away["pace_control"]),
        "script_clash_edge": clash * (home["script_ypp"] - away["script_ypp"]),
    }


def build_rolling_drive_features(years=GAME_YEARS) -> pd.DataFrame:
    """Start-of-week snapshots for every game with drive coverage."""
    metrics = build_team_game_metrics(years)
    history = defaultdict(list)
    rows = []
    by_week = defaultdict(list)
    for key in metrics:
        by_week[(key[0], key[1])].append(key)
    for season_week in sorted(by_week):
        pending = []
        for key in sorted(by_week[season_week], key=lambda k: (k[2], k[3])):
            _, _, home_team, away_team = key
            home = _summary(history[home_team])
            away = _summary(history[away_team])
            rows.append(_matchup_row(key, home, away))
            actual = metrics[key]
            actual_spp = actual["home"].get("game_spp", np.nan)
            gap = abs(home["off_spp"] - away["off_spp"])
            if np.isfinite(actual_spp) and gap >= 1.0:
                control = ((abs(actual_spp - away["off_spp"]) -
                            abs(actual_spp - home["off_spp"])) / gap)
                control = float(np.clip(control, -1.0, 1.0))
            else:
                control = 0.0
            actual["home"]["pace_control"] = control
            actual["away"]["pace_control"] = -control
            pending.append((home_team, actual["home"]))
            pending.append((away_team, actual["away"]))
        # Never let an early game in the provider's arbitrary same-week order leak
        # into another prediction from that slate.
        for team, game in pending:
            history[team].append(game)
    return pd.DataFrame(rows).drop_duplicates(KEYS)


def _weighted(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    weights = pd.to_numeric(group["dropbacks"], errors="coerce").fillna(0.0)
    ok = values.notna() & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan


def build_entering_quick_pressure(years=GAME_YEARS) -> dict[int, pd.DataFrame]:
    """N-1 team quick-pass and pressure style, standardized within entering N."""
    out = {}
    tm_all = trumedia.load()
    pff_fields = ["avg_time_to_throw", "avg_depth_of_target",
                  "pressure_to_sack_rate", "positive_epa_percent"]
    for year in years:
        prior = year - 1
        path = Path(PFF_DIR) / f"passing_{prior}.csv"
        passing = pd.read_csv(path)
        passing["team"] = passing.team_name.map(TEAM_MAP)
        passing = passing.dropna(subset=["team"])
        pff_rows = []
        for team, group in passing.groupby("team"):
            pff_rows.append({"team": team, **{f: _weighted(group, f)
                                              for f in pff_fields}})
        pff = pd.DataFrame(pff_rows).set_index("team")
        tm = tm_all[tm_all.season == prior].set_index("team")[[
            "press_allowed", "press_gen", "early_pass_rate", "blitz_pg"]]
        frame = pff.join(tm, how="outer")
        rename = {
            "avg_time_to_throw": "qb_ttt", "avg_depth_of_target": "qb_adot",
            "pressure_to_sack_rate": "qb_pressure_sack",
            "positive_epa_percent": "qb_positive_epa",
        }
        frame = frame.rename(columns=rename)
        for col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            scale = float(values.std(ddof=0)) or 1.0
            frame[col] = (values - values.mean()) / scale
        frame["qb_quickness"] = -frame.qb_ttt
        out[year] = frame.fillna(0.0)
    return out


def attach_quick_pressure(data: pd.DataFrame) -> pd.DataFrame:
    styles = build_entering_quick_pressure(sorted(data.season.unique()))
    rows = []
    zero = pd.Series(dtype=float)
    for r in data.itertuples():
        frame = styles[int(r.season)]
        home = frame.loc[r.home_team] if r.home_team in frame.index else zero
        away = frame.loc[r.away_team] if r.away_team in frame.index else zero

        rows.append(_quick_pressure_row(home, away))
    return pd.concat([data.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _quick_pressure_row(home: pd.Series, away: pd.Series) -> dict[str, float]:
    def value(row, name):
        return float(row.get(name, 0.0))

    hq, aq = value(home, "qb_quickness"), value(away, "qb_quickness")
    hp, ap = value(home, "press_gen"), value(away, "press_gen")
    hb, ab = value(home, "blitz_pg"), value(away, "blitz_pg")
    return {
        "qb_quickness_diff": hq - aq,
        "early_pass_diff": (value(home, "early_pass_rate") -
                            value(away, "early_pass_rate")),
        "quick_vs_pressure_edge": hq * ap - aq * hp,
        "quick_vs_blitz_edge": hq * ab - aq * hb,
        "pressure_exposure_edge": (value(home, "press_allowed") * ap -
                                   value(away, "press_allowed") * hp),
        "deep_time_pressure_edge": (value(home, "qb_adot") * ap -
                                    value(away, "qb_adot") * hp),
        "sack_conversion_edge": (value(home, "qb_pressure_sack") * ap -
                                 value(away, "qb_pressure_sack") * hp),
        "quick_efficiency_edge": (hq * value(home, "qb_positive_epa") -
                                  aq * value(away, "qb_positive_epa")),
    }
