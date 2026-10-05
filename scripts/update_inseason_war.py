"""Publish in-season player WAR to viz/data/players_inseason.json.

    python -m scripts.update_inseason_war                 # weeks 1..last completed week
    python -m scripts.update_inseason_war --week 3        # explicit cutoff
    python -m scripts.update_inseason_war --refit-params  # re-fit the rule from the backtest

Pulls the ten PFF reports for weeks 1..W of 2026 (re-pulled every run, because PFF
keeps charting late games), runs them through the production WAR facet path, and
applies the rule in src/inseason_war.py. Needs PFF_API_KEY and the local PFF source
files, so it runs on Mark's machine, not in the six-hourly GitHub job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from config import ROOT  # noqa: E402
from src import inseason_war as IW  # noqa: E402

SEASON = 2026


def last_completed_week() -> int:
    games = json.loads((ROOT / "viz" / "data" / "schedule.json").read_text())
    tot, fin = defaultdict(int), defaultdict(int)
    for g in games:
        if g.get("st", "regular") != "regular" or g.get("w") is None:
            continue
        tot[g["w"]] += 1
        fin[g["w"]] += g.get("f", 0)
    week = 0
    for w in sorted(tot):
        if fin[w] < tot[w]:
            break
        week = w
    return week


def ensure_key():
    if os.environ.get("PFF_API_KEY") or os.name != "nt":
        return
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        os.environ["PFF_API_KEY"] = winreg.QueryValueEx(k, "PFF_API_KEY")[0]


def refit_params():
    from scripts import war_inseason_backtest as B
    from scripts import playing_time_backtest as PT
    hist = B.history()
    fr = B.frame(hist)
    # Opponent slopes first, then the update rule on schedule-neutral rates.
    opp_beta = B.fit_opponent_beta(fr)
    fr = B.apply_opponent(fr, opp_beta)
    pr26 = B.priors(hist, SEASON)
    war = pd.read_csv(ROOT / "war_model" / "hybrid_player_war.csv")
    war = war[(war.season == SEASON - 1) & (war.snaps >= 100)]
    war["group"] = war.position.map(__import__("src.war_window", fromlist=["GROUP"]).GROUP)
    repl = ((war.war - war.waa) / war.snaps * 1000).groupby(war.group).median()
    params = {
        "fitted_on": sorted(int(s) for s in fr.season.unique()),
        "cuts": sorted(int(c) for c in fr.cut.unique()),
        "rule": IW.fit_rule(fr),
        "k": IW.war_scale(hist),
        "full_time_snaps": IW.full_time_snaps(hist),
        "mu_2026": pr26.groupby("group").mu.first().to_dict(),
        "tau2_2026": pr26.groupby("group").tau2.first().to_dict(),
        "opp_beta": opp_beta,
        "full_time_per_game": PT.full_time_per_game(PT.weekly(SEASON - 1)),
        "repl_per_1000": repl.to_dict(),
        "pt_n0": 1.0,
        # Validated deployment adjustments (scripts/deployment_war_backtest.py) are
        # fitted there, not here; a refit carries them forward.
        "deployment": (json.loads(IW.PARAMS.read_text()).get("deployment")
                       if IW.PARAMS.exists() else None),
    }
    IW.PARAMS.write_text(json.dumps(params, indent=1))
    print(f"-> {IW.PARAMS}")
    return params


def write_team_tables(params: dict, week: int, history: bool, skip_pull: bool):
    """The v5.2 model column's inputs: team WAR change per week at cuts 3/6/9.

    ``history`` rebuilds data/live/inseason_war_team_history.csv (2022-25, the
    training rows) from the staged windows. The 2026 payload is rewritten every run
    with every model cut already reached; the latest cut is re-pulled by main(), and
    an earlier cut's window is pulled once if it is missing and then left alone.
    """
    from scripts import sync_pff_war_windows as SW
    from scripts import war_inseason_backtest as B
    hist = B.history()
    rule, k = params["rule"], params["k"]
    if history:
        parts = []
        for s in (2022, 2023, 2024, 2025):
            pr = B.priors(hist, s)
            mu = pr.groupby("group").mu.first().to_dict()
            for c in IW.MODEL_CUTS:
                parts.append(IW.window_rows(s, c, pr, mu,
                                            opp_beta=params.get("opp_beta")))
        D = IW.team_deltas(pd.concat(parts, ignore_index=True), rule, k)
        D.to_csv(IW.TEAM_HISTORY, index=False)
        print(f"-> {IW.TEAM_HISTORY} ({len(D)} team-cuts)")
    pr = B.priors(hist, SEASON)
    mu = pr.groupby("group").mu.first().to_dict()
    cutoffs = {}
    # Keep the fitted checkpoints for historical replay, plus the latest completed
    # week so current power ratings never wait three weeks for a roster change.
    publish_cuts = sorted(set(c for c in IW.MODEL_CUTS if c <= week) | {week})
    # Cuts before FROZEN_BEFORE_CUT fed games that are already graded. They keep the
    # values they were published with; the October 2026 method (pooled WAR, opponent
    # adjustment, sample-weighted update, 2026 playing time) applies from there on.
    old = {}
    if IW.team_payload_path(SEASON).exists():
        old = json.loads(IW.team_payload_path(SEASON).read_text()).get("cutoffs", {})
    for c in publish_cuts:
        if c < IW.FROZEN_BEFORE_CUT and str(c) in old:
            cutoffs[str(c)] = old[str(c)]
            continue
        d = SW.window_dir(SEASON, 1, c)
        if not skip_pull and len(list(d.glob("*.csv"))) < 10:
            SW.main([SEASON], [(1, c)])
        rows = IW.window_rows(SEASON, c, pr, mu, weight_season=SEASON - 1,
                              opp_beta=params.get("opp_beta"))
        D = IW.team_deltas(rows, rule, k)
        cutoffs[str(c)] = {r.team: round(float(r.D), 8) for r in D.itertuples()}
    # Availability is intentionally current-only. Applying today's injury news to an
    # old cut would rewrite the model's pregame view of already completed games.
    from src.data import war
    injury = IW.availability_team_deltas(war.player_contributions())
    latest = cutoffs[str(week)]
    for team, delta in injury.items():
        latest[team] = round(latest.get(team, 0.0) + delta, 8)
    # The 2026 playing-time term (payload "team_playing_time") is NOT added here. It
    # is in every player's WAR, but summed to a team it tracks blowouts: strong teams
    # rest starters, so their shares fall (Miami, Georgia, Texas A&M lowest at week 5)
    # and weak teams' rise, which says nothing about close games to come. D already
    # counts each player's actual 2026 snaps through the performance term.
    payload = {"season": SEASON, "through_week": int(week),
               "definition": "sum over players of k*(updated rate - calibrated prior)"
                             "*snaps per week/1000, weeks 1..cut; current cutoff also "
                             "removes unavailable WAR; rates are opponent-adjusted and "
                             "sample-weighted; src/inseason_war.py",
               "cutoffs": cutoffs}
    IW.team_payload_path(SEASON).write_text(json.dumps(payload, indent=1))
    print(f"-> {IW.team_payload_path(SEASON)} (cuts {sorted(int(c) for c in cutoffs)})")


def main(week: int | None, refit: bool, skip_pull: bool, history: bool = False):
    ensure_key()
    week = week if week is not None else last_completed_week()
    if week < 1:
        raise SystemExit("no completed week yet")
    params = refit_params() if refit or not IW.PARAMS.exists() else \
        json.loads(IW.PARAMS.read_text())

    from scripts.sync_pff_war_windows import LEGACY, POSITION, window_dir
    from scripts import sync_pff_war_windows as SW
    from scripts import war_inseason_backtest as B
    from src import war_window as ww
    from src.data import war

    d = window_dir(SEASON, 1, week)
    if not skip_pull:
        # PFF may fail partway through a refresh. Fetch a complete replacement
        # first, then replace cached reports; never erase the last valid window.
        SW.WINDOW_DIR.mkdir(parents=True, exist_ok=True)
        original_dir = SW.WINDOW_DIR
        with tempfile.TemporaryDirectory(prefix="war-refresh-", dir=original_dir) as temp:
            SW.WINDOW_DIR = Path(temp)
            try:
                SW.main([SEASON], [(1, week)])
            finally:
                SW.WINDOW_DIR = original_dir
            staged = Path(temp) / d.name
            required = [f"{name}.csv" for name in (*LEGACY, *POSITION)]
            missing_stage = [name for name in required if not (staged / name).exists()]
            if missing_stage:
                raise SystemExit(
                    f"PFF refresh incomplete through week {week}: "
                    f"missing {', '.join(missing_stage)}; cached and published WAR unchanged")
            d.mkdir(parents=True, exist_ok=True)
            for name in required:
                (staged / name).replace(d / name)
    files = {B.PREFIX[n]: d / f"{n}.csv" for n in (*LEGACY, *POSITION)}
    missing = [path.name for path in files.values() if not path.exists()]
    if missing:
        raise SystemExit(
            f"incomplete PFF window through week {week}: missing {', '.join(missing)}; "
            "published player WAR was not changed")
    players = ww.load_players_from(files, SEASON)
    fc = ww.facet_contrib(players, SEASON, weight_season=SEASON - 1)
    hist = B.history()
    priors = B.priors(hist, SEASON)
    roster = war.player_contributions()
    payload = IW.build(week, players, fc, priors, params, roster)
    IW.OUT.write_text(json.dumps(payload, allow_nan=False))
    print(f"-> {IW.OUT}: through week {week}, {payload['matched_players']} of "
          f"{payload['roster_players']} two-deep players matched to "
          f"{payload['pff_players']} PFF players")
    write_team_tables(params, week, history, skip_pull)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--week", type=int)
    ap.add_argument("--refit-params", action="store_true")
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--team-history", action="store_true",
                    help="also rebuild the 2022-25 team WAR history the trainer reads")
    a = ap.parse_args()
    main(a.week, a.refit_params, a.skip_pull, a.team_history)
