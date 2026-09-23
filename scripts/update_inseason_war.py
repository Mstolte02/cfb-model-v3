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
    hist = B.history()
    fr = B.frame(hist)
    pr26 = B.priors(hist, SEASON)
    params = {
        "fitted_on": sorted(int(s) for s in fr.season.unique()),
        "cuts": sorted(int(c) for c in fr.cut.unique()),
        "rule": IW.fit_rule(fr),
        "k": IW.war_scale(hist),
        "full_time_snaps": IW.full_time_snaps(hist),
        "mu_2026": pr26.groupby("group").mu.first().to_dict(),
    }
    IW.PARAMS.write_text(json.dumps(params, indent=1))
    print(f"-> {IW.PARAMS}")
    return params


def main(week: int | None, refit: bool, skip_pull: bool):
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
        # late charting: always replace this season's window
        for f in d.glob("*.csv"):
            f.unlink()
        SW.main([SEASON], [(1, week)])
    files = {B.PREFIX[n]: d / f"{n}.csv" for n in (*LEGACY, *POSITION)}
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--week", type=int)
    ap.add_argument("--refit-params", action="store_true")
    ap.add_argument("--skip-pull", action="store_true")
    a = ap.parse_args()
    main(a.week, a.refit_params, a.skip_pull)
