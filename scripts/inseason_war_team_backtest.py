"""Does in-season player WAR improve the live team model? Forward test against v5.1.

The live model (v5.1) walks a rating from the preseason level on scores and adds PFF's
season-to-date team form. Player WAR reaches it only through the week-0 level. This
asks whether moving each team by the in-season player WAR change (the rule in
src/inseason_war.py) adds anything on top, scored the way every v5 extension is scored
(scripts/v5_extension_backtest.py): the same four members, the same 2,189 games of
2023-25, one extra stack column, season-week block bootstrap against v5.1 itself.

The team signal, for a game in week w, uses the latest cut c < w (3, 6, 9):

    war_delta_diff = D(home) - D(away)
    D(team)        = sum over its players with 20+ snaps in weeks 1..c of
                     k * (updated rate - calibrated prior) * snaps per week / 1000

It uses only weeks 1..c and the player's own snaps in them - no rest-of-season
filter - and the rule's lambda and calibration for test season T are fitted on the
OTHER seasons' backtest rows, so the test season never shapes its own feature.

    python -m scripts.inseason_war_team_backtest
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import ARTIFACTS  # noqa: E402
from scripts import prior_decay_backtest as PD  # noqa: E402
from scripts import v5_extension_backtest as V  # noqa: E402
from scripts import war_inseason_backtest as B  # noqa: E402
from src import inseason_war as IW  # noqa: E402

OUT = ARTIFACTS / "inseason_war_team_backtest.json"
WAR_SEASONS = [2022, 2023, 2024, 2025]
COL = "war_delta_diff"


def available_cuts():
    return [c for c in (3, 6, 9)
            if all(B.window_dir(s, 1, c).exists() and
                   len(list(B.window_dir(s, 1, c).glob("*.csv"))) == 10
                   for s in WAR_SEASONS)]


def player_deltas(hist, cuts):
    """Per (season, cut, player): obs, prior m, snaps, team - no future information."""
    import json as _j
    team_map = _j.load(open(IW.ROOT / "war_model" / "team_map.json"))
    rows = []
    for s in WAR_SEASONS:
        pr = B.priors(hist, s)
        mu = pr.groupby("group").mu.first()
        for c in cuts:
            d = B.window_dir(s, 1, c)
            files = {B.PREFIX[n]: d / f"{n}.csv" for n in (*B.LEGACY, *B.POSITION)}
            pl = B.ww.load_players_from(files, s)
            fc = B.ww.facet_contrib(pl, s)
            teams = (pl.assign(player_id=pl.player_id.astype(str)).groupby("player_id")
                     .team_name.agg(lambda x: IW.canonical_team(
                         list(dict.fromkeys(x.dropna())), team_map)))
            fc["team"] = fc.player_id.map(teams)
            fc = fc[fc.group.notna() & (fc.snaps >= IW.MIN_SNAPS) & fc.team.notna()]
            fc["obs"] = fc.fc / fc.snaps * 1000.0
            fc = fc.merge(pr[["player_id", "group", "m"]], on=["player_id", "group"],
                          how="left")
            fc["m"] = fc.m.fillna(fc.group.map(mu))
            rows.append(fc.assign(season=s, cut=c))
        print(f"deltas {s}", flush=True)
    return pd.concat(rows, ignore_index=True)


def team_signal(pd_rows, rule, k):
    out = []
    for (g, c), d in pd_rows.groupby(["group", "cut"]):
        p = rule.get(g, {}).get(str(c))
        if p is None:
            continue
        upd = p["a"] + p["b"] * ((1 - p["lam"]) * d.m + p["lam"] * d.obs)
        base = p["a_p"] + p["b_p"] * d.m
        per_week = d.snaps / c
        out.append(d.assign(D=k[g] * (upd - base) * per_week / 1000.0))
    x = pd.concat(out)
    return x.groupby(["season", "cut", "team"]).D.sum().reset_index()


def game_column(games, D, cuts):
    lookup = {(r.season, r.cut, r.team): r.D for r in D.itertuples()}
    vals = []
    for g in games.itertuples(index=False):
        usable = [c for c in cuts if c < g.week]
        if not usable or g.season not in WAR_SEASONS:
            vals.append(0.0)
            continue
        c = max(usable)
        vals.append(lookup.get((g.season, c, g.home_team), 0.0)
                    - lookup.get((g.season, c, g.away_team), 0.0))
    return games.assign(**{COL: vals})


def run(payload, extras_by_test, extra_cols, name):
    """V.run_variant, but the extra frame may differ by outer test season."""
    preds, coefs = [], {}
    for test in V.OUTER:
        extra = extras_by_test[test]
        member_p = []
        for spec in V.SPECS:
            designs = V.member_designs(payload, test, spec)
            designs = {y: d.merge(extra, on=V.KEYS, how="left") for y, d in designs.items()}
            for d in designs.values():
                d[list(extra_cols)] = d[list(extra_cols)].fillna(0.0)
            train = pd.concat([d for y, d in designs.items() if y != test],
                              ignore_index=True)
            cols = [*V.BASE_COLS, *extra_cols]
            C = payload["ctx"][(test, spec)]["sel"]["C"]
            p, coef = V.fit_predict_stack(train, designs[test], cols, C)
            member_p.append(p)
            coefs[f"{test}:{spec}"] = coef
        base = designs[test][V.KEYS + ["y"]].copy()
        base["p"] = np.mean(member_p, axis=0)
        preds.append(base)
    out = pd.concat(preds, ignore_index=True)
    out["variant"] = name
    return out, coefs


def main():
    cuts = available_cuts()
    print("cuts", cuts, flush=True)
    payload = V.build_contexts()
    reg = V.variants(payload)
    pff = reg["pff_outcome_composite"]()
    games = pff["extra"][V.KEYS].drop_duplicates()
    pff_cols = list(pff["extra_cols"])

    hist = B.history()
    fr = _frame_with(hist, cuts)
    k = IW.war_scale(hist)
    rows = player_deltas(hist, cuts)

    extras_v51, extras_war, used = {}, {}, {}
    for test in V.OUTER:
        rule = IW.fit_rule(fr[fr.season != test])      # never the test season
        D = team_signal(rows, rule, k)
        col = game_column(games, D, cuts)
        extras_v51[test] = pff["extra"]
        extras_war[test] = pff["extra"].merge(col, on=V.KEYS, how="left")
        used[test] = {"rule_seasons": sorted(int(s) for s in fr[fr.season != test].season.unique()),
                      "nonzero_games": int((col[COL] != 0).sum())}

    res = {}
    res["v51"], c1 = run(payload, extras_v51, pff_cols, "v51")
    res["v51_war"], c2 = run(payload, extras_war, [*pff_cols, COL], "v51_war")
    m = res["v51"].rename(columns={"p": "p_base"}).merge(
        res["v51_war"][V.KEYS + ["p"]], on=V.KEYS)
    report = {
        "cuts": cuts, "n": int(len(m)), "used": used,
        "brier_v51": float(((m.p_base - m.y) ** 2).mean()),
        "brier_v51_war": float(((m.p - m.y) ** 2).mean()),
        "vs_v51": PD.bootstrap(m, "p", "p_base"),
        "by_season": {str(s): {"v51": float(((g.p_base - g.y) ** 2).mean()),
                               "v51_war": float(((g.p - g.y) ** 2).mean()),
                               "n": int(len(g))} for s, g in m.groupby("season")},
        "by_week_band": {},
        "war_coef": {kk: float(v.get(COL, np.nan)) for kk, v in c2.items()},
    }
    m["band"] = pd.cut(m.week, [0, 3, 6, 9, 99], labels=["1-3", "4-6", "7-9", "10+"])
    for b_, g in m.groupby("band", observed=True):
        report["by_week_band"][str(b_)] = {
            "v51": float(((g.p_base - g.y) ** 2).mean()),
            "v51_war": float(((g.p - g.y) ** 2).mean()), "n": int(len(g))}
    OUT.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=1, default=float))


def _frame_with(hist, cuts):
    B.CUTS = {c: ((1, c), (c + 1, 16)) for c in cuts}
    return B.frame(hist)


if __name__ == "__main__":
    main()
