"""Backtest the in-season Elo layer: does updating ratings during the season beat
the static preseason model, and by how much, as the season progresses?

For a held-out season, seed Elo from a model trained WITHOUT that season, then walk
the games in week order: predict each game pregame (static model vs current Elo),
then update Elo with the result. Compare Brier by week bucket.
Run: ./venv/bin/python -m scripts.backtest_elo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, OPP_ADJ_ALPHA, UNCERTAINTY_LAMBDA
from src.data import load, pff
from src import matchup as MU, oppadj as OA, model as M, elo as E
from scripts.train import load_bundle, raw_returning, blended_talent

HFA_ELO = 65.0
K = 40.0


def backtest_season(TEST, std, cfbd_tal, ret, games, pyth, ret_raw, talent, od):
    tr = [g for g in GAME_YEARS if g != TEST]
    b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)

    def frame(N):
        u = MU.uncertainty_u(ret_raw[N]) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        return MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc, od_by_year=od)

    # Train seeding model on other seasons.
    parts = MU.assemble(tr, std, pyth, talent, ret, games, lam=UNCERTAINTY_LAMBDA,
                        b_o=b_o, b_d=b_d, ret_raw_by_year=ret_raw, od_by_year=od)
    Xtr = np.vstack([parts[g][0] for g in tr]); ytr = np.concatenate([parts[g][1] for g in tr])
    hf = np.concatenate([parts[g][2] for g in tr])
    cfb_model, _ = M.train(Xtr, ytr, hf)

    fr = frame(TEST)
    teams = list(fr.index)
    # Seed Elo from preseason win-vs-average probability.
    elo = {t: E.to_elo(cfb_model.win_prob(MU.vs_average_vector(fr, t), 0.0)) for t in teams}

    g = games[TEST].dropna(subset=["week"]).sort_values("week")
    rows = []
    for _, gm in g.iterrows():
        h, a = gm["home_team"], gm["away_team"]
        if h not in elo or a not in elo or gm["home_points"] == gm["away_points"]:
            continue
        neutral = bool(gm.get("neutral_site", False))
        y = 1 if gm["home_points"] > gm["away_points"] else 0
        # static model pregame prob
        p_model = cfb_model.win_prob(MU.matchup_vector(fr, h, a), 0.0 if neutral else 1.0)
        # in-season Elo pregame prob, then update
        p_elo = E.update(elo, h, a, gm["home_points"], gm["away_points"],
                         HFA_ELO, k=K, neutral=neutral)
        rows.append((gm["week"], y, p_model, p_elo))

    return pd.DataFrame(rows, columns=["week", "y", "p_model", "p_elo"])


def main():
    load.require_key()
    print("Loading data ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)

    def brier(s, col):
        return np.mean((s[col] - s["y"]) ** 2)

    seasons = [y for y in GAME_YEARS if y >= 2022]
    print(f"\nIn-season Elo backtest (seeded from a held-out preseason model)\n")
    print(f"{'season':<8}{'n':>5}{'static':>9}{'Elo':>9}{'improve':>9}")
    print("-" * 40)
    alld = []
    for s in seasons:
        d = backtest_season(s, std, cfbd_tal, ret, games, pyth, ret_raw, talent, od)
        d["season"] = s; alld.append(d)
        bm, be = brier(d, "p_model"), brier(d, "p_elo")
        print(f"{s:<8}{len(d):>5}{bm:>9.4f}{be:>9.4f}{(be-bm)/bm*100:>8.1f}%")
    d = pd.concat(alld)
    bm, be = brier(d, "p_model"), brier(d, "p_elo")
    print("-" * 40)
    print(f"{'ALL':<8}{len(d):>5}{bm:>9.4f}{be:>9.4f}{(be-bm)/bm*100:>8.1f}%")

    d["bucket"] = pd.cut(d["week"], [0, 4, 9, 99], labels=["wk1-4", "wk5-9", "wk10+"])
    print("\nPooled by week window (static -> Elo):")
    for b, sub in d.groupby("bucket", observed=True):
        print(f"  {b:<7} n={len(sub):<5} {brier(sub,'p_model'):.4f} -> {brier(sub,'p_elo'):.4f}")


if __name__ == "__main__":
    main()
