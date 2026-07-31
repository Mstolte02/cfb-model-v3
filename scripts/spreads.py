"""Margin / spread model (doc §5.2): predict each side's points (offense vs
opponent defense), derive spread + total. Run: ./venv/bin/python -m scripts.spreads

Reports (LOSO over 2021-25):
  - margin MAE/RMSE vs a home-field-only baseline
  - the points model's implied win-prob Brier vs the dedicated logistic (sanity:
    a coherent spread model should roughly match the classifier)
Then prints example 2026 lines.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, PROJECTION_YEAR
from src.data import load
from src import matchup as MU
from src import model as M
from src import spread as SP
from scripts.train import load_bundle, raw_returning, build_projection_frame


def build_frames(std, pyth, talent, ret, ret_raw, train_years):
    """Entering-season frames for every game-year, uncertainty fit on train_years."""
    b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent)
    frames = {}
    for N in GAME_YEARS:
        unc = ((UNCERTAINTY_LAMBDA, b_o, b_d, MU.uncertainty_u(ret_raw[N]))
               if N in ret_raw else None)
        f = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc)
        if f is not None:
            frames[N] = f
    return frames, b_o, b_d


def main():
    load.require_key()
    print("Pulling data from CFBD ...")
    std, talent, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    folds = [y for y in GAME_YEARS if (y - 1) in std]

    rows = []
    for test_year in folds:
        train_years = [g for g in GAME_YEARS if g != test_year]
        frames, b_o, b_d = build_frames(std, pyth, talent, ret, ret_raw, train_years)

        # Points (spread) model on training seasons.
        Xtr = np.vstack([SP.build_points_rows(frames[g], games[g])[0] for g in train_years])
        ytr = np.concatenate([SP.build_points_rows(frames[g], games[g])[1] for g in train_years])
        pmodel, alpha = SP.fit(Xtr, ytr)
        sigma = float(np.sqrt(np.mean((pmodel.predict(Xtr) - ytr) ** 2)))

        base_margin = np.mean([g["home_points"] - g["away_points"]
                               for yr in train_years for _, g in games[yr].iterrows()
                               if not pd.isna(g["home_points"])])

        # Dedicated win logistic (reference), same uncertainty settings.
        wparts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                             lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                             ret_raw_by_year=ret_raw)
        wmodel, _ = M.train(np.vstack([wparts[g][0] for g in train_years]),
                            np.concatenate([wparts[g][1] for g in train_years]),
                            np.concatenate([wparts[g][2] for g in train_years]))

        fr = frames[test_year]
        ae, se, base_ae, br_spread, br_logit = [], [], [], [], []
        for _, g in games[test_year].iterrows():
            h, a = g["home_team"], g["away_team"]
            if h not in fr.index or a not in fr.index or pd.isna(g["home_points"]):
                continue
            neutral = bool(g.get("neutral_site", False))
            actual = g["home_points"] - g["away_points"]
            pred = SP.game(pmodel, fr, h, a, neutral)["margin"]
            ae.append(abs(actual - pred)); se.append((actual - pred) ** 2)
            base_ae.append(abs(actual - (0 if neutral else base_margin)))
            y = 1 if actual > 0 else 0
            br_spread.append((SP.win_prob_from_margin(pred, sigma) - y) ** 2)
            p_logit = wmodel.win_prob(MU.matchup_vector(fr, h, a), 0.0 if neutral else 1.0)
            br_logit.append((p_logit - y) ** 2)

        rows.append({"year": test_year, "mae": np.mean(ae), "rmse": np.sqrt(np.mean(se)),
                     "base_mae": np.mean(base_ae), "brier_spread": np.mean(br_spread),
                     "brier_logit": np.mean(br_logit), "alpha": alpha})

    r = pd.DataFrame(rows)
    print(f"\nLOSO folds: {folds}\n")
    print(f"{'year':<6}{'MAE':>7}{'RMSE':>7}{'baseMAE':>9}{'Brier(spr)':>12}{'Brier(logit)':>14}")
    for _, x in r.iterrows():
        print(f"  {int(x.year):<4}{x.mae:>7.2f}{x.rmse:>7.2f}{x.base_mae:>9.2f}"
              f"{x.brier_spread:>12.4f}{x.brier_logit:>14.4f}")
    print(f"\nMEAN  margin MAE {r.mae.mean():.2f} pts  vs home-field baseline "
          f"{r.base_mae.mean():.2f}  ({(r.mae.mean()-r.base_mae.mean())/r.base_mae.mean()*100:+.0f}%)"
          f"  |  RMSE {r.rmse.mean():.2f}")
    print(f"      implied win-prob Brier: spread-model {r.brier_spread.mean():.4f}  "
          f"vs dedicated logistic {r.brier_logit.mean():.4f}")

    # ---- Final model on all seasons -> 2026 example lines ----
    print("\nBuilding 2026 lines (points model fit on 2021-2025) ...")
    frames_all, _, _ = build_frames(std, pyth, talent, ret, ret_raw, GAME_YEARS)
    Xall = np.vstack([SP.build_points_rows(frames_all[g], games[g])[0] for g in GAME_YEARS])
    yall = np.concatenate([SP.build_points_rows(frames_all[g], games[g])[1] for g in GAME_YEARS])
    pmodel, _ = SP.fit(Xall, yall)

    fr26 = build_projection_frame()
    avg = pd.Series({"O": 0.0, "D": 0.0, "pythag": 0.0, "talent": 0.0, "returning": 0.0},
                    name="(avg team)")
    fr26 = pd.concat([fr26, avg.to_frame().T])

    print(f"\n=== {PROJECTION_YEAR} example lines (neg = home favored) ===")
    print("Top teams at home vs an AVERAGE team:")
    for t in ["Ohio State", "Notre Dame", "Georgia", "Oregon", "Texas", "Alabama"]:
        if t in fr26.index:
            gp = SP.game(pmodel, fr26, t, "(avg team)", neutral=False)
            print(f"  {t:<13} {gp['spread_home']:+5.1f}   "
                  f"(proj {gp['home_pts']:.0f}-{gp['away_pts']:.0f}, total {gp['total']:.0f})")
    print("\nMarquee neutral-site matchups:")
    for h, a in [("Ohio State", "Notre Dame"), ("Georgia", "Texas"), ("Oregon", "Alabama")]:
        if h in fr26.index and a in fr26.index:
            gp = SP.game(pmodel, fr26, h, a, neutral=True)
            fav = h if gp["margin"] > 0 else a
            print(f"  {h} vs {a}:  {fav} by {abs(gp['margin']):.1f}   (total {gp['total']:.0f})")


if __name__ == "__main__":
    main()
