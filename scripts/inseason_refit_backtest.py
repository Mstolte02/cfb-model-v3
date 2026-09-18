"""Is refreshing the preseason model's inputs better than walking a rating on results?

Two different ideas about what "in-season" should mean, and the repo only implements
one of them.

  WALK THE RATING (shipping). Keep one number per team and move it by the residual
  between the score and what was expected. The preseason model sets the week-0 value
  and is never consulted again, because `dynamic_blend` is 1.0.

  REFRESH THE INPUTS. Keep the preseason model's structure and recompute the part of
  it that the season has actually changed - the opponent-adjusted offence and defence
  composites - from the games played so far, then predict with that.

The two use different information. A rating walk sees only final scores. A refresh
sees efficiency, and it sees it the way the preseason model already understands it.
Neither dominates on its face: the walk is closer to what wins games, the refresh is
built out of a per-play signal that stabilises faster than scores do.

Four arms, all predicted strictly from games before the current week:

  preseason     the static model, never updated. blend = 0. The floor.
  elo           the shipping rating walk. blend = 1. The incumbent.
  refit         the model's O and D columns replaced by this season's to-date
                opponent-adjusted composites, other features left at preseason,
                coefficients refit forward-only on that representation.
  refit_elo     both, side by side, which is the arrangement
                audit/RATING_ARCHITECTURE_EXPERIMENTS.md found beats folding a
                feature into the rating.

The in-season composites are built exactly the way the preseason ones are - the same
five offensive and three defensive fields from `game_advanced`, z-scored within the
season, then run through the same SRS-style opponent adjustment in `src/oppadj.py` -
so `refit` really is the preseason model reading fresher inputs rather than a
different model wearing its name.

Run: python -m scripts.inseason_refit_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri
from sklearn.linear_model import LogisticRegression

from config import ARTIFACTS, GAME_YEARS
from scripts.inseason_update_backtest import bootstrap_difference, build_frames
from scripts.train import load_bundle
from scripts.v4_backtest import CANDIDATES, choose_candidate, fit_predict, metric, tune
from src import v4 as V4
from src.data import load

OUT_JSON = ARTIFACTS / "inseason_refit_backtest.json"
OUT_CSV = ARTIFACTS / "inseason_refit_backtest_predictions.csv"

OFF = ["off_ppa", "off_pass_ppa", "off_rush_ppa", "off_success_rate",
       "off_explosiveness"]
DEF = ["def_ppa", "def_success_rate", "def_explosiveness"]
ARMS = ["preseason", "elo", "refit", "refit_elo"]
ELO_K = .20
OPPADJ_ALPHA = .5      # src/oppadj's fixed-point weight; 0 raw, 1 full SRS
MIN_GAMES = 2          # below this a to-date composite is noise, so the arm holds
                       # the preseason value rather than pretending to know


def _z(s):
    s = pd.Series(s).astype(float)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def game_composites(year):
    """Per team-game O and D composite, z-scored over the season's FBS team-games.
    Same construction as src/ewma._game_composites, lifted so this script can cut it
    by week without rebuilding the whole season each time."""
    g = load.game_advanced(year)
    fbs = set(load.team_stats(year)["team"])
    g = g[g["team"].isin(fbs) & g["opponent"].isin(fbs)].dropna(subset=OFF + DEF).copy()
    for c in OFF + DEF:
        g[c] = _z(g[c])
    g["O_game"] = g[OFF].mean(axis=1)
    g["D_game"] = -g[DEF].mean(axis=1)       # negate so larger is a better defence
    return g[["week", "team", "opponent", "O_game", "D_game"]]


def todate_od(comp, week):
    """Opponent-adjusted O/D from every game STRICTLY BEFORE `week`.

    The adjustment is the same fixed point src/oppadj uses: a team that faced hard
    defences has its offence revised up, and the other way round. Two passes is
    enough at alpha = .5 - the third moves nothing that survives the re-standardise.
    """
    past = comp[comp.week < week]
    if past.empty:
        return None
    raw = past.groupby("team").agg(O=("O_game", "mean"), D=("D_game", "mean"),
                                   n=("O_game", "size"))
    sched = {}
    for r in past.itertuples():
        sched.setdefault(r.team, []).append(r.opponent)
    O, D = raw.O.to_dict(), raw.D.to_dict()
    for _ in range(2):
        newO = {t: O[t] + OPPADJ_ALPHA * np.mean([D.get(o, 0.0) for o in sched[t]])
                for t in O}
        newD = {t: D[t] + OPPADJ_ALPHA * np.mean([O.get(o, 0.0) for o in sched[t]])
                for t in D}
        O, D = newO, newD
    out = pd.DataFrame({"O": pd.Series(O), "D": pd.Series(D), "n": raw.n})
    out["O"] = _z(out.O)
    out["D"] = _z(out.D)
    return out


def season_design(model, frame, part, comp, ratings_out=None):
    """One row per game: the preseason static logit, the Elo rating gap as of that
    week, and the to-date O/D differences. Built in week order with a whole slate
    resolved before any of its results are applied."""
    X, y, home_flag, margins, meta = part
    ratings = {t: model.team_logit_strength(frame, t) for t in frame.index}
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    n = len(y)
    static = np.zeros(n); elo = np.zeros(n)
    dO = np.zeros(n); dD = np.zeros(n); have = np.zeros(n)
    sigma = model.margin_sigma
    for week, slate in order.groupby("week", sort=True, dropna=False):
        od = todate_od(comp, float(week))
        changes = {}
        for _, row in slate.iterrows():
            i, home, away = int(row._row), row.home_team, row.away_team
            pw = float(np.clip(model.win_prob(X[i], home_flag[i]), 1e-6, 1 - 1e-6))
            static[i] = float(np.log(pw / (1 - pw)))
            elo[i] = ratings[home] - ratings[away] + model.hfa_coef * home_flag[i]
            if od is not None and home in od.index and away in od.index \
                    and od.at[home, "n"] >= MIN_GAMES and od.at[away, "n"] >= MIN_GAMES:
                dO[i] = od.at[home, "O"] - od.at[away, "O"]
                dD[i] = od.at[home, "D"] - od.at[away, "D"]
                have[i] = 1.0
            p = float(expit(elo[i]))
            expected = sigma * float(ndtri(np.clip(p, .01, .99)))
            score = float(np.clip((float(margins[i]) - expected) / sigma, -2.5, 2.5))
            d = ELO_K * score
            changes[home] = changes.get(home, 0.0) + d
            changes[away] = changes.get(away, 0.0) - d
        for t, d in changes.items():
            ratings[t] += d
    if ratings_out is not None:
        ratings_out.update(ratings)
    return pd.DataFrame({"static": static, "elo": elo, "dO": dO, "dD": dD,
                         "have": have, "hfa": home_flag, "y": y,
                         "week": meta.week.to_numpy()})


ARM_COLUMNS = {
    "preseason": ["static"],
    "elo": ["elo"],
    # `have` lets the fit keep an intercept shift for the rows where the season is
    # still too young to have a composite, instead of reading a zero as "average".
    "refit": ["static_core", "dO", "dD", "have", "hfa"],
    "refit_elo": ["static_core", "elo", "dO", "dD", "have", "hfa"],
}


def add_core(design, model, frame, part):
    """`static_core` is the preseason logit with its O/D contribution removed, so the
    refit arms carry talent, returning production, WAR, portal and recruiting WITHOUT
    also carrying last season's efficiency, which the to-date columns replace."""
    X = part[0]
    names = list(model.feature_names)
    keep = np.array([0.0 if n in ("O", "D") else 1.0 for n in names])
    design = design.copy()
    design["static_core"] = (X @ (np.asarray(model.coef, float) * keep)
                             + model.hfa_coef * part[2])
    return design


def fit_arm(train, test, arm):
    cols = ARM_COLUMNS[arm]
    if arm in ("preseason", "elo"):
        # Nothing to fit: these arms ARE their column, read as a logit.
        return expit(test[cols[0]].to_numpy())
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(train[cols].to_numpy(), train.y.to_numpy())
    return m.predict_proba(test[cols].to_numpy())[:, 1]


def main():
    std, talent, ret, games, _ = load_bundle()
    frames = build_frames(std, talent, ret, games)
    all_parts = {name: V4.assemble(GAME_YEARS, frames, games, columns)
                 for name, columns in CANDIDATES.items()}
    comps = {y: game_composites(y) for y in GAME_YEARS}

    folds, outputs = [], []
    # Two earlier seasons are the minimum: one to fit the team model on and one
    # to generate in-season rows for the arm coefficients. 2022 has only 2021
    # behind it, so it cannot be a fold here. The primary window is 2023-25
    # regardless, which is the same window every other in-season audit reports.
    for test in [2023, 2024, 2025]:
        pool = [y for y in GAME_YEARS if y < test]
        selected, _ = choose_candidate(all_parts, pool)
        names, parts = CANDIDATES[selected], all_parts[selected]
        knobs, _ = tune(parts, pool, names)

        train_rows = []
        for i in range(1, len(pool)):
            val, tr = pool[i], pool[:i]
            vm, _, _ = fit_predict(parts, tr, val, names, knobs)
            d = season_design(vm, frames[val], parts[val], comps[val])
            train_rows.append(add_core(d, vm, frames[val], parts[val]))
        train = pd.concat(train_rows, ignore_index=True)

        model, _, _ = fit_predict(parts, pool, test, names, knobs)
        d = season_design(model, frames[test], parts[test], comps[test])
        d = add_core(d, model, frames[test], parts[test])

        rows = parts[test][4].copy()
        rows["season"], rows["y"] = test, parts[test][1]
        fold = {"season": test, "selected_team_model": selected, "arms": {}}
        for arm in ARMS:
            p = fit_arm(train, d, arm)
            rows[arm] = p
            fold["arms"][arm] = metric(parts[test][1], p)
        rows["have_composite"] = d.have.to_numpy()
        folds.append(fold)
        outputs.append(rows)
        print(f"{test}: " + "  ".join(
            f"{a}={fold['arms'][a]['brier']:.5f}" for a in ARMS), flush=True)

    predictions = pd.concat(outputs, ignore_index=True)
    primary = predictions[predictions.season >= 2023].copy()
    result = {
        "contract": "strict expanding replay; arm coefficients fit on earlier seasons only",
        "primary_window": "2023-2025",
        "n_primary": int(len(primary)),
        "oppadj_alpha": OPPADJ_ALPHA,
        "folds": folds,
        "pooled_2023_2025": {a: metric(primary.y, primary[a]) for a in ARMS},
        "bootstrap_vs_elo": {a: bootstrap_difference(primary, a, right="elo")
                             for a in ARMS if a != "elo"},
    }
    # Where the season is young the composite is absent by design; split so the
    # comparison is not carried by rows where `refit` had nothing to say.
    for label, sub in (("with_composite", primary[primary.have_composite == 1]),
                       ("without_composite", primary[primary.have_composite == 0])):
        result[label] = {"n": int(len(sub)),
                         **{a: metric(sub.y, sub[a]) for a in ARMS}} if len(sub) else {}

    OUT_JSON.write_text(json.dumps(result, indent=2))
    predictions.to_csv(OUT_CSV, index=False)
    print("\nPrimary 2023-25:")
    for a, m in result["pooled_2023_2025"].items():
        print(f"  {a:<12} Brier={m['brier']:.6f}  logloss={m['logloss']:.6f}")
    for a, item in result["bootstrap_vs_elo"].items():
        lo, hi = item["ci95"]
        print(f"  {a:<12} vs elo {item['brier_difference_vs_current']:+.6f}  "
              f"CI=[{lo:+.6f}, {hi:+.6f}]")
    for label in ("with_composite", "without_composite"):
        if result.get(label):
            print(f"  {label} (n={result[label]['n']}): " + "  ".join(
                f"{a}={result[label][a]['brier']:.5f}" for a in ARMS))
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
