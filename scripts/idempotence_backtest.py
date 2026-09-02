"""Backtest an idempotence-shaped football rating without touching production.

The repository's season-level opponent adjustment already solves an exact fixed
point, so applying it twice cannot distinguish teams.  This experiment instead
measures the path to that fixed point:

* pass 1: each team's average capped, home-field-adjusted scoring margin;
* pass 2: pass 1 plus the average pass-1 rating of its opponents;
* fixed point: the regularised Massey/SRS solution using the same games.

The predictive test asks whether the recursive correction (fixed point - pass 1)
adds information beside a pregame Elo.  Every feature is rebuilt at the start of a
week from earlier weeks only.  The resume test asks whether fixed-point persistence
or result disruption improves leave-one-season-out agreement with the final CFP
committee ranking beside record, quality, SOS, and power-conference membership.

This is deliberately a research script.  It downloads no data and writes only the
requested output artifact.  The public schedule and rankings CSVs used for the audit
can be supplied with ``--schedules-dir`` and ``--rankings-dir``.

Run:
  python -m scripts.idempotence_backtest \
      --schedules-dir /path/to/schedules \
      --rankings-dir /path/to/rankings
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

from config import ARTIFACTS
from src.elo import expected, update


HFA_ELO = 65.0
HFA_POINTS = 2.5
ELO_K = 40.0
MARGIN_CAP = 28.0
RIDGE = 0.25
MIN_FEATURE_GAMES = 3
PREDICT_YEARS = list(range(2018, 2026))
RESUME_YEARS = list(range(2015, 2026))
POWER_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10"}
POWER_INDEPENDENTS = {"Notre Dame"}


def paired_week_bootstrap(df: pd.DataFrame, left: str, right: str,
                          draws: int = 5000, seed: int = 20260902) -> dict:
    """Paired Brier difference with season-week blocks, matching the v4 audit."""
    d = df.dropna(subset=[left, right]).copy()
    d["loss_diff"] = (d[left] - d.y) ** 2 - (d[right] - d.y) ** 2
    blocks = [g.loss_diff.to_numpy(float) for _, g in
              d.groupby(["season", "week"], sort=True, dropna=False)]
    observed = float(d.loss_diff.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[i] = np.concatenate([blocks[j] for j in chosen]).mean()
    low, high = np.quantile(samples, [.025, .975])
    return {"difference": observed, "ci95": [float(low), float(high)],
            "probability_left_better": float(np.mean(samples < 0)),
            "n_games": int(len(d)), "n_season_week_blocks": int(len(blocks)),
            "draws": int(draws), "seed": int(seed)}


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def load_schedule(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["completed"] = _as_bool(d["completed"])
    d["neutral_site"] = _as_bool(d["neutral_site"])
    d["fbs_game"] = _as_bool(d["fbs_game"])
    d = d[
        d.completed
        & d.home_points.notna()
        & d.away_points.notna()
        & d.week.notna()
    ].copy()
    d["week"] = d.week.astype(int)
    d["season"] = d.season.astype(int)
    return d.sort_values(["season", "week", "start_date", "game_id"], kind="stable")


def _game_arrays(games: pd.DataFrame, teams: list[str]):
    idx = {team: i for i, team in enumerate(teams)}
    rows, margins, results = [], [], []
    counts = np.zeros(len(teams), float)
    for g in games.itertuples(index=False):
        if g.home_team not in idx or g.away_team not in idx:
            continue
        row = np.zeros(len(teams), float)
        row[idx[g.home_team]], row[idx[g.away_team]] = 1.0, -1.0
        margin = np.clip(float(g.home_points) - float(g.away_points),
                         -MARGIN_CAP, MARGIN_CAP)
        if not bool(g.neutral_site):
            margin -= HFA_POINTS
        rows.append(row)
        margins.append(margin)
        results.append(np.sign(float(g.home_points) - float(g.away_points)))
        counts[idx[g.home_team]] += 1
        counts[idx[g.away_team]] += 1
    if not rows:
        return (np.empty((0, len(teams))), np.empty(0), np.empty(0), counts)
    return (np.vstack(rows), np.asarray(margins, float),
            np.asarray(results, float), counts)


@dataclass
class FixedPoint:
    teams: list[str]
    first: np.ndarray
    second: np.ndarray
    fixed: np.ndarray
    games: np.ndarray
    disruption: np.ndarray

    def series(self, values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=self.teams, dtype=float)


def fixed_point(games: pd.DataFrame, teams: list[str] | None = None,
                ridge: float = RIDGE) -> FixedPoint:
    if teams is None:
        teams = sorted(set(games.home_team) | set(games.away_team))
    teams = list(teams)
    B, y, results, counts = _game_arrays(games, teams)
    n = len(teams)
    if len(B) == 0:
        zero = np.zeros(n, float)
        return FixedPoint(teams, zero, zero, zero, counts, zero)

    direct_total = B.T @ y
    first = np.divide(direct_total, counts, out=np.zeros(n), where=counts > 0)

    # r2 = mean(team-perspective margin + opponent's r1).
    idx = {team: i for i, team in enumerate(teams)}
    opponent_total = np.zeros(n, float)
    for g in games.itertuples(index=False):
        i, j = idx.get(g.home_team), idx.get(g.away_team)
        if i is None or j is None:
            continue
        opponent_total[i] += first[j]
        opponent_total[j] += first[i]
    second = first + np.divide(opponent_total, counts, out=np.zeros(n),
                               where=counts > 0)

    normal = B.T @ B + ridge * np.eye(n)
    inverse = np.linalg.inv(normal)
    fixed = inverse @ B.T @ y
    fixed -= fixed.mean()

    # Exact leave-one-observation influence for ridge least squares.  The L1 norm
    # says how far the entire rating universe moves when the game is removed.
    fitted = B @ fixed
    residual = y - fitted
    leverage = np.einsum("ij,jk,ik->i", B, inverse, B)
    direction = B @ inverse
    global_shift = (np.abs(residual) / np.maximum(1.0 - leverage, 1e-6)
                    * np.abs(direction).sum(axis=1))
    disruption = np.zeros(n, float)
    for row, shift, winner_sign in zip(B, global_shift, results):
        disruption += row * winner_sign * shift
    disruption = np.divide(disruption, counts, out=np.zeros(n), where=counts > 0)
    return FixedPoint(teams, first, second, fixed, counts, disruption)


def _lookup(fp: FixedPoint, values: np.ndarray) -> dict[str, float]:
    return dict(zip(fp.teams, values))


def predictive_rows(schedules: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for season, all_games in sorted(schedules.items()):
        games = all_games[(all_games.season_type == "regular") & all_games.fbs_game]
        teams = sorted(set(games.home_team) | set(games.away_team))
        ratings = {team: 1500.0 for team in teams}
        history = games.iloc[0:0].copy()
        for week, slate in games.groupby("week", sort=True):
            fp = fixed_point(history, teams)
            first = _lookup(fp, fp.first)
            second_correction = _lookup(fp, fp.second - fp.first)
            recursive_correction = _lookup(fp, fp.fixed - fp.first)
            agrees = np.sign(fp.first) == np.sign(fp.fixed)
            stable = _lookup(fp, np.where(
                agrees,
                np.sign(fp.fixed) * np.minimum(np.abs(fp.first), np.abs(fp.fixed)),
                0.0))
            counts = _lookup(fp, fp.games)
            pending = []
            for g in slate.itertuples(index=False):
                home, away = g.home_team, g.away_team
                hfa = 0.0 if bool(g.neutral_site) else HFA_ELO
                p = expected(ratings[home], ratings[away], hfa)
                enough = min(counts[home], counts[away]) >= MIN_FEATURE_GAMES
                rows.append({
                    "season": season, "week": int(week),
                    "home_team": home, "away_team": away,
                    "y": float(g.home_points > g.away_points),
                    "p_elo": p,
                    "elo_logit": np.log(np.clip(p, 1e-6, 1-1e-6) /
                                         np.clip(1-p, 1e-6, 1-1e-6)),
                    "first_gap": (first[home] - first[away]) if enough else 0.0,
                    "second_correction_gap": (
                        second_correction[home] - second_correction[away]
                        if enough else 0.0),
                    "recursive_correction_gap": (
                        recursive_correction[home] - recursive_correction[away]
                        if enough else 0.0),
                    "stable_gap": (stable[home] - stable[away]) if enough else 0.0,
                    "feature_available": bool(enough),
                })
                pending.append(g)
            # Preserve the repo's start-of-week contract: predict the complete slate
            # before updating Elo or the recursive game graph.
            for g in pending:
                update(ratings, g.home_team, g.away_team, g.home_points,
                       g.away_points, HFA_ELO, k=ELO_K,
                       neutral=bool(g.neutral_site))
            history = pd.concat([history, slate], ignore_index=True)
    return pd.DataFrame(rows)


def _prob(model, x: pd.DataFrame, names: list[str]) -> np.ndarray:
    return model.predict_proba(x[names].to_numpy(float))[:, 1]


def predictive_backtest(rows: pd.DataFrame) -> dict:
    families = {
        "elo_recalibrated": [],
        "first_pass_strength": ["first_gap"],
        "second_pass_correction": ["second_correction_gap"],
        "recursive_correction": ["recursive_correction_gap"],
        "stable_strength": ["stable_gap"],
        "recursive_plus_stable": ["recursive_correction_gap", "stable_gap"],
    }
    predictions, folds = [], []
    for test in PREDICT_YEARS:
        train = rows[rows.season < test]
        holdout = rows[rows.season == test]
        if len(train) == 0 or len(holdout) == 0:
            continue
        fold = {"season": test, "n": int(len(holdout)), "models": {}}
        out = holdout[["season", "week", "home_team", "away_team", "y",
                       "feature_available"]].copy()
        for name, extras in families.items():
            cols = ["elo_logit", *extras]
            model = LogisticRegression(C=1.0, max_iter=2000)
            model.fit(train[cols].to_numpy(float), train.y.to_numpy(int))
            p = _prob(model, holdout, cols)
            out[f"p_{name}"] = p
            fold["models"][name] = {
                "brier": float(np.mean((p - holdout.y) ** 2)),
                "logloss": float(log_loss(holdout.y, p)),
                "coefficients": dict(zip(cols, model.coef_[0].astype(float))),
            }
        predictions.append(out)
        folds.append(fold)

    pred = pd.concat(predictions, ignore_index=True)
    base = "p_elo_recalibrated"
    pooled = {}
    for name in families:
        col = f"p_{name}"
        pooled[name] = {
            "brier": float(np.mean((pred[col] - pred.y) ** 2)),
            "logloss": float(log_loss(pred.y, pred[col])),
            "brier_change_vs_elo": float(np.mean((pred[col] - pred.y) ** 2)
                                          - np.mean((pred[base] - pred.y) ** 2)),
        }
        if col != base:
            pooled[name]["paired_week_bootstrap"] = paired_week_bootstrap(
                pred, left=col, right=base)
    return {"folds": folds, "pooled": pooled,
            "n": int(len(pred)), "predictions": pred}


def _z(s: pd.Series) -> pd.Series:
    sd = float(s.std(ddof=0))
    return (s - s.mean()) / (sd if sd else 1.0)


def resume_features(schedule: pd.DataFrame) -> pd.DataFrame:
    regular = schedule[schedule.season_type == "regular"].copy()
    fbs_games = regular[regular.fbs_game]
    teams = sorted(set(fbs_games.home_team) | set(fbs_games.away_team))
    fp = fixed_point(fbs_games, teams)
    first, second, fixed = map(fp.series, (fp.first, fp.second, fp.fixed))
    games = fp.series(fp.games)
    disruption = fp.series(fp.disruption)
    wins = pd.Series(0.0, index=teams)
    total = pd.Series(0.0, index=teams)
    opponents = {team: [] for team in teams}
    conference: dict[str, str] = {}
    for g in regular.itertuples(index=False):
        for team, conf in ((g.home_team, g.home_conference),
                           (g.away_team, g.away_conference)):
            if team in total.index and isinstance(conf, str):
                conference[team] = conf
        hin, ain = g.home_team in total.index, g.away_team in total.index
        if not (hin or ain):
            continue
        if hin:
            total[g.home_team] += 1
            wins[g.home_team] += float(g.home_points > g.away_points)
            opponents[g.home_team].append(
                fixed.get(g.away_team, -2.0 * fixed.std(ddof=0)))
        if ain:
            total[g.away_team] += 1
            wins[g.away_team] += float(g.away_points > g.home_points)
            opponents[g.away_team].append(
                fixed.get(g.home_team, -2.0 * fixed.std(ddof=0)))

    d = pd.DataFrame({"team": teams})
    d["win_pct"] = d.team.map(wins / total.replace(0, np.nan)).fillna(0.0)
    d["rating_z"] = d.team.map(_z(fixed))
    d["sos"] = d.team.map(lambda t: float(np.mean(opponents[t]))
                           if opponents[t] else 0.0)
    d["sos_z"] = _z(d.sos)
    d["p4"] = d.team.map(lambda t: float(
        conference.get(t) in POWER_CONFERENCES or t in POWER_INDEPENDENTS))

    first_z, second_z, fixed_z = _z(first), _z(second), _z(fixed)
    # Signed quality that survives both the raw and recursively adjusted views.
    stable = pd.Series(np.where(
        np.sign(first_z) == np.sign(fixed_z),
        np.sign(fixed_z) * np.minimum(np.abs(first_z), np.abs(fixed_z)),
        0.0), index=teams)
    d["persistence"] = d.team.map(stable)
    d["recursive_lift"] = d.team.map(_z(fixed_z - first_z))
    d["second_pass_lift"] = d.team.map(_z(second_z - first_z))
    d["instability"] = d.team.map(_z(-(fixed_z - first_z).abs()))
    d["disruption"] = d.team.map(_z(disruption))
    return d


def _h2h_feature(feats: pd.DataFrame, results: dict[int, list[tuple[str, str]]],
                 score: pd.Series, k: int = 10) -> pd.Series:
    out = pd.Series(0.0, index=feats.index)
    for season, group in feats.groupby("season"):
        index = {team: i for i, team in enumerate(group.team)}
        matrix = np.zeros((len(index), len(index)), float)
        for winner, loser in results.get(int(season), []):
            i, j = index.get(winner), index.get(loser)
            if i is not None and j is not None:
                matrix[i, j], matrix[j, i] = 1.0, -1.0
        rank = np.argsort(np.argsort(-score.loc[group.index].to_numpy(float)))
        near = np.abs(rank[:, None] - rank[None, :]) <= k
        out.loc[group.index] = (matrix * near).sum(axis=1)
    return out


def _resume_loso(ranked: pd.DataFrame, all_teams: pd.DataFrame,
                 results: dict[int, list[tuple[str, str]]],
                 names: list[str], h2h_within: int = 10) -> np.ndarray:
    rhos = []
    for season in sorted(ranked.season.unique()):
        train = ranked[ranked.season != season]
        test = ranked[ranked.season == season]
        plain = [name for name in names if name != "h2h"]
        provisional = LinearRegression().fit(train[plain], -train["rank"])

        train_all = all_teams[all_teams.season != season].copy()
        provisional_score = pd.Series(
            provisional.predict(train_all[plain]), index=train_all.index)
        train_all["h2h"] = _h2h_feature(
            train_all, results, provisional_score, h2h_within)
        fitted_train = train.drop(columns="h2h", errors="ignore").merge(
            train_all[["season", "team", "h2h"]],
            on=["season", "team"], how="left")
        model = LinearRegression().fit(fitted_train[names], -fitted_train["rank"])

        test_all = all_teams[all_teams.season == season].copy()
        provisional_score = pd.Series(
            provisional.predict(test_all[plain]), index=test_all.index)
        test_all["h2h"] = _h2h_feature(
            test_all, results, provisional_score, h2h_within)
        fitted_test = test.drop(columns="h2h", errors="ignore").merge(
            test_all[["season", "team", "h2h"]],
            on=["season", "team"], how="left")
        score = pd.Series(model.predict(fitted_test[names]), index=fitted_test.index)
        truth = -fitted_test["rank"]
        rhos.append(float(score.corr(truth, method="spearman")))
    return np.asarray(rhos)


def resume_backtest(schedules: dict[int, pd.DataFrame], rankings_dir: Path) -> dict:
    features = []
    ranks = []
    results = {}
    for year in RESUME_YEARS:
        if year not in schedules:
            continue
        f = resume_features(schedules[year])
        f.insert(0, "season", year)
        features.append(f)
        regular = schedules[year]
        regular = regular[(regular.season_type == "regular") & regular.fbs_game]
        results[year] = [
            ((g.home_team, g.away_team) if g.home_points > g.away_points
             else (g.away_team, g.home_team))
            for g in regular.itertuples(index=False)
            if g.home_points != g.away_points
        ]
        path = rankings_dir / f"{year}.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path)
        r = r[r.poll == "Playoff Committee Rankings"]
        if len(r):
            final_week = int(r.week.max())
            ranks.append(r[r.week == final_week][["season", "team", "rank"]])
    all_teams = pd.concat(features, ignore_index=True)
    ranked = all_teams.merge(pd.concat(ranks, ignore_index=True),
                             on=["season", "team"], how="inner")
    base_names = ["win_pct", "rating_z", "sos_z", "p4", "h2h"]
    candidates = {
        "persistence": ["persistence"],
        "recursive_lift": ["recursive_lift"],
        "second_pass_lift": ["second_pass_lift"],
        "instability": ["instability"],
        "disruption": ["disruption"],
        "second_pass_plus_disruption": ["second_pass_lift", "disruption"],
    }
    base = _resume_loso(ranked, all_teams, results, base_names)
    out = {
        "n": int(len(ranked)),
        "seasons": sorted(int(x) for x in ranked.season.unique()),
        "base_features": base_names,
        "base_loso_spearman": float(base.mean()),
        "candidates": {},
    }
    for candidate, columns in candidates.items():
        score = _resume_loso(ranked, all_teams, results,
                             [*base_names, *columns])
        delta = score - base
        leave_one_out = [float(np.delete(delta, i).mean())
                         for i in range(len(delta))]
        out["candidates"][candidate] = {
            "features": columns,
            "loso_spearman": float(score.mean()),
            "change": float(delta.mean()),
            "fold_changes": dict(zip(map(str, out["seasons"]),
                                     map(float, delta))),
            "seasons_improved": int((delta > 0).sum()),
            "worst_leave_one_season_out_change": float(min(leave_one_out)),
        }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--schedules-dir", type=Path, required=True,
                   help="Directory containing YEAR.csv or YEAR.csv.gz schedule files")
    p.add_argument("--rankings-dir", type=Path, required=True,
                   help="Directory containing YEAR.csv rankings files")
    p.add_argument("--output", type=Path,
                   default=ARTIFACTS / "idempotence_backtest.json")
    return p.parse_args()


def main():
    args = parse_args()
    schedules = {}
    for year in range(2014, 2026):
        candidates = [args.schedules_dir / f"{year}.csv.gz",
                      args.schedules_dir / f"{year}.csv"]
        path = next((p for p in candidates if p.exists()), None)
        if path is not None:
            schedules[year] = load_schedule(path)
    if not schedules:
        raise FileNotFoundError("no schedule files found")

    rows = predictive_rows(schedules)
    prediction = predictive_backtest(rows)
    resume = resume_backtest(schedules, args.rankings_dir)
    pred_frame = prediction.pop("predictions")
    payload = {
        "contract": ("weekly predictive features use prior weeks only; resume "
                     "features use completed regular seasons"),
        "definitions": {
            "persistence": ("signed minimum absolute strength shared by pass 1 "
                            "and the fixed point"),
            "recursive_correction": "fixed-point SRS minus pass-1 average margin",
            "disruption": ("mean signed global L1 rating change under exact "
                           "leave-one-game-out ridge influence"),
        },
        "predictive": prediction,
        "resume": resume,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    pred_frame.to_csv(args.output.with_suffix(".predictions.csv"), index=False)
    print(json.dumps({"predictive": prediction["pooled"], "resume": resume}, indent=2))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
