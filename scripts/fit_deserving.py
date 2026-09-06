"""Fit the Most Deserving model from the audited second-pass experiment.

Input: sportsdataverse cfb_schedules_YEAR.csv.gz and CFBD-derived rankings_YEAR.csv,
2015–2025. No downloads. Run from repository root:
  python -m scripts.fit_deserving --schedules-dir DIR --rankings-dir DIR
Feature definitions copied from research/idempotence-rating, idempotence_backtest.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
MARGIN_CAP=28.0
HFA_POINTS=2.5
RIDGE=0.25
POWER_CONFERENCES={"SEC","Big Ten","Big 12","ACC","Pac-12","Pac-10"}
POWER_INDEPENDENTS={"Notre Dame"}


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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--schedules-dir',type=Path,required=True)
    parser.add_argument('--rankings-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('viz/data/deserving-model.json'))
    args=parser.parse_args()
    features=[];ranks=[];results={}
    for y in range(2015,2026):
     games=load_schedule(args.schedules_dir/f'cfb_schedules_{y}.csv.gz');f=resume_features(games);f.insert(0,'season',y);features.append(f)
     reg=games[(games.season_type=='regular')&games.fbs_game]
     results[y]=[(g.home_team,g.away_team) if g.home_points>g.away_points else (g.away_team,g.home_team) for g in reg.itertuples() if g.home_points!=g.away_points]
     r=pd.read_csv(args.rankings_dir/f'rankings_{y}.csv');r=r[r.poll=='Playoff Committee Rankings'];ranks.append(r[r.week==r.week.max()][['season','team','rank']])
    allteams=pd.concat(features,ignore_index=True);ranked=allteams.merge(pd.concat(ranks),on=['season','team'])
    plain=['win_pct','rating_z','sos_z','p4','second_pass_lift'];names=plain+['h2h']
    prov=LinearRegression().fit(ranked[plain],-ranked['rank']);allteams['h2h']=_h2h_feature(allteams,results,pd.Series(prov.predict(allteams[plain]),index=allteams.index),10)
    ranked=ranked.merge(allteams[['season','team','h2h']],on=['season','team']);fit=LinearRegression().fit(ranked[names],-ranked['rank']);scale=10/fit.coef_[0]
    res={'weights':dict(zip(names,(fit.coef_*scale).tolist())),'provisional_weights':dict(zip(plain,(prov.coef_*scale).tolist())),'h2h_within':10,'seasons':list(range(2015,2026)),'n':len(ranked),'source_branch':'research/idempotence-rating','method':'Second-pass opponent lift added to committee-style resume model','margin_cap':28,'home_field_points':2.5,'ridge':.25}
    args.output.write_text(json.dumps(res,indent=2)+'\n');print(res,flush=True)

if __name__ == "__main__":
    main()
