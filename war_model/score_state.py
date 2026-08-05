"""How much of each team's season was played in a decided game.

The facet weights are fitted against the FOLLOWING season's wins, and the reason is
in build_hybrid's own notes: fitting on the same season loads the model onto facets
contaminated by the result. Coverage grade is the clearest case - it explains the
season it happened in better than pass rush does and repeats less than half as well,
because a defence that is ahead forces obvious passing downs and PFF rewards the
coverage that follows.

Dodging that by moving the target to next season costs something real, though.
Next-season wins can only be predicted by what PERSISTS, so the fitted "value" of a
job becomes value x persistence, and a job that decides games without carrying over -
pass protection, most obviously - is charged for its own volatility. That is not what
positional value means.

So the contamination is measured instead of dodged. Quarter-by-quarter line scores
give the score at the start of every quarter of every game, and from those, per
team-season:

    lead_share    share of quarters entered ahead by 14+
    trail_share   share of quarters entered behind by 14+
    mean_state    mean signed margin entering a quarter, in scores

BAD CONTROL IS THE TRAP HERE, and it is worth being explicit about. Score state is
DOWNSTREAM of ability: good teams lead because they are good. Residualising a facet on
raw lead_share would strip out the ability along with the artifact and leave the
weights worse than they started.

What is actually contaminating is the part of a team's game states it did NOT earn
with the play we are measuring - leads handed over by special teams, turnover luck,
an opponent's injuries, a soft week. So lead_share and trail_share are themselves
first regressed on the team's own total measured production, and only the RESIDUAL
game state - lopsidedness the team's own play does not account for - is projected out
of the facets. That leaves the earned part of a lead where it belongs and removes the
part that is decorating the grades.
"""
import ast
import json
import os

import numpy as np
import pandas as pd

from paths import GAMES_CSV, require

HERE = os.path.dirname(os.path.abspath(__file__))
BLOWOUT = 14


def _parse(v):
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return [float(x) for x in ast.literal_eval(v)]
        except (ValueError, SyntaxError):
            return []
    return []


def load_line_scores(years):
    """[season, team, opp, quarter, margin_entering] over FBS-vs-FBS games."""
    rows = []

    d = pd.read_csv(require(GAMES_CSV, "the CFB games table", "CFB_GAMES_CSV"),
                    low_memory=False)
    d = d[d.season.isin([y for y in years if y < 2025]) & (d.status == "completed")]
    d = d[(d.home_classification == "fbs") & (d.away_classification == "fbs")]
    for r in d.itertuples():
        rows.append((r.season, r.home_team, r.away_team,
                     _parse(r.home_line_scores), _parse(r.away_line_scores)))

    if 2025 in years:
        j = json.load(open(f"{HERE}/games_2025.json"))
        for g in j:
            if not (g.get("completed") and g.get("homeClassification") == "fbs"
                    and g.get("awayClassification") == "fbs"):
                continue
            rows.append((2025, g["homeTeam"], g["awayTeam"],
                         _parse(g.get("homeLineScores")),
                         _parse(g.get("awayLineScores"))))

    out = []
    for season, home, away, hs, as_ in rows:
        n = min(len(hs), len(as_))
        if n < 2:
            continue
        hc = np.cumsum(hs[:n])
        ac = np.cumsum(as_[:n])
        # margin ENTERING each quarter after the first: what the players on the field
        # for that quarter were actually playing in front of
        for q in range(1, n):
            m = float(hc[q - 1] - ac[q - 1])
            out.append((season, home, away, q + 1, m))
            out.append((season, away, home, q + 1, -m))
    return pd.DataFrame(out, columns=["season", "team", "opp", "quarter", "margin"])


def team_profile(years, team_map=None):
    """[season, team, lead_share, trail_share, mean_state] on CFBD school names."""
    ls = load_line_scores(years)
    if team_map:
        ls["team"] = ls.team.map(team_map).fillna(ls.team)
    g = ls.groupby(["season", "team"])
    prof = pd.DataFrame({
        "lead_share": g.margin.apply(lambda s: float((s >= BLOWOUT).mean())),
        "trail_share": g.margin.apply(lambda s: float((s <= -BLOWOUT).mean())),
        "mean_state": g.margin.mean() / 7.0,
        "quarters": g.margin.size(),
    }).reset_index()
    return prof


def unearned(prof, production):
    """The part of each team's game state its own measured production does not explain.

    production  Series indexed like prof's rows: the team's weighted facet total, or
                any single summary of how well it actually played.

    Regressing the state variables on production within season and keeping the
    residual is what stops this from being a bad control. See the module docstring.
    """
    cols = ["lead_share", "trail_share", "mean_state"]
    out = prof.copy()
    p = np.asarray(production, float)
    for c in cols:
        res = np.full(len(out), np.nan)
        for s, idx in out.groupby("season").groups.items():
            i = out.index.get_indexer(idx)
            X = np.column_stack([np.ones(len(i)), p[i]])
            yv = out[c].to_numpy(float)[i]
            ok = np.isfinite(yv) & np.isfinite(p[i])
            if ok.sum() < 10:
                res[i] = yv - np.nanmean(yv)
                continue
            beta, *_ = np.linalg.lstsq(X[ok], yv[ok], rcond=None)
            res[i] = yv - X @ beta
        out[f"{c}_resid"] = res
    return out


def project_out(Z, state, cols=("lead_share_resid", "trail_share_resid",
                                "mean_state_resid")):
    """Remove the unearned-game-state component from every facet, within season.

    Z      team-seasons x facets, standardized within season
    state  frame indexed the same way, carrying `cols`
    """
    S = state.reindex(Z.index)
    out = Z.copy()
    seasons = Z.index.get_level_values("season")
    for s in sorted(set(seasons)):
        m = seasons == s
        A = np.column_stack([np.ones(int(m.sum()))]
                            + [S.loc[m, c].fillna(0.0).to_numpy(float) for c in cols])
        Y = Z.loc[m].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        out.loc[m, :] = Y - A @ beta
    return out
