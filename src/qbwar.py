"""QB value ("WAR"-style) from CFBD player PPA.

Stage 1 of the WAR build. Each QB-game gives an EPA/play (`averagePPA.all`).
We fit a ridge (= mixed-effects with a Gaussian random-effect prior) model on
QB + opponent one-hot indicators:

    ppa_game  =  intercept  +  qb_effect[i]  +  opp_effect[j]  +  e

The QB coefficient is that quarterback's **opponent-adjusted per-play value**
(EPA/play above an average QB, given the schedule faced). The ridge penalty is
the shrinkage: low-volume QBs are pulled toward the mean (empirical Bayes), so a
two-game backup doesn't post an extreme rating. `id` is stable across seasons and
teams, so the same frame supports aging curves and transfers later.

Opponent adjustment + player isolation + shrinkage in one fit — the RAPM idea,
scoped to the one position where it's cleanly identified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

from src.data import cfbd_client as c

REG_WEEKS = range(1, 16)
ALPHA_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]


def load_qb_games(years) -> pd.DataFrame:
    """Per-QB-per-game EPA/play with opponent, pooled over the given seasons."""
    rows = []
    for y in years:
        for wk in REG_WEEKS:
            for p in c.player_ppa_games(y, wk):
                if p.get("position") != "QB":
                    continue
                a = p.get("averagePPA") or {}
                if a.get("all") is None or not p.get("opponent"):
                    continue
                rows.append({"season": y, "id": str(p["id"]), "name": p["name"],
                             "team": p["team"], "opponent": p["opponent"],
                             "ppa": float(a["all"])})
    return pd.DataFrame(rows)


def fbs_set(year: int) -> set:
    """Canonical FBS school names for a season."""
    return {t["school"] for t in c.fbs_teams(year)}


def load_qb_volume(years) -> pd.DataFrame:
    """Season pass attempts per player id (dropback proxy / value weight)."""
    rows = []
    for y in years:
        for r in c.player_season_stats(y, "passing"):
            if r.get("statType") == "ATT":
                rows.append({"season": y, "id": str(r["playerId"]), "att": int(r["stat"])})
    return pd.DataFrame(rows)


def fit_season_values(dfy: pd.DataFrame):
    """Ridge on QB + opponent one-hot -> opponent-adjusted per-play QB values.
    Returns (DataFrame[id, qb_value, n_games], chosen alpha)."""
    encq = OneHotEncoder(handle_unknown="ignore")
    Q = encq.fit_transform(dfy[["id"]])
    enco = OneHotEncoder(handle_unknown="ignore")
    O = enco.fit_transform(dfy[["opponent"]])
    X = sparse.hstack([Q, O]).tocsr()
    y = dfy["ppa"].to_numpy()

    alpha = max(ALPHA_GRID, key=lambda a: cross_val_score(
        Ridge(alpha=a), X, y, cv=5, scoring="neg_mean_squared_error").mean())
    m = Ridge(alpha=alpha).fit(X, y)

    val = pd.DataFrame({"id": encq.categories_[0],
                        "qb_value": m.coef_[:Q.shape[1]]})
    ng = dfy.groupby("id").size().rename("n_games")
    return val.merge(ng, on="id"), alpha


def build_qb_values(years) -> pd.DataFrame:
    """Per (id, season) opponent-adjusted QB value + team, name, volume."""
    games = load_qb_games(years)
    vol = load_qb_volume(years)
    out = []
    for y, dfy in games.groupby("season"):
        # Confine to the FBS universe: FBS QB vs FBS opponent. FCS teams and buy
        # games otherwise inflate values (weak, unmodeled competition) and pollute
        # the opponent effects.
        fbs = fbs_set(y)
        dfy = dfy[dfy["team"].isin(fbs) & dfy["opponent"].isin(fbs)]
        val, alpha = fit_season_values(dfy)
        val["season"] = y
        val["alpha"] = alpha
        info = dfy.groupby("id").agg(
            name=("name", "first"),
            team=("team", lambda s: s.mode().iat[0])).reset_index()
        out.append(val.merge(info, on="id"))
    res = pd.concat(out, ignore_index=True)
    res = res.merge(vol, on=["season", "id"], how="left")
    res["att"] = res["att"].fillna(0).astype(int)
    return res.sort_values(["season", "qb_value"], ascending=[True, False]).reset_index(drop=True)
