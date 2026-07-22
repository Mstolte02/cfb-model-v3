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


def load_player_games(years, positions) -> pd.DataFrame:
    """Per-player-per-game EPA/play (FBS vs FBS) for the given positions. Same
    cached game files as QB — CFBD credits ball-touchers (QB/RB/WR/TE) and
    box-score defenders; OL / non-targeted coverage never appear (CFBD can't
    value them — that's where PFF stays)."""
    positions = set(positions)
    rows = []
    for y in years:
        fbs = fbs_set(y)
        for wk in REG_WEEKS:
            for p in c.player_ppa_games(y, wk):
                if p.get("position") not in positions:
                    continue
                a = p.get("averagePPA") or {}
                if a.get("all") is None or not p.get("opponent"):
                    continue
                if p["team"] not in fbs or p["opponent"] not in fbs:
                    continue
                rows.append({"season": y, "id": str(p["id"]), "name": p["name"],
                             "position": p["position"], "team": p["team"],
                             "opponent": p["opponent"], "ppa": float(a["all"])})
    return pd.DataFrame(rows)


def build_position_values(years, positions) -> pd.DataFrame:
    """Opponent-adjusted, shrunk per-play value per (id, season) for each position
    (ridge on player + opponent, fit within season AND position so opponent
    effects and the shrinkage scale are position-specific)."""
    games = load_player_games(years, positions)
    out = []
    for (y, pos), dfy in games.groupby(["season", "position"]):
        val, alpha = fit_season_values(dfy)
        val["season"], val["position"], val["alpha"] = y, pos, alpha
        info = dfy.groupby("id").agg(
            name=("name", "first"),
            team=("team", lambda s: s.mode().iat[0])).reset_index()
        out.append(val.merge(info, on="id"))
    return pd.concat(out, ignore_index=True)


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


# --- team-level, leakage-free QB feature ---------------------------------------

def _within_season_z(df: pd.DataFrame, min_att: int = 150) -> pd.DataFrame:
    """Normalize qb_value within each season (scale from real starters, applied to
    all) so magnitudes are comparable across years (fixes per-season ridge alpha)."""
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["qb_z"] = np.nan
    for s, g in df.groupby("season"):
        base = g.loc[g["att"] >= min_att, "qb_value"]
        mu, sd = base.mean(), (base.std(ddof=0) or 1.0)
        df.loc[g.index, "qb_z"] = (g["qb_value"] - mu) / sd
    return df


def build_team_qb_feature(value_csv, seasons) -> dict:
    """{season N: Series(team -> QB feature)} — each team's entering-N starter,
    valued from their MOST RECENT PRIOR season (leakage-free on performance).

    "Starter" = the team's max-attempts QB that season (identity is preseason-
    knowable via depth charts; only the VALUE must be leakage-free, and it comes
    strictly from season < N). New starters with no prior FBS season are imputed
    to the season mean. The feature is re-standardized within season.
    """
    df = _within_season_z(pd.read_csv(value_csv))
    val = {(r.id, int(r.season)): r.qb_z for r in df.itertuples()
           if not np.isnan(r.qb_z)}
    starters = df.sort_values("att").groupby(["season", "team"]).tail(1)

    out = {}
    for N in seasons:
        feats = {}
        for r in starters[starters["season"] == N].itertuples():
            prior = next((val[(r.id, s)] for s in range(N - 1, 2018, -1)
                          if (r.id, s) in val), None)
            feats[r.team] = prior
        s = pd.Series(feats, dtype=float)
        s = s.fillna(s.mean())                       # new starters -> neutral prior
        out[N] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    return out


def war_qb_grades(value_csv, grade_season: int = 2025) -> dict:
    """{normalized_player_name: grade} — opponent-adjusted QB WAR from `grade_season`,
    rescaled onto that season's PFF QB grade distribution so it drops straight into
    the roster-talent math in place of the PFF grade. Names normalized with
    pff._norm for matching against the two-deep. QBs with no CFBD season aren't in
    the map (callers keep their PFF grade)."""
    from src.data import pff
    df = _within_season_z(pd.read_csv(value_csv))
    s = df[df["season"] == grade_season]
    z = {pff._norm(n): zz for n, zz in zip(s["name"], s["qb_z"]) if not np.isnan(zz)}

    g = pff.load_player_grades()
    qbg = g[(g["season"] == grade_season) & (g["group"] == "QB")]["grade"]
    mu = float(qbg.mean()); sd = float(qbg.std(ddof=0) or 10.0)
    return {name: float(np.clip(mu + sd * zz, 0, 100)) for name, zz in z.items()}


def build_team_qb_delta(value_csv, seasons) -> dict:
    """{season N: Series(team -> QB *change*)}: the entering-N starter's prior
    value minus the value of the QB last year's offense rating was actually built
    on (the team's season-(N-1) starter). ~0 if the QB returns; negative on a
    downgrade, positive on an upgrade. This is the QB signal that is NOT already
    priced into the stale prior-year offense rating.
    """
    df = _within_season_z(pd.read_csv(value_csv))
    val = {(r.id, int(r.season)): r.qb_z for r in df.itertuples() if not np.isnan(r.qb_z)}
    starters = df.sort_values("att").groupby(["season", "team"]).tail(1)
    starter_id = {(int(r.season), r.team): r.id for r in starters.itertuples()}

    out = {}
    for N in seasons:
        rows = {}
        for r in starters[starters["season"] == N].itertuples():
            new_val = next((val[(r.id, s)] for s in range(N - 1, 2018, -1)
                            if (r.id, s) in val), None)
            prev_id = starter_id.get((N - 1, r.team))
            prev_val = val.get((prev_id, N - 1)) if prev_id else None
            rows[r.team] = (new_val, prev_val)
        nv = pd.Series({t: v[0] for t, v in rows.items()}, dtype=float)
        pv = pd.Series({t: v[1] for t, v in rows.items()}, dtype=float)
        nv = nv.fillna(nv.mean()); pv = pv.fillna(pv.mean())
        d = nv - pv
        out[N] = (d - d.mean()) / (d.std(ddof=0) or 1.0)
    return out
