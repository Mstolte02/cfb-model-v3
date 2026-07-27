"""True matchup-adjusted, team-level model (offense vs opponent defense).

Each team collapses to two ratings per season:
    O = offensive rating  (mean of standardized offensive stats)
    D = defensive rating   (mean of standardized defensive stats; already
                            sign-flipped in features.standardize so higher=better)

A game's features (home perspective), all team-level:
    off_edge = O_home(N-1) - D_away(N-1)     # home offense vs away defense
    def_edge = D_home(N-1) - O_away(N-1)     # home defense vs away offense
    pythag   = pythag_home(N-1) - pythag_away(N-1)
    talent   = talent_home(N)   - talent_away(N)
    returning= return_home(N)   - return_away(N)
    (+ home-field indicator, appended by model.train)

The L2 logistic fits the weights on game outcomes (target B), which blends the
prior-year ratings with the talent/returning/Pythagorean priors -> that blend IS
the regression to a calibrated mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES

OFF_STATS = [f for f, m in FEATURES.items() if f.startswith("off_")]
DEF_STATS = [f for f, m in FEATURES.items() if f.startswith("def_")]

# fp_margin is a team-level term (offense + defense + special teams), so it sits
# alongside the talent/pythag/returning priors rather than in the O/D composites.
MATCHUP_COLS = ["off_edge", "def_edge", "fp_margin", "pythag", "talent", "returning"]


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def od_ratings(std_prev: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Offense and defense composite ratings from a season's standardized stats."""
    O = _z(std_prev[OFF_STATS].mean(axis=1))
    D = _z(std_prev[DEF_STATS].mean(axis=1))
    return O, D


def fit_talent_od_slopes(train_years, std_by_year, talent_by_year, od_by_year=None):
    """Slopes of the prior-year O/D composites on entering-year talent_z.
    Used as each team's talent-implied baseline for uncertainty shrinkage."""
    xs_o, ys_o, xs_d, ys_d = [], [], [], []
    for N in train_years:
        prior = N - 1
        if N not in talent_by_year:
            continue
        if od_by_year is not None:
            if prior not in od_by_year:
                continue
            O, D = od_by_year[prior]["O"], od_by_year[prior]["D"]
        elif prior in std_by_year:
            O, D = od_ratings(std_by_year[prior])
        else:
            continue
        tz = talent_by_year[N]
        common = O.index.intersection(tz.index)
        xs_o.append(tz.loc[common].values); ys_o.append(O.loc[common].values)
        xs_d.append(tz.loc[common].values); ys_d.append(D.loc[common].values)
    import numpy as _np
    xo = _np.concatenate(xs_o); yo = _np.concatenate(ys_o)
    xd = _np.concatenate(xs_d); yd = _np.concatenate(ys_d)
    b_o = float((xo @ yo) / (xo @ xo)) if (xo @ xo) else 0.0
    b_d = float((xd @ yd) / (xd @ xd)) if (xd @ xd) else 0.0
    return b_o, b_d


def team_frame(N, std_by_year, pythag_by_year, talent_by_year, returning_by_year,
               uncertainty=None, od_by_year=None):
    """Per-team rating components entering season N (None if inputs missing).

    od_by_year={year: DataFrame[O, D]} overrides the season-aggregate O/D with a
    precomputed source (e.g. EWMA recency-weighted ratings from src/ewma.py).

    uncertainty=(lam, b_o, b_d, ret_raw_series) applies the doc's §C per-team
    regression: teams with little returning production are pulled toward their
    talent-implied rating. u = 1 - returning_fraction (more missing = more shrink):
        O_adj = (1 - lam*u)*O + lam*u*(b_o*talent_z)
    """
    prior = N - 1
    if (prior not in pythag_by_year or N not in talent_by_year
            or N not in returning_by_year):
        return None
    if od_by_year is not None:
        if prior not in od_by_year:
            return None
        O, D = od_by_year[prior]["O"], od_by_year[prior]["D"]
    elif prior in std_by_year:
        O, D = od_ratings(std_by_year[prior])
    else:
        return None
    # fp_margin: team-level term, taken from the standardized prior-season stats.
    fp = std_by_year[prior]["fp_margin"] if (prior in std_by_year and
         "fp_margin" in std_by_year[prior]) else pd.Series(0.0, index=O.index)
    df = pd.DataFrame({"O": O, "D": D, "fp_margin": fp,
                       "pythag": pythag_by_year[prior],
                       "talent": talent_by_year[N],
                       "returning": returning_by_year[N]})
    if uncertainty is not None:
        # 4th element is the per-team uncertainty u in [0,1] (assemble builds it).
        lam, b_o, b_d, u_src = uncertainty
        u = u_src.reindex(df.index).fillna(0.0).clip(lower=0, upper=1)
        df["O"] = (1 - lam * u) * df["O"] + lam * u * (b_o * df["talent"])
        df["D"] = (1 - lam * u) * df["D"] + lam * u * (b_d * df["talent"])

    # Retired features are zeroed rather than removed. A column of zeros differences
    # to zero and fits a coefficient of exactly zero, so the feature contributes
    # nothing while every downstream consumer - the six-wide model.json, the JS port,
    # the playoff simulator - keeps the same shape and stays verifiably in sync.
    # See config.DROPPED_FEATURES for why these two went.
    from config import DROPPED_FEATURES
    for c in DROPPED_FEATURES:
        if c in df.columns:
            df[c] = 0.0
    return df.dropna()


def build_year(N, frame, games_df):
    """(X, y, home_flag, margins) for season N's games, entering-N team frame."""
    teams = set(frame.index)
    X, y, hf, mg = [], [], [], []
    for _, g in games_df.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in teams or a not in teams or g["home_points"] == g["away_points"]:
            continue
        fh, fa = frame.loc[h], frame.loc[a]
        X.append([
            fh.O - fa.D,                  # off_edge: home offense vs away defense
            fh.D - fa.O,                  # def_edge: home defense vs away offense
            fh.fp_margin - fa.fp_margin,  # field-position edge (team-level)
            fh.pythag - fa.pythag,
            fh.talent - fa.talent,
            fh.returning - fa.returning,
        ])
        y.append(1 if g["home_points"] > g["away_points"] else 0)
        hf.append(0 if g.get("neutral_site", False) else 1)
        mg.append(float(g["home_points"] - g["away_points"]))
    return np.array(X, dtype=float), np.array(y), np.array(hf), np.array(mg)


def assemble(game_years, std_by_year, pythag_by_year, talent_by_year,
             returning_by_year, games_by_year,
             lam=0.0, b_o=0.0, b_d=0.0, ret_raw_by_year=None,
             od_by_year=None, u_by_year=None) -> dict:
    """Build {year: (X, y, home_flag)}. lam>0 enables per-team uncertainty
    shrinkage. u defaults to (1 - returning production); pass u_by_year to test
    alternative mean-regression formulas. od_by_year overrides the O/D source."""
    parts = {}
    for N in game_years:
        unc = None
        if lam > 0:
            if u_by_year is not None and N in u_by_year:
                u = u_by_year[N]
            elif ret_raw_by_year and N in ret_raw_by_year:
                u = (1.0 - ret_raw_by_year[N]).clip(lower=0, upper=1)
            else:
                u = None
            if u is not None:
                unc = (lam, b_o, b_d, u)
        frame = team_frame(N, std_by_year, pythag_by_year, talent_by_year,
                           returning_by_year, uncertainty=unc, od_by_year=od_by_year)
        if frame is None:
            continue
        parts[N] = build_year(N, frame, games_by_year[N])
    return parts


def vs_average_vector(frame: pd.DataFrame, team: str) -> np.ndarray:
    """Matchup features for `team` vs an average opponent (all-zero ratings)."""
    f = frame.loc[team]
    return np.array([f.O, f.D, f.fp_margin, f.pythag, f.talent, f.returning], dtype=float)


def matchup_vector(frame: pd.DataFrame, a: str, b: str) -> np.ndarray:
    fa, fb = frame.loc[a], frame.loc[b]
    return np.array([fa.O - fb.D, fa.D - fb.O, fa.fp_margin - fb.fp_margin,
                     fa.pythag - fb.pythag, fa.talent - fb.talent,
                     fa.returning - fb.returning], dtype=float)


def power_ratings(model, frame: pd.DataFrame) -> pd.DataFrame:
    teams = list(frame.index)
    rows = []
    for t in teams:
        probs = [model.win_prob(matchup_vector(frame, t, o), is_home=0.0)
                 for o in teams if o != t]
        rows.append({"team": t, "power": float(np.mean(probs)),
                     "vs_average": model.win_prob(vs_average_vector(frame, t), 0.0)})
    out = pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out
