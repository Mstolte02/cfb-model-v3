"""Stage 1: merged player-season frame, normalized facet values, and the PFF Massey ratings.

Follows Eager & Chahrouri (PFF WAR, Sloan submission):
  - normalize each facet grade so the average player contributes zero
  - weight facets by random-forest importance against adjusted team wins
  - build f, solve M r = f for the Massey ratings
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from facets import FACETS, SOURCES, POS_GROUP, YEARS, PFF_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["season", "player_id", "player", "position", "team_name"]


# ---------------------------------------------------------------- player frame
def load_players():
    """Merge all five exports into one row per player-team-season."""
    frames = []
    for pref, fname in SOURCES.items():
        parts = []
        for y in YEARS:
            d = pd.read_csv(f"{PFF_DIR}/{fname}_{y}.csv")
            d["season"] = y
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        # a player can appear once per team per season; collapse any exact dupes
        d = d.drop_duplicates(subset=["season", "player_id", "team_name"], keep="first")
        idx = ["season", "player_id", "team_name"]
        keep = [c for c in d.columns if c not in idx]
        d = d[idx + keep].rename(columns={c: f"{pref}__{c}" for c in keep})
        frames.append(d.set_index(idx))

    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.reset_index()

    # player/position appear in every export; coalesce to the first non-null.
    # Position is taken from the export where the player logged the most snaps,
    # but PFF is consistent across files, so first non-null is sufficient.
    for base in ("player", "position"):
        cols = [f"{p}__{base}" for p in SOURCES if f"{p}__{base}" in out.columns]
        out[base] = out[cols].bfill(axis=1).iloc[:, 0]
        out = out.drop(columns=cols)
    return out


# ------------------------------------------------------- facet normalization
def facet_values(players):
    """Per-player facet contribution: snap-weighted z-score of the grade x snaps.

    Snap weighting makes the *average snap* the zero point, which is what
    play-level normalization achieves in the paper.
    """
    rows = []
    norms = {}
    for name, (gcol, scol, positions, side) in FACETS.items():
        if gcol not in players.columns or scol not in players.columns:
            raise KeyError(f"{name}: missing {gcol} or {scol}")
        d = players[players.position.isin(positions)][KEY + [gcol, scol]].copy()
        d = d.rename(columns={gcol: "grade", scol: "snaps"})
        d["snaps"] = pd.to_numeric(d["snaps"], errors="coerce").fillna(0.0)
        d["grade"] = pd.to_numeric(d["grade"], errors="coerce")
        # no snaps or no grade => no contribution (i.e. exactly average)
        d = d[(d.snaps > 0) & d.grade.notna()]
        if d.empty:
            continue
        for season, g in d.groupby("season"):
            w = g.snaps.to_numpy(float)
            x = g.grade.to_numpy(float)
            mu = np.average(x, weights=w)
            sd = np.sqrt(np.average((x - mu) ** 2, weights=w))
            if sd == 0:
                continue
            norms[(name, season)] = (mu, sd)
            z = (x - mu) / sd
            rows.append(pd.DataFrame({
                "season": season, "player_id": g.player_id.to_numpy(),
                "player": g.player.to_numpy(), "position": g.position.to_numpy(),
                "team_name": g.team_name.to_numpy(), "facet": name, "side": side,
                "grade": x, "snaps": w, "z": z, "value": z * w,
            }))
    return pd.concat(rows, ignore_index=True), norms


# ------------------------------------------------------------- schedule / M
def load_schedule():
    """FBS-vs-FBS games only: games.csv through 2024, 2025 from the CFBD pull.

    The season list follows facets.YEARS rather than being written out, so extending
    the PFF span extends the schedule with it - otherwise the ratings quietly keep
    solving on five seasons while the player frame carries twelve.
    """
    games = []
    src = "/Users/markstolte/Downloads/CFB_Data/data/games.csv"
    d = pd.read_csv(src, low_memory=False)
    csv_years = [y for y in YEARS if y < 2025]
    d = d[d.season.isin(csv_years) & (d.status == "completed")]
    d = d[(d.home_classification == "fbs") & (d.away_classification == "fbs")]
    d = d[d.home_points.notna() & d.away_points.notna()]
    games.append(d[["season", "home_team", "home_points", "away_team", "away_points"]])

    j = pd.DataFrame(json.load(open(f"{HERE}/games_2025.json")))
    j = j[j.completed & (j.homeClassification == "fbs") & (j.awayClassification == "fbs")]
    j = j[j.homePoints.notna() & j.awayPoints.notna()]
    games.append(pd.DataFrame({
        "season": 2025, "home_team": j.homeTeam, "home_points": j.homePoints,
        "away_team": j.awayTeam, "away_points": j.awayPoints,
    }))

    g = pd.concat(games, ignore_index=True)
    return g[g.home_points != g.away_points].reset_index(drop=True)


def adjusted_records(sched):
    """Adjusted wins: full credit if the margin is 9+, half a win/loss otherwise."""
    rec = {}
    for r in sched.itertuples():
        blowout = abs(r.home_points - r.away_points) >= 9
        hw = (1.0 if blowout else 0.5) if r.home_points > r.away_points else (0.0 if blowout else 0.5)
        for team, credit in ((r.home_team, hw), (r.away_team, 1.0 - hw)):
            k = (r.season, team)
            d = rec.setdefault(k, {"g": 0, "adj_w": 0.0})
            d["g"] += 1
            d["adj_w"] += credit
    out = pd.DataFrame([
        {"season": s, "team": t, "fbs_games": v["g"], "adj_wins": v["adj_w"],
         "adj_win_pct": v["adj_w"] / v["g"]}
        for (s, t), v in rec.items()
    ])
    wins = {}
    for r in sched.itertuples():
        w = r.home_team if r.home_points > r.away_points else r.away_team
        l = r.away_team if r.home_points > r.away_points else r.home_team
        wins[(r.season, w)] = wins.get((r.season, w), 0) + 1
        wins.setdefault((r.season, l), 0)
    out["fbs_wins"] = [wins.get((s, t), 0) for s, t in zip(out.season, out.team)]
    out["fbs_win_pct"] = out.fbs_wins / out.fbs_games
    return out


def massey_matrix(sched, season, teams):
    """M: diagonal = games played, off-diagonal = -(games between i and j)."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    M = np.zeros((n, n))
    s = sched[sched.season == season]
    for r in s.itertuples():
        if r.home_team not in idx or r.away_team not in idx:
            continue
        i, j = idx[r.home_team], idx[r.away_team]
        M[i, i] += 1
        M[j, j] += 1
        M[i, j] -= 1
        M[j, i] -= 1
    return M, idx


def solve_massey(M, f):
    """Pin the singular system with an all-ones row (ratings sum to zero)."""
    A = M.copy()
    b = f.copy()
    A[-1, :] = 1.0
    b[-1] = 0.0
    return np.linalg.solve(A, b), np.linalg.inv(A)


if __name__ == "__main__":
    players = load_players()
    print(f"player-team-seasons: {len(players)}")
    fv, norms = facet_values(players)
    print(f"player-facet rows:   {len(fv)}   facets: {fv.facet.nunique()}")

    sched = load_schedule()
    print(f"FBS-vs-FBS games:    {len(sched)}")
    recs = adjusted_records(sched)
    print(f"team-seasons:        {len(recs)}")

    players.to_parquet(f"{HERE}/players.parquet")
    fv.to_parquet(f"{HERE}/facet_values.parquet")
    sched.to_csv(f"{HERE}/schedule.csv", index=False)
    recs.to_csv(f"{HERE}/records.csv", index=False)
    print("stage 1 written")
