"""Who actually played: a pregame personnel signal from PFF weekly snap counts.

The rating walk learns that a team got worse only after it loses. A starter who did
not play last week is visible before that. For week W of season N, from the PFF
`offense` and `defense` position reports of single weeks before W:

* a player is a **regular** if, in the team's games before its most recent one, he
  took at least ``REGULAR_SHARE`` of the side's snaps on average over at least two
  games;
* a regular is **missing** if he took under ``MISSING_SHARE`` of the snaps in the
  team's most recent game;
* each missing regular counts his **season N-1 WAR** (war_model/hybrid_player_war.csv,
  joined on PFF player id), so an absent all-conference quarterback weighs far more
  than an absent rotational guard. Players with no prior season count zero.

Every quantity is known before week W kicks off: the most recent game is at latest
week W-1, and N-1 WAR is a completed season. Matchup columns are home minus away.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import CFB_EXTERNAL, GAME_YEARS

WEEKLY_DIR = Path(CFB_EXTERNAL) / "pff_api" / "player_weekly"
PLAYER_WAR = Path(__file__).resolve().parents[1] / "war_model" / "hybrid_player_war.csv"
KEYS = ["season", "week", "home_team", "away_team"]
REGULAR_SHARE = .60
MISSING_SHARE = .10
COLUMNS = ["missing_war_diff", "missing_qb_war_diff", "missing_count_diff"]


def load_weekly(season: int) -> pd.DataFrame:
    """One row per (week, team, side, player) with that week's snap count."""
    frames = []
    # The offense report's all-snaps column is snap_counts_total.
    for side, column in (("offense", "snap_counts_total"),
                         ("defense", "snap_counts_defense")):
        for path in sorted(WEEKLY_DIR.glob(f"{side}_{season}_w*.csv")):
            week = int(path.stem.rsplit("_w", 1)[1])
            d = pd.read_csv(path, low_memory=False)
            if d.empty or column not in d:
                continue
            frames.append(pd.DataFrame({
                "week": week, "team": d["team"], "side": side,
                "player_id": d["player_id"].astype(str),
                "position": d.get("position"),
                "snaps": pd.to_numeric(d[column], errors="coerce").fillna(0.0)}))
    if not frames:
        return pd.DataFrame(columns=["week", "team", "side", "player_id",
                                     "position", "snaps"])
    out = pd.concat(frames, ignore_index=True)
    side_total = out.groupby(["week", "team", "side"]).snaps.transform("max")
    out["share"] = np.where(side_total > 0, out.snaps / side_total, 0.0)
    return out


def _prior_war(season: int) -> dict[str, float]:
    # The WAR build keys players by PFF id as text (a few ids are CFBD-prefixed).
    w = pd.read_csv(PLAYER_WAR, usecols=["season", "player_id", "war"],
                    dtype={"player_id": str})
    w = w[w.season == season - 1].groupby("player_id").war.sum()
    return w.clip(lower=0.0).to_dict()


def team_states(season: int) -> dict[tuple[int, str], dict]:
    """{(week W, team): missing-regular summary as of the start of week W}."""
    weekly = load_weekly(season)
    if weekly.empty:
        return {}
    war = _prior_war(season)
    out = {}
    for team, rows in weekly.groupby("team"):
        weeks = sorted(rows.week.unique())
        for i in range(1, len(weeks)):
            last, earlier = weeks[i], weeks[:i]
            history = rows[rows.week.isin(earlier)]
            n_games = history.groupby(["player_id", "side"]).week.nunique()
            mean_share = (history.groupby(["player_id", "side"]).share.sum()
                          / len(earlier))
            regular = mean_share[(mean_share >= REGULAR_SHARE) & (n_games >= 2)].index
            recent = rows[rows.week == last].set_index(["player_id", "side"]).share
            missing = [pid for pid, side in regular
                       if recent.get((pid, side), 0.0) < MISSING_SHARE]
            positions = rows.drop_duplicates("player_id").set_index("player_id").position
            qbs = [pid for pid in missing if positions.get(pid) == "QB"]
            state = {"missing_war": float(sum(war.get(p, 0.0) for p in missing)),
                     "missing_qb_war": float(sum(war.get(p, 0.0) for p in qbs)),
                     "missing_count": float(len(missing))}
            # The state holds until the team plays again.
            upto = weeks[i + 1] if i + 1 < len(weeks) else 99
            for w in range(last + 1, upto + 1):
                out[(w, team)] = state
    return out


def build_availability(games: pd.DataFrame, years=GAME_YEARS) -> pd.DataFrame:
    """Matchup columns for ``games`` (season, week, home_team, away_team)."""
    states = {year: team_states(year) for year in years}
    zero = {"missing_war": 0.0, "missing_qb_war": 0.0, "missing_count": 0.0}
    rows = []
    for g in games[KEYS].drop_duplicates().itertuples(index=False):
        s = states.get(int(g.season), {})
        h = s.get((int(g.week), g.home_team), zero)
        a = s.get((int(g.week), g.away_team), zero)
        rows.append({**g._asdict(),
                     "missing_war_diff": h["missing_war"] - a["missing_war"],
                     "missing_qb_war_diff": h["missing_qb_war"] - a["missing_qb_war"],
                     "missing_count_diff": h["missing_count"] - a["missing_count"]})
    return pd.DataFrame(rows)
