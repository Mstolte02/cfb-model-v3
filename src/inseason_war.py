"""In-season player WAR: the rule audit/WAR_SAMPLE_SIZE_UPDATING.md chose.

    prior      a per-player filter over full PFF seasons; a season counts by its snaps,
               so a player with a long record is hard to move (the sample-size prior)
    update     (1 - lam) * prior + lam * season-to-date, one lam per position and cut
    calibrate  a + b * estimate, and the prior alone gets its own a_p + b_p * m, both
               fitted to rest-of-season PFF WAR on 2022-25

The published number keeps the preseason projection's playing time and moves only
the estimate of how good the player is per snap:

    in-season WAR = preseason expected WAR + k * (updated rate - calibrated prior) * S / 1000

where k turns facet-WAR into production WAR for the position and S is the player's
expected season snaps (expected snap share x a full-time season at that position).

Everything here runs on this machine: the PFF source files are not in git. The fitted
parameters are committed to data/live/inseason_war_params.json so a rebuild does not
have to re-run the backtest, and the result is committed as viz/data/players_inseason.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT

PARAMS = ROOT / "data" / "live" / "inseason_war_params.json"
OUT = ROOT / "viz" / "data" / "players_inseason.json"
LAM_GRID = np.linspace(0, 1, 21)
# The backtest graded players with 20+ snaps in the window (war_inseason_backtest
# MIN_WINDOW). Below that the rule is untested, so those players are not moved.
MIN_SNAPS = 20


def _wls(x, y, w):
    X = np.c_[np.ones(len(x)), x]
    sw = np.sqrt(np.asarray(w, float))
    return np.linalg.lstsq(X * sw[:, None], np.asarray(y) * sw, rcond=None)[0]


def fit_rule(fr: pd.DataFrame) -> dict:
    """{group: {cut: {lam, a, b, a_p, b_p, n}}} from backtest rows (any seasons given)."""
    out = {}
    for (g, cut), d in fr.groupby(["group", "cut"]):
        w = d.snaps_t.to_numpy()
        lam = float(min(LAM_GRID, key=lambda l: np.average(
            ((1 - l) * d.m + l * d.obs - d.target) ** 2, weights=w)))
        blend = (1 - lam) * d.m + lam * d.obs
        a, b = _wls(blend, d.target, w)
        a_p, b_p = _wls(d.m, d.target, w)
        out.setdefault(g, {})[str(int(cut))] = dict(
            lam=lam, a=float(a), b=float(b), a_p=float(a_p), b_p=float(b_p), n=int(len(d)))
    return out


def war_scale(hist: pd.DataFrame, seasons=(2023, 2024, 2025)) -> dict:
    """k per group: production WAA per unit of facet-WAR (f_contrib summed).

    Production waa = games * slope * (c_t * f_contrib + schedule share), so a player's
    WAA is close to proportional to his facet-WAR; the slope is fitted on the last three
    seasons with the production build's own numbers.
    """
    import sys
    sys.path.insert(0, str(ROOT / "war_model"))
    war = pd.read_csv(ROOT / "war_model" / "hybrid_player_war.csv", dtype={"player_id": str})
    war = war.groupby(["season", "player_id"], as_index=False).waa.sum()
    h = hist[hist.season.isin(seasons)].merge(war, on=["season", "player_id"])
    return {g: float(np.polyfit(d.fc, d.waa, 1)[0]) for g, d in h.groupby("group")}


def full_time_snaps(hist: pd.DataFrame, season: int = 2025) -> dict:
    """A full-time season at each position: median over teams of the busiest player's
    snaps, so expected_snap_share x this is a season of snaps."""
    import sys
    sys.path.insert(0, str(ROOT / "war_model"))
    from src.war_window import GROUP  # noqa: F401  (same vocabulary)
    h = hist[hist.season == season]
    team = h.team_name
    top = h.assign(team=team).groupby(["group", "team"]).snaps.max()
    return {g: float(v.median()) for g, v in top.groupby(level=0)}


def pick_cut(params_group: dict, week: int) -> str:
    """Nearest fitted cut to the current week, ties to the earlier one."""
    cuts = sorted(int(c) for c in params_group)
    return str(min(cuts, key=lambda c: (abs(c - week), c)))


# 2026 FBS newcomers that war_model/team_map.json (the production build's map, built
# on seasons they were not FBS in) does not know yet. Kept here rather than added to
# that file so the in-season display cannot change the production WAR build.
NEW_FBS = {"SACRAMENTO": "Sacramento State", "N DAK ST": "North Dakota State"}


def canonical_team(names, team_map) -> str | None:
    team_map = {**team_map, **NEW_FBS}
    canonical = set(team_map.values())
    for n in names:
        if n in canonical:
            return n
        if n in team_map:
            return team_map[n]
    return None


# ---------------------------------------------------------------- team signal (v5.2)
# The live ensemble's in-season WAR column. For a game in week w it reads the latest
# cut c < w and takes D(home) - D(away), where D is the team's summed change in player
# WAR per week: k * (updated rate - calibrated prior) * snaps per week / 1000, over
# players with MIN_SNAPS+ snaps in weeks 1..c. Measured as a stack column in
# scripts/inseason_war_team_backtest.py.
MODEL_CUTS = (3, 6, 9)
TEAM_HISTORY = ROOT / "data" / "live" / "inseason_war_team_history.csv"


def team_payload_path(season: int) -> Path:
    return ROOT / "data" / "live" / f"inseason_war_team_{season}.json"


def window_rows(season: int, cut: int, priors: pd.DataFrame, mu: dict,
                weight_season: int | None = None) -> pd.DataFrame:
    """Per player for weeks 1..cut: obs rate, prior mean, snaps, canonical team."""
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts import war_inseason_backtest as B
    from src import war_window as ww
    team_map = json.load(open(ROOT / "war_model" / "team_map.json"))
    d = B.window_dir(season, 1, cut)
    files = {B.PREFIX[n]: d / f"{n}.csv" for n in (*B.LEGACY, *B.POSITION)}
    pl = ww.load_players_from(files, season)
    fc = ww.facet_contrib(pl, season, weight_season=weight_season)
    teams = (pl.assign(player_id=pl.player_id.astype(str)).groupby("player_id")
             .team_name.agg(lambda x: canonical_team(list(dict.fromkeys(x.dropna())),
                                                     team_map)))
    fc["team"] = fc.player_id.map(teams)
    fc = fc[fc.group.notna() & (fc.snaps >= MIN_SNAPS) & fc.team.notna()].copy()
    fc["obs"] = fc.fc / fc.snaps * 1000.0
    fc = fc.merge(priors[["player_id", "group", "m"]], on=["player_id", "group"],
                  how="left")
    fc["m"] = fc.m.fillna(fc.group.map(mu))
    return fc.assign(season=season, cut=cut)


def team_deltas(rows: pd.DataFrame, rule: dict, k: dict) -> pd.DataFrame:
    """(season, cut, team, D) from window_rows output."""
    out = []
    for (g, c), d in rows.groupby(["group", "cut"]):
        p = rule.get(g, {}).get(str(int(c)))
        if p is None:
            continue
        upd = p["a"] + p["b"] * ((1 - p["lam"]) * d.m + p["lam"] * d.obs)
        base = p["a_p"] + p["b_p"] * d.m
        out.append(d.assign(D=k[g] * (upd - base) * (d.snaps / c) / 1000.0))
    x = pd.concat(out)
    return x.groupby(["season", "cut", "team"], as_index=False).D.sum()


def game_war_column(games: pd.DataFrame, deltas: pd.DataFrame) -> pd.Series:
    """war_delta_diff per (season, week, home_team, away_team) row, regular season."""
    lookup = {(int(r.season), int(r.cut), r.team): float(r.D) for r in deltas.itertuples()}
    cuts = {s: sorted(int(c) for c in g.cut.unique()) for s, g in deltas.groupby("season")}
    vals = []
    for g in games.itertuples(index=False):
        usable = [c for c in cuts.get(int(g.season), []) if c < int(g.week)]
        if not usable:
            vals.append(0.0)
            continue
        c = max(usable)
        vals.append(lookup.get((int(g.season), c, g.home_team), 0.0)
                    - lookup.get((int(g.season), c, g.away_team), 0.0))
    return pd.Series(vals, index=games.index, name="war_delta_diff")


def build(week: int, window_players: pd.DataFrame, window_fc: pd.DataFrame,
          priors: pd.DataFrame, params: dict, roster: pd.DataFrame) -> dict:
    """Assemble the site payload.

    window_players  the merged PFF rows for weeks 1..week (for each player's team names)
    window_fc       src.war_window.facet_contrib on them
    priors          scripts.war_inseason_backtest.priors(hist, 2026)
    roster          src.data.war.player_contributions() (the two-deep the site shows)
    """
    import sys
    sys.path.insert(0, str(ROOT / "war_model"))
    from build_roster_2026 import norm_name

    team_map = json.load(open(ROOT / "war_model" / "team_map.json"))
    teams = (window_players.assign(player_id=window_players.player_id.astype(str))
             .groupby("player_id").team_name
             .agg(lambda s: canonical_team(list(dict.fromkeys(s.dropna())), team_map)))
    w = window_fc.copy()
    w["team"] = w.player_id.map(teams)
    w = w[w.group.notna() & (w.snaps > 0)]
    w["obs"] = w.fc / w.snaps * 1000.0
    w = w.merge(priors[["player_id", "group", "m", "prior_snaps"]],
                on=["player_id", "group"], how="left")
    rule, k, S = params["rule"], params["k"], params["full_time_snaps"]
    mu = params["mu_2026"]
    w["m"] = w.m.fillna(w.group.map(mu))
    rows = []
    for g, d in w.groupby("group"):
        if g not in rule:
            continue
        c = rule[g][pick_cut(rule[g], week)]
        upd = c["a"] + c["b"] * ((1 - c["lam"]) * d.m + c["lam"] * d.obs)
        base = c["a_p"] + c["b_p"] * d.m
        rows.append(d.assign(delta_rate=upd - base, k=k[g], S=S[g]))
    w = pd.concat(rows, ignore_index=True)
    w["key"] = w.player.map(norm_name)
    w.loc[w.snaps < MIN_SNAPS, "delta_rate"] = 0.0
    # one PFF row per (team, name); a name shared by two players on one team is
    # ambiguous and dropped rather than guessed
    w = w[~w.duplicated(["team", "key"], keep=False)]

    # Joined on team and name only. The two-deep and PFF disagree on position for
    # some players (a safety listed at corner), and requiring the group to match threw
    # those away. The rule uses PFF's group, because that is whose facets the rate was
    # built from; the playing time comes from the roster, as in preseason.
    r = roster.copy()
    r["key"] = r.player.map(norm_name)
    j = r.merge(w[["team", "key", "group", "snaps", "delta_rate", "k", "S", "obs"]],
                on=["team", "key"], how="left")
    share = j.expected_snap_share.fillna(0.0)
    j["delta_war"] = j.k * j.delta_rate * share * j.S / 1000.0
    j["war_inseason"] = j.proj_war + j.delta_war.fillna(0.0)

    players, matched = {}, 0
    for row in j.itertuples():
        if pd.isna(row.snaps):
            continue
        matched += 1
        players.setdefault(row.team, {})[row.player] = {
            "sn": int(row.snaps),
            "war": round(float(row.war_inseason), 3),
            "d": round(float(row.delta_war if pd.notna(row.delta_war) else 0.0), 3),
        }
    return {
        "schema": 1, "season": 2026, "through_week": int(week),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cut_used": {g: pick_cut(rule[g], week) for g in rule},
        "matched_players": matched, "roster_players": int(len(r)),
        "pff_players": int(w.player_id.nunique()),
        "method": ("Preseason expected WAR, moved by how this season's PFF grades change "
                   "the per-snap estimate. The prior counts each past season by its "
                   "snaps; the season so far gets one weight per position and week. "
                   "Playing time is held at the preseason projection."),
        "players": players,
    }
