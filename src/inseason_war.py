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
AVAILABILITY = ROOT / "war_model" / "availability_2026.csv"
# Expected share of the next game a reported status leaves a player. Out is a hard
# zero; the report grades between keep his contribution continuous rather than
# forcing a play/sit call (scripts/sync_injury_reports.py uses the same scale).
AVAIL_SHARE = {"out": 0.0, "doubtful": 0.25, "questionable": 0.5}
# Regular-season games a preseason proj_war is spread over (see availability_team_deltas).
SEASON_GAMES = 12
STARTERS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OT": 2, "IOL": 3,
            "DT": 2, "EDGE": 2, "LB": 2, "CB": 3, "SAF": 2}


def _wls(x, y, w):
    X = np.c_[np.ones(len(x)), x]
    sw = np.sqrt(np.asarray(w, float))
    return np.linalg.lstsq(X * sw[:, None], np.asarray(y) * sw, rcond=None)[0]


# Fine grids matter: on the old 9-point s2 grid the gain could not land where the
# offensive line needs it and the rule was 5-8% worse there than a single blend.
G_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
S2_GRID = tuple(np.round(np.logspace(-1, 5, 49), 4))


def updated_rate(c: dict, m, P, obs, snaps):
    """Calibrated per-snap rate after the season so far.

    Sample-weighted (shipped October 2026, Mark's rule): each player's gain is
    K = P / (P + s2 / snaps). P is the variance of his prior, small when years of snaps
    sit behind it; s2 / snaps is the noise in this season's rate, small when he has
    played a lot. So a newcomer or a player with a thin record moves quickly, a
    four-year starter slowly, and anyone moves faster as 2026 snaps pile up. g rescales
    the window to the season scale. The older one-weight blend remains for any rule
    fitted before this (no "s2" key).
    """
    if "s2" in c:
        K = P / (P + c["s2"] / np.maximum(snaps, 1.0))
        est = m + K * (c["g"] * obs - m)
    else:
        est = (1 - c["lam"]) * m + c["lam"] * obs
    return c["a"] + c["b"] * est


def fit_rule(fr: pd.DataFrame) -> dict:
    """{group: {cut: {g, s2, a, b, a_p, b_p, lam, n}}}: the sample-weighted update."""
    out = {}
    for (g, cut), d in fr.groupby(["group", "cut"]):
        w = d.snaps_t.to_numpy()
        n = np.maximum(d.snaps_w.to_numpy(), 1.0)

        def est(gg, s2):
            K = d.P.to_numpy() / (d.P.to_numpy() + s2 / n)
            return d.m.to_numpy() + K * (gg * d.obs.to_numpy() - d.m.to_numpy())
        gg, s2 = min(((x, y) for x in G_GRID for y in S2_GRID),
                     key=lambda p: np.average((est(*p) - d.target) ** 2, weights=w))
        a, b = _wls(est(gg, s2), d.target, w)
        a_p, b_p = _wls(d.m, d.target, w)
        lam = float(min(LAM_GRID, key=lambda l: np.average(
            ((1 - l) * d.m + l * d.obs - d.target) ** 2, weights=w)))
        out.setdefault(g, {})[str(int(cut))] = dict(
            g=float(gg), s2=float(s2), a=float(a), b=float(b), a_p=float(a_p),
            b_p=float(b_p), lam=lam, n=int(len(d)))
    return out


def fit_rule_blend(fr: pd.DataFrame) -> dict:
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


def availability_overrides(path: Path = AVAILABILITY) -> dict[tuple[str, str], str]:
    """Current verified player overrides, keyed the same way as the PFF name join."""
    if not path.exists():
        return {}
    import sys
    sys.path.insert(0, str(ROOT / "war_model"))
    from build_roster_2026 import norm_name
    rows = pd.read_csv(path).fillna("")
    return {(r.team, norm_name(r.player)): str(r.status).lower()
            for r in rows.itertuples()}


def availability_team_deltas(roster: pd.DataFrame,
                             path: Path = AVAILABILITY) -> dict[str, float]:
    """WAR removed by confirmed current absences without rewriting the week-0 roster.

    Keeping this as a live delta preserves the temporal contract: a new injury changes
    the next prediction, not the preseason baseline or already-played games.

    Units matter here. The team signal D is WAR per week (rate change x snaps per
    week / 1000), and the stacks weight it at roughly 4 logits per unit. proj_war is a
    whole season, so it is spread over SEASON_GAMES before it joins D; adding season
    WAR directly would have let one injured starter swing a game by tens of points.
    """
    import sys
    sys.path.insert(0, str(ROOT / "war_model"))
    from build_roster_2026 import norm_name
    status = availability_overrides(path)
    out: dict[str, float] = {}
    for r in roster.itertuples():
        share = AVAIL_SHARE.get(status.get((r.team, norm_name(r.player))), 1.0)
        # An absence already baked into the base roster has no additional live cost.
        if share >= 1.0 or not bool(r.available):
            continue
        out[r.team] = (out.get(r.team, 0.0)
                       - (1.0 - share) * float(r.proj_war) / SEASON_GAMES)
    return out


def current_starters(frame: pd.DataFrame) -> pd.Series:
    """Infer today's first units from cumulative PFF participation.

    This is a usage chart, not a speculative weekly depth chart. Once PFF has charted
    enough players at a position, the busiest available players fill the normal unit;
    sparse groups retain the preseason starter flags. Verified absences always lose
    their starter designation.
    """
    result = frame.is_starter.fillna(False).astype(bool).copy()
    for (_, group), idx in frame.groupby(["team", "broad_group"]).groups.items():
        cap = STARTERS.get(group)
        if not cap:
            continue
        group_rows = frame.loc[idx]
        eligible = group_rows[(group_rows.available_now) & group_rows.snaps.notna()]
        if len(eligible) < cap:
            continue
        result.loc[idx] = False
        chosen = eligible.sort_values(["snaps", "proj_war"], ascending=False).head(cap)
        result.loc[chosen.index] = True
    result.loc[~frame.available_now] = False
    return result


# ---------------------------------------------------------------- team signal (v5.2)
# The live ensemble's in-season WAR column. For a game in week w it reads the latest
# cut c < w and takes D(home) - D(away), where D is the team's summed change in player
# WAR per week: k * (updated rate - calibrated prior) * snaps per week / 1000, over
# players with MIN_SNAPS+ snaps in weeks 1..c. Measured as a stack column in
# scripts/inseason_war_team_backtest.py.
MODEL_CUTS = (3, 6, 9)
# Graded weeks keep their inputs: see scripts/update_inseason_war.write_team_tables.
FROZEN_BEFORE_CUT = 5
TEAM_HISTORY = ROOT / "data" / "live" / "inseason_war_team_history.csv"


def team_payload_path(season: int) -> Path:
    return ROOT / "data" / "live" / f"inseason_war_team_{season}.json"


def opponent_adjust(frame: pd.DataFrame, season: int, cut: int,
                    beta: dict | None) -> pd.Series:
    """obs made schedule-neutral: rate - beta[group] * difficulty the unit faced.

    Difficulty is src/opponent_strength over weeks 1..cut, with team strength measured
    only from games through the cut, so a live number never looks ahead.
    """
    if not beta:
        return frame.obs
    from src import opponent_strength as O
    diff = O.window_difficulty(season, 1, cut, through=cut)
    d = O.player_difficulty(frame.team, frame.group, diff)
    return frame.obs - frame.group.map(beta).fillna(0.0) * d


def window_rows(season: int, cut: int, priors: pd.DataFrame, mu: dict,
                weight_season: int | None = None,
                opp_beta: dict | None = None) -> pd.DataFrame:
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
    fc["obs"] = opponent_adjust(fc, season, cut, opp_beta)
    fc = fc.merge(priors[["player_id", "group", "m", "P"]], on=["player_id", "group"],
                  how="left")
    fc["m"] = fc.m.fillna(fc.group.map(mu))
    fc["P"] = fc.P.fillna(fc.group.map(priors.groupby("group").tau2.first()))
    return fc.assign(season=season, cut=cut)


def team_deltas(rows: pd.DataFrame, rule: dict, k: dict) -> pd.DataFrame:
    """(season, cut, team, D) from window_rows output."""
    out = []
    for (g, c), d in rows.groupby(["group", "cut"]):
        group_rule = rule.get(g, {})
        if not group_rule:
            continue
        p = group_rule[pick_cut(group_rule, int(c))]
        upd = updated_rate(p, d.m, d.P, d.obs, d.snaps)
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


WEEKLY = ROOT.parent / "source-data" / "pff_api" / "player_weekly"
# A projected share below this makes proj_war / share an unstable per-share value (a
# 5% backup with 0.01 WAR would be "worth" a starter); those players are valued from
# their per-snap rate instead, calibrated to the projection.
V_SHARE_FLOOR = 0.15
# Mark's rule (October 2026): who is in the model is decided by real 2026 snaps, never
# the preseason two-deep. Any PFF player with this many snaps is included.
EMERGED_SNAPS = MIN_SNAPS


def season_snaps(season: int, week: int) -> pd.DataFrame:
    """Per (team, name key): scrimmage snaps in weeks 1..week, team games, PFF group."""
    import sys
    sys.path.insert(0, str(ROOT / "war_model"))
    from build_roster_2026 import norm_name
    from scripts.playing_time_backtest import weekly
    w = weekly(season)
    w = w[w.week <= week]
    games = w[["team", "week"]].drop_duplicates().groupby("team").size()
    t = w.groupby(["team", "player_id"]).agg(player=("player", "first"),
                                             group=("group", "first"),
                                             sn=("snaps", "sum")).reset_index()
    t["G"] = t.team.map(games)
    t["key"] = t.player.map(norm_name)
    t["player_id"] = t.player_id.astype(str)
    return t


def _rate_value(group: pd.Series, m: pd.Series, week: int, params: dict) -> pd.Series:
    """WAR per full-time share at the prior per-snap rate (before calibration)."""
    rule, k, S, R = (params["rule"], params["k"], params["full_time_snaps"],
                     params["repl_per_1000"])
    out = []
    for g, mi in zip(group, m):
        if g not in rule or pd.isna(mi):
            out.append(np.nan)
            continue
        c = rule[g][pick_cut(rule[g], week)]
        rate = c["a_p"] + c["b_p"] * mi
        out.append(S[g] / 1000.0 * (k[g] * rate + R[g]))
    return pd.Series(out, index=group.index)


def playing_time(j: pd.DataFrame, week: int, params: dict) -> pd.DataFrame:
    """share_pre, share_now and V (WAR per full-time share) for each roster row.

    share_now is a Bayesian average of the preseason share and the 2026 snaps:
    (n0 * prior + G * observed) / (n0 + G) in snaps per team game, as a share of a
    full-time player at the position. scripts/playing_time_backtest.py fitted n0 at
    0.5-1 game for every position (observed snaps cut rest-of-season error by two
    thirds); the live prior is the projection, so n0 = 1 leans slightly toward it.
    """
    snaps = season_snaps(2026, week)
    by_key = snaps.drop_duplicates(["team", "key"], keep=False).set_index(["team", "key"])
    games = snaps.groupby("team").G.first()
    unit = snaps.groupby(["team", "group"]).sn.sum()
    n0 = params.get("pt_n0", 1.0)
    grp = j.broad_group
    slots = grp.map(STARTERS)
    share_pre = j.expected_snap_share.fillna(0.0).clip(0.0, 1.0)
    idx = pd.MultiIndex.from_arrays([j.team, j.key])
    sn = pd.Series(by_key.sn.reindex(idx).to_numpy(), index=j.index).fillna(0.0)
    G = j.team.map(games).fillna(0.0)
    # His fraction of every snap his room has taken in 2026, by anyone, times the
    # room's starter slots. The two-deep has no say in who counts (Mark, Oct 2026).
    # The preseason share is on the projection's two-deep footing, which overstates
    # starters; it is only a one-game prior, so after a few games it barely matters.
    U = pd.Series(unit.reindex(pd.MultiIndex.from_arrays([j.team, grp])).to_numpy(),
                  index=j.index)
    obs_opp = (sn / U).where(U > 0, 0.0)
    pre_opp = share_pre / slots
    opp_now = (n0 * pre_opp + G * obs_opp) / (n0 + G)
    share_now = (opp_now * slots).clip(0.0, 1.0).where(slots.notna() & (G > 0), share_pre)

    m = j.m.fillna(grp.map(params["mu_2026"]))
    v_rate = _rate_value(grp, m, week, params)
    starters = (share_pre >= .5) & v_rate.gt(0) & j.proj_war.gt(0)
    calib = ((j.proj_war / share_pre) / v_rate)[starters].groupby(grp[starters]).median()
    v_low = (v_rate * grp.map(calib)).clip(lower=0.0).fillna(0.0)
    V = np.where(share_pre >= V_SHARE_FLOOR, j.proj_war / share_pre.clip(lower=1e-9), v_low)
    out = pd.DataFrame({"share_pre": share_pre, "share_now": share_now, "V": V,
                        "sn26": sn, "G": G}, index=j.index)
    out.attrs["calib"] = calib.to_dict()
    return out


# Where a player lined up, by side: PFF column -> label. Position-appropriate, so a
# receiver reports wide/slot/inline and a lineman his spot on the line.
ALIGN = {
    "recv": {"recv__wide_snaps": "wide", "recv__slot_snaps": "slot", "recv__inline_snaps": "inline"},
    "blk": {"blk__snap_counts_lt": "LT", "blk__snap_counts_lg": "LG", "blk__snap_counts_ce": "C",
            "blk__snap_counts_rg": "RG", "blk__snap_counts_rt": "RT"},
    "def": {"def__snap_counts_dl_a_gap": "A-gap", "def__snap_counts_dl_b_gap": "B-gap",
            "def__snap_counts_dl_over_t": "over T", "def__snap_counts_dl_outside_t": "outside T",
            "def__snap_counts_box": "box", "def__snap_counts_slot": "slot",
            "def__snap_counts_fs": "deep", "def__snap_counts_corner": "outside CB"},
}
# RB is left out: PFF reports no backfield alignment, so his split would omit his main spot.
SIDE_OF = {"WR": "recv", "TE": "recv", "OT": "blk", "IOL": "blk",
           "DT": "def", "EDGE": "def", "LB": "def", "CB": "def", "SAF": "def"}


def alignments(window_players: pd.DataFrame) -> dict:
    """{(player_id, side): {label: share}} for shares of at least 5%."""
    out = {}
    w = window_players.assign(player_id=window_players.player_id.astype(str))
    for side, cols in ALIGN.items():
        have = [c for c in cols if c in w.columns]
        if not have:
            continue
        a = w.groupby("player_id")[have].sum()
        tot = a.sum(axis=1)
        sh = a.div(tot.replace(0, np.nan), axis=0)
        for pid, row in sh.iterrows():
            d = {cols[c]: round(float(v), 2) for c, v in row.items() if v >= .05}
            if d:
                out[(pid, side)] = dict(sorted(d.items(), key=lambda kv: -kv[1]))
    return out


# Alignment pools per group, in window-report column names (see
# scripts/deployment_rate_backtest.shares, which estimated the slopes on these).
DEPLOY_POOLS = {
    "WR": {"wide": "recv__wide_snaps", "slot": "recv__slot_snaps", "inline": "recv__inline_snaps"},
    "TE": {"wide": "recv__wide_snaps", "slot": "recv__slot_snaps", "inline": "recv__inline_snaps"},
    "OT": {"lt": "pblk__snap_counts_lt", "rt": "pblk__snap_counts_rt"},
    "IOL": {"lg": "pblk__snap_counts_lg", "c": "pblk__snap_counts_ce", "rg": "pblk__snap_counts_rg"},
    "DT": {"a_gap": "def__snap_counts_dl_a_gap", "b_gap": "def__snap_counts_dl_b_gap",
           "over_t": "def__snap_counts_dl_over_t", "outside_t": "def__snap_counts_dl_outside_t"},
    "EDGE": {"b_gap": "def__snap_counts_dl_b_gap", "over_t": "def__snap_counts_dl_over_t",
             "outside_t": "def__snap_counts_dl_outside_t", "off_ball": "def__snap_counts_box"},
    "LB": {"box": "def__snap_counts_box", "slot_d": "def__snap_counts_slot",
           "a_gap": "def__snap_counts_dl_a_gap", "b_gap": "def__snap_counts_dl_b_gap",
           "over_t": "def__snap_counts_dl_over_t", "outside_t": "def__snap_counts_dl_outside_t"},
    "CB": {"slot_d": "def__snap_counts_slot", "outside_cb": "def__snap_counts_corner"},
    "SAF": {"deep": "def__snap_counts_fs", "box": "def__snap_counts_box", "slot_d": "def__snap_counts_slot"},
}


def deployment_shift(j: pd.DataFrame, window_players: pd.DataFrame, params: dict) -> pd.Series:
    """Per-snap rate shift from where each player actually lined up in 2026.

    scripts/deployment_rate_backtest.py estimated, per position and in WAR-rate units,
    (a) assignment difficulty - how the same player's rate moves with his alignment
    mix, within player-season - and (b) replacement level - how much worse the fill-ins
    are when a regular from that alignment misses a game. Both are ridge-shrunk with
    the penalty chosen by leave-one-season-out, no significance gates. The shift is
    sum over alignments of (replacement slope - difficulty slope) * (his share - mean).
    Nothing is a hand-set multiplier: WR, OT and IOL shrink to about zero because the
    data gives them nothing; safeties, edges, tight ends and linebackers move.
    """
    cfg = (params.get("deployment") or {}).get("rate") or {}
    out = pd.Series(0.0, index=j.index)
    w = window_players.assign(player_id=window_players.player_id.astype(str))
    for g, c in cfg.items():
        pool = {k: col for k, col in DEPLOY_POOLS[g].items() if col in w.columns}
        if not pool:
            continue
        a = w.groupby("player_id")[list(pool.values())].sum().rename(
            columns={v: k for k, v in pool.items()})
        tot = a.sum(axis=1).replace(0, np.nan)
        sh = pd.DataFrame(index=a.index)
        for k in c["net"]:
            if k == "off_ball":
                sh[k] = a["off_ball"] / tot
            else:
                sh[k] = a[k] / tot
        rows = j.broad_group.eq(g) & j.player_id.notna()
        pid = j.loc[rows, "player_id"].astype(str)
        shift = sum(c["net"][k] * (pid.map(sh[k]) - c["mean"][k]) for k in c["net"])
        out.loc[rows] = shift.fillna(0.0).to_numpy()
    return out


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
    w["obs"] = opponent_adjust(w, 2026, week, params.get("opp_beta"))
    w = w.merge(priors[["player_id", "group", "m", "P", "prior_snaps"]],
                on=["player_id", "group"], how="left")
    rule, k, S = params["rule"], params["k"], params["full_time_snaps"]
    mu = params["mu_2026"]
    w["m"] = w.m.fillna(w.group.map(mu))
    w["P"] = w.P.fillna(w.group.map(params["tau2_2026"]))
    rows = []
    for g, d in w.groupby("group"):
        if g not in rule:
            continue
        c = rule[g][pick_cut(rule[g], week)]
        upd = updated_rate(c, d.m, d.P, d.obs, d.snaps)
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
    j = r.merge(w[["team", "key", "player_id", "group", "snaps", "delta_rate", "k", "S", "obs", "m"]],
                on=["team", "key"], how="left").assign(emerged=False)
    # Players who were not on the preseason two-deep but are playing. Without this
    # a breakout (Jesse Legree, Oregon State, third string in July) never appears.
    # They start from zero projected WAR, so their whole WAR is change since opening.
    on_chart = set(zip(r.team, r.key))
    ex = w[[(t, k) not in on_chart for t, k in zip(w.team, w.key)]]
    ex = ex[(ex.snaps >= EMERGED_SNAPS) & ex.team.isin(set(r.team)) & ex.group.isin(STARTERS)]
    j = pd.concat([j, ex[["team", "player", "key", "player_id", "group", "snaps", "delta_rate", "k", "S",
                          "obs", "m"]].assign(broad_group=ex.group, expected_snap_share=0.0,
                                              proj_war=0.0, available=True, is_starter=False,
                                              emerged=True)], ignore_index=True)
    pt = playing_time(j, week, params)
    j = j.join(pt)
    # Performance: the per-snap change, now applied to the snaps he is getting.
    perf = (j.k * j.delta_rate * j.share_now * j.S / 1000.0).fillna(0.0)
    # Playing time: the change in share, valued at his preseason per-share value.
    j["war_inseason"] = j.proj_war + (j.share_now - j.share_pre) * j.V + perf
    # Deployment: applied to opening and current alike, so it never shows as change.
    shift = deployment_shift(j, window_players, params)
    dep = (j.k.fillna(j.broad_group.map(params["k"])) * shift * j.share_now
           * j.broad_group.map(S).fillna(0.0) / 1000.0).fillna(0.0)
    j["proj_war"] = j.proj_war + dep
    j["war_inseason"] = j.war_inseason + dep
    j["delta_war"] = j.war_inseason - j.proj_war

    status = availability_overrides()
    j["status"] = [status.get((t, k), "") for t, k in zip(j.team, j.key)]
    # WAR before availability: what an injured player is worth when he plays. The
    # Team Overview's injury report shows it beside the counted figure.
    j["war_base"] = j.war_inseason.copy()
    j["available_now"] = j.available.fillna(True) & j.status.ne("out")
    j.loc[~j.available_now, "war_inseason"] = 0.0
    share = j.status.map(AVAIL_SHARE).fillna(1.0)
    j["war_inseason"] = j.war_inseason * share
    j["delta_war"] = j.war_inseason - j.proj_war
    # Team playing-time signal, WAR per week like D: the roster's share changes plus
    # snaps taken by players off the chart. Players on an injury report are left to
    # availability_team_deltas, which already removes their value.
    pt_ok = ~j.status.isin(list(AVAIL_SHARE))
    team_pt = (((j.share_now - j.share_pre) * j.V)[pt_ok].groupby(j.team[pt_ok]).sum()
               / SEASON_GAMES)
    j["starter_now"] = current_starters(j)
    j.loc[j.status.eq("starter") & j.available_now, "starter_now"] = True

    al = alignments(window_players)
    players, matched = {}, 0
    for row in j.itertuples():
        if pd.notna(row.snaps):
            matched += 1
        players.setdefault(row.team, {})[row.player] = {
            "sn": int(row.snaps) if pd.notna(row.snaps) else 0,
            "war": round(float(row.war_inseason), 3),
            "d": round(float(row.delta_war if pd.notna(row.delta_war) else 0.0), 3),
            "sh": round(float(row.share_now), 3),
            **({"new": True, "g": row.broad_group} if row.emerged else {}),
            "st": bool(row.starter_now),
            "out": not bool(row.available_now),
            **({"inj": row.status} if row.status in AVAIL_SHARE else {}),
            # healthy WAR (if he plays) vs the WAR counted now; the gap is availability
            "base": round(float(row.war_base), 3),
            **({"al": al[(str(row.player_id), SIDE_OF[row.broad_group])]}
               if pd.notna(row.player_id) and (str(row.player_id), SIDE_OF.get(row.broad_group))
               in al else {}),
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
                   "Playing time is the 2026 snap share, blended with the preseason "
                   "share as if it were one game; per-snap rates are adjusted for "
                   "opponent difficulty; injury reports scale availability."),
        "team_playing_time": {t: round(float(v), 6) for t, v in team_pt.items()},
        "players": players,
    }
