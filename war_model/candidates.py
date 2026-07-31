"""Candidate football skills, generated rather than hand-picked.

facets.py holds twenty facets I chose by hand. That list is the least defensible part
of the model - it is why a receiver earns 29% of his measured value from run blocking
while a running back gets four separate channels - so this replaces it with the full
cross-product of (metric x position group) and lets selection prune it.

ONE CONSTRAINT SHAPES EVERYTHING HERE. WAR has to decompose to individual players, so
every candidate must be a *player-season metric with a snap denominator*: value is a
snap-weighted z-score times snaps, which sums from players to units to teams without a
remainder. That rules out team-level stats - CFBD success rate, havoc, line yards -
however predictive they are, because there is no principled way to hand a team rate
back to the players who produced it. Those live in cfb-model-v3's O/D composites,
which is a different model with a different job.

So a candidate is (metric column, denominator column, position group). Metrics are
grades and rates, never counting stats: a count is volume, and volume is already
carried by the denominator. Feeding both in would double-count playing time and hand
every facet to whoever was on the field most.

Run: ./rbenv/bin/python candidates.py
"""
import json, os
import numpy as np
import pandas as pd

from facets import YEARS, PFF_DIR, POS_GROUP

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- position groups
GROUPS = {
    "QB": ("QB",),
    "RB": ("HB", "FB"),
    "WR": ("WR",),
    "TE": ("TE",),
    "OL": ("T", "G", "C"),
    "DI": ("DI",),
    "ED": ("ED",),
    "LB": ("LB",),
    "CB": ("CB",),
    "S":  ("S",),
}
OFFENSE = {"QB", "RB", "WR", "TE", "OL"}

# Which export each prefix comes from, matching build_massey.SOURCES.
SOURCES = {"pass": "passing", "rush": "rushing", "recv": "receiving",
           "blk": "blocking", "def": "defense"}

# ------------------------------------------------------------------- the catalogue
# metric -> (denominator, groups it is defined for). A metric only becomes a
# candidate where it means something: passing accuracy for a guard is noise, and
# coverage grade for a centre is undefined.
#
# Denominators are the honest exposure for that metric. Route grade is per route,
# not per snap; pass-rush grade is per pass-rush snap; a drop grade is per target.
# Getting this wrong is how a receiver ends up rated on run-blocking volume.
SKILL = "skill"      # graded quality
RATE = "rate"        # observed rate, already normalized by opportunity

CATALOGUE = [
    # ---- quarterback ----------------------------------------------------------
    ("pass__grades_pass",            "pass__passing_snaps", ("QB",), SKILL),
    ("pass__accuracy_percent",       "pass__attempts",      ("QB",), RATE),
    ("pass__btt_rate",               "pass__attempts",      ("QB",), RATE),
    ("pass__twp_rate",               "pass__attempts",      ("QB",), RATE),
    ("pass__completion_percent",     "pass__attempts",      ("QB",), RATE),
    ("pass__positive_epa_percent",   "pass__attempts",      ("QB",), RATE),
    ("pass__sack_percent",           "pass__dropbacks",     ("QB",), RATE),
    ("pass__pressure_to_sack_rate",  "pass__dropbacks",     ("QB",), RATE),
    ("pass__ypa",                    "pass__attempts",      ("QB",), RATE),
    ("pass__qb_rating",              "pass__attempts",      ("QB",), RATE),
    ("rush__grades_run",             "rush__attempts",      ("QB",), SKILL),
    ("pass__grades_hands_fumble",    "pass__dropbacks",     ("QB",), SKILL),

    # ---- ball carriers --------------------------------------------------------
    ("rush__grades_run",             "rush__attempts",      ("HB", "FB"), SKILL),
    ("rush__ypa",                    "rush__attempts",      ("HB", "FB"), RATE),
    ("rush__yco_attempt",            "rush__attempts",      ("HB", "FB"), RATE),
    ("rush__breakaway_percent",      "rush__attempts",      ("HB", "FB"), RATE),
    ("rush__elusive_rating",         "rush__attempts",      ("HB", "FB"), RATE),
    ("rush__grades_hands_fumble",    "rush__total_touches", ("HB", "FB"), SKILL),

    # ---- receiving (all three receiving groups) -------------------------------
    ("recv__grades_pass_route",      "recv__routes",  ("WR",),       SKILL),
    ("recv__grades_pass_route",      "recv__routes",  ("TE",),       SKILL),
    ("recv__grades_pass_route",      "recv__routes",  ("HB", "FB"),  SKILL),
    ("recv__grades_hands_drop",      "recv__targets", ("WR",),       SKILL),
    ("recv__grades_hands_drop",      "recv__targets", ("TE",),       SKILL),
    ("recv__yprr",                   "recv__routes",  ("WR",),       RATE),
    ("recv__yprr",                   "recv__routes",  ("TE",),       RATE),
    ("recv__contested_catch_rate",   "recv__targets", ("WR",),       RATE),
    ("recv__caught_percent",         "recv__targets", ("WR",),       RATE),
    ("recv__drop_rate",              "recv__targets", ("WR",),       RATE),
    ("recv__positive_epa_percent",   "recv__targets", ("WR",),       RATE),
    ("recv__positive_epa_percent",   "recv__targets", ("TE",),       RATE),
    ("recv__avg_depth_of_target",    "recv__targets", ("WR",),       RATE),
    ("recv__targeted_qb_rating",     "recv__targets", ("WR",),       RATE),
    ("recv__targeted_qb_rating",     "recv__targets", ("TE",),       RATE),
    ("recv__yards_after_catch",      "recv__receptions", ("WR",),    RATE),
    ("recv__yards_after_catch",      "recv__receptions", ("TE",),    RATE),
    ("recv__yards_after_catch",      "recv__receptions", ("HB","FB"), RATE),
    ("recv__avoided_tackles",        "recv__receptions", ("WR",),    RATE),
    ("recv__yards_per_reception",    "recv__receptions", ("WR",),    RATE),
    ("recv__first_downs",            "recv__targets",    ("WR",),    RATE),
    ("recv__first_downs",            "recv__targets",    ("TE",),    RATE),
    ("recv__touchdowns",             "recv__targets",    ("WR",),    RATE),
    ("recv__grades_offense",         "recv__routes",     ("WR",),    SKILL),

    # ---- blocking -------------------------------------------------------------
    ("blk__grades_pass_block",  "blk__snap_counts_pass_block", ("T", "G", "C"), SKILL),
    ("blk__grades_run_block",   "blk__snap_counts_run_block",  ("T", "G", "C"), SKILL),
    ("blk__grades_pass_block",  "blk__snap_counts_pass_block", ("TE",),         SKILL),
    ("blk__grades_run_block",   "blk__snap_counts_run_block",  ("TE",),         SKILL),
    ("blk__grades_run_block",   "blk__snap_counts_run_block",  ("HB", "FB"),    SKILL),
    ("blk__pressures_allowed",  "blk__snap_counts_pass_block", ("T", "G", "C"), RATE),
    ("blk__sacks_allowed",      "blk__snap_counts_pass_block", ("T", "G", "C"), RATE),
    ("blk__penalties",          "blk__snap_counts_offense",    ("T", "G", "C"), RATE),

    # ---- pass rush ------------------------------------------------------------
    ("def__grades_pass_rush_defense", "def__snap_counts_pass_rush", ("DI",), SKILL),
    ("def__grades_pass_rush_defense", "def__snap_counts_pass_rush", ("ED",), SKILL),
    ("def__grades_pass_rush_defense", "def__snap_counts_pass_rush", ("LB",), SKILL),
    ("def__grades_pass_rush_defense", "def__snap_counts_pass_rush", ("CB", "S"), SKILL),
    ("def__total_pressures",          "def__snap_counts_pass_rush", ("DI",), RATE),
    ("def__total_pressures",          "def__snap_counts_pass_rush", ("ED",), RATE),
    ("def__sacks",                    "def__snap_counts_pass_rush", ("DI",), RATE),
    ("def__sacks",                    "def__snap_counts_pass_rush", ("ED",), RATE),
    ("def__hurries",                  "def__snap_counts_pass_rush", ("ED",), RATE),
    ("def__qb_rating_against",        "def__snap_counts_pass_rush", ("ED",), RATE),

    # ---- run defence ----------------------------------------------------------
    ("def__grades_run_defense", "def__snap_counts_run_defense", ("DI",), SKILL),
    ("def__grades_run_defense", "def__snap_counts_run_defense", ("ED",), SKILL),
    ("def__grades_run_defense", "def__snap_counts_run_defense", ("LB",), SKILL),
    ("def__grades_run_defense", "def__snap_counts_run_defense", ("CB", "S"), SKILL),
    ("def__stops",              "def__snap_counts_run_defense", ("DI",), RATE),
    ("def__stops",              "def__snap_counts_run_defense", ("LB",), RATE),

    # ---- coverage -------------------------------------------------------------
    ("def__grades_coverage_defense", "def__snap_counts_coverage", ("CB",), SKILL),
    ("def__grades_coverage_defense", "def__snap_counts_coverage", ("S",),  SKILL),
    ("def__grades_coverage_defense", "def__snap_counts_coverage", ("LB",), SKILL),
    ("def__catch_rate",              "def__targets",             ("CB",), RATE),
    ("def__catch_rate",              "def__targets",             ("S",),  RATE),
    ("def__yards",                   "def__snap_counts_coverage", ("CB",), RATE),
    ("def__yards",                   "def__snap_counts_coverage", ("S",),  RATE),
    ("def__yards_after_catch",       "def__targets",              ("CB",), RATE),
    ("def__interceptions",           "def__targets",             ("CB", "S"), RATE),
    ("def__pass_break_ups",          "def__targets",             ("CB",), RATE),
    ("def__qb_rating_against",       "def__targets",             ("CB",), RATE),

    # ---- tackling -------------------------------------------------------------
    ("def__grades_tackle",     "def__snap_counts_defense", ("DI", "ED"), SKILL),
    ("def__grades_tackle",     "def__snap_counts_defense", ("LB",),      SKILL),
    ("def__grades_tackle",     "def__snap_counts_defense", ("CB", "S"),  SKILL),
    ("def__missed_tackle_rate", "def__snap_counts_defense", ("LB",),     RATE),
    ("def__missed_tackle_rate", "def__snap_counts_defense", ("CB", "S"), RATE),
    ("def__tackles",           "def__snap_counts_defense", ("LB",),      RATE),

    # ---- discipline -----------------------------------------------------------
    ("def__grades_defense_penalty", "def__snap_counts_defense", ("DI", "ED"), SKILL),
    ("def__grades_defense_penalty", "def__snap_counts_defense", ("LB", "CB", "S"), SKILL),
    ("rush__grades_offense_penalty", "blk__snap_counts_offense",
     ("QB", "HB", "FB", "WR", "TE"), SKILL),
]

# Metrics where a LOWER raw value is better. Signs are flipped at normalization so
# every feature reads the same direction, which is what makes a positive coefficient
# interpretable and a negative one a red flag rather than an artefact.
LOWER_IS_BETTER = {
    "pass__twp_rate", "pass__sack_percent", "pass__pressure_to_sack_rate",
    "recv__drop_rate", "blk__pressures_allowed", "blk__sacks_allowed",
    "blk__penalties", "def__catch_rate", "def__yards_per_coverage_snap",
    "def__qb_rating_against", "def__missed_tackle_rate",
    "def__yards", "def__yards_after_catch",
}

# Volume floors. A rate computed on four targets is noise dressed as a measurement,
# and it will not be regularized away because it can take extreme values.
MIN_DENOM = {"rate": 25, "skill": 12}


def group_of(positions):
    """Short label for a set of PFF positions, e.g. ('T','G','C') -> OL."""
    for name, members in GROUPS.items():
        if tuple(positions) == members:
            return name
    grps = {POS_GROUP.get(p, p) for p in positions}
    return "-".join(sorted(grps))


def feature_name(metric, positions):
    """Readable, stable id: the metric with its export prefix and unit stripped."""
    m = metric.split("__", 1)[1]
    m = m.replace("grades_", "").replace("_defense", "").replace("_percent", "_pct")
    return f"{group_of(positions)}_{m}"


def build_catalogue():
    """[(name, metric, denominator, positions, kind)] with duplicates dropped."""
    out, seen = [], set()
    for metric, denom, positions, kind in CATALOGUE:
        name = feature_name(metric, positions)
        if name in seen:
            continue
        seen.add(name)
        out.append((name, metric, denom, tuple(positions), kind))
    return out


def load_players():
    """Merged player-team-season frame, one row per player per team per season."""
    frames = []
    for pref, fname in SOURCES.items():
        parts = []
        for y in YEARS:
            d = pd.read_csv(f"{PFF_DIR}/{fname}_{y}.csv", low_memory=False)
            d["season"] = y
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        d = d.drop_duplicates(subset=["season", "player_id", "team_name"], keep="first")
        idx = ["season", "player_id", "team_name"]
        keep = [c for c in d.columns if c not in idx]
        d = d[idx + keep].rename(columns={c: f"{pref}__{c}" for c in keep})
        frames.append(d.set_index(idx))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.reset_index()
    for base in ("player", "position"):
        cols = [f"{p}__{base}" for p in SOURCES if f"{p}__{base}" in out.columns]
        out[base] = out[cols].bfill(axis=1).iloc[:, 0]
        out = out.drop(columns=cols)
    return out


def facet_values(players, catalogue=None, verbose=True):
    """Player-level value rows, in the same schema build_massey.facet_values emits.

    value = snap-weighted z of the metric, times the denominator. Identical grammar to
    the hand-built facets, so everything downstream - the Massey solve, the closed-form
    WAA, the replacement pool - works unchanged on a learned feature set.
    """
    catalogue = catalogue or build_catalogue()
    rows, skipped = [], []
    for name, metric, denom, positions, kind in catalogue:
        if metric not in players.columns or denom not in players.columns:
            skipped.append((name, "missing column"))
            continue
        d = players[players.position.isin(positions)][
            ["season", "player_id", "player", "position", "team_name"]].copy()
        d["metric"] = pd.to_numeric(players.loc[d.index, metric], errors="coerce")
        d["snaps"] = pd.to_numeric(players.loc[d.index, denom], errors="coerce")
        d = d[(d.snaps >= MIN_DENOM[kind]) & d.metric.notna()]
        if len(d) < 200:
            skipped.append((name, f"only {len(d)} qualifying player-seasons"))
            continue
        if metric in LOWER_IS_BETTER:
            d["metric"] = -d["metric"]
        for season, g in d.groupby("season"):
            wt = g.snaps.to_numpy(float)
            x = g.metric.to_numpy(float)
            mu = np.average(x, weights=wt)
            sd = np.sqrt(np.average((x - mu) ** 2, weights=wt))
            if sd == 0 or not np.isfinite(sd):
                continue
            z = np.clip((x - mu) / sd, -6, 6)     # a z of 40 is a data error, not a season
            rows.append(pd.DataFrame({
                "season": season, "player_id": g.player_id.to_numpy(),
                "player": g.player.to_numpy(), "position": g.position.to_numpy(),
                "team_name": g.team_name.to_numpy(), "facet": name,
                "group": group_of(positions),
                "side": "off" if group_of(positions) in OFFENSE else "def",
                "kind": kind, "snaps": wt, "z": z, "value": z * wt,
            }))
    if verbose and skipped:
        print(f"  dropped {len(skipped)} candidates:")
        for n, why in skipped:
            print(f"    {n:<28} {why}")
    return pd.concat(rows, ignore_index=True)


def team_matrix(fv, team_map=None):
    """Team-season totals per candidate, standardized within season."""
    team_map = team_map or json.load(open(f"{HERE}/team_map.json"))
    fv = fv.copy()
    fv["team"] = fv.team_name.map(team_map)
    names = sorted(fv.facet.unique())
    tot = (fv.groupby(["season", "team", "facet"], as_index=False)["value"].sum()
             .pivot(index=["season", "team"], columns="facet", values="value")
             .reindex(columns=names).fillna(0.0))
    Z = tot.groupby(level="season").transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)).fillna(0.0)
    return tot, Z, names


def main():
    cat = build_catalogue()
    print(f"candidate features declared: {len(cat)}")
    by_group = pd.Series([c[0].split("_")[0] for c in cat]).value_counts()
    print("  by position group: " + ", ".join(f"{k} {v}" for k, v in by_group.items()))

    print("\nloading PFF exports ...")
    players = load_players()
    print(f"  player-team-seasons: {len(players)}")

    print("\nbuilding candidate values ...")
    fv = facet_values(players)
    names = sorted(fv.facet.unique())
    print(f"  live candidates: {len(names)}   player-facet rows: {len(fv):,}")

    tot, Z, _ = team_matrix(fv)
    print(f"  team-seasons: {len(Z)}")

    fv.to_parquet(f"{HERE}/candidate_values.parquet")
    Z.to_csv(f"{HERE}/candidate_team_z.csv")
    tot.to_csv(f"{HERE}/candidate_team_totals.csv")
    json.dump({"features": names,
               "catalogue": [{"name": n, "metric": m, "denominator": d,
                              "positions": list(p), "kind": k}
                             for n, m, d, p, k in cat if n in names]},
              open(f"{HERE}/candidate_catalogue.json", "w"), indent=1)
    print(f"\nwritten: candidate_values.parquet, candidate_team_z.csv, "
          f"candidate_catalogue.json")


if __name__ == "__main__":
    main()
