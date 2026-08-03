"""A CFBD-sourced facet set built to slot into the same pipeline as facets.py.

The two sources are not interchangeable, and the shape of the gap decides the
architecture:

  offense skill  PPA is genuinely competitive with PFF grades. It is EPA per play,
                 already play-level and outcome-anchored, and it comes with a usage
                 share that acts as the snap denominator.
  defense        CFBD has counting stats only (tackles, TFL, sacks, hurries, passes
                 defensed). There is no coverage or run-defense quality measure, and
                 no snap denominator, so a corner nobody throws at is invisible.
  offensive line CFBD has nothing at the player level at all.

So this module deliberately covers only what CFBD can actually measure. Facets it
cannot construct are left to the PFF set; build_hybrid.py does the joining.

Facet value follows the PFF convention exactly: a snap-weighted z-score of a rate
metric, times a volume term, so the average snap contributes zero.

THE DEFENSIVE DENOMINATOR IS PFF'S INDIVIDUAL SNAP COUNT, NOT TEAM PLAYS, and that
is the one place this module borrows from the other source. CFBD publishes no
individual snap counts on defense, so these rates used to divide by the team's
defensive plays - every defender on the roster carrying the same ~870 - and the
result was a facet that measured PLAYING TIME rather than quality. Correlation of
the facet z against the player's real snap count, before:

    tackle_lb .917   tackle_db .877   havoc_lb .786   havoc_db .777
    havoc_dl  .741   cov_db    .713        vs PFF's own LB_tackle at .134

A rotational defender was charged three quarters of a standard deviation of negative
value for not being on the field, and the two purest playing-time proxies were being
driven to exactly zero weight by the fit. It also cost these facets their share of
the replacement pool: build_hybrid withheld per-snap credit from them, because paying
it per a team denominator hands a 1-100 snap defender 67% of what a 600-snap starter
collects. That was 6.4% of total facet weight sitting outside the pool.

PFF's defense export carries snap_counts_defense per player, so the denominator
exists - it just lives in the other source. Joining it makes these true per-snap
rates, which fixes both problems at once and lets build_hybrid pay them normally.

The join is (season, CFBD team, normalized name), and it is required rather than
imputed: a player PFF never graded on defense has NO snap count, and inventing one
would be inventing the denominator. Those rows are dropped (~11.6%, and they already
contribute nothing to any PFF facet, so this makes their treatment consistent rather
than inconsistent). Team totals are standardized within season downstream, so the
season-to-season variation in coverage washes out there.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# PPA with garbage time excluded lives in a parallel cache. Backups accumulate a
# disproportionate share of their value in blowouts already decided, where the
# defense has emptied out; counting those snaps at face value had a 261-snap backup
# out-earning full-time starters. CFBD_GARBAGE=include reverts to the raw pull.
CACHE = f"{HERE}/cfbd_cache"
PPA_CACHE = (CACHE if os.environ.get("CFBD_GARBAGE") == "include"
             else f"{HERE}/cfbd_cache_gt")

# name -> (rate metric, volume metric, positions)
# Volume is a plays-equivalent so that the value column is comparable to PFF's
# z-score x snaps.
CFBD_FACETS = {
    "pass_qb":  ("ppa_pass",  "plays_pass", ("QB",)),
    "run_qb":   ("ppa_rush",  "plays_rush", ("QB",)),
    "run_rb":   ("ppa_rush",  "plays_rush", ("RB", "FB")),
    "recv_wr":  ("ppa_all",   "plays_recv", ("WR",)),
    "recv_te":  ("ppa_all",   "plays_recv", ("TE",)),
    "recv_rb":  ("ppa_all",   "plays_recv", ("RB", "FB")),
    # defense: rate = disruption per the player's OWN snap, volume = those snaps,
    # with the snap count borrowed from PFF. See the module docstring.
    "havoc_dl": ("havoc_rate", "def_snaps", ("DL", "DE", "DT", "EDGE")),
    "havoc_lb": ("havoc_rate", "def_snaps", ("LB",)),
    "havoc_db": ("havoc_rate", "def_snaps", ("DB", "CB", "S")),
    "tackle_lb": ("tackle_rate", "def_snaps", ("LB",)),
    "tackle_db": ("tackle_rate", "def_snaps", ("DB", "CB", "S")),
    "cov_db":   ("pd_rate",    "def_snaps", ("DB", "CB", "S")),
}

DEF_POS = ("DL", "DE", "DT", "EDGE", "LB", "DB", "CB", "S")


def load_ppa(years):
    """Player-season PPA joined to usage, flattened out of the nested JSON."""
    rows = []
    for y in years:
        src = PPA_CACHE if os.path.exists(f"{PPA_CACHE}/ppa_{y}.json") else CACHE
        ppa = json.load(open(f"{src}/ppa_{y}.json"))
        use = {str(r["id"]): r["usage"] for r in json.load(open(f"{CACHE}/usage_{y}.json"))}
        for r in ppa:
            a, u = r["averagePPA"], use.get(str(r["id"]), {})
            rows.append({
                "season": y, "player_id": str(r["id"]), "player": r["name"],
                "position": r["position"], "team": r["team"],
                "ppa_all": a.get("all"), "ppa_pass": a.get("pass"), "ppa_rush": a.get("rush"),
                "tot_all": r["totalPPA"].get("all"),
                "tot_pass": r["totalPPA"].get("pass"), "tot_rush": r["totalPPA"].get("rush"),
                "use_overall": u.get("overall"), "use_pass": u.get("pass"),
                "use_rush": u.get("rush"),
            })
    d = pd.DataFrame(rows)
    # plays = totalPPA / averagePPA, which recovers the denominator CFBD divided by.
    # Guard the near-zero average case, where the ratio explodes.
    for kind in ("all", "pass", "rush"):
        r, t = d[f"ppa_{kind}"], d[f"tot_{kind}"]
        d[f"plays_{kind}"] = np.where(r.abs() > 1e-6, t / r.where(r.abs() > 1e-6), np.nan)
    d["plays_recv"] = d["plays_all"]
    return d


def load_stats(years, categories):
    """stats/player/season is long-format; pivot to one column per statType."""
    rows = []
    for y in years:
        for cat in categories:
            p = f"{CACHE}/stats_{y}_{cat}.json"
            if os.path.exists(p):
                rows.extend(json.load(open(p)))
    d = pd.DataFrame(rows)
    d["stat"] = pd.to_numeric(d.stat, errors="coerce")
    d["playerId"] = d.playerId.astype(str)
    wide = d.pivot_table(index=["season", "playerId", "player", "position", "team"],
                         columns="statType", values="stat", aggfunc="sum").reset_index()
    return wide.rename(columns={"playerId": "player_id"})


def pff_def_snaps(years):
    """{(season, CFBD team, normalized name): defensive snaps} from the PFF export.

    Restricted to keys that identify exactly ONE PFF player. A repeated name on the
    same team in the same season cannot be resolved from the CFBD side - the two
    sources' position labels disagree far too often to break the tie with - so an
    ambiguous key yields no snap count and the row is dropped like any other miss.
    Only 16 of 53,442 PFF defensive player-seasons are ambiguous this way.
    """
    from facets import PFF_DIR
    from build_roster_2026 import norm_name
    team_map = json.load(open(f"{HERE}/team_map.json"))
    frames = []
    for y in years:
        p = f"{PFF_DIR}/defense_{y}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["season"] = y
        d["team"] = d.team_name.map(team_map)
        d["key"] = d.player.map(norm_name)
        frames.append(d[["season", "team", "key", "snap_counts_defense"]])
    if not frames:
        return {}
    pf = pd.concat(frames, ignore_index=True)
    pf = pf[pf.team.notna() & (pf.snap_counts_defense > 0)]
    n = pf.groupby(["season", "team", "key"]).size()
    pf = pf[[n.get(k, 0) == 1 for k in zip(pf.season, pf.team, pf.key)]]
    return dict(zip(zip(pf.season, pf.team, pf.key),
                    pf.snap_counts_defense.astype(float)))


def load_defense(years, team_plays):
    """Per-player defensive rates, denominated by the player's own PFF snap count.

    team_plays is still taken and still reported, because it is what tells us how
    much of a team's defense a player was actually on the field for - but it is no
    longer the denominator. See the module docstring for why that changed.
    """
    from build_roster_2026 import norm_name
    d = load_stats(years, ["defensive"])
    d = d[d.position.isin(DEF_POS)].copy()
    for c in ("TOT", "SOLO", "TFL", "SACKS", "PD", "QB HUR"):
        if c not in d.columns:
            d[c] = 0.0
        d[c] = d[c].fillna(0.0)
    d["team_def_plays"] = [team_plays.get((s, t), np.nan)
                           for s, t in zip(d.season, d.team)]

    snaps = pff_def_snaps(years)
    d["key"] = d.player.map(norm_name)
    d["def_snaps"] = [snaps.get((s, t, k), np.nan)
                      for s, t, k in zip(d.season, d.team, d.key)]
    # scoped to FBS for the report, because facet_values() drops everyone else
    # anyway and counting FCS misses here would overstate what is actually lost
    fbs = fbs_teams()
    scope = d[d.team.isin(fbs)] if fbs is not None else d
    kept = scope.def_snaps > 0
    print(f"  FBS defensive rows with a PFF snap count: {int(kept.sum())} of "
          f"{len(scope)} ({kept.mean()*100:.1f}%); the rest have no denominator "
          f"and are dropped")
    d = d[d.def_snaps > 0].copy()

    d["havoc"] = d["TFL"] + d["SACKS"] + d["QB HUR"] + d["PD"]
    d["havoc_rate"] = d.havoc / d.def_snaps
    d["tackle_rate"] = d["TOT"] / d.def_snaps
    d["pd_rate"] = d["PD"] / d.def_snaps
    return d


def team_def_plays(years):
    """Defensive plays faced, from the team advanced-stats endpoint."""
    out = {}
    for y in years:
        p = f"{CACHE}/team_adv_{y}.json"
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            dv = (r.get("defense") or {}).get("plays")
            if dv:
                out[(y, r["team"])] = float(dv)
    return out


def fbs_teams():
    """Teams with an FBS record, so normalization is not polluted by FCS opponents.

    CFBD returns every player it has, including FCS. Standardizing across that pool
    puts the zero point below FBS average and lets FCS outliers set the scale.
    """
    p = f"{HERE}/records.csv"
    return set(pd.read_csv(p).team) if os.path.exists(p) else None


# Volume below which a rate is mostly noise. PPA plays for a receiver are targets,
# not routes, so a five-target season can post a huge per-play average on one long
# touchdown; without shrinkage those players dominate the facet. k is the number of
# league-average plays mixed in, i.e. the volume at which a player gets half weight.
#
# The defensive facets need this now and did not before. Under the old team
# denominator every player divided by the same ~870, so a 12-snap defender simply
# scored near zero; dividing by his OWN 12 snaps instead lets one tackle-for-loss
# post a rate no starter can touch. k = 200 follows the same convention the offensive
# facets use - half weight at roughly the median volume, which for defensive snaps is
# 227 - and sits in the flat part of the reliability curve.
#
# NOT chosen by maximizing year-over-year rho, which keeps climbing all the way to
# k = 400 and beyond. That is an artefact: heavier shrinkage makes z more a function
# of snap count, and snap count is stable year to year, so rho measures the shrinkage
# as much as the player. Between k = 150 and k = 400 rho moves .503 -> .511 on
# havoc_dl while |r(snaps, z)| moves the wrong way throughout.
SHRINK_K = {
    "pass_qb": 60, "run_qb": 25, "run_rb": 40, "recv_wr": 25, "recv_te": 20,
    "recv_rb": 20,
    "havoc_dl": 200, "havoc_lb": 200, "havoc_db": 200,
    "tackle_lb": 200, "tackle_db": 200, "cov_db": 200,
}


def facet_values(years):
    """Snap-weighted z-scores per facet, in the same schema as facet_values.parquet."""
    ppa = load_ppa(years)
    tp = team_def_plays(years)
    dfn = load_defense(years, tp)
    fbs = fbs_teams()

    frames = []
    for name, (rate_col, vol_col, positions) in CFBD_FACETS.items():
        src = ppa if rate_col.startswith("ppa_") else dfn
        d = src[src.position.isin(positions)].copy()
        if fbs is not None:
            d = d[d.team.isin(fbs)]
        if d.empty:
            continue
        d = d.rename(columns={rate_col: "grade", vol_col: "snaps"})
        d["grade"] = pd.to_numeric(d.grade, errors="coerce")
        d["snaps"] = pd.to_numeric(d.snaps, errors="coerce")
        d = d[(d.snaps > 0) & d.grade.notna() & np.isfinite(d.snaps) & np.isfinite(d.grade)]
        # PPA plays reconstructed from a ratio can blow up; cap at a plausible max
        d = d[d.snaps < 2000]
        if d.empty:
            continue
        k = SHRINK_K.get(name, 0)
        for season, g in d.groupby("season"):
            w = g.snaps.to_numpy(float)
            x = g.grade.to_numpy(float)
            mu0 = np.average(x, weights=w)
            if k:
                x = (w * x + k * mu0) / (w + k)
            mu = np.average(x, weights=w)
            sd = np.sqrt(np.average((x - mu) ** 2, weights=w))
            if sd == 0:
                continue
            z = (x - mu) / sd
            frames.append(pd.DataFrame({
                "season": season, "player_id": g.player_id.to_numpy(),
                "player": g.player.to_numpy(), "position": g.position.to_numpy(),
                "cfbd_team": g.team.to_numpy(), "facet": name,
                "grade": x, "snaps": w, "z": z, "value": z * w,
            }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # default to whatever span the PFF facets cover, so the two halves of the
    # hybrid can never be built over different seasons
    from facets import YEARS as PFF_YEARS
    years = ([int(y) for y in os.environ["CFBD_YEARS"].split(",")]
             if os.environ.get("CFBD_YEARS") else list(PFF_YEARS))
    fv = facet_values(years)
    print(f"cfbd facet rows: {len(fv)}   facets: {fv.facet.nunique()}")
    print(fv.groupby("facet").agg(n=("value", "size"),
                                  mean_snaps=("snaps", "mean")).round(1).to_string())
    fv.to_parquet(f"{HERE}/cfbd_facet_values.parquet")
    print("cfbd_facet_values.parquet written")
