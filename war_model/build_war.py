"""Stage 3: wins above average, then wins above replacement, for every player-season.

WAA follows the paper: recompute the team's Massey rating with the player's grades
replaced by an average (zero) player at the same snap counts, and take the difference
in implied wins. Because M is fixed within a season and removing one player perturbs
exactly one entry of f, this is done exactly in closed form via A^-1 rather than by
re-solving the system 50,000 times.

WAR = WAA + the player's per-snap share of the league-wide pool of wins between an
average team and a replacement-level team.
"""
import json, os
import numpy as np
import pandas as pd

from facets import FACETS, YEARS
from build_massey import massey_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FACET_NAMES = list(FACETS)
TEAM_MAP = json.load(open(f"{HERE}/team_map.json"))

# Win pct of a roster made entirely of replacement-level FBS players. The NFL paper
# uses 3 of 16 (.188); FBS is far more stratified than the NFL, and the weakest real
# rosters in this sample already imply 2-3 wins of ~12, so a true replacement roster
# belongs below them. Rankings are invariant to this choice; only the level moves.
REPL_WIN_PCT = 0.15


def team_sensitivity(sched, season, teams):
    """c_t = d(rating_t) / d(f_t), holding every other team's f fixed.

    f is mean-centered before solving, so bumping one team's raw f by delta moves
    the centered vector by delta * (e_t - 1/n). The pinned last row of b stays 0.
    """
    M, idx = massey_matrix(sched, season, teams)
    A = M.copy()
    A[-1, :] = 1.0
    Ainv = np.linalg.inv(A)
    n = len(teams)
    c = np.zeros(n)
    for t in range(n):
        v = -np.ones(n) / n
        v[t] += 1.0
        v[-1] = 0.0  # b[-1] is pinned to 0 and never moves
        c[t] = (Ainv @ v)[t]
    return pd.Series(c, index=teams)


def main():
    fv = pd.read_parquet(f"{HERE}/facet_values.parquet")
    recs = pd.read_csv(f"{HERE}/records.csv")
    sched = pd.read_csv(f"{HERE}/schedule.csv")
    weights = pd.read_csv(f"{HERE}/facet_weights.csv", index_col=0)
    tot = pd.read_csv(f"{HERE}/team_facet_totals.csv", index_col=[0, 1])
    wmap = json.load(open(f"{HERE}/wins_map.json"))
    slope, intercept = wmap["slope"], wmap["intercept"]
    w = weights["rf"]

    fv["team"] = fv.team_name.map(TEAM_MAP)
    rec_idx = recs.set_index(["season", "team"])
    fv = fv[[(s, t) in rec_idx.index for s, t in zip(fv.season, fv.team)]]

    # sigma used when standardizing team facet totals within a season
    sigma = tot.groupby(level="season").std(ddof=0)

    # ---- WAA ----------------------------------------------------------------
    sens, games = {}, {}
    for season in YEARS:
        teams = sorted(recs[recs.season == season].team)
        sens[season] = team_sensitivity(sched, season, teams)
        games[season] = recs[recs.season == season].set_index("team").fbs_games

    fv["sigma"] = [sigma.loc[s, f] for s, f in zip(fv.season, fv.facet)]
    fv["w"] = fv.facet.map(w)
    # the player's contribution to his team's standardized facet total
    fv["f_contrib"] = fv.w * fv.value / fv.sigma
    fv["c_t"] = [sens[s][t] for s, t in zip(fv.season, fv.team)]
    fv["games"] = [games[s][t] for s, t in zip(fv.season, fv.team)]
    fv["waa"] = fv.games * slope * fv.c_t * fv.f_contrib

    print(f"c_t range: {min(s.min() for s in sens.values()):.4f} "
          f"to {max(s.max() for s in sens.values()):.4f}")

    # ---- replacement level --------------------------------------------------
    # The paper fixes its anchors by assumption ("a team of average players will win
    # an average of eight games, while a team of replacement-level players will win
    # an average of three"), and we do the same, for a concrete reason: the
    # rating -> wins map is fitted on real rosters, so extrapolating it to a
    # hypothetical all-replacement team lands ~2 SD outside the fitted range where
    # OLS attenuation pulls the estimate toward .500. The empirical figure is still
    # computed below and reported, but it is not what sets the pool.
    repl_z = {}
    for (season, facet), g in fv.groupby(["season", "facet"]):
        cut = g.snaps.quantile(0.25)
        fringe = g[g.snaps <= cut]
        repl_z[(season, facet)] = np.average(fringe.z, weights=fringe.snaps) if len(fringe) else 0.0

    avg_snaps = fv.groupby(["season", "facet", "team"]).snaps.sum().groupby(["season", "facet"]).mean()
    empirical = {}
    for season in YEARS:
        f_repl = sum(w[facet] * repl_z[(season, facet)] * avg_snaps.loc[season, facet]
                     / sigma.loc[season, facet]
                     for facet in FACET_NAMES if (season, facet) in repl_z)
        g_avg = recs[recs.season == season].fbs_games.mean()
        empirical[season] = slope * (f_repl / g_avg) + intercept

    repl_pct = {s: REPL_WIN_PCT for s in YEARS}
    print(f"\nreplacement anchor: {REPL_WIN_PCT:.3f} win pct "
          f"(~{REPL_WIN_PCT * 11.9:.1f} wins of 11.9)")
    print("  attenuated empirical estimate, for reference: "
          + ", ".join(f"{s}={empirical[s]:.3f}" for s in YEARS))

    # ---- apportion the replacement pool per snap ----------------------------
    pool_rows = []
    for season in YEARS:
        r = recs[recs.season == season]
        pool = float(((0.5 - repl_pct[season]) * r.fbs_games).sum())
        league_snaps = fv[fv.season == season].groupby("facet").snaps.sum()
        for facet in FACET_NAMES:
            if facet not in league_snaps or league_snaps[facet] == 0:
                continue
            pool_rows.append({"season": season, "facet": facet,
                              "per_snap": pool * w[facet] / league_snaps[facet]})
        print(f"  {season}: WAR pool = {pool:.0f} wins across {len(r)} teams")
    per_snap = pd.DataFrame(pool_rows).set_index(["season", "facet"]).per_snap

    fv["repl_credit"] = [per_snap.get((s, f), 0.0) for s, f in zip(fv.season, fv.facet)] * fv.snaps
    fv["war"] = fv.waa + fv.repl_credit

    # ---- collapse to player-seasons ----------------------------------------
    key = ["season", "player_id", "player", "position", "team_name", "team"]
    war = fv.groupby(key, as_index=False).agg(
        waa=("waa", "sum"), war=("war", "sum"), snaps=("snaps", "sum"))
    facet_war = fv.pivot_table(index=key, columns="facet", values="war", aggfunc="sum").fillna(0.0)

    war.to_csv(f"{HERE}/player_war.csv", index=False)
    facet_war.to_csv(f"{HERE}/player_war_by_facet.csv")
    fv.to_parquet(f"{HERE}/facet_war.parquet")
    json.dump({str(k): v for k, v in repl_pct.items()},
              open(f"{HERE}/replacement.json", "w"), indent=1)

    print(f"\nplayer-seasons: {len(war)}   total WAR: {war.war.sum():.0f}")
    print("\ntop 15 by WAR, all positions, 2025:")
    print(war[war.season == 2025].nlargest(15, "war")[
        ["player", "position", "team", "waa", "war"]].to_string(index=False))


if __name__ == "__main__":
    main()
