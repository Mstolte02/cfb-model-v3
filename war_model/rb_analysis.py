"""Stage 5: the running back deliverable.

Turns the league-wide WAR build into RB-specific outputs: leaderboards, a decomposition
of where each back's wins come from, and the divergence between WAR and the counting
stats a yards-based ranking would reward.
"""
import json, os
import numpy as np
import pandas as pd

import artifacts

from facets import YEARS, PFF_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
RB_FACETS = ["run_rb", "recv_rb", "pblk_skill", "rblk_skill", "fumble", "pen_off"]
PRETTY = {"run_rb": "Rushing", "recv_rb": "Receiving", "pblk_skill": "Pass blocking",
          "rblk_skill": "Run blocking", "fumble": "Ball security", "pen_off": "Penalties"}


def load():
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    fw = pd.read_parquet(f"{HERE}/{artifacts.FACET_WAR}")
    rush = pd.concat([pd.read_csv(f"{PFF_DIR}/rushing_{y}.csv").assign(season=y)
                      for y in YEARS], ignore_index=True)
    recv = pd.concat([pd.read_csv(f"{PFF_DIR}/receiving_{y}.csv").assign(season=y)
                      for y in YEARS], ignore_index=True)

    rb = war[war.position.isin(["HB", "FB"])].copy()
    # the hybrid build keys players by a string uid (a PFF id, or cfbd:<id> for a
    # player PFF never graded), so the PFF box scores have to be cast to match
    rb["player_id"] = rb.player_id.astype(str)
    for d in (rush, recv):
        d["player_id"] = d.player_id.astype(str)
    box = rush[["season", "player_id", "attempts", "yards", "ypa", "touchdowns",
                "yards_after_contact", "yco_attempt", "avoided_tackles", "fumbles",
                "grades_run", "player_game_count", "breakaway_percent", "first_downs"]]
    rb = rb.merge(box, on=["season", "player_id"], how="left")
    rb = rb.merge(recv[["season", "player_id", "receptions", "rec_yards_r"]]
                  .rename(columns={"rec_yards_r": "rec_yards"})
                  if "rec_yards_r" in recv.columns else
                  recv[["season", "player_id", "receptions", "yards"]]
                  .rename(columns={"yards": "rec_yards"}),
                  on=["season", "player_id"], how="left")

    # per-facet WAR for these players only
    f = fw[fw.facet.isin(RB_FACETS)].pivot_table(
        index=["season", "player_id"], columns="facet", values="war", aggfunc="sum").fillna(0.0)
    f.columns = [f"war_{c}" for c in f.columns]
    rb = rb.merge(f.reset_index(), on=["season", "player_id"], how="left")
    for c in [f"war_{c}" for c in RB_FACETS]:
        if c not in rb.columns:
            rb[c] = 0.0
        rb[c] = rb[c].fillna(0.0)
    return rb


def main():
    rb = load()
    rb["carries"] = rb.attempts.fillna(0)
    qual = rb[rb.carries >= 75].copy()
    print(f"RB player-seasons: {len(rb)}   with 75+ carries: {len(qual)}")

    # ---- rank divergence: WAR vs a pure rushing-yards ranking ---------------
    qual["rank_war"] = qual.groupby("season").war.rank(ascending=False)
    qual["rank_yards"] = qual.groupby("season").yards.rank(ascending=False)
    qual["rank_shift"] = qual.rank_yards - qual.rank_war  # positive = WAR likes him more

    print("\nWAR vs rushing-yards ranking, Spearman by season:")
    for s in YEARS:
        d = qual[qual.season == s]
        print(f"  {s}: rho = {d.rank_war.corr(d.rank_yards, method='spearman'):.3f}  (n={len(d)})")

    print("\nbiggest risers under WAR (yards rank -> WAR rank):")
    print(qual.nlargest(10, "rank_shift")[
        ["season", "player", "team", "yards", "rank_yards", "rank_war", "war"]]
        .to_string(index=False))
    print("\nbiggest fallers under WAR:")
    print(qual.nsmallest(10, "rank_shift")[
        ["season", "player", "team", "yards", "rank_yards", "rank_war", "war"]]
        .to_string(index=False))

    # ---- what drives RB wins ------------------------------------------------
    wcols = [f"war_{c}" for c in RB_FACETS]
    share = qual[wcols].sum()
    share = (share / share.sum() * 100).round(1)
    print("\nshare of total qualified-RB WAR by facet (%):")
    for c in wcols:
        print(f"  {PRETTY[c[4:]]:<15} {share[c]:5.1f}")

    # ---- correlation of WAR with the usual box-score suspects ---------------
    print("\ncorrelation of RB WAR with conventional metrics (75+ carries):")
    for c in ["yards", "ypa", "grades_run", "yco_attempt", "touchdowns",
              "avoided_tackles", "breakaway_percent", "carries"]:
        d = qual[[c, "war"]].dropna()
        print(f"  {c:<18} r = {d[c].corr(d.war):+.3f}")

    # ---- career totals ------------------------------------------------------
    career = (rb.groupby(["player_id", "player"], as_index=False)
                .agg(war=("war", "sum"), waa=("waa", "sum"), seasons=("season", "nunique"),
                     carries=("carries", "sum"), yards=("yards", "sum"),
                     teams=("team", lambda s: " / ".join(sorted(set(s.dropna()))))))
    print("\ntop 15 RBs, 2021-2025 cumulative WAR:")
    print(career.nlargest(15, "war")[["player", "teams", "seasons", "carries", "yards", "war"]]
          .to_string(index=False))

    qual.to_csv(f"{HERE}/rb_qualified.csv", index=False)
    rb.to_csv(f"{HERE}/rb_all.csv", index=False)
    career.to_csv(f"{HERE}/rb_career.csv", index=False)
    print("\nstage 5 written")


if __name__ == "__main__":
    main()
