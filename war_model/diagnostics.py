"""Stage 4: replicate the paper's validation checks on the CFB build."""
import json, os
import numpy as np
import pandas as pd

import artifacts

from facets import POS_GROUP, YEARS, PFF_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 200)


def yoy(df, value, snap_min):
    """Year-over-year correlation, thresholded on snaps in the first season."""
    a = df[df.snaps >= snap_min][["season", "player_id", value]].copy()
    b = df[["season", "player_id", value]].copy()
    b["season"] -= 1
    j = a.merge(b, on=["season", "player_id"], suffixes=("", "_n1"))
    if len(j) < 30:
        return np.nan, len(j)
    return np.corrcoef(j[value], j[f"{value}_n1"])[0, 1], len(j)


def main():
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    recs = pd.read_csv(f"{HERE}/records.csv")
    repl = {int(k): v for k, v in json.load(open(f"{HERE}/replacement.json")).items()}
    war["group"] = war.position.map(POS_GROUP)
    war = war[war.group.notna()]

    print("=" * 78)
    print("CHECK 1  WAA sums to zero (every player measured against the average)")
    print("=" * 78)
    for s in YEARS:
        d = war[war.season == s]
        print(f"  {s}: sum(WAA) = {d.waa.sum():+.3f}   sum(WAR) = {d.war.sum():7.1f}")

    print()
    print("=" * 78)
    print("CHECK 2  replacement level vs the actual distribution of team quality")
    print("=" * 78)
    team = war.groupby(["season", "team"], as_index=False).war.sum()
    team = team.merge(recs, on=["season", "team"])
    team["implied_wins"] = team.war + team.season.map(repl) * team.fbs_games
    print(f"  implied wins vs actual FBS wins: r = "
          f"{np.corrcoef(team.implied_wins, team.fbs_wins)[0,1]:.3f}")
    print(f"  mean abs error: {np.abs(team.implied_wins - team.fbs_wins).mean():.2f} wins")
    worst = team.nsmallest(5, "implied_wins")[["season", "team", "fbs_wins", "fbs_games", "implied_wins"]]
    print("\n  weakest rosters by implied wins (replacement floor should sit at or below these):")
    print(worst.to_string(index=False))
    print(f"\n  replacement floor: {np.mean(list(repl.values())) * 11.9:.1f} wins of ~11.9")

    print()
    print("=" * 78)
    print("CHECK 3  paper Table 2 analogue: value and stability by position")
    print("=" * 78)
    rows = []
    for g, d in war.groupby("group"):
        thr = d.snaps.quantile(0.60)
        q = d[d.snaps >= thr]
        r, n = yoy(d, "war", thr)
        rows.append({"pos": g, "n": len(q), "mean_WAR": q.war.mean(),
                     "CV": q.war.std() / abs(q.war.mean()) if q.war.mean() else np.nan,
                     "yoy_r": r, "yoy_n": n})
    t2 = pd.DataFrame(rows).set_index("pos").reindex(
        ["QB", "RB", "WR", "TE", "T", "G", "C", "DI", "ED", "LB", "CB", "S"])
    print(t2.round(3).to_string())
    print("\n  NFL paper for reference:  QB 1.63 mean / 0.62 yoy | RB 0.10 / 0.53 | WR 0.28 / 0.52")

    print()
    print("=" * 78)
    print("CHECK 4  is WAR more stable than the raw inputs it is built from?")
    print("=" * 78)
    rush = pd.concat([pd.read_csv(f"{PFF_DIR}/rushing_{y}.csv")
                      .assign(season=y) for y in YEARS], ignore_index=True)
    # the hybrid build keys players by a string uid (a PFF id, or cfbd:<id>), so the
    # PFF box scores have to be cast to match before joining
    war = war.copy(); war["player_id"] = war.player_id.astype(str)
    rush = rush.copy(); rush["player_id"] = rush.player_id.astype(str)
    rb = war[war.group == "RB"].merge(
        rush[["season", "player_id", "grades_run", "yards", "ypa", "attempts"]],
        on=["season", "player_id"], how="left")
    rb = rb[rb.attempts.notna()]
    print(f"  RBs with 100+ carries in season n, measured into season n+1:")
    for col in ["war", "waa", "grades_run", "yards", "ypa"]:
        d = rb.rename(columns={"attempts": "_a"}).copy()
        a = d[d._a >= 100][["season", "player_id", col]]
        b = d[["season", "player_id", col]].copy()
        b["season"] -= 1
        j = a.merge(b, on=["season", "player_id"], suffixes=("", "_n1")).dropna()
        r = np.corrcoef(j[col], j[f"{col}_n1"])[0, 1] if len(j) > 30 else np.nan
        print(f"    {col:<11} r = {r:.3f}   (n = {len(j)})")

    team.to_csv(f"{HERE}/team_implied_wins.csv", index=False)
    t2.to_csv(f"{HERE}/position_table.csv")


if __name__ == "__main__":
    main()
