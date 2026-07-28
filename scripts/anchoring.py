"""Is the projection just a copy of last season's finish?

The rating feels like an echo of the previous year, which is a testable claim rather
than a matter of taste. The test is not "does the projection correlate with last
season" - it should, teams are persistent - it is whether it correlates with last
season MORE THAN REALITY DOES.

Reality sets the ceiling: across the historical seasons, how well does a team's
finish in year N predict its finish in N+1? If the model's year-N-to-projection
correlation exceeds that, it is over-anchored - claiming more persistence than the
sport actually has. If it sits below, it is doing real work.

Run: ./venv/bin/python -m scripts.anchoring
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ROOT, ARTIFACTS, GAME_YEARS
from src.data import load, cfbd_client as c


def season_records(years):
    """Adjusted win pct per team-season, FBS games only."""
    out = {}
    for y in years:
        fbs = {t["school"] for t in c.fbs_teams(y)}
        w, g = {}, {}
        for gm in c.games(y):
            h, a = gm.get("homeTeam"), gm.get("awayTeam")
            hp, ap = gm.get("homePoints"), gm.get("awayPoints")
            if hp is None or ap is None or h not in fbs or a not in fbs:
                continue
            for t in (h, a):
                w.setdefault(t, 0); g.setdefault(t, 0)
            g[h] += 1; g[a] += 1
            if hp > ap:
                w[h] += 1
            else:
                w[a] += 1
        out[y] = pd.Series({t: w[t] / g[t] for t in w if g[t] >= 8})
    return out


def main():
    load.require_key()
    hist = [y for y in range(2014, 2026) if y != 2020]
    rec = season_records(hist)

    # ---- 1. how persistent is college football, actually? -------------------
    print("=" * 68)
    print("1. REALITY'S CEILING: year N win pct vs year N+1 win pct")
    print("=" * 68)
    rs = []
    for y in hist:
        if (y + 1) not in rec:
            continue
        a, b = rec[y], rec[y + 1]
        common = a.index.intersection(b.index)
        if len(common) < 80:
            continue
        r = float(np.corrcoef(a[common], b[common])[0, 1])
        rs.append(r)
        print(f"  {y} -> {y+1}: r = {r:+.3f}   (n={len(common)})")
    real = float(np.mean(rs))
    print(f"\n  mean season-to-season persistence: r = {real:.3f}")
    print(f"  -> a projection correlating with last season ABOVE {real:.3f} is")
    print(f"     claiming more persistence than the sport has.")

    # ---- 2. where does the 2026 projection sit? -----------------------------
    print("\n" + "=" * 68)
    print("2. THE 2026 PROJECTION vs 2025 RESULTS")
    print("=" * 68)
    ratings = json.loads((ROOT / "viz" / "data" / "ratings.json").read_text())["teams"]
    proj = pd.Series({t["team"]: t["power"] for t in ratings})
    projw = pd.Series({t["team"]: t.get("avg_wins") for t in ratings}).dropna()
    last = rec[2025]

    common = proj.index.intersection(last.index)
    r_power = float(np.corrcoef(proj[common], last[common])[0, 1])
    cw = projw.index.intersection(last.index)
    r_wins = float(np.corrcoef(projw[cw], last[cw])[0, 1])
    print(f"  2026 power rating vs 2025 win pct : r = {r_power:+.3f}  (n={len(common)})")
    print(f"  2026 projected wins vs 2025 win pct: r = {r_wins:+.3f}  (n={len(cw)})")
    print(f"  reality's season-to-season figure  : r = {real:+.3f}")
    gap = r_power - real
    print(f"\n  gap: {gap:+.3f}  -> "
          + ("OVER-ANCHORED: the projection is more persistent than the sport"
             if gap > 0.05 else
             "UNDER-anchored: the projection moves more than the sport does"
             if gap < -0.05 else
             "about right: the projection is as persistent as the sport is"))

    # ---- 3. how much does the board actually move? --------------------------
    print("\n" + "=" * 68)
    print("3. HOW MUCH MOVEMENT IS THERE?")
    print("=" * 68)
    last_rank = last.rank(ascending=False)
    proj_rank = proj.rank(ascending=False)
    d = pd.DataFrame({"last": last_rank, "proj": proj_rank}).dropna()
    d["move"] = d["last"] - d["proj"]
    print(f"  mean absolute rank change: {d.move.abs().mean():.1f} places")
    print(f"  teams moving 20+ places  : {(d.move.abs() >= 20).sum()} of {len(d)}")
    print("\n  biggest risers (2025 finish -> 2026 projection):")
    for t, r in d.nlargest(6, "move").iterrows():
        print(f"    {t:<22} #{int(r['last']):>3} -> #{int(r['proj']):>3}   +{int(r['move'])}")
    print("\n  biggest fallers:")
    for t, r in d.nsmallest(6, "move").iterrows():
        print(f"    {t:<22} #{int(r['last']):>3} -> #{int(r['proj']):>3}   {int(r['move'])}")


if __name__ == "__main__":
    main()
