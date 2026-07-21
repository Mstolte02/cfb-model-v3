"""Compare all non-model talent/prior signals: how they correlate with each other
and with the target (next-season win%). Also re-optimizes the PFF position-group
weights (NNLS on win%), since PFF grades differ from PFSN.

All signals are "entering season N" (preseason-known); target = season-N win%.
Prior-year signals use N-1; recruiting/returning/roster use the N-entering value.
Run: ./venv/bin/python -m scripts.compare_signals
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from config import STAT_YEARS, ARTIFACTS
from src.data import load, cfbd_client, trumedia, pff
from src import projection as P
from scripts.validate_pfsn import load_pfsn


def _z(s):
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def win_pct(years):
    out = {}
    for y in years:
        g = load.games(y); w = {}; n = {}
        for _, r in g.iterrows():
            hw = r.home_points > r.away_points
            for t, win in [(r.home_team, hw), (r.away_team, not hw)]:
                w[t] = w.get(t, 0) + int(win); n[t] = n.get(t, 0) + 1
        out[y] = pd.Series({t: w[t] / n[t] for t in w})
    return out


def prior_strength(year):
    raw = cfbd_client.advanced_season_stats(year)
    rows = [(t["team"], (t.get("offense") or {}).get("ppa", np.nan)
             - (t.get("defense") or {}).get("ppa", np.nan)) for t in raw]
    return pd.DataFrame(rows, columns=["team", "v"]).dropna().drop_duplicates(
        "team").set_index("team")["v"]


def optimize_pff_weights(gscores, wp):
    """NNLS weights so the weighted group score best predicts win%."""
    wide = gscores.pivot_table(index=["season", "team"], columns="group",
                               values="group_score")
    groups = list(wide.columns)
    rows, y = [], []
    for (season, team), r in wide.iterrows():
        if season in wp and team in wp[season].index and r.notna().all():
            rows.append(r.values); y.append(wp[season][team])
    X = np.array(rows); y = np.array(y)
    Xz = (X - X.mean(0)) / X.std(0)
    yz = (y - y.mean()) / y.std()
    coef, _ = nnls(Xz, yz)
    w = coef / coef.sum()
    return dict(zip(groups, w)), len(y)


def main():
    load.require_key()
    years = [y for y in STAT_YEARS if y >= 2022]
    print("Assembling signals ...")
    wp = win_pct(STAT_YEARS)

    # Re-optimize PFF weights on win%, then build roster talent with them.
    gscores = pff.build_group_scores()
    pff_w, n = optimize_pff_weights(gscores, wp)
    print(f"\n=== PFF-optimized position weights (NNLS on win%, n={n}) ===")
    print("  " + "  ".join(f"{g} {pff_w.get(g,0)*100:.0f}%" for g in
          ["QB", "CB", "EDGE", "DT", "SAF", "LB", "WR", "OL", "TE", "RB"]))
    print("  PFSN weights:      QB 28% CB 22% EDGE 15% DT 12% SAF 9% LB 6% WR 3% OL 3% TE 1% RB 1%")
    pff_roster = pff.build_roster_talent(weights=pff_w, group_scores=gscores)

    # Other signals
    cfbd_names = set().union(*[set(prior_strength(y).index) for y in STAT_YEARS])
    pfsn_team, _ = load_pfsn(cfbd_names)
    tm = trumedia.load()
    pythag = P.build_pythag({y: load.games(y) for y in STAT_YEARS})

    rows = []
    for N in years:
        fp = tm[tm.season == N - 1].set_index("team")["fp_margin"]
        rec = load.returning_production(N).set_index("team")["rp"]
        cf = load.talent(N).set_index("team")["talent"]
        teams = [t for t in wp[N].index if t in cf.index]   # FBS only
        for t in teams:
            rows.append({
                "season": N, "team": t, "win_pct": wp[N][t],
                "CFBD_talent": cf.get(t), "returning": rec.get(t),
                "PFF_roster": pff_roster.get(N, pd.Series()).get(t),
                "PFSN_team_prev": pfsn_team.get(N - 1, pd.Series()).get(t),
                "pythag_prev": pythag.get(N - 1, pd.Series()).get(t),
                "fp_margin_prev": fp.get(t),
                "prior_strength": prior_strength(N - 1).get(t),
            })
    df = pd.DataFrame(rows)
    sigs = ["CFBD_talent", "returning", "PFF_roster", "PFSN_team_prev",
            "pythag_prev", "fp_margin_prev", "prior_strength"]
    # standardize each signal within season
    for s in sigs:
        df[s] = df.groupby("season")[s].transform(_z)

    print("\n=== Correlation with target (next-season win%) ===")
    tcorr = df[sigs].corrwith(df["win_pct"]).sort_values(key=abs, ascending=False)
    for s, r in tcorr.items():
        print(f"  {s:<16}{r:>7.3f}")

    cols = sigs + ["win_pct"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Signal correlations (each other + target win%)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(ARTIFACTS / "signal_correlations.png", dpi=120)
    print(f"\nSaved -> artifacts/signal_correlations.png   (n={len(df)} team-seasons)")


if __name__ == "__main__":
    main()
