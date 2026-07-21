"""Feature-assembly analysis over a BROAD pool of CFBD season-advanced stats,
using the methodology doc's §4.2 criteria:

  - predictive signal : corr( feature in N-1 , team strength in N )   (forward!)
  - stability         : corr( feature in N-1 , feature in N )         (carry-over)
  - collinearity      : feature-feature correlation matrix            (redundancy)

A good feature is high on BOTH predictive signal and stability, and not redundant
with others. Saves two plots + prints a ranked table.
Run: ./venv/bin/python -m scripts.feature_analysis
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import STAT_YEARS, ARTIFACTS
from src.data import cfbd_client


# Broad candidate pool (flattened from /stats/season/advanced).
def _flatten(raw, year):
    rows = []
    for t in raw:
        o, d = t.get("offense") or {}, t.get("defense") or {}
        def g(b, *keys):
            for k in keys:
                b = (b or {}).get(k) if isinstance(b, dict) else None
            return b
        rows.append({
            "team": t["team"], "season": year,
            # offense
            "off_ppa": o.get("ppa"), "off_pass_ppa": g(o, "passingPlays", "ppa"),
            "off_rush_ppa": g(o, "rushingPlays", "ppa"), "off_success": o.get("successRate"),
            "off_expl": o.get("explosiveness"), "off_pts_opp": o.get("pointsPerOpportunity"),
            "off_power": o.get("powerSuccess"), "off_stuff": o.get("stuffRate"),
            "off_line_yds": o.get("lineYards"), "off_open_field": o.get("openFieldYards"),
            "off_havoc": g(o, "havoc", "total"), "off_std_ppa": g(o, "standardDowns", "ppa"),
            # defense
            "def_ppa": d.get("ppa"), "def_pass_ppa": g(d, "passingPlays", "ppa"),
            "def_rush_ppa": g(d, "rushingPlays", "ppa"), "def_success": d.get("successRate"),
            "def_expl": d.get("explosiveness"), "def_pts_opp": d.get("pointsPerOpportunity"),
            "def_power": d.get("powerSuccess"), "def_stuff": d.get("stuffRate"),
            "def_line_yds": d.get("lineYards"), "def_havoc": g(d, "havoc", "total"),
        })
    return pd.DataFrame(rows)


FEATS = ["off_ppa", "off_pass_ppa", "off_rush_ppa", "off_success", "off_expl",
         "off_pts_opp", "off_power", "off_stuff", "off_line_yds", "off_open_field",
         "off_havoc", "off_std_ppa", "def_ppa", "def_pass_ppa", "def_rush_ppa",
         "def_success", "def_expl", "def_pts_opp", "def_power", "def_stuff",
         "def_line_yds", "def_havoc"]


def _z(s):
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) or 1.0)


def main():
    from src.data import load
    load.require_key()
    print("Pulling broad season-advanced pool ...")
    by_year = {}
    for y in STAT_YEARS:
        df = _flatten(cfbd_client.advanced_season_stats(y), y).dropna(subset=FEATS)
        df = df.drop_duplicates("team").set_index("team")
        for f in FEATS:
            df[f] = _z(df[f])
        df["strength"] = _z(df["off_ppa"] - df["def_ppa"])  # team quality proxy
        by_year[y] = df

    # predictive signal (feature N-1 -> strength N) and stability (N-1 -> N)
    pred, stab = {}, {}
    for f in FEATS:
        px, py, sx, sy = [], [], [], []
        for N in STAT_YEARS:
            if (N - 1) not in by_year or N not in by_year:
                continue
            a, b = by_year[N - 1], by_year[N]
            common = a.index.intersection(b.index)
            px.extend(a.loc[common, f]); py.extend(b.loc[common, "strength"])
            sx.extend(a.loc[common, f]); sy.extend(b.loc[common, f])
        pred[f] = np.corrcoef(px, py)[0, 1]
        stab[f] = np.corrcoef(sx, sy)[0, 1]

    tbl = pd.DataFrame({"predictive_r": pred, "stability_r": stab})
    tbl["abs_pred"] = tbl["predictive_r"].abs()
    tbl = tbl.sort_values("abs_pred", ascending=False)
    print("\n=== Feature signal (sorted by |predictive r| with next-year strength) ===")
    print(f"{'feature':<15}{'pred_r':>9}{'stability_r':>13}")
    for f, r in tbl.iterrows():
        print(f"  {f:<13}{r['predictive_r']:>9.3f}{r['stability_r']:>13.3f}")

    # collinearity matrix (pooled across years)
    allz = pd.concat([by_year[y][FEATS] for y in by_year])
    corr = allz.corr()

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=7)
    ax.set_title("Feature collinearity (|r| near 1 = redundant)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(ARTIFACTS / "feature_corr.png", dpi=120)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    ax2.scatter(tbl["stability_r"], tbl["abs_pred"], s=30, color="#1E407C")
    for f, r in tbl.iterrows():
        ax2.annotate(f, (r["stability_r"], r["abs_pred"]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel("Stability (year-to-year r)")
    ax2.set_ylabel("Predictive signal (|r| with next-year strength)")
    ax2.set_title("Keep features toward the TOP-RIGHT (stable AND predictive)")
    ax2.grid(alpha=0.3); fig2.tight_layout()
    fig2.savefig(ARTIFACTS / "feature_signal.png", dpi=120)
    print(f"\nSaved -> artifacts/feature_corr.png, artifacts/feature_signal.png")


if __name__ == "__main__":
    main()
