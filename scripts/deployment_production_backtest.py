"""Part 1: does adjusting production for the difficulty of its alignment predict better?

For each job (route running, coverage, pass rush, run stopping, pass protection), the
within-player slope of production on alignment shares is fitted on the other seasons
(src/deployment.within_slopes) and shrunk toward zero by its own uncertainty
(beta * b^2 / (b^2 + se^2)). Then, for every player with games on both sides of a
cut (weeks 1-c and c+1-15), his late production is predicted from his early
production two ways:

  raw        early mean
  deployed   early mean - beta.(early shares - late shares)

i.e. the deployed arm knows that his late games put him in a harder or easier mix.
The error is weighted by late opportunity and reported as a percentage change; no
pass/fail gate. Graded seasons 2022-25, leave one season out.

    python -m scripts.deployment_production_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ARTIFACTS  # noqa: E402
from src import deployment as DP  # noqa: E402

OUT = ARTIFACTS / "deployment_production_backtest.json"
SEASONS = [2022, 2023, 2024, 2025]
CUTS = (3, 6, 9)


def windows(g, cut, shares):
    def agg(x):
        w = x.wt.to_numpy(float)
        return pd.Series({"y": np.average(x.y, weights=w), "wt": w.sum(),
                          **{s: np.average(x[s].fillna(0), weights=w) for s in shares}})
    e = g[g.week <= cut].groupby(["season", "player_id", "group"]).apply(agg)
    l = g[g.week > cut].groupby(["season", "player_id", "group"]).apply(agg)
    return e.join(l, lsuffix="_e", rsuffix="_l", how="inner").reset_index()


def main():
    report = {}
    for job, spec in DP.SPECS.items():
        g = DP.player_games(job, SEASONS)
        shares = [c for c in g.columns if c.startswith("sh_")]
        for grp in sorted(g.group.unique()):
            gg = g[g.group == grp]
            use = [s for s in shares if gg[s].fillna(0).mean() > .02]
            if len(use) < 2 or len(gg) < 500:
                continue
            kept = use[1:]                       # first share is the base
            err = {"raw": 0.0, "deployed": 0.0, "w": 0.0}
            betas = []
            for T in SEASONS:
                tr = gg[gg.season != T].dropna(subset=kept)
                b, se = DP.within_slopes(tr, kept)
                b = b * b ** 2 / (b ** 2 + se ** 2)          # shrink by uncertainty
                betas.append(dict(zip([k[3:] for k in kept], np.round(b, 4))))
                for c in CUTS:
                    wdw = windows(gg[gg.season == T], c, kept)
                    wdw = wdw[(wdw.wt_e > 0) & (wdw.wt_l > 0)]
                    if wdw.empty:
                        continue
                    dsh = np.column_stack([wdw[f"{s}_e"] - wdw[f"{s}_l"] for s in kept])
                    pred = {"raw": wdw.y_e, "deployed": wdw.y_e - dsh @ b}
                    for k, p in pred.items():
                        err[k] += float(np.sum(wdw.wt_l * (p - wdw.y_l) ** 2))
                    err["w"] += float(wdw.wt_l.sum())
            if not err["w"]:
                continue
            b_all, se_all = DP.within_slopes(gg.dropna(subset=kept), kept)
            res = {"base_alignment": use[0][3:],
                   "beta": dict(zip([k[3:] for k in kept], np.round(b_all, 4).tolist())),
                   "se": dict(zip([k[3:] for k in kept], np.round(se_all, 4).tolist())),
                   "error_change_pct": round(100 * (err["deployed"] / err["raw"] - 1), 3),
                   "player_games": int(len(gg)), "beta_by_left_out_season": betas}
            report[f"{job}:{grp}"] = res
            print(f"{job:15}{grp:5} base={res['base_alignment']:9} beta={res['beta']} "
                  f"se={res['se']}  error {res['error_change_pct']:+.3f}%", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
