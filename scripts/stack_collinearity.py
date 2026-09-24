"""How much do the live stack's inputs overlap? VIF and correlations, v5.2.

The stack reads eight pregame columns: prior_level, elo_change, dO, dD, hfa (v5),
pff_O_diff, pff_D_diff (v5.1) and war_delta_diff (v5.2). Several measure the same
thing a second way - CFBD opponent-adjusted offence (dO) and PFF offence (pff_O_diff)
most obviously - and a logistic stack given two near-duplicates splits the credit
between them arbitrarily, which is how Ole Miss's +2.0 SD offence ended up costing it
0.9pp. This measures the overlap on the rows the stacks are trained on.

VIF_j = 1 / (1 - R^2_j), R^2_j from regressing column j on the others with an
intercept. Reported on all rows and on the rows where every in-season input is live
(week >= 4 of 2022-25: form needs two games, PFF one, WAR a cut at week 3).

    python -m scripts.stack_collinearity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS
from scripts import v5_extension_backtest as V
from src import inseason_war as IW

OUT = ARTIFACTS / "stack_collinearity.json"
PFF = ["pff_O_diff", "pff_D_diff"]
WAR = ["war_delta_diff"]


def designs(payload, test=2025):
    """Every season's pregame design rows for each member, with the v5.1/v5.2 columns."""
    reg = V.variants(payload)
    pff = reg["pff_outcome_composite"]()["extra"]
    games = pff[V.KEYS].drop_duplicates()
    war = games.assign(war_delta_diff=IW.game_war_column(
        games, pd.read_csv(IW.TEAM_HISTORY)))
    out = {}
    for spec in V.SPECS:
        d = pd.concat([x.assign(season=y) if "season" not in x else x
                       for y, x in V.member_designs(payload, test, spec).items()],
                      ignore_index=True)
        d = d.merge(pff, on=V.KEYS, how="left").merge(war, on=V.KEYS, how="left")
        d[PFF + WAR] = d[PFF + WAR].fillna(0.0)
        out[spec] = d
    return out


def vif(frame: pd.DataFrame, cols) -> dict:
    X = frame[cols].to_numpy(float)
    out = {}
    for j, c in enumerate(cols):
        y = X[:, j]
        others = np.c_[np.ones(len(X)), np.delete(X, j, axis=1)]
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss if ss > 0 else np.nan
        out[c] = {"r2": float(r2), "vif": float(1 / (1 - r2)) if r2 < 1 else float("inf")}
    return out


def main():
    payload = V.build_contexts()
    cols = [*V.BASE_COLS, *PFF, *WAR]
    report = {}
    for spec, d in designs(payload).items():
        live = d[(d.week >= 4) & (d.season >= 2022)]
        report[spec] = {
            "n_all": int(len(d)), "n_live": int(len(live)),
            "vif_all": vif(d, cols), "vif_live": vif(live, cols),
            "corr_live": live[cols].corr().round(3).to_dict(),
        }
    OUT.write_text(json.dumps(report, indent=2))
    spec = V.SPECS[0]
    r = report[spec]
    print(f"{spec}: {r['n_all']} rows, {r['n_live']} with every input live\n")
    print(f"{'column':<16}{'VIF all':>9}{'VIF live':>10}   R^2 live")
    for c in cols:
        print(f"{c:<16}{r['vif_all'][c]['vif']:>9.2f}{r['vif_live'][c]['vif']:>10.2f}"
              f"   {r['vif_live'][c]['r2']:.3f}")
    print("\ncorrelations, live rows:")
    print(pd.DataFrame(r["corr_live"]).loc[cols, cols].to_string())
    print("\nVIF live by member:")
    print(pd.DataFrame({s: {c: round(v["vif_live"][c]["vif"], 2) for c in cols}
                        for s, v in report.items()}).to_string())


if __name__ == "__main__":
    main()
