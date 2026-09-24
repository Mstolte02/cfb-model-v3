"""Do the overlapping stack inputs each carry unique signal? Forward test, v5.2.

scripts/stack_collinearity.py found CFBD form and PFF form close to duplicates
(offence r .81, defence r .70 on live rows). Two near-copies in a logistic stack
split the credit arbitrarily, so the question is whether each adds anything the
other does not. Every variant is scored the way v5 extensions are
(scripts/v5_extension_backtest.py): the same four members, the same 2,189 games of
2023-25, the member's own penalty C, season-week block bootstrap against v5.2.

  v52                 shipped: prior_level, elo_change, dO, dD, hfa, pff_O, pff_D, war
  drop_dO / drop_pffO drop one offence measure
  drop_dD / drop_pffD drop one defence measure
  drop_both_cfbd      drop dO and dD (PFF carries form alone)
  x_off / x_def / x_both   v52 plus dO*pff_O and/or dD*pff_D
  drop_war            v5.1 columns on the v5.2 preseason models

Unique signal = dropping the column makes the forward score worse. If dropping one
of a pair costs nothing, the pair is redundant and that column goes; if both cost,
both stay and the interaction is the next thing to try.

    python -m scripts.stack_overlap_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS
from scripts import prior_decay_backtest as PD
from scripts import v5_extension_backtest as V
from scripts.stack_collinearity import PFF, WAR
from src import inseason_war as IW

OUT = ARTIFACTS / "stack_overlap_backtest.json"
BASE = list(V.BASE_COLS)                      # prior_level, elo_change, dO, dD, hfa
V52 = [*BASE, *PFF, *WAR]


def without(*cols):
    return [c for c in V52 if c not in cols]


VARIANTS = {
    "v52": V52,
    "drop_dO": without("dO"),
    "drop_pffO": without("pff_O_diff"),
    "drop_dD": without("dD"),
    "drop_pffD": without("pff_D_diff"),
    "drop_both_cfbd": without("dO", "dD"),
    "drop_both_pff": without("pff_O_diff", "pff_D_diff"),
    "x_off": [*V52, "x_off"],
    "x_def": [*V52, "x_def"],
    "x_both": [*V52, "x_off", "x_def"],
    "drop_war": without("war_delta_diff"),
}


def extra_frame(payload):
    reg = V.variants(payload)
    pff = reg["pff_outcome_composite"]()["extra"]
    games = pff[V.KEYS].drop_duplicates()
    war = games.assign(war_delta_diff=IW.game_war_column(
        games, pd.read_csv(IW.TEAM_HISTORY)))
    return pff.merge(war, on=V.KEYS, how="left")


def run(payload, extra, cols, name):
    preds = []
    for test in V.OUTER:
        member_p = []
        for spec in V.SPECS:
            designs = V.member_designs(payload, test, spec)
            prepared = {}
            for y, d in designs.items():
                d = d.merge(extra, on=V.KEYS, how="left")
                d[PFF + WAR] = d[PFF + WAR].fillna(0.0)
                d["x_off"] = d.dO * d.pff_O_diff
                d["x_def"] = d.dD * d.pff_D_diff
                prepared[y] = d
            train = pd.concat([d for y, d in prepared.items() if y != test],
                              ignore_index=True)
            C = payload["ctx"][(test, spec)]["sel"]["C"]
            p, _ = V.fit_predict_stack(train, prepared[test], cols, C)
            member_p.append(p)
        base = prepared[test][V.KEYS + ["y"]].copy()
        base["p"] = np.mean(member_p, axis=0)
        preds.append(base)
    return pd.concat(preds, ignore_index=True).assign(variant=name)


def main():
    payload = V.build_contexts()
    extra = extra_frame(payload)
    res = {name: run(payload, extra, cols, name) for name, cols in VARIANTS.items()}
    base = res["v52"].rename(columns={"p": "p_base"})
    report = {}
    for name, frame in res.items():
        m = base.merge(frame[V.KEYS + ["p"]], on=V.KEYS)
        y = m.y.to_numpy(float)
        p = np.clip(m.p.to_numpy(float), 1e-9, 1 - 1e-9)
        e = {"brier": float(np.mean((p - y) ** 2)),
             "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
             "by_season": {str(s): float(((g.p - g.y) ** 2).mean())
                           for s, g in m.groupby("season")}}
        if name != "v52":
            e["vs_v52"] = PD.bootstrap(m, "p", "p_base")
        report[name] = e
    OUT.write_text(json.dumps(report, indent=2, default=float))
    b = report["v52"]["brier"]
    print(f"{'variant':<16}{'Brier':>10}{'vs v5.2':>11}   95% CI               P(better)  by season")
    for name, e in report.items():
        if name == "v52":
            print(f"{name:<16}{e['brier']:>10.6f}{'':>11}   {'':<21}{'':>9}  "
                  + " ".join(f"{v:.5f}" for v in e["by_season"].values()))
            continue
        v = e["vs_v52"]
        print(f"{name:<16}{e['brier']:>10.6f}{v['difference']:>+11.6f}   "
              f"[{v['ci95'][0]:+.6f},{v['ci95'][1]:+.6f}] {v['probability_left_better']:>6.2f}  "
              + " ".join(f"{x:.5f}" for x in e["by_season"].values()))


if __name__ == "__main__":
    main()
