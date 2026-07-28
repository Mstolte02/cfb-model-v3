"""How much should the roster matter, and how much should last season?

Two knobs control that balance and they have never been tuned together:

  WAR share          how much of the talent signal is player WAR, against PFF
                     roster grades and recruiting. Last swept BEFORE the facet
                     reweighting, which changed what WAR measures - its
                     next-season correlation went .436 -> .477 - so the old
                     optimum is stale by construction.

  uncertainty lambda how far a team's prior-season O/D composites regress toward
                     its talent baseline, scaled by how little production it
                     returns. lambda = 0 trusts last season completely; lambda = 1
                     sends a low-continuity team all the way to its roster
                     baseline. It sits at 0.25 today, lowered from 1.0 when the
                     feature set was trimmed and never revisited.

Sweeping them jointly matters because they are substitutes: both move weight from
last season's results toward this year's roster, so tuning either alone understates
how far the pair can go.

Run: ./venv/bin/python -m scripts.roster_vs_results
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, OPP_ADJ_ALPHA, TALENT_BLEND, WAR_BLEND, ARTIFACTS
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning
from scripts.talent_sweep import sources

WAR_GRID = [0.0, 0.25, 0.40, 0.55, 0.70, 1.0]
LAM_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]


def loso(talent, lam, std, ret, games, pyth, ret_raw, od):
    res = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != ty]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                            lam=lam, b_o=b_o, b_d=b_d,
                            ret_raw_by_year=ret_raw, od_by_year=od)
        if ty not in parts:
            continue
        av = [g for g in tr if g in parts]
        Xtr = np.vstack([parts[g][0] for g in av])
        ytr = np.concatenate([parts[g][1] for g in av])
        hf = np.concatenate([parts[g][2] for g in av])
        mdl, _ = M.train(Xtr, ytr, hf)
        ev = M.evaluate(mdl, parts[ty][0], parts[ty][1], parts[ty][2])
        for k in res:
            res[k].append(ev[k])
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    load.require_key()
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    src, _ = sources()

    def talent_at(war_share):
        """PFF and recruiting keep their 50/50 split of whatever WAR leaves."""
        rest = 1.0 - war_share
        wp, wc = rest * TALENT_BLEND, rest * (1 - TALENT_BLEND)
        return {N: wp * src["PFF"][N] + wc * src["CFBD"][N] + war_share * src["WAR"][N]
                for N in src["CFBD"]}

    print("=" * 74)
    print("JOINT SWEEP: WAR share (rows) x how far teams regress off last season")
    print("Brier, lower is better. lambda 0 = trust last season, 1 = trust roster.")
    print("=" * 74)
    header = "  WAR   " + "".join(f"  lam={l:.2f}" for l in LAM_GRID)
    print(header)
    print("-" * len(header))

    rows = []
    best = None
    for ws in WAR_GRID:
        talent = talent_at(ws)
        line = f"  {ws:.2f}  "
        for lam in LAM_GRID:
            m = loso(talent, lam, std, ret, games, pyth, ret_raw, od)
            rows.append({"war": ws, "lam": lam, **m})
            line += f"  {m['brier']:.4f}"
            if best is None or m["brier"] < best["brier"]:
                best = {"war": ws, "lam": lam, **m}
        print(line, flush=True)

    r = pd.DataFrame(rows)
    ship = r[(r.war == WAR_BLEND) & (r.lam == 0.25)]
    print(f"\nships today:  WAR {WAR_BLEND:.2f}, lambda 0.25 -> "
          f"Brier {float(ship.brier.iloc[0]):.4f}  acc {float(ship.accuracy.iloc[0]):.3f}")
    print(f"best of grid: WAR {best['war']:.2f}, lambda {best['lam']:.2f} -> "
          f"Brier {best['brier']:.4f}  acc {best['accuracy']:.3f}")
    d = best["brier"] - float(ship.brier.iloc[0])
    print(f"difference:   {d:+.4f}  "
          + ("worth adopting" if d < -0.0003 else "within noise - the surface is flat"))

    print("\ntop 8 combinations:")
    print(r.sort_values("brier").head(8).round(4).to_string(index=False))

    print("\nmarginal effect of each knob at the other's best value:")
    bl = r[r.lam == best["lam"]].sort_values("war")
    print("  holding lambda at %.2f:" % best["lam"])
    for _, x in bl.iterrows():
        print(f"    WAR {x.war:.2f} -> {x.brier:.4f}")
    bw = r[r.war == best["war"]].sort_values("lam")
    print("  holding WAR share at %.2f:" % best["war"])
    for _, x in bw.iterrows():
        print(f"    lambda {x.lam:.2f} -> {x.brier:.4f}")

    r.to_csv(ARTIFACTS / "roster_vs_results.csv", index=False)
    json.dump(best, open(ARTIFACTS / "roster_vs_results.json", "w"), indent=1)
    print(f"\n-> {ARTIFACTS / 'roster_vs_results.csv'}")


if __name__ == "__main__":
    main()
