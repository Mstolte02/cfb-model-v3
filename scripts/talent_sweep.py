"""How much of each talent source should the model actually use?

Three signals compete for the talent slot:

  PFF     roster-aware grade average - this year's roster carrying last year's grades
  CFBD    247 recruiting composite
  WAR     summed prior-year WAR of this year's roster

The weights that ship were never jointly fitted: PFF/CFBD came from a 50/50 test and
WAR was bolted on at 25% by a one-dimensional sweep holding that 50/50 fixed. Worse,
WAR is *built from* PFF grades - the facets underneath it are PFF pass, coverage,
blocking and rush grades plus CFBD play value - so PFF and WAR may be close to the
same measurement twice, which a one-at-a-time sweep cannot reveal.

Part 1 measures how correlated the three actually are. Part 2 grid-searches the whole
simplex under LOSO, so the answer is jointly fitted rather than assembled.

Run: ./venv/bin/python -m scripts.talent_sweep [step]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA, ARTIFACTS
from src.data import load, pff, war
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning

FEATURES = ["O", "D", "fp_margin", "pythag", "talent", "returning"]


def sources():
    """{name: {season: Series}} for the three talent signals, on a common index."""
    std, cfbd_tal, ret, games, pyth = load_bundle()
    roster = pff.build_roster_talent()
    war_tal = war.talent_by_year(
        {y: s.index for y, s in cfbd_tal.items() if s is not None})

    out = {"CFBD": {}, "PFF": {}, "WAR": {}}
    for N, base in cfbd_tal.items():
        if base is None:
            continue
        out["CFBD"][N] = base
        r = roster.get(N)
        out["PFF"][N] = base if r is None else r.reindex(base.index).fillna(base)
        w = war_tal.get(N)
        out["WAR"][N] = out["PFF"][N] if w is None else w.reindex(base.index).fillna(out["PFF"][N])
    return out, (std, cfbd_tal, ret, games, pyth)


def run(talent, std, ret, games, pyth, ret_raw, od):
    """LOSO for one talent vector, refitting slopes and model per fold."""
    out = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != ty]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                            lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                            ret_raw_by_year=ret_raw, od_by_year=od)
        if ty not in parts:
            continue
        Xtr = np.vstack([parts[g][0] for g in tr if g in parts])
        ytr = np.concatenate([parts[g][1] for g in tr if g in parts])
        hf = np.concatenate([parts[g][2] for g in tr if g in parts])
        mdl, _ = M.train(Xtr, ytr, hf)
        m = M.evaluate(mdl, *parts[ty])
        for k in out:
            out[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main(step=0.125):
    load.require_key()
    src, (std, cfbd_tal, ret, games, pyth) = sources()
    ret_raw = raw_returning()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    args = (std, ret, games, pyth, ret_raw, od)

    # ---- 1. how different are these three, really? --------------------------
    print("=" * 66)
    print("1. CORRELATION BETWEEN THE TALENT SOURCES (pooled across seasons)")
    print("=" * 66)
    names = ["PFF", "CFBD", "WAR"]
    stacked = {}
    for n in names:
        stacked[n] = pd.concat([src[n][y].rename(y) for y in sorted(src[n])],
                               axis=0, keys=sorted(src[n]))
    df = pd.DataFrame(stacked).dropna()
    C = df.corr()
    print(C.round(3).to_string())
    print(f"\n  n = {len(df)} team-seasons")
    print("\n  by season:")
    for y in sorted(src["PFF"]):
        d = pd.DataFrame({n: src[n][y] for n in names}).dropna()
        print(f"    {y}:  PFF~WAR {d.PFF.corr(d.WAR):+.3f}   "
              f"PFF~CFBD {d.PFF.corr(d.CFBD):+.3f}   WAR~CFBD {d.WAR.corr(d.CFBD):+.3f}")

    # ---- 2. joint sweep over the simplex ------------------------------------
    print("\n" + "=" * 66)
    print(f"2. JOINT WEIGHT SWEEP (step {step}), LOSO 2021-25")
    print("=" * 66)
    grid = []
    k = int(round(1 / step))
    for i in range(k + 1):
        for j in range(k + 1 - i):
            w = (i * step, j * step, 1 - i * step - j * step)
            grid.append(tuple(round(x, 4) for x in w))

    results = []
    for wp, wc, ww in grid:
        talent = {}
        for N in src["CFBD"]:
            talent[N] = wp * src["PFF"][N] + wc * src["CFBD"][N] + ww * src["WAR"][N]
        m = run(talent, *args)
        results.append({"pff": wp, "cfbd": wc, "war": ww, **m})
        print(f"  PFF {wp:.3f}  CFBD {wc:.3f}  WAR {ww:.3f}   "
              f"Brier {m['brier']:.4f}  acc {m['accuracy']:.3f}", flush=True)

    r = pd.DataFrame(results).sort_values("brier")
    print("\n  top 8 blends:")
    print(r.head(8).round(4).to_string(index=False))
    best = r.iloc[0]
    ship = r[(r.pff.round(3) == 0.375) & (r.cfbd.round(3) == 0.375)]
    print(f"\n  best: PFF {best.pff:.3f} / CFBD {best.cfbd:.3f} / WAR {best.war:.3f} "
          f"-> Brier {best.brier:.4f}, acc {best.accuracy:.3f}")
    if len(ship):
        s = ship.iloc[0]
        print(f"  nearest to what ships (0.375/0.375/0.25): Brier {s.brier:.4f}")

    r.to_csv(ARTIFACTS / "talent_sweep.csv", index=False)
    json.dump({"best": {k: float(best[k]) for k in
                        ("pff", "cfbd", "war", "brier", "log_loss", "accuracy")},
               "corr": {a: {b: float(C.loc[a, b]) for b in names} for a in names},
               "n_team_seasons": int(len(df))},
              open(ARTIFACTS / "talent_sweep.json", "w"), indent=1)
    print(f"\n-> {ARTIFACTS / 'talent_sweep.csv'}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.125)
