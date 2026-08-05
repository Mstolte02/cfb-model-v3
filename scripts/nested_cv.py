"""The honest forward number: every tuned knob selected INSIDE a held-out season.

Every accuracy this project has published was measured on the same leave-one-season-out
split that the knobs were chosen on. There are at least a dozen of them - the opponent
adjustment strength, the uncertainty lambda, two talent blend weights, the ensemble
weight, the shrinkage lambda, the Pythagorean exponent, the dropped-feature list, the
EA blend threshold, the score-shape switch, the logistic C, the ridge alpha - and each
was picked by looking at a LOSO score and keeping the best. A score selected on a split
is not a forward estimate of performance on that split, and with twelve knobs the gap
is not a rounding error.

This measures the gap. The outer loop holds out one season completely. Inside it, the
knobs are chosen by a LOSO run over the REMAINING seasons only - the held-out season is
not looked at, not once - and then a single model is fitted on those seasons with the
chosen settings and scored on the held-out one. The mean of those outer scores is the
only number here that means anything, and it is the only one printed as a headline.

WHAT IS TUNED INSIDE, and what is not. The four knobs with real surface and cheap
re-evaluation are swept here: OPP_ADJ_ALPHA, UNCERTAINTY_LAMBDA, TALENT_BLEND and
WAR_BLEND. The logistic C and the ridge alpha were already chosen inside the training
fold by model.train, so they were never part of the problem. The rest - the feature
list, the Pythagorean exponent, the score-shape table - are held at their configured
values, so the number below still flatters the model by whatever those are worth. It is
an upper bound on honesty, not a certificate.

Run: ./venv/bin/python -m scripts.nested_cv
"""
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import ARTIFACTS, GAME_YEARS
from src.data import load
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from src.data import pff
from scripts.train import blended_talent, load_bundle, raw_returning

# Deliberately coarse. A fine grid inside a nested loop costs a great deal of compute
# to select a knob to a precision the data does not support, and the inner selection
# is itself noisy - see the spread of chosen values printed per fold.
GRID = {
    "alpha": [0.5, 0.75, 0.85, 1.0],
    "lam": [0.5, 0.7, 0.9],
    "talent_blend": [0.3, 0.5, 0.7],
    "war_blend": [0.0, 0.4, 0.6],
}
DEFAULTS = {"alpha": 0.85, "lam": 0.70, "talent_blend": 0.5, "war_blend": 0.40}


class Bench:
    """Everything expensive is built once and cached by the knob it depends on."""

    def __init__(self):
        self.std, self.cfbd_tal, self.ret, self.games, self.pyth = load_bundle()
        self.ret_raw = raw_returning()
        self.pff_roster = pff.build_roster_talent()
        self._od, self._talent = {}, {}

    def od(self, alpha):
        if alpha not in self._od:
            self._od[alpha] = OA.build_od_by_year(self.std, self.games, alpha)
        return self._od[alpha]

    def talent(self, tb, wb):
        if (tb, wb) not in self._talent:
            self._talent[(tb, wb)] = blended_talent(
                self.cfbd_tal, self.pff_roster, w=tb, war_w=wb)
        return self._talent[(tb, wb)]

    def score(self, train_years, test_year, knobs):
        od = self.od(knobs["alpha"])
        tal = self.talent(knobs["talent_blend"], knobs["war_blend"])
        b_o, b_d = MU.fit_talent_od_slopes(train_years, self.std, tal, od_by_year=od)
        parts = MU.assemble(GAME_YEARS, self.std, self.pyth, tal, self.ret, self.games,
                            lam=knobs["lam"], b_o=b_o, b_d=b_d,
                            ret_raw_by_year=self.ret_raw, od_by_year=od)
        tr = [g for g in train_years if g in parts]
        if test_year not in parts or not tr:
            return None
        X = np.vstack([parts[g][0] for g in tr])
        y = np.concatenate([parts[g][1] for g in tr])
        hf = np.concatenate([parts[g][2] for g in tr])
        mdl, _ = M.train(X, y, hf)
        return M.evaluate(mdl, *parts[test_year])


def choose(bench, pool, verbose=False):
    """Coordinate descent over GRID, scored by LOSO WITHIN `pool` only."""
    knobs = dict(DEFAULTS)

    def inner(k):
        s = [bench.score([y for y in pool if y != h], h, k) for h in pool]
        s = [m["brier"] for m in s if m]
        return float(np.mean(s)) if s else np.inf

    best = inner(knobs)
    for _ in range(2):                       # two passes is enough to settle
        moved = False
        for name, values in GRID.items():
            for v in values:
                if v == knobs[name]:
                    continue
                trial = dict(knobs, **{name: v})
                sc = inner(trial)
                if sc < best - 1e-6:
                    best, knobs, moved = sc, trial, True
        if not moved:
            break
    return knobs, best


def main():
    print("Loading (cached CFBD pulls) ...")
    bench = Bench()
    folds = [y for y in GAME_YEARS if (y - 1) in bench.std]

    rows = []
    for outer in folds:
        pool = [y for y in folds if y != outer]
        knobs, inner_brier = choose(bench, pool)
        m = bench.score(pool, outer, knobs)
        rows.append({"held_out": outer, "inner_brier": inner_brier,
                     "outer_brier": m["brier"], "outer_logloss": m["log_loss"],
                     "outer_accuracy": m["accuracy"], **knobs})
        print(f"  {outer}: chose " +
              " ".join(f"{k}={knobs[k]}" for k in GRID) +
              f"   inner {inner_brier:.4f} -> OUTER {m['brier']:.4f}")

    ob = float(np.mean([r["outer_brier"] for r in rows]))
    ib = float(np.mean([r["inner_brier"] for r in rows]))
    acc = float(np.mean([r["outer_accuracy"] for r in rows]))

    # the number the project has been reporting: knobs fixed at their configured
    # values, scored on the very split they were selected on
    flat = [bench.score([y for y in folds if y != t], t, DEFAULTS) for t in folds]
    flat_b = float(np.mean([m["brier"] for m in flat if m]))

    print(f"\n{'':<34}{'Brier':>8}{'Acc':>8}")
    print(f"{'LOSO at the tuned config (reported)':<34}{flat_b:>8.4f}{'':>8}")
    print(f"{'inner-loop best (selection score)':<34}{ib:>8.4f}")
    print(f"{'NESTED, honest forward estimate':<34}{ob:>8.4f}{acc:>8.3f}")
    print(f"\nselection optimism: {ob - flat_b:+.4f} Brier "
          f"({(ob - flat_b) / flat_b * 100:+.1f}%)")
    chosen = {k: sorted({r[k] for r in rows}) for k in GRID}
    print("values chosen across outer folds "
          "(a knob that moves is a knob the data does not pin down):")
    for k, v in chosen.items():
        print(f"    {k:<14}{v}")

    out = {"folds": rows, "nested_brier": ob, "nested_accuracy": acc,
           "loso_at_configured": flat_b, "inner_best": ib,
           "selection_optimism": ob - flat_b, "grid": GRID, "defaults": DEFAULTS,
           "tuned_inside": list(GRID),
           "not_tuned_inside": ["FEATURES", "PYTHAG_EXP", "SHRINKAGE_LAMBDA",
                                "ENSEMBLE_W", "SCORE_SHAPE", "EA_BLEND_SNAPS"]}
    (ARTIFACTS / "nested_cv.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {ARTIFACTS / 'nested_cv.json'}")


if __name__ == "__main__":
    main()
