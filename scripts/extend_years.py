"""Should the win-probability model train on 2015-19 as well as 2021-25?

PFF now reaches back to 2014, so the talent signal could cover nine seasons instead of
four. The blocker is that the O and D composites are not the same measurement in both
eras: four of their ten inputs come from TruMedia, which only exists for 2021-25, so
older seasons would be built from six features and neutral-filled on the rest.

That is a real hazard rather than a hypothetical - the model would fit one coefficient
across two different definitions of the same variable - so this measures it instead of
assuming either way. Both arms are scored the same way, leave-one-season-out over
whatever seasons they contain, on the seasons they share.

Run: ./venv/bin/python -m scripts.extend_years
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning, blended_talent

RECENT = [2021, 2022, 2023, 2024, 2025]
EXTENDED = [2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025]


def loso(years, score_on):
    """Train on all-but-one of `years`, score the held-out season. Only seasons in
    `score_on` are scored, so both arms are judged on the same games."""
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, config.OPP_ADJ_ALPHA)

    res = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in score_on:
        tr = [g for g in years if g != ty]
        if not tr:
            continue
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        parts = MU.assemble(years + [ty], std, pyth, talent, ret, games,
                            lam=config.UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                            ret_raw_by_year=ret_raw, od_by_year=od)
        if ty not in parts:
            continue
        av = [g for g in tr if g in parts]
        if not av:
            continue
        Xtr = np.vstack([parts[g][0] for g in av])
        ytr = np.concatenate([parts[g][1] for g in av])
        hf = np.concatenate([parts[g][2] for g in av])
        mdl, _ = M.train(Xtr, ytr, hf)
        ev = M.evaluate(mdl, parts[ty][0], parts[ty][1], parts[ty][2])
        res["brier"].append(ev["brier"])
        res["log_loss"].append(ev["log_loss"])
        res["accuracy"].append(ev["accuracy"])
    n = len(res["brier"])
    return {k: float(np.mean(v)) for k, v in res.items()} | {"folds": n}


def main():
    load.require_key()
    # score both arms on the seasons they have in common, so the comparison is fair
    common = [y for y in RECENT if y in EXTENDED] or [2022, 2023, 2024, 2025]
    print(f"scoring both arms on: {common}\n")

    print(f"{'training seasons':<44}{'folds':>6}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}")
    print("-" * 77)

    for label, yrs in (("2021-25 only (ships)", RECENT),
                       ("2015-19 + 2022-25 (TruMedia neutral pre-2021)", EXTENDED)):
        saved = config.GAME_YEARS
        config.GAME_YEARS = yrs
        try:
            m = loso(yrs, common)
            print(f"{label:<44}{m['folds']:>6}{m['brier']:>9.4f}"
                  f"{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}")
        finally:
            config.GAME_YEARS = saved


if __name__ == "__main__":
    main()
