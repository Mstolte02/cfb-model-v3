"""Does player WAR beat the PFF grade average as the model's talent signal?

Same harness as scripts/validate_roster_talent.py, same folds, same everything
downstream - only the talent vector changes. Candidates:

  CFBD (base)         247 composite recruiting, the model's original talent input
  PFF roster-aware    the current default: this year's roster x last year's grades
  blend 50/50         what config.TALENT_BLEND ships today
  WAR                 summed prior-year WAR of this year's roster
  WAR + CFBD 50/50    WAR given the same recruiting blend the PFF signal gets
  WAR + PFF 50/50     both roster signals, no recruiting

Run: ./venv/bin/python -m scripts.validate_war_talent
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS
from src.data import load, pff, war
from scripts.train import load_bundle, raw_returning
from scripts.validate_roster_talent import run_loso


def main():
    load.require_key()
    if not war.available():
        sys.exit(f"no WAR build found at {war.PLAYER_WAR}")

    print("Building bundle + talent variants ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    roster = pff.build_roster_talent()
    war_tal = war.talent_by_year({y: cfbd_tal[y].index for y in cfbd_tal
                                  if cfbd_tal[y] is not None})
    print(f"  WAR talent seasons: {sorted(war_tal)} "
          f"(teams/season: {[len(war_tal[y]) for y in sorted(war_tal)]})")

    folds = [y for y in GAME_YEARS if y in roster and y in war_tal]
    print(f"  LOSO folds (all signals available): {folds}\n")

    variants = {k: {} for k in
                ("cfbd", "pff", "blend", "war", "war_cfbd", "war_pff")}
    for N in std:
        base = cfbd_tal.get(N)
        if base is None:
            continue
        r = roster.get(N)
        r = base if r is None else r.reindex(base.index).fillna(base)
        wv = war_tal.get(N)
        # a team the WAR build has no roster for falls back to the PFF signal, the
        # same way the PFF signal falls back to CFBD
        wv = r if wv is None else wv.reindex(base.index).fillna(r)
        variants["cfbd"][N] = base
        variants["pff"][N] = r
        variants["blend"][N] = 0.5 * r + 0.5 * base
        variants["war"][N] = wv
        variants["war_cfbd"][N] = 0.5 * wv + 0.5 * base
        variants["war_pff"][N] = 0.5 * wv + 0.5 * r

    labels = {"cfbd": "CFBD (base)", "pff": "PFF roster-aware",
              "blend": "blend 50/50 (ships)", "war": "WAR",
              "war_cfbd": "WAR + CFBD 50/50", "war_pff": "WAR + PFF 50/50"}
    print(f"{'talent':<22}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}   (folds {folds})")
    print("-" * 54)
    results = {}
    for k, var in variants.items():
        m = run_loso(var, std, ret, games, pyth, ret_raw, folds)
        results[k] = m
        print(f"{labels[k]:<22}{m['brier']:>9.4f}{m['log_loss']:>10.4f}"
              f"{m['accuracy']:>8.3f}")

    best = min(results, key=lambda k: results[k]["brier"])
    ship = results["blend"]["brier"]
    print(f"\nbest by Brier: {labels[best]} ({results[best]['brier']:.4f}); "
          f"currently shipping {labels['blend']} ({ship:.4f}) -> "
          f"{'adopt' if results[best]['brier'] < ship - 1e-4 else 'no change'}")


if __name__ == "__main__":
    main()
