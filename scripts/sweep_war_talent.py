"""WAR as a third talent axis rather than a replacement.

validate_war_talent.py asks whether WAR should replace the PFF grade signal and the
answer is no. This asks the different question the swap cannot: whether WAR carries
anything the PFF+CFBD blend does not already have. The blend that ships is
0.5*PFF + 0.5*CFBD; here a WAR share w is mixed in as

    talent = (1-w) * (0.5*PFF + 0.5*CFBD) + w * WAR

so w=0 reproduces the shipping model exactly and any gain at w>0 is WAR earning its
place on top of what is already there.

Run: ./venv/bin/python -m scripts.sweep_war_talent
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS
from src.data import load, pff, war
from scripts.train import load_bundle, raw_returning
from scripts.validate_roster_talent import run_loso

GRID = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8]


def main():
    load.require_key()
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    roster = pff.build_roster_talent()
    war_tal = war.talent_by_year({y: cfbd_tal[y].index for y in cfbd_tal
                                  if cfbd_tal[y] is not None})
    folds = [y for y in GAME_YEARS if y in roster and y in war_tal]
    print(f"LOSO folds: {folds}\n")

    base_by_year, war_by_year = {}, {}
    for N in std:
        base = cfbd_tal.get(N)
        if base is None:
            continue
        r = roster.get(N)
        r = base if r is None else r.reindex(base.index).fillna(base)
        blend = 0.5 * r + 0.5 * base
        wv = war_tal.get(N)
        base_by_year[N] = blend
        war_by_year[N] = blend if wv is None else wv.reindex(base.index).fillna(blend)

    print(f"{'WAR share':>10}{'Brier':>10}{'LogLoss':>10}{'Acc':>8}")
    print("-" * 38)
    rows = []
    for w in GRID:
        var = {N: (1 - w) * base_by_year[N] + w * war_by_year[N] for N in base_by_year}
        m = run_loso(var, std, ret, games, pyth, ret_raw, folds)
        rows.append((w, m))
        print(f"{w:>10.2f}{m['brier']:>10.4f}{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}")

    b0 = rows[0][1]["brier"]
    bw, bm = min(rows, key=lambda t: t[1]["brier"])
    print(f"\nbaseline (w=0, ships today): Brier {b0:.4f}")
    print(f"best: w={bw:.2f} Brier {bm['brier']:.4f}  (delta {bm['brier']-b0:+.4f})")
    print("adopt" if bm["brier"] < b0 - 1e-4 else "no material gain - keep w=0")


if __name__ == "__main__":
    main()
