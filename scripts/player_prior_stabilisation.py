"""Should an in-season player value shrink toward the league mean, or toward himself?

`src/qbwar.py` estimates a player's opponent-adjusted per-play value with a ridge on
player + opponent one-hots. The ridge penalty IS the shrinkage, and with no offset it
pulls every player toward zero, which is the league mean. That is the right answer
when nothing is known about a player and the wrong one when something is - and by the
time a season is under way, something always is, because the same routine produced a
value for him last year.

This measures the difference. Two estimators, same data, same alpha, same folds:

  flat    ridge on ppa                    -> shrinks toward the league mean (today)
  prior   ridge on ppa - prior[player]    -> shrinks toward his own last-season value

graded against the rest of the same season, which is estimated the same way from the
games that come after the cut so that the target is opponent-adjusted too. An earlier
version of this script compared RAW per-game PPA against an adjusted prior, which
loaded the comparison in the prior's favour; nothing here is raw.

RUN ACROSS EVERY POSITION CFBD PRICES PER GAME, not only quarterback. That turns out
to be QB, RB, WR and TE (and a trickle of FB). There are NO DEFENDERS in the per-game
PPA feed at all - 2022, 2024 and 2025 all return ball-carriers only - so no defensive
player can be updated in season from this source, whatever the season-level endpoint
suggests. That is a limit of the data, not of the method, and it is where PFF keeps
its value.

Reads only the cached CFBD files under data/raw, so it needs no key and no network.

Run:  venv/Scripts/python -m scripts.player_prior_stabilisation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, ROOT
from src.qbwar import fit_season_values

RAW = ROOT / "data" / "raw"
OUT_JSON = ARTIFACTS / "player_prior_stabilisation.json"
POSITIONS = ["QB", "RB", "WR", "TE"]
PRIOR_SEASONS = [2021, 2022, 2023, 2024]     # each supplies the prior for the next
TEST_SEASONS = [2022, 2023, 2024, 2025]
CUTS = [2, 3, 4, 5, 6, 7, 8]
MIN_BEFORE = 2        # an in-season estimate needs at least this many games
MIN_AFTER = 3         # so does the target
ALPHA = 10.0          # middle of qbwar.ALPHA_GRID, held fixed so the two estimators
                      # differ only in what they shrink toward


def load_games(year, position):
    fbs = {t["school"] for t in json.loads((RAW / f"teams_{year}.json").read_text())}
    rows = []
    for wk in range(1, 16):
        f = RAW / f"ppa_players_games_{year}_wk{wk}.json"
        if not f.exists():
            continue
        for p in json.loads(f.read_text()):
            if p.get("position") != position:
                continue
            a = p.get("averagePPA") or {}
            if a.get("all") is None or not p.get("opponent"):
                continue
            if p["team"] not in fbs or p["opponent"] not in fbs:
                continue
            rows.append({"week": wk, "id": str(p["id"]),
                         "opponent": p["opponent"], "ppa": float(a["all"])})
    return pd.DataFrame(rows)


def values(frame, prior=None):
    """Opponent-adjusted value per id. Returns an empty Series on a frame too thin
    to identify anything."""
    if len(frame) < 20 or frame.id.nunique() < 5:
        return pd.Series(dtype=float)
    val, _ = fit_season_values(frame, prior=prior, alpha=ALPHA)
    return val.set_index("id")["qb_value"]


def main():
    out = {"alpha": ALPHA, "positions": {}}
    print(f"cut = week the season is split at. n = players graded. "
          f"r/MAE are against the rest of the season,\nitself opponent-adjusted. "
          f"'flat' shrinks to the league mean, 'prior' to the player's own "
          f"last-season value.\n")
    for pos in POSITIONS:
        by_year = {y: load_games(y, pos) for y in [*PRIOR_SEASONS, *TEST_SEASONS]}
        priors = {}
        for y in PRIOR_SEASONS:
            priors[y + 1] = values(by_year[y]).to_dict()
        rows = []
        print(f"--- {pos} ---")
        print(f"{'cut':>4} {'n':>6} {'r flat':>8} {'r prior':>8} "
              f"{'MAE flat':>9} {'MAE prior':>10} {'MAE gain':>9}")
        for cut in CUTS:
            obs_f, obs_p, tgt = [], [], []
            for y in TEST_SEASONS:
                g = by_year[y]
                if not len(g):
                    continue
                before, after = g[g.week <= cut], g[g.week > cut]
                counts_b = before.groupby("id").size()
                counts_a = after.groupby("id").size()
                keep = set(counts_b[counts_b >= MIN_BEFORE].index) & \
                    set(counts_a[counts_a >= MIN_AFTER].index) & set(priors[y])
                if len(keep) < 10:
                    continue
                vf = values(before)
                vp = values(before, prior=priors[y])
                vt = values(after)
                for pid in keep:
                    if pid in vf.index and pid in vp.index and pid in vt.index:
                        obs_f.append(vf[pid]); obs_p.append(vp[pid]); tgt.append(vt[pid])
            if len(obs_f) < 40:
                print(f"{cut:>4} {len(obs_f):>6}   (too few)")
                continue
            a, b, t = np.array(obs_f), np.array(obs_p), np.array(tgt)
            z = lambda x: (x - x.mean()) / (x.std() or 1.0)
            za, zb, zt = z(a), z(b), z(t)
            r_f, r_p = float(np.corrcoef(a, t)[0, 1]), float(np.corrcoef(b, t)[0, 1])
            m_f, m_p = float(np.abs(za - zt).mean()), float(np.abs(zb - zt).mean())
            print(f"{cut:>4} {len(a):>6} {r_f:>8.3f} {r_p:>8.3f} "
                  f"{m_f:>9.3f} {m_p:>10.3f} {100 * (m_f - m_p) / m_f:>8.1f}%")
            rows.append({"cut": cut, "n": len(a), "r_flat": r_f, "r_prior": r_p,
                         "mae_flat": m_f, "mae_prior": m_p,
                         "mae_gain_pct": 100 * (m_f - m_p) / m_f})
        out["positions"][pos] = rows
        print()
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
