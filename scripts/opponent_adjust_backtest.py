"""Should a player's in-season PFF rate be adjusted for the opponents he faced?

Mark's request (4 October 2026): a grade earned against a weak schedule should count
for less than the same grade against a strong one, at every position.

1. ESTIMATE. The effect of opponent difficulty on a player's facet rate is estimated
   WITHIN players: the same player's early window (weeks 1-c) against his late window
   (weeks c+1-16), regressed on the difference in the difficulty his unit faced
   (src/opponent_strength.py). Comparing a player with himself removes the confound
   that better teams also tend to play better schedules. One slope per position
   group, fitted on earlier seasons only.

2. TEST. Does the adjusted early window predict the rest of the season better? The
   target is the raw late-window rate, so the adjusted arm adds back the late
   schedule it knows (beta * late difficulty) - a fair comparison on the raw scale.
   Both arms blend with the same Kalman prior and get the same calibration, every
   parameter fitted per group and cut on earlier seasons; 2023-25 are graded.

    python -m scripts.opponent_adjust_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ARTIFACTS  # noqa: E402
from scripts import war_inseason_backtest as B  # noqa: E402
from src import inseason_war as IW  # noqa: E402
from src import opponent_strength as O  # noqa: E402

OUT = ARTIFACTS / "opponent_adjust_backtest.json"
LAM = np.linspace(0, 1, 21)


def frame() -> pd.DataFrame:
    team_map = json.load(open(Path(B.__file__).resolve().parents[1] / "war_model" / "team_map.json"))
    hist = B.history()
    rows = []
    for s in B.TEST:
        pr = B.priors(hist, s)
        gp = pr.groupby("group").mu.first()
        for cut, (wa, wb) in B.CUTS.items():
            w, r = B.window(s, *wa), B.window(s, *wb)
            d = (w[["player_id", "group", "snaps", "fc", "team_name"]]
                 .merge(r[["player_id", "snaps", "fc"]], on="player_id", suffixes=("_w", "_t")))
            d = d[(d.snaps_w >= B.MIN_WINDOW) & (d.snaps_t >= B.MIN_TARGET) & d.group.notna()]
            d["team"] = d.team_name.map(lambda t: IW.canonical_team([t], team_map))
            d["obs"] = d.fc_w / d.snaps_w * 1000
            d["target"] = d.fc_t / d.snaps_t * 1000
            de, dl = O.window_difficulty(s, *wa), O.window_difficulty(s, *wb)
            d["diff_w"] = O.player_difficulty(d.team, d.group, de)
            d["diff_t"] = O.player_difficulty(d.team, d.group, dl)
            d = d.merge(pr[["player_id", "group", "m"]], on=["player_id", "group"], how="left")
            d["m"] = d.m.fillna(d.group.map(gp))
            rows.append(d.assign(season=s, cut=cut))
        print(f"frame {s}", flush=True)
    return pd.concat(rows, ignore_index=True)


def fit_beta(tr: pd.DataFrame) -> float:
    """Within-player slope of rate on difficulty: (obs - target) on (diff_w - diff_t)."""
    y = (tr.obs - tr.target).to_numpy()
    x = (tr.diff_w - tr.diff_t).to_numpy()
    w = 1.0 / (1.0 / tr.snaps_w + 1.0 / tr.snaps_t).to_numpy()   # harmonic snaps
    X = np.c_[np.ones_like(x), x]
    sw = np.sqrt(w)
    return float(np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0][1])


def blend_fit(tr, x_tr, w):
    best = min(LAM, key=lambda l: np.average(((1 - l) * tr.m + l * x_tr - tr.target) ** 2,
                                             weights=w))
    z = (1 - best) * tr.m + best * x_tr
    a, b = np.polyfit(z, tr.target, 1, w=np.sqrt(w))[::-1]
    return best, a, b


def main():
    fr = frame()
    out, betas = {"by_group": {}}, {}
    pooled = {"raw": 0.0, "adj": 0.0, "n": 0.0}
    for g, d in fr.groupby("group"):
        acc = {"raw": 0.0, "adj": 0.0, "n": 0.0}
        for T in B.GRADED:
            for c in B.CUTS:
                tr = d[(d.season < T) & (d.cut == c)]
                te = d[(d.season == T) & (d.cut == c)]
                if len(tr) < 50 or len(te) < 20:
                    continue
                beta = fit_beta(d[d.season < T])
                for arm in ("raw", "adj"):
                    sh = beta if arm == "adj" else 0.0
                    x_tr = tr.obs - sh * tr.diff_w + sh * tr.diff_t
                    x_te = te.obs - sh * te.diff_w + sh * te.diff_t
                    lam, a, b = blend_fit(tr, x_tr, tr.snaps_t)
                    p = a + b * ((1 - lam) * te.m + lam * x_te)
                    e = float(np.average((p - te.target) ** 2, weights=te.snaps_t))
                    acc[arm] += e * len(te)
                    pooled[arm] += e * len(te)
                acc["n"] += len(te)
                pooled["n"] += len(te)
        betas[g] = fit_beta(d)
        if acc["n"]:
            out["by_group"][g] = {"raw": acc["raw"] / acc["n"], "adj": acc["adj"] / acc["n"],
                                  "change_pct": 100 * (acc["adj"] / acc["raw"] - 1),
                                  "beta_all_seasons": betas[g], "n": int(acc["n"])}
    out["pooled_change_pct"] = 100 * (pooled["adj"] / pooled["raw"] - 1)
    out["beta"] = betas
    OUT.write_text(json.dumps(out, indent=1))
    print(f"pooled change in rest-of-season error: {out['pooled_change_pct']:+.2f}%")
    for g, r in sorted(out["by_group"].items()):
        print(f"  {g:5} beta {r['beta_all_seasons']:+8.3f}  error {r['change_pct']:+6.2f}%  n={r['n']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
