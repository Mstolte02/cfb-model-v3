"""The sample-size prior as projection features: kal_m and kal_sd.

A per-player filter over full PFF seasons (scripts/war_credibility_backtest.py). The
player's true value per 1,000 snaps drifts between seasons, and each season is a
noisy look at it with variance sigma2 / snaps + omega2, so a season counts by its
snaps and a player with a long, heavy record has a tight prior. For target season T
it returns the filter's mean (kal_m) and standard deviation (kal_sd) BEFORE season T
is seen, with the filter's parameters fitted on seasons before T.

Measured inside the shipped projection in scripts/war_projection_prior_test.py:
holdout MAE -16% / -14% / -11% (2023 / 2024 / 2025), player r .58 -> .64 in 2025,
team-sum r .78 -> .81.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import war_credibility_backtest as K  # noqa: E402

# The earliest seasons have too little history to fit drift and noise; their targets
# use parameters fitted through 2016. Every target season still reads only its own
# past through the filter itself.
MIN_FIT_BEFORE = 2017
COLUMNS = ["kal_m", "kal_sd"]


@lru_cache(maxsize=1)
def table(last_target: int = 2026) -> pd.DataFrame:
    """(player_id, group, target_season) -> kal_m, kal_sd for every target season."""
    d = K.load()
    seasons = sorted(d.season.unique())
    targets = [s for s in seasons[1:]] + [last_target]
    fits = {}
    out = []
    for T in targets:
        fit_before = max(T, MIN_FIT_BEFORE)
        hist = [s for s in seasons if s < T]
        cols = hist + [T]
        t = len(hist)
        for g in sorted(d.group.unique()):
            ids, R, S = K.panel(d, g, cols)
            if not len(ids):
                continue
            key = (fit_before, g)
            if key not in fits:
                fs = [s for s in seasons if s < fit_before]
                fids, fR, fS = K.panel(d, g, fs + [fit_before])
                tf = len(fs)
                train = (fS[:, :tf] >= K.MIN_TARGET_SNAPS) & np.isfinite(fR[:, :tf])
                mu = float(np.average(fR[:, :tf][train], weights=fS[:, :tf][train]))
                fits[key] = (K.fit_kalman(fR, fS, tf, mu), mu)
            params, mu = fits[key]
            m, P = K.kalman(R, S, params, mu, t)
            seen = (np.where(S[:, :t] >= K.MIN_OBS_SNAPS, 1, 0).sum(1) > 0)
            out.append(pd.DataFrame({"player_id": ids[seen], "group": g,
                                     "target_season": T, "kal_m": m[seen, t],
                                     "kal_sd": np.sqrt(P[seen, t])}))
    t = pd.concat(out, ignore_index=True)
    t["player_id"] = t.player_id.astype(str)
    return t
