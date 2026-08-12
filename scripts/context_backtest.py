"""Strict online ablation for rest/travel/schedule context.

The base predictions are the already locked expanding-window v4 predictions.  For
season N, the context overlay is fitted only to locked predictions from seasons < N.
This tests whether contextual information adds anything after team strength and the
weekly dynamic update, rather than refitting on the same games it is judged on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS
from scripts.v4_backtest import metric, paired_week_bootstrap
from src.context import (CONTEXT_FEATURES, OffsetLogit, attach_context,
                         build_context)
from src.data import cfbd_client


PRED_PATH = ARTIFACTS / "v4_backtest_predictions.csv"
OUT_JSON = ARTIFACTS / "context_backtest.json"
OUT_CSV = ARTIFACTS / "context_backtest_predictions.csv"


def main():
    pred = pd.read_csv(PRED_PATH)
    cfbd_client.venues()  # fetch once when the reproducible cache is absent
    context = build_context(sorted(pred.season.unique()))
    data = attach_context(pred, context)
    if data[CONTEXT_FEATURES].isna().any(axis=None):
        missing = int(data[CONTEXT_FEATURES].isna().any(axis=1).sum())
        raise RuntimeError(f"context join missed {missing} modeled games")

    folds, pieces = [], []
    seasons = sorted(data.season.unique())
    for test in seasons[1:]:
        train = data[data.season < test]
        holdout = data[data.season == test].copy()
        overlay = OffsetLogit(penalty=20.0).fit(train)
        holdout["p_context"] = overlay.predict(holdout)
        base = metric(holdout.y, holdout.p_dynamic)
        augmented = metric(holdout.y, holdout.p_context)
        folds.append({"season": int(test), "train_seasons": sorted(train.season.unique().tolist()),
                      "base": base, "context": augmented,
                      "brier_change": augmented["brier"] - base["brier"],
                      "model": overlay.payload()})
        pieces.append(holdout)
        print(f"{test}: base={base['brier']:.5f} context={augmented['brier']:.5f} "
              f"change={augmented['brier'] - base['brier']:+.5f}")

    out = pd.concat(pieces, ignore_index=True)
    base = metric(out.y, out.p_dynamic)
    augmented = metric(out.y, out.p_context)
    paired = paired_week_bootstrap(out, left="p_context", right="p_dynamic")
    final = OffsetLogit(penalty=20.0).fit(data)
    result = {
        "contract": "season N context coefficients use only locked v4 predictions from seasons < N",
        "features": CONTEXT_FEATURES,
        "weather_policy": "realized historical weather excluded; it is not a pregame forecast",
        "travel_policy": "great-circle team-home-to-venue distance; no fabricated itinerary",
        "folds": folds, "pooled_base": base, "pooled_context": augmented,
        "pooled_brier_change": augmented["brier"] - base["brier"],
        "paired_week_bootstrap": paired, "production_candidate": final.payload(),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    out.to_csv(OUT_CSV, index=False)
    print(f"\npooled: base={base['brier']:.5f} context={augmented['brier']:.5f} "
          f"change={result['pooled_brier_change']:+.5f}")
    print(f"paired: {paired}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
