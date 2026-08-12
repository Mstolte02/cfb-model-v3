"""Leakage-safe player-level audit of intrinsic and role-adjusted WAR forecasts.

The 2025 roster is an untouched holdout and every feature comes from 2024 or earlier.
The exact 2026 July depth charts do not have historical archives, so the role test
uses the observable preseason proxy available in every fold: prior-season room rank
and prior snap share.  It answers whether a second role allocation adds signal after
the player model has already seen prior usage; it does not pretend to validate a
specific 2026 depth-chart call.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import artifacts
from build_roster_2026 import PFF_TO_GROUP
from build_recruiting import load_recruits
from facets import YEARS as WAR_YEARS
from project_2026_v2 import (FEATURES, build_history, build_population, fit,
                             load_rosters, make_training, slot_counts)

OUT = HERE / "player_projection_validation.json"


def healthy_targets(war: pd.DataFrame, starters: dict[str, int]) -> dict[str, float]:
    """Healthy backup contribution shares learned only before the holdout."""
    w = war[(war.season < 2025) & (war.snaps > 0)].copy()
    if "group" not in w:
        w["group"] = w.position.map(PFF_TO_GROUP)
    w = w[w.group.notna()]
    key = ["season", "team", "group"]
    w["rank"] = w.groupby(key).snaps.rank(ascending=False, method="first")
    w["share"] = w.snaps / w.groupby(key).snaps.transform("sum")
    w["backup"] = w["rank"] > w.group.map(starters).fillna(1)
    st = w[~w.backup].groupby(key).share.sum().rename("starter_share")
    w = w.merge(st, on=key)
    out = {}
    for group, rows in w.groupby("group"):
        healthy = rows[rows.starter_share >= rows.starter_share.quantile(.75)]
        out[group] = float(healthy[healthy.backup].share.sum() / healthy.share.sum())
    return out


def role_overlay(rows: pd.DataFrame, pred: np.ndarray, starters: dict[str, int],
                 targets: dict[str, float], weight: float) -> np.ndarray:
    """Allocate room value with an ex-ante prior-usage depth proxy."""
    d = rows[["team", "group", "prior_rank", "snaps_lag1"]].copy()
    d["intrinsic"] = np.clip(np.asarray(pred, float), 0.0, None)
    d["adjusted"] = d.intrinsic
    for (_, group), unit in d.groupby(["team", "group"]):
        total = float(unit.intrinsic.sum())
        target = targets.get(group)
        n_starters = int(starters.get(group, 1))
        first = unit[unit.prior_rank <= n_starters]
        second = unit[unit.prior_rank > n_starters]
        if total <= 0 or target is None or first.empty or second.empty:
            continue
        template = pd.Series(0.0, index=unit.index)
        for tier, share in ((first, 1.0 - target), (second, target)):
            base = tier.snaps_lag1.fillna(0.0).clip(lower=0.0)
            if base.sum() <= 0:
                base = tier.intrinsic
            if base.sum() <= 0:
                base = pd.Series(1.0, index=tier.index)
            template.loc[tier.index] = share * base / base.sum()
        raw_share = unit.intrinsic / total
        d.loc[unit.index, "adjusted"] = total * (
            (1.0 - weight) * raw_share + weight * template)
    return d.adjusted.to_numpy()


def metrics(actual: pd.Series, pred: np.ndarray) -> dict:
    y, p = np.asarray(actual, float), np.asarray(pred, float)
    q = np.quantile(p, .90)
    aq = np.quantile(y, .90)
    chosen = p >= q
    precision = float(np.mean(y[chosen] >= aq)) if chosen.any() else 0.0
    slope, intercept = np.polyfit(p, y, 1)
    return {
        "players": int(len(y)),
        "pearson_r": float(np.corrcoef(p, y)[0, 1]),
        "spearman_r": float(pd.Series(p).corr(pd.Series(y), method="spearman")),
        "mae_wins": float(mean_absolute_error(y, p)),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "top_decile_precision": precision,
        "top_decile_lift": precision / .10,
    }


def main():
    player_war = pd.read_csv(HERE / artifacts.PLAYER_WAR)
    ratings = pd.read_csv(HERE / artifacts.TEAM_RATINGS)
    records = pd.read_csv(HERE / "records.csv")
    recruits = load_recruits()
    roster_2026 = pd.read_csv(HERE / "roster_2026.csv")
    _, starters = slot_counts(roster_2026)

    history = build_history(player_war)
    fbs = set(records.team.unique())
    rosters = load_rosters(WAR_YEARS, fbs)
    population = build_population(history, rosters)
    targets = [y for y in WAR_YEARS if (y - 1) in set(WAR_YEARS)]
    training = make_training(population, history, ratings, recruits, rosters,
                             starters, targets)
    groups = sorted(history.group.dropna().unique())
    training["group_code"] = training.group.map({g: i for i, g in enumerate(groups)})
    training["share_lag1"] = training.share_lag1.fillna(0.0)

    tr = training[training.target_season < 2025]
    te = training[training.target_season == 2025].copy()
    model = fit().fit(tr[FEATURES], tr.war)
    intrinsic = model.predict(te[FEATURES])
    intrinsic_by_index = pd.Series(intrinsic, index=te.index)
    healthy = healthy_targets(player_war, starters)
    soft = role_overlay(te, intrinsic, starters, healthy, .50)
    hard = role_overlay(te, intrinsic, starters, healthy, 1.00)

    active = te.snaps > 0
    payload = {
        "contract": "2025 holdout; trained on player seasons through 2024; all 2025 roster members included",
        "intrinsic_definition": "preseason expected player WAR before a current depth-chart overlay",
        "role_proxy": "prior-season room rank and snap share; exact archived preseason depth charts unavailable",
        "all_roster_players": {
            "intrinsic": metrics(te.war, intrinsic),
            "soft_role_overlay": metrics(te.war, soft),
            "hard_role_overlay": metrics(te.war, hard),
        },
        "players_with_2025_snaps": {
            "intrinsic": metrics(te.loc[active, "war"], intrinsic[active]),
            "soft_role_overlay": metrics(te.loc[active, "war"], soft[active]),
            "hard_role_overlay": metrics(te.loc[active, "war"], hard[active]),
        },
        "by_position_intrinsic": {
            group: metrics(rows.war, intrinsic_by_index.loc[rows.index].to_numpy())
            for group, rows in te.groupby("group")
        },
        "limitations": [
            "Player WAR is realized seasonal contribution, not a per-snap scouting grade.",
            "The role overlay test cannot validate individual 2026 depth-chart disagreements.",
            "Correlation measures ordering; calibration and top-decile precision measure scale and useful ranking separately.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2))
    a = payload["all_roster_players"]
    for name in ("intrinsic", "soft_role_overlay", "hard_role_overlay"):
        m = a[name]
        print(f"{name:<20} r={m['pearson_r']:.3f} rho={m['spearman_r']:.3f} "
              f"MAE={m['mae_wins']:.4f} top-decile precision={m['top_decile_precision']:.1%}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
