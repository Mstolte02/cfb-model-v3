"""Is home field worth the same everywhere?

The model fits one home-field coefficient for all 136 teams. That is a strong claim.
Altitude at Laramie and Provo, a genuinely hostile night crowd, a long eastward trip
into an early kick and an empty stadium in a bad year are not the same advantage, and
the sport has believed so for decades.

The obvious fix - a free intercept per team - is the wrong one. A team plays six or
seven home games a season, so an unpooled home-field estimate is mostly noise, and
the noisiest teams would get the largest adjustments. This uses the same treatment
the coach phase used: **partial pooling**, where each team's home-field estimate is
shrunk toward the league mean by its own sample size,

    hfa_team = league_hfa + (n / (n + k)) * (raw_team_margin_edge - league_hfa)

with k fitted by the same expanding-fold selection as every other knob. k -> infinity
is the shipping single coefficient, so the flat model is nested inside this one and
the comparison is honest by construction.

Estimates for season N use completed home games through N-1 only.

Candidates sweep k, plus two references: the shipping flat coefficient, and an
unpooled per-team estimate, which is there to show what happens without the shrinkage
rather than as a serious contender.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA
from scripts import v4_backtest as BT
from scripts.train import load_bundle
from src import oppadj as OA
from src import v4 as V4
from src.data import pff, war


OUT_JSON = ARTIFACTS / "team_hfa_backtest.json"
CORE = ["O", "D", "talent", "returning", "war_projected"]
# k in games. 6 is about one season of home games, 40 about six seasons.
SHRINK_K = [6, 12, 25, 40]


def home_margin_edges(games_by_year: dict, through_year: int,
                      rated: set | None = None) -> pd.DataFrame:
    """Per-team average home margin minus average away margin, through a season.

    Differencing a team's own home and away margins removes team strength: a good team
    wins everywhere, and what is left is how much more it wins at home. Neutral-site
    games are excluded because they are neither.

    `rated` restricts the table to teams the model actually rates. Without it the
    schedule drags in every FCS and Division II visitor an FBS team ever hosted; they
    appear almost entirely as road teams, so their "home edge" is a handful of games
    against nobody, and they were producing +12 and -12 point estimates and dragging
    the league mean that every other team is shrunk toward.
    """
    home_margin, away_margin = defaultdict(list), defaultdict(list)
    for year, frame in games_by_year.items():
        if year > through_year:
            continue
        for game in frame.itertuples():
            if bool(getattr(game, "neutral_site", False)):
                continue
            hp, ap = game.home_points, game.away_points
            if pd.isna(hp) or pd.isna(ap):
                continue
            if rated is not None and (game.home_team not in rated or
                                      game.away_team not in rated):
                continue
            home_margin[game.home_team].append(float(hp - ap))
            away_margin[game.away_team].append(float(ap - hp))
    teams = sorted(set(home_margin) | set(away_margin))
    rows = []
    for team in teams:
        h, a = home_margin.get(team, []), away_margin.get(team, [])
        if not h or not a:
            continue
        rows.append({"team": team, "home_games": len(h),
                     "edge": float(np.mean(h) - np.mean(a))})
    return pd.DataFrame(rows).set_index("team")


def shrunk_hfa(edges: pd.DataFrame, k: float | None) -> pd.Series:
    """Partial-pool each team's edge toward the league mean.

    k=None returns the raw unpooled edge. The league mean is the weighted mean, so a
    team with two home games does not drag the centre it is shrunk toward.
    """
    league = float(np.average(edges.edge, weights=edges.home_games))
    if k is None:
        return edges.edge - league
    weight = edges.home_games / (edges.home_games + k)
    return weight * (edges.edge - league)


def build_frames():
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {y: s.index for y, s in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    war_projected = war.projected_team_talent(indices)
    frames = {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od, pff_lag, war_lag,
                               granular=True)
        if frame is None:
            continue
        wp = war_projected.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = wp.fillna(0.0)
        frames[year] = frame
    return frames, games


def apply_hfa(parts_year, hfa: pd.Series, scale: float):
    """Replace the flat home indicator with a per-team multiplier.

    build_year hands back `home` as 1.0 for a home game and 0.0 for a neutral. The
    model multiplies it by one fitted coefficient, so scaling the indicator by
    (1 + shrunk_edge/scale) lets the same coefficient carry a team-specific advantage
    while the league mean stays exactly where it was.
    """
    X, y, home, margin, meta = parts_year
    adjusted = home.copy()
    for i, team in enumerate(meta.home_team):
        if home[i] == 0.0:
            continue
        adjusted[i] = 1.0 + float(hfa.get(team, 0.0)) / scale
    return X, y, adjusted, margin, meta


def main():
    frames, games = build_frames()
    years = sorted(frames)
    base_parts = V4.assemble(years, frames, games, CORE)

    # One scale for the whole study: the SD of the shrunk edges over the full history,
    # so the multiplier is order 1 and the fitted coefficient stays interpretable.
    rated = set().union(*(set(f.index) for f in frames.values()))
    full_edges = home_margin_edges(games, max(years), rated)
    scale = float(shrunk_hfa(full_edges, 12).std()) or 1.0

    specs = {"flat": None, "unpooled": "raw",
             **{f"pooled_k{k}": k for k in SHRINK_K}}
    rows, spread = [], {}
    for test in (2022, 2023, 2024, 2025):
        pool = [y for y in years if y < test]
        if not pool or test not in base_parts:
            continue
        edges = home_margin_edges(games, test - 1, rated)
        meta = None
        for name, k in specs.items():
            if name == "flat":
                parts = base_parts
            else:
                hfa = shrunk_hfa(edges, None if k == "raw" else float(k))
                parts = {y: (apply_hfa(base_parts[y], hfa, scale)
                             if y in base_parts else None) for y in years}
                parts = {y: v for y, v in parts.items() if v is not None}
                if test == 2025:
                    spread[name] = {
                        "sd": float(hfa.std()), "min": float(hfa.min()),
                        "max": float(hfa.max()),
                        "top": hfa.sort_values(ascending=False).head(6).round(3)
                                  .to_dict(),
                        "bottom": hfa.sort_values().head(6).round(3).to_dict()}
            knobs, _ = BT.tune(parts, pool, CORE)
            k_dyn, blend, _ = BT.tune_dynamic(parts, frames, pool, CORE, knobs)
            model, static, margin = BT.fit_predict(parts, pool, test, CORE, knobs)
            dynamic, _ = BT.dynamic_predictions(model, frames[test], parts[test],
                                                 k_dyn, blend)
            if meta is None:
                meta = parts[test][4].copy()
                meta["season"], meta["y"] = test, parts[test][1]
            meta[f"p_static_{name}"] = static
            meta[f"p_dynamic_{name}"] = dynamic
        rows.append(meta)
        print(f"{test} done", flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    pooled = {n: {"static": BT.metric(predictions.y, predictions[f"p_static_{n}"]),
                  "dynamic": BT.metric(predictions.y, predictions[f"p_dynamic_{n}"])}
              for n in specs}

    def boot(left, right, draws=5000, seed=20260825):
        data = predictions.dropna(subset=[left, right]).copy()
        data["delta"] = (data[left] - data.y) ** 2 - (data[right] - data.y) ** 2
        blocks = [g.delta.to_numpy(float) for _, g in
                  data.groupby(["season", "week"], sort=True)]
        rng = np.random.default_rng(seed)
        samples = np.asarray([
            np.concatenate([blocks[i] for i in
                            rng.integers(0, len(blocks), len(blocks))]).mean()
            for _ in range(draws)])
        return {"difference": float(data.delta.mean()),
                "ci95": [float(v) for v in np.quantile(samples, [.025, .975])]}

    vs_flat = {n: {"static": boot(f"p_static_{n}", "p_static_flat"),
                   "dynamic": boot(f"p_dynamic_{n}", "p_dynamic_flat")}
               for n in specs if n != "flat"}
    result = {"question": "is home field team-specific enough to pay for itself",
              "shrinkage": "hfa_team = w*(team home-minus-away margin - league), "
                           "w = n/(n+k)",
              "scale": scale, "pooled": pooled, "vs_flat": vs_flat,
              "hfa_spread_2025": spread}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    bs = pooled["flat"]["static"]["brier"]
    bo = pooled["flat"]["dynamic"]["brier"]
    print(f"\n{'candidate':14s} {'static':>9} {'delta':>9} {'online':>9} {'delta':>9}")
    for name in specs:
        s, o = pooled[name]["static"]["brier"], pooled[name]["dynamic"]["brier"]
        print(f"{name:14s} {s:>9.5f} {s-bs:>+9.5f} {o:>9.5f} {o-bo:>+9.5f}")
    print("\nvs flat (negative = better):")
    for name, block in vs_flat.items():
        st, dy = block["static"], block["dynamic"]
        print(f"  {name:14s} static {st['difference']:+.5f} "
              f"[{st['ci95'][0]:+.5f},{st['ci95'][1]:+.5f}]  online "
              f"{dy['difference']:+.5f} [{dy['ci95'][0]:+.5f},{dy['ci95'][1]:+.5f}]")
    if "pooled_k12" in spread:
        block = spread["pooled_k12"]
        print(f"\nteam home-field spread at k=12 (points of margin, vs league mean):"
              f"  sd {block['sd']:.2f}  range {block['min']:+.2f} to {block['max']:+.2f}")
        print("  strongest:", ", ".join(f"{t} {v:+.2f}"
                                        for t, v in block["top"].items()))
        print("  weakest:  ", ", ".join(f"{t} {v:+.2f}"
                                        for t, v in block["bottom"].items()))
    print(f"\n-> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
