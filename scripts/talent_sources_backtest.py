"""Are there talent inputs the model is not using?

Team talent today is three axes blended into one number: the CFBD recruiting
composite, PFF roster-aware grades, and roster WAR. Two things CFBD publishes are not
in that blend at all, and both describe roster quality that the existing axes cannot
see:

* **The transfer portal.** A team that lost four rated starters and replaced them with
  three looks identical to the recruiting composite as one that stood still. Portal
  entries carry origin, destination and a recruit rating, so incoming and outgoing
  talent can be priced separately.
* **Recruiting by position group.** The shipping talent figure is one number per team.
  The same composite split by quarterback, line, skill and secondary makes it possible
  to say a roster is stacked in the wrong places.

Both enter as team-level features the reciprocal architecture differences itself, and
both are lagged: season N sees the portal class that arrived for N and recruiting
classes through N, which are preseason facts, never season-N results.

Candidates are additive against the clean core so the question is what each one adds,
and the answer is scored on the same +.001 bar as every other phase.
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
from src.data import cfbd_client, fbs, pff, war


OUT_JSON = ARTIFACTS / "talent_sources_backtest.json"
CORE = ["O", "D", "talent", "returning"]
PORTAL = ["portal_in", "portal_out", "portal_net"]
# Rated-only: the first build imputed the 44% of entries with no recruit rating at the
# league mean, which turned portal_in into a head count with a quality label. These
# count only players the recruiting services actually rated, plus the blue-chip tally
# separately, so quality and volume are not the same column.
PORTAL_RATED = ["portal_in_rated", "portal_out_rated", "portal_net_rated",
                "portal_blue_in", "portal_blue_out"]
GROUPS = ["rec_qb", "rec_ol", "rec_skill", "rec_front7", "rec_secondary"]
SPECS = {
    "clean_core": CORE,
    "core_portal": [*CORE, *PORTAL],
    "core_portal_rated": [*CORE, *PORTAL_RATED],
    "core_groups": [*CORE, *GROUPS],
    "core_both": [*CORE, *PORTAL, *GROUPS],
    "core_groups_portal_rated": [*CORE, *GROUPS, *PORTAL_RATED],
}

GROUP_MAP = {
    "Quarterback": "rec_qb",
    "Offensive Line": "rec_ol",
    "Receiver": "rec_skill", "Running Back": "rec_skill",
    "Defensive Line": "rec_front7", "Linebacker": "rec_front7",
    "Defensive Back": "rec_secondary",
}


def portal_features(years) -> dict[int, pd.DataFrame]:
    """Incoming, outgoing and net rated portal talent per team, per season.

    A portal entry is dated to the season it is listed under, which is the season the
    player arrives for. Unrated entries are counted at the league's rated mean rather
    than dropped: a walk-on transfer is not a zero-talent player, and dropping them
    would make a team that took ten unrated transfers look like it took none.
    """
    out = {}
    for year in years:
        rows = cfbd_client.transfer_portal(year)
        # The feed covers every division. A move between two FCS programmes is not an
        # FBS roster event and must not set the imputed rating that FBS arrivals are
        # scored against, so the fallback is the mean of moves landing in FBS.
        members = fbs.teams(year)
        rated = [float(r["rating"]) for r in rows
                 if r.get("rating") and r.get("destination") in members]
        fallback = float(np.mean(rated)) if rated else 0.85
        incoming, outgoing = defaultdict(float), defaultdict(float)
        in_rated, out_rated = defaultdict(float), defaultdict(float)
        blue_in, blue_out = defaultdict(float), defaultdict(float)
        # A blue-chip portal move is the one everybody actually notices. Four stars
        # and up, which is roughly the top decile of rated entries.
        for entry in rows:
            rating = float(entry["rating"]) if entry.get("rating") else None
            stars = float(entry["stars"]) if entry.get("stars") else 0.0
            value = rating if rating is not None else fallback
            dest, origin = entry.get("destination"), entry.get("origin")
            if dest:
                incoming[dest] += value
                if rating is not None:
                    in_rated[dest] += rating
                if stars >= 4:
                    blue_in[dest] += 1.0
            if origin:
                outgoing[origin] += value
                if rating is not None:
                    out_rated[origin] += rating
                if stars >= 4:
                    blue_out[origin] += 1.0
        teams = sorted(set(incoming) | set(outgoing))
        frame = pd.DataFrame({
            "portal_in": [incoming.get(t, 0.0) for t in teams],
            "portal_out": [outgoing.get(t, 0.0) for t in teams],
            "portal_in_rated": [in_rated.get(t, 0.0) for t in teams],
            "portal_out_rated": [out_rated.get(t, 0.0) for t in teams],
            "portal_blue_in": [blue_in.get(t, 0.0) for t in teams],
            "portal_blue_out": [blue_out.get(t, 0.0) for t in teams],
        }, index=teams)
        frame["portal_net"] = frame.portal_in - frame.portal_out
        frame["portal_net_rated"] = frame.portal_in_rated - frame.portal_out_rated
        out[year] = frame
    return out


def group_features(years) -> dict[int, pd.DataFrame]:
    """Three-year rolling recruiting rating per position group, per team.

    One class is a small sample and most of a roster is not freshmen, so each season
    averages the three classes that make up the bulk of it.
    """
    raw = cfbd_client.recruiting_groups(min(years) - 4, max(years))
    frame = pd.DataFrame(raw)
    frame["column"] = frame.positionGroup.map(GROUP_MAP)
    frame = frame.dropna(subset=["column", "averageRating"])
    frame["year"] = pd.to_numeric(frame.year, errors="coerce") if "year" in frame \
        else np.nan
    if frame.year.isna().all():
        # The endpoint omits the year when a range is requested; fall back to the
        # single-season call so each class keeps its date.
        blocks = []
        for year in range(min(years) - 3, max(years) + 1):
            block = pd.DataFrame(cfbd_client.recruiting_groups(year, year))
            block["year"] = year
            blocks.append(block)
        frame = pd.concat(blocks, ignore_index=True)
        frame["column"] = frame.positionGroup.map(GROUP_MAP)
        frame = frame.dropna(subset=["column", "averageRating"])
    frame["averageRating"] = pd.to_numeric(frame.averageRating, errors="coerce")

    out = {}
    for year in years:
        window = frame[(frame.year <= year) & (frame.year >= year - 2)]
        pivot = window.pivot_table(index="team", columns="column",
                                   values="averageRating", aggfunc="mean")
        for column in GROUPS:
            if column not in pivot:
                pivot[column] = np.nan
        out[year] = pivot[GROUPS]
    return out


def build_frames():
    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    pff_lag = pff.build_lagged_team_talent()
    indices = {y: s.index for y, s in talent.items()}
    war_lag = war.lagged_team_talent(indices)
    portal = portal_features(GAME_YEARS)
    groups = group_features(GAME_YEARS)

    frames, coverage = {}, {}
    for year in GAME_YEARS:
        frame = V4.build_frame(year, std, talent, ret, od, pff_lag, war_lag,
                               granular=True)
        if frame is None:
            continue
        p = portal[year].reindex(frame.index)
        g = groups[year].reindex(frame.index)
        coverage[str(year)] = {
            "portal": float(p.portal_in.notna().mean()),
            "groups": float(g[GROUPS[0]].notna().mean()),
        }
        # Standardize within season so a portal sum and a recruit rating enter on the
        # same scale as the rest of the frame, and centre missing at zero.
        for column in [*PORTAL, *PORTAL_RATED]:
            values = p[column]
            frame[column] = ((values - values.mean()) /
                             (values.std(ddof=0) or 1.0)).fillna(0.0)
        for column in GROUPS:
            values = g[column]
            frame[column] = ((values - values.mean()) /
                             (values.std(ddof=0) or 1.0)).fillna(0.0)
        frames[year] = frame
    return frames, games, coverage


def _bootstrap(frame, left, right, draws=5000, seed=20260825):
    data = frame.dropna(subset=[left, right]).copy()
    data["delta"] = (data[left] - data.y) ** 2 - (data[right] - data.y) ** 2
    blocks = [g.delta.to_numpy(float) for _, g in
              data.groupby(["season", "week"], sort=True)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        np.concatenate([blocks[i] for i in
                        rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(draws)])
    return {"difference": float(data.delta.mean()),
            "ci95": [float(v) for v in np.quantile(samples, [.025, .975])],
            "probability_left_better": float(np.mean(samples < 0))}


def main():
    frames, games, coverage = build_frames()
    years = [y for y in GAME_YEARS if y in frames]
    parts = {n: V4.assemble(years, frames, games, c) for n, c in SPECS.items()}
    rows = []
    for test in (2022, 2023, 2024, 2025):
        pool = [y for y in years if y < test]
        if not pool:
            continue
        meta = None
        for name, cols in SPECS.items():
            knobs, _ = BT.tune(parts[name], pool, cols)
            k, blend, _ = BT.tune_dynamic(parts[name], frames, pool, cols, knobs)
            model, static, margin = BT.fit_predict(parts[name], pool, test, cols,
                                                    knobs)
            dynamic, _ = BT.dynamic_predictions(model, frames[test],
                                                 parts[name][test], k, blend)
            if meta is None:
                meta = parts[name][test][4].copy()
                meta["season"], meta["y"] = test, parts[name][test][1]
            meta[f"p_static_{name}"] = static
            meta[f"p_dynamic_{name}"] = dynamic
        rows.append(meta)
        print(f"{test} done", flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    pooled = {n: {"static": BT.metric(predictions.y, predictions[f"p_static_{n}"]),
                  "dynamic": BT.metric(predictions.y, predictions[f"p_dynamic_{n}"])}
              for n in SPECS}
    vs_core = {n: {"static": _bootstrap(predictions, f"p_static_{n}",
                                        "p_static_clean_core"),
                   "dynamic": _bootstrap(predictions, f"p_dynamic_{n}",
                                         "p_dynamic_clean_core")}
               for n in SPECS if n != "clean_core"}
    result = {"specs": SPECS, "coverage": coverage, "pooled": pooled,
              "vs_clean_core": vs_core,
              "selection_bar": BT.SELECTION_MIN_GAIN}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    base_s = pooled["clean_core"]["static"]["brier"]
    base_o = pooled["clean_core"]["dynamic"]["brier"]
    print(f"\n{'candidate':16s} {'static':>9} {'delta':>9} {'online':>9} {'delta':>9}")
    for name in SPECS:
        s = pooled[name]["static"]["brier"]
        o = pooled[name]["dynamic"]["brier"]
        print(f"{name:16s} {s:>9.5f} {s-base_s:>+9.5f} {o:>9.5f} {o-base_o:>+9.5f}")
    print("\nvs clean core (negative = better):")
    for name, block in vs_core.items():
        st, dy = block["static"], block["dynamic"]
        print(f"  {name:14s} static {st['difference']:+.5f} "
              f"[{st['ci95'][0]:+.5f},{st['ci95'][1]:+.5f}]   "
              f"online {dy['difference']:+.5f} "
              f"[{dy['ci95'][0]:+.5f},{dy['ci95'][1]:+.5f}]")
    print(f"\n-> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
