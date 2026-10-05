"""Historical team-week Power Ratings for the Team Overview's Historical Standing card.

Every completed regular-season week of 2021-2025 is replayed through the live v5
ensemble (scripts/ensemble_replay, the same stdlib code the site's 2026 numbers come
from), with each season's own week-0 frame and PFF form, and the Power Rating (mean
neutral-site win probability against every other rated team) is recorded for every
team after every slate, preseason included. The 2026 snapshots already published in
viz/data/ratings.json are appended as they stand.

The reference population is therefore "team-week model Power Ratings, 2021 to date".
The past seasons use the production ensemble, which was fitted on 2021-25, so their
ratings are descriptive (in-sample), not the forecasts the model made at the time.
The in-season WAR column is not replayed for past seasons (it is zero), which is how
the stack treated weeks before the first WAR cut.

Writes viz/data/power_history.json: observations, fixed 2.5-point bins, and the sorted
power values the browser uses for an empirical percentile.

    python -m scripts.export_power_history
"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GAME_YEARS, ROOT  # noqa: E402
from scripts import ensemble_replay as ER  # noqa: E402
from scripts import train_live_ensemble as TL  # noqa: E402
from src import live_ensemble as LE  # noqa: E402

OUT = ROOT / "viz" / "data" / "power_history.json"
BIN = 0.025


def season_snapshots(season, manifest, frames, parts_by_spec, raw) -> list[dict]:
    frame = frames[season]
    block = LE.ensemble_block(manifest, frame, frame)
    games = parts_by_spec[TL.SPECS[0]][season][4]
    margins = parts_by_spec[TL.SPECS[0]][season][3]
    finals = [{"week": int(g.week), "season_type": "regular", "home": g.home_team,
               "away": g.away_team, "neutral": bool(g.neutral_site),
               "home_score": float(m), "away_score": 0.0}
              for g, m in zip(games.itertuples(), margins)]
    stats = raw[season]
    rows = [dict(zip(ER.FORM_FIELDS, v)) for v in
            stats[list(ER.FORM_FIELDS)].itertuples(index=False, name=None)]
    names = list(frame.index)
    pff = TL.staged_pff_payload(season, names)
    res = ER.replay(block, finals, rows, pff=pff)
    pre = ER.initial_state(block)
    pre["pff"] = ER.pff_table(pff, 1)
    out = [(0, ER.power_table(block, pre, names))]
    for key, snap in res["snapshots"]:
        out.append((key, ER.power_table(block, snap, names)))
    return [{"season": season, "week": wk, "team": r["team"], "power": round(r["power"], 5)}
            for wk, table in out for r in table]


def histogram(values: list[float]) -> list[dict]:
    lo = int(min(values) / BIN)
    hi = int(max(values) / BIN) + 1
    bins = [{"lo": round(i * BIN, 4), "hi": round((i + 1) * BIN, 4), "n": 0}
            for i in range(lo, hi)]
    for v in values:
        bins[min(int(v / BIN) - lo, len(bins) - 1)]["n"] += 1
    return bins


def percentile(sorted_values: list[float], x: float) -> float:
    """Empirical percentile rank: share of observations at or below x (midrank ties)."""
    lo = bisect.bisect_left(sorted_values, x)
    hi = bisect.bisect_right(sorted_values, x)
    return 100.0 * (lo + 0.5 * (hi - lo)) / len(sorted_values)


def main():
    frames, parts_by_spec, raw, _ = TL.build(include_projection=False)
    manifest = LE.load_manifest()
    obs = []
    for season in GAME_YEARS:
        obs += season_snapshots(season, manifest, frames, parts_by_spec, raw)
        print(f"{season}: {len(obs)} observations", flush=True)
    current = json.loads((ROOT / "viz" / "data" / "ratings.json").read_text(encoding="utf-8"))
    for snap in current.get("history", []):
        for r in snap["teams"]:
            obs.append({"season": current["season"], "week": snap["week"],
                        "team": r["team"], "power": round(r["power"], 5)})
    values = sorted(o["power"] for o in obs)
    payload = {"population": "team-week model Power Ratings (mean neutral-site win "
                             "probability vs every rated team), "
                             f"{GAME_YEARS[0]}-{current['season']}, preseason included",
               "note": "past seasons replayed through the production v5 ensemble",
               "bin_width": BIN, "bins": histogram(values),
               "values": [round(v, 4) for v in values],
               "observations": [[o["season"], o["week"], o["team"], o["power"]] for o in obs]}
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"-> {OUT} ({len(obs)} observations)")


if __name__ == "__main__":
    main()
