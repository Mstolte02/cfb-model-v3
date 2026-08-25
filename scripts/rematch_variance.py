"""How far apart are two meetings of the SAME two teams?

The model is graded on a residual standard deviation of about 17 points, and the
matchup page used that number to shape its simulated margins. It should not have.
That 17 is measured across every game the model called +7 - a pool of different
teams - so it carries both the noise of the sport and the model's error about who
these two teams are. Replay one matchup and the second part does not roll again:
it is a fixed error, identical in all 20,000 games.

This measures the first part on its own. Take every pair that met twice in the
same season, put both margins on the same team's side of the ledger, take out home
field, and look at how far apart the two results are. For independent draws from a
common distribution, var(difference) = 2 * var(single game), so the within-matchup
standard deviation is sd(difference) / sqrt(2).

The estimate is an UPPER bound on the noise of the sport. Those two meetings sit
weeks apart, so injuries, form and roster change are inside the difference too.

    python scripts/rematch_variance.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
HFA = 2.7          # the model's fitted home-field edge, in points


def main() -> None:
    pairs: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    games = 0

    for path in sorted(RAW.glob("games_*.json")):
        for g in json.loads(path.read_text(encoding="utf-8")):
            hp, ap = g.get("homePoints"), g.get("awayPoints")
            if hp is None or ap is None:
                continue
            if g.get("homeClassification") != "fbs" or g.get("awayClassification") != "fbs":
                continue
            games += 1
            home, away = g["homeTeam"], g["awayTeam"]
            first, second = sorted((home, away))
            margin = (hp - ap) if home == first else (ap - hp)
            if not g.get("neutralSite"):
                margin -= HFA if home == first else -HFA
            pairs[(g["season"], first, second)].append(margin)

    diffs = [v[0] - v[1] for v in pairs.values() if len(v) == 2]
    sd = statistics.stdev(diffs)
    within = sd / (2 ** 0.5)
    # Standard error of a standard deviation, for n draws: sd / sqrt(2(n-1)).
    se = within / (2 * (len(diffs) - 1)) ** 0.5

    print(f"FBS-vs-FBS games read      : {games}")
    print(f"pairs that met twice       : {len(diffs)}")
    print(f"sd of the two margins' gap : {sd:.2f}")
    print(f"within-matchup sd          : {within:.2f} +/- {se:.2f}")
    print("model residual sd          : 17.01  (viz/data/model_v4.json, margin.sigma)")


if __name__ == "__main__":
    main()
