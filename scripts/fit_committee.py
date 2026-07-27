"""Fit the playoff committee's selection behaviour on its own history (2014-2025).

The simulation has to rank a simulated season the way the committee would, and until
now it did that with three hand-set weights weighted against a single year's final
ranking. The committee has published a weekly ranking every year since 2014, so the
behaviour can be fitted properly.

Target: the FINAL regular-season committee ranking (selection day), which is the one
that actually picks the field. Features are the things the sim knows about a
simulated season and the committee knew about the real one:

  win_pct     record, the dominant term
  rating_z    team quality independent of record - the sim's own preseason rating,
              used here rather than an end-of-season rating so the fitted weight is
              calibrated to the same noisy quantity the sim will feed it
  sos_z       strength of schedule, mean opponent rating
  p4          whether the team plays in a power conference

Fit is a linear score against negated rank, then validated by Spearman correlation
against the real final ranking on held-out seasons.

Run: ./venv/bin/python -m scripts.fit_committee
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import ROOT, ARTIFACTS
from src.data import load, cfbd_client as c

YEARS = list(range(2014, 2026))
POLL = "Playoff Committee Rankings"
CACHE = ROOT / "data" / "raw" / "committee_rankings.json"
P4_BY_ERA = {  # conference names change; membership is what matters
    "SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10",
}
# Notre Dame is an independent but has a CFP contract slot and is treated by the
# committee as a power program, so the "power" flag has to include it or the model
# systematically under-ranks the one team the flag was never meant to catch.
POWER_INDEPENDENTS = {"Notre Dame"}


def pull_rankings():
    """Final regular-season committee ranking per season, cached."""
    if CACHE.exists():
        return json.load(open(CACHE))
    out = {}
    for y in YEARS:
        weeks = c._get("/rankings", {"year": y}, f"rankings_{y}.json") or []
        best = None
        for wk in weeks:
            for p in wk.get("polls", []):
                if p.get("poll") != POLL:
                    continue
                # selection day is the last committee poll of the regular season
                key = (wk.get("seasonType") == "postseason", wk.get("week") or 0)
                if best is None or key > best[0]:
                    best = (key, [{"rank": r["rank"], "team": r["school"]}
                                  for r in p["ranks"]])
        if best:
            out[str(y)] = best[1]
            print(f"  {y}: {len(best[1])} ranked teams")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def massey_rating(games, teams):
    """Opponent-adjusted point-differential rating, capped at 28 per game.

    The sim's `rating` is a preseason projection, but inside a simulated season it is
    the parameter the games were generated from - i.e. that team's true quality. The
    honest analogue in real history is the best available estimate of true quality
    that season, which is a full-season opponent-adjusted margin, not a preseason
    guess. That is also the only version of the feature available back to 2014.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    M = np.zeros((n, n))
    b = np.zeros(n)
    for h, a, hp, ap in games:
        if h not in idx or a not in idx:
            continue
        i, j = idx[h], idx[a]
        m = float(np.clip(hp - ap, -28, 28))
        M[i, i] += 1; M[j, j] += 1
        M[i, j] -= 1; M[j, i] -= 1
        b[i] += m; b[j] -= m
    M[-1, :] = 1.0
    b[-1] = 0.0
    try:
        r = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        r = np.linalg.lstsq(M, b, rcond=None)[0]
    return pd.Series(r, index=teams)


def season_features(years):
    """Per team-season: realized record, season-long quality rating, and SOS."""
    rows = []
    for N in years:
        raw = c.games(N)
        played = [(g["homeTeam"], g["awayTeam"], g["homePoints"], g["awayPoints"])
                  for g in raw
                  if g.get("homePoints") is not None and g.get("awayPoints") is not None]
        fbs = {t["school"] for t in c.fbs_teams(N)}
        teams = sorted({t for g in played for t in g[:2]} & fbs)
        if len(teams) < 100:
            print(f"  [warn] {N}: only {len(teams)} FBS teams with results; skipped")
            continue
        rating = massey_rating(played, teams)
        rating = (rating - rating.mean()) / rating.std()

        w = {t: 0 for t in teams}
        gp = {t: 0 for t in teams}
        opp = {t: [] for t in teams}
        for h, a, hp, ap in played:
            hin, ain = h in w, a in w
            if hin and ain:
                gp[h] += 1; gp[a] += 1
                opp[h].append(rating[a]); opp[a].append(rating[h])
                if hp > ap: w[h] += 1
                else: w[a] += 1
            elif hin or ain:
                t = h if hin else a
                gp[t] += 1
                opp[t].append(-2.0)          # non-FBS opponent, as the sim assumes
                if (hp > ap) == (t == h):
                    w[t] += 1
        for t in teams:
            if gp[t] < 8:
                continue
            rows.append({"season": N, "team": t, "wins": w[t], "games": gp[t],
                         "win_pct": w[t] / gp[t], "rating_z": float(rating[t]),
                         "sos": float(np.mean(opp[t])) if opp[t] else 0.0})
        print(f"  {N}: {len(teams)} teams, {len(played)} games")
    d = pd.DataFrame(rows)
    d["sos_z"] = d.groupby("season").sos.transform(lambda s: (s - s.mean()) / s.std())
    return d


def main():
    load.require_key()
    print("Pulling committee rankings ...")
    ranks = pull_rankings()

    print("\nBuilding season features ...")
    feats = season_features(YEARS)
    conf = {t["school"]: t.get("conference") for t in
            json.load(open(ROOT / "data" / "raw" / "teams_2026.json"))}
    feats["p4"] = feats.team.map(
        lambda t: 1.0 if (conf.get(t) in P4_BY_ERA or t in POWER_INDEPENDENTS) else 0.0)

    rk = pd.DataFrame([{"season": int(y), "team": r["team"], "rank": r["rank"]}
                       for y, rs in ranks.items() for r in rs])
    d = feats.merge(rk, on=["season", "team"], how="inner")
    print(f"  matched team-seasons with a committee rank: {len(d)} "
          f"across {d.season.nunique()} seasons")
    if len(d) < 100:
        sys.exit("too few matched rows to fit")

    X = d[["win_pct", "rating_z", "sos_z", "p4"]].to_numpy(float)
    y = (-d["rank"]).to_numpy(float)

    # leave-one-season-out: fit elsewhere, score the held-out year by Spearman
    rhos_new, rhos_old = [], []
    for s in sorted(d.season.unique()):
        tr, te = d.season != s, d.season == s
        m = LinearRegression().fit(X[tr], y[tr])
        p = m.predict(X[te])
        rhos_new.append(pd.Series(p).corr(pd.Series(y[te]), method="spearman"))
        # the weights that ship today, for comparison
        old = 10 * d.loc[te, "win_pct"] + 1.0 * d.loc[te, "rating_z"] + 0.75 * d.loc[te, "sos_z"]
        rhos_old.append(old.corr(pd.Series(y[te], index=old.index), method="spearman"))

    print(f"\nleave-one-season-out Spearman vs the real final ranking:")
    print(f"  current hand-set weights : {np.mean(rhos_old):.3f}")
    print(f"  fitted on 2014-2025      : {np.mean(rhos_new):.3f}")

    full = LinearRegression().fit(X, y)
    names = ["win_pct", "rating_z", "sos_z", "p4"]
    # rescale so win_pct keeps a weight of 10, matching the existing config units
    scale = 10.0 / full.coef_[0]
    w = {n: round(float(cf * scale), 4) for n, cf in zip(names, full.coef_)}
    print("\nfitted committee weights (win_pct pinned at 10 for comparability):")
    for n in names:
        print(f"  {n:<10} {w[n]:+.3f}")

    out = {"weights": w, "seasons": sorted(int(s) for s in d.season.unique()),
           "n": int(len(d)),
           "loso_spearman_fitted": round(float(np.mean(rhos_new)), 4),
           "loso_spearman_previous": round(float(np.mean(rhos_old)), 4)}
    (ARTIFACTS / "committee_model.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {ARTIFACTS / 'committee_model.json'}")


if __name__ == "__main__":
    main()
