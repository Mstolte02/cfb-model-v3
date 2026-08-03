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
  h2h         net record against the teams ranked within 15 places of you

Fit is a linear score against negated rank, then validated by Spearman correlation
against the real final ranking on held-out seasons.

THREE CANDIDATES WERE TESTED AND ONE SURVIVED. All three are things the committee is
widely believed to weigh, and the difference between them is only what twelve seasons
of published rankings will actually support. LOSO Spearman, base model 0.9084:

  h2h         0.9126 (+0.0042)  KEPT. The only candidate that stays positive whichever
                                season is deleted (worst 11/12 +0.0020), and it helps
                                in 7 of 12. Fitted weight +0.056, i.e. beating one
                                near-neighbour is worth about half a tenth of a win.
  preseason   0.9083 (-0.0001)  DROPPED. Worthless on its own, and the +0.0057 it looks
                                worth alongside h2h is 2020 and nothing else: delete
                                that season and it turns negative. 2020 is the season
                                with 563 games and teams playing between four and
                                eleven of them, where record and SOS stop meaning the
                                same thing and ANY stable prior would help. That is a
                                fact about 2020, not evidence the committee anchors.
  loss_late   0.9075 (-0.0008)  DROPPED. Negative alone and negative in every subset
                                that excludes preseason. Whatever the committee does
                                about a November loss, it is already inside win_pct
                                and strength of schedule.

The selection gate is the JACKKNIFE - does the gain survive deleting any one season -
rather than a t-test, which on twelve seasons rejects everything including h2h. Both
rejected features are still computed and still scored on every run, so the finding is
reproducible rather than remembered.

Run: ./venv/bin/python -m scripts.fit_committee
"""
import json
import sys
from itertools import combinations
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


def pull_preseason(years):
    """{season: {team: rank}} from the week-1 AP Top 25, already in the rankings cache.

    The preseason poll is the committee's starting point in the sense that matters
    here: it is published before anyone has played, so nothing about it can be caused
    by the season, and any weight it earns is evidence of anchoring rather than of the
    poll being right.
    """
    out = {}
    for y in years:
        weeks = c._get("/rankings", {"year": y}, f"rankings_{y}.json") or []
        for wk in weeks:
            if wk.get("seasonType") != "regular" or (wk.get("week") or 0) != 1:
                continue
            for p in wk.get("polls", []):
                if p.get("poll") == "AP Top 25":
                    out[y] = {r["school"]: r["rank"] for r in p["ranks"]}
    return out


def preseason_score(rank):
    """25 for the preseason No. 1 down to 1 for No. 25, 0 for everyone unranked."""
    return 0.0 if rank is None else float(max(0, 26 - rank))


def season_features(years):
    """Per team-season: realized record, season-long quality rating, SOS, loss timing,
    the preseason poll, and the raw results needed to build head-to-head later."""
    rows = []
    results = {}          # season -> list of (winner, loser), FBS vs FBS only
    pre = pull_preseason(years)
    for N in years:
        raw = c.games(N)
        played = [(g["homeTeam"], g["awayTeam"], g["homePoints"], g["awayPoints"],
                   g.get("week") or 0)
                  for g in raw
                  if g.get("homePoints") is not None and g.get("awayPoints") is not None]
        fbs = {t["school"] for t in c.fbs_teams(N)}
        teams = sorted({t for g in played for t in g[:2]} & fbs)
        if len(teams) < 100:
            print(f"  [warn] {N}: only {len(teams)} FBS teams with results; skipped")
            continue
        rating = massey_rating([g[:4] for g in played], teams)
        rating = (rating - rating.mean()) / rating.std()
        last_week = max((g[4] for g in played), default=1) or 1

        w = {t: 0 for t in teams}
        gp = {t: 0 for t in teams}
        opp = {t: [] for t in teams}
        # How LATE a team's losses were. Each loss scores a ramp running -0.5 at the
        # opener to +0.5 at the finale, so two November losses are positive, two
        # September losses are negative, and an undefeated team is zero.
        #
        # The centring does NOT change the fit, and it is worth being clear about why:
        # the uncentred sum differs from this one by 0.5 x the loss count, which is an
        # exact linear function of win_pct and the intercept, both already in the model.
        # OLS is invariant to that reparametrisation - swapping one for the other
        # reproduces the LOSO Spearman to four decimals. It is centred anyway because
        # the shipped weight then means "the extra cost of losing late" on its own, and
        # because the per-team score decomposition on the playoff page shows this term
        # directly: an undefeated team has to contribute zero there, not an offset that
        # silently cancels against win_pct.
        late = {t: 0.0 for t in teams}
        res = []
        for h, a, hp, ap, wk in played:
            hin, ain = h in w, a in w
            ramp = (wk - 1) / max(last_week - 1, 1) - 0.5
            if hin and ain:
                gp[h] += 1; gp[a] += 1
                opp[h].append(rating[a]); opp[a].append(rating[h])
                win, lose = (h, a) if hp > ap else (a, h)
                w[win] += 1
                late[lose] += ramp
                res.append((win, lose))
            elif hin or ain:
                t = h if hin else a
                gp[t] += 1
                opp[t].append(-2.0)          # non-FBS opponent, as the sim assumes
                if (hp > ap) == (t == h):
                    w[t] += 1
                else:
                    late[t] += ramp
        results[N] = res
        for t in teams:
            if gp[t] < 8:
                continue
            rows.append({"season": N, "team": t, "wins": w[t], "games": gp[t],
                         "win_pct": w[t] / gp[t], "rating_z": float(rating[t]),
                         "sos": float(np.mean(opp[t])) if opp[t] else 0.0,
                         "loss_late": late[t] / gp[t],
                         "preseason": preseason_score(pre.get(N, {}).get(t))})
        print(f"  {N}: {len(teams)} teams, {len(played)} games, "
              f"{len(pre.get(N, {}))} preseason-ranked")
    d = pd.DataFrame(rows)
    d["sos_z"] = d.groupby("season").sos.transform(lambda s: (s - s.mean()) / s.std())
    return d, results


def h2h_feature(d, results, score, k=10):
    """Net head-to-head record against the teams a team is actually being compared to.

    Head-to-head is the committee's stated tiebreaker, and a tiebreaker is by definition
    circular: it only matters between teams that are already close, and "close" is what
    the ranking decides. So it is computed in two passes - `score` is the ranking the
    rest of the model produces with this feature switched off, and the mask keeps only
    pairs within `k` places of each other in it.

    Vectorized as the signed result matrix: R[i,j] = 1 if i beat j and -1 if j beat i,
    masked by the proximity matrix and summed along the row. A team that beat two
    neighbours and lost to one scores +1.
    """
    out = pd.Series(0.0, index=d.index)
    for N, g in d.groupby("season"):
        idx = {t: i for i, t in enumerate(g.team)}
        n = len(idx)
        R = np.zeros((n, n))
        for win, lose in results.get(N, []):
            i, j = idx.get(win), idx.get(lose)
            if i is not None and j is not None:
                R[i, j], R[j, i] = 1.0, -1.0
        rank = np.argsort(np.argsort(-score.loc[g.index].to_numpy()))
        near = np.abs(rank[:, None] - rank[None, :]) <= k
        out.loc[g.index] = (R * near).sum(1)
    return out


BASE = ["win_pct", "rating_z", "sos_z", "p4"]


def fit_with_h2h(d, feats, results, names, k, iters=3):
    """Least squares on `names`, with the h2h column rebuilt from its own fit.

    h2h needs a ranking to decide which pairs are close enough to matter, and that
    ranking is what the fit produces, so the two are solved together: score with h2h
    off, rebuild h2h from that score, refit, repeat. Three passes is comfortably past
    the point where the weights stop moving.

    The provisional score is built on `feats` - EVERY FBS team - rather than on the
    ranked subset, because a team's head-to-head record includes the unranked teams it
    lost to, and those losses are exactly the ones the committee punishes.
    """
    use_h2h = "h2h" in names
    plain = [n for n in names if n != "h2h"]
    m = LinearRegression().fit(d[plain].to_numpy(float), (-d["rank"]).to_numpy(float))
    if not use_h2h:
        return m, plain, None
    provisional = m
    for _ in range(iters):
        score = pd.Series(provisional.predict(feats[plain].to_numpy(float)),
                          index=feats.index)
        feats = feats.copy()
        feats["h2h"] = h2h_feature(feats, results, score, k)
        d = d.drop(columns=["h2h"], errors="ignore").merge(
            feats[["season", "team", "h2h"]], on=["season", "team"], how="left")
        y = (-d["rank"]).to_numpy(float)
        m = LinearRegression().fit(d[names].to_numpy(float), y)
        provisional = LinearRegression().fit(d[plain].to_numpy(float), y)
    return m, names, (provisional, k)


def loso(d, feats, results, names, k=10):
    """Per-season leave-one-season-out Spearman against the real final ranking.

    Returns the twelve season values rather than their mean, because deciding whether a
    feature is worth keeping means asking whether it helps CONSISTENTLY, and a mean
    over twelve numbers cannot answer that.
    """
    rhos = []
    for s in sorted(d.season.unique()):
        tr = d[d.season != s]
        ftr = feats[feats.season != s]
        m, cols, prov = fit_with_h2h(tr, ftr, results, names, k)
        te = d[d.season == s].copy()
        if prov is not None:
            # the held-out season's h2h uses the FOLD's weights, never its own ranking
            pm, kk = prov
            fte = feats[feats.season == s].copy()
            score = pd.Series(pm.predict(fte[[n for n in names if n != "h2h"]]
                                         .to_numpy(float)), index=fte.index)
            fte["h2h"] = h2h_feature(fte, results, score, kk)
            te = te.drop(columns=["h2h"], errors="ignore").merge(
                fte[["season", "team", "h2h"]], on=["season", "team"], how="left")
        p = pd.Series(m.predict(te[cols].to_numpy(float)))
        rhos.append(p.corr(pd.Series((-te["rank"]).to_numpy(float)), method="spearman"))
    return np.array(rhos)


def paired(a, b):
    """(mean difference, its standard error) over the seasons, a against b.

    Paired because both arms see the same twelve seasons and the season-to-season
    spread of Spearman (roughly 0.87-0.95) dwarfs the differences being judged. An
    unpaired comparison of two means that close would be uninformative; the paired
    spread is what says whether a feature helps in most seasons or in one.
    """
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))


def leverage(a, b, years):
    """(worst-case mean gain after deleting one season, which season that was).

    Twelve seasons is few enough that one of them can carry a feature single-handed,
    and 2020 is the obvious candidate: 563 games, teams playing between four and eleven
    of them, and a base model that falls from ~0.93 to 0.71 because win_pct and SOS
    stop meaning the same thing. A feature whose entire case is one season has not been
    shown to work; it has been shown to patch that season.
    """
    diff = np.asarray(a) - np.asarray(b)
    means = [(np.delete(diff, i).mean(), years[i]) for i in range(len(diff))]
    return min(means)


def main():
    load.require_key()
    print("Pulling committee rankings ...")
    ranks = pull_rankings()

    print("\nBuilding season features ...")
    feats, results = season_features(YEARS)
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

    # the weights that shipped before any of this, as the floor to beat
    rhos_old = []
    for s in sorted(d.season.unique()):
        te = d[d.season == s]
        old = 10 * te.win_pct + 1.0 * te.rating_z + 0.75 * te.sos_z
        rhos_old.append(old.corr(pd.Series((-te["rank"]).to_numpy(float),
                                           index=old.index), method="spearman"))
    base = loso(d, feats, results, BASE)
    print(f"\nleave-one-season-out Spearman vs the real final ranking:")
    print(f"  hand-set weights, pre-2026     : {np.mean(rhos_old):.4f}")
    print(f"  fitted, {'+'.join(BASE)} : {base.mean():.4f}   <- the number to beat")

    # ---- how close does head-to-head have to be to count? ------------------------
    print("\nhead-to-head proximity (how many places apart two teams can be and still "
          "have their meeting matter):")
    best_k, best_k_rho = 10, -1.0
    for k in (3, 5, 8, 10, 15, 25):
        r = loso(d, feats, results, BASE + ["h2h"], k).mean()
        print(f"  within {k:>2} places: {r:.4f}")
        if r > best_k_rho:
            best_k, best_k_rho = k, r

    CANDS = ("preseason", "loss_late", "h2h")
    print(f"\nadd-one-in (h2h at its best proximity, {best_k} places). "
          f"+/- is the paired standard error over the 12 seasons:")
    for f in CANDS:
        r = loso(d, feats, results, BASE + [f], best_k)
        m, se = paired(r, base)
        print(f"  + {f:<10} {r.mean():.4f}   ({m:+.4f} +/- {se:.4f})")

    full_names = BASE + list(CANDS)
    full = loso(d, feats, results, full_names, best_k)
    m, se = paired(full, base)
    print(f"\nall three together: {full.mean():.4f}   ({m:+.4f} +/- {se:.4f})")

    # ---- every subset, because the candidates interact -----------------------------
    # Drop-one-out from the full set is not enough to choose with: it says what a
    # feature is worth ALONGSIDE THE OTHER TWO, and that is not the question when one
    # of the other two is about to be thrown out. loss_late is the case in point - it
    # looks worth +0.0014 next to preseason and h2h, and costs 0.0029 next to h2h
    # alone. Three candidates is eight subsets, so just score them all.
    years = sorted(d.season.unique())
    print("\nevery subset. `worst 11/12` deletes the single season most favourable to "
          "the subset, which is the check that stops one odd year carrying a feature:")
    print(f"  {'features added to the base':<34}{'LOSO':>8}{'vs base':>10}"
          f"{'+/-':>8}{'worst 11/12':>13}{'seasons up':>12}")
    subsets = []
    for r_ in range(len(CANDS) + 1):
        for combo in combinations(CANDS, r_):
            rr = base if not combo else loso(d, feats, results, BASE + list(combo), best_k)
            m, se = paired(rr, base)
            lv, worst_year = leverage(rr, base, years)
            up = int((np.asarray(rr) > np.asarray(base)).sum())
            subsets.append((list(combo), rr, m, se, lv, worst_year))
            label = " + ".join(combo) if combo else "(base only)"
            print(f"  {label:<34}{rr.mean():>8.4f}{m:>+10.4f}{se:>8.4f}"
                  f"{lv:>+11.4f} ({worst_year}){up:>7}/{len(years)}")

    # THE GATE IS THE JACKKNIFE, NOT THE STANDARD ERROR. On twelve seasons a t-test is
    # too blunt to be worth deferring to - it rejects everything here, including a
    # feature that helps in the majority of seasons and survives every deletion - and
    # its threshold would be arbitrary either way. What can be asked of twelve seasons
    # is whether the gain depends on any ONE of them, and that question separates these
    # candidates cleanly: exactly one subset stays positive no matter which season is
    # removed. The standard error is printed for information and not used to choose.
    ok = [s for s in subsets if s[0] and s[2] > 0 and s[4] > 0]
    if ok:
        keep, chosen, m, se, lv, _ = max(ok, key=lambda s: s[4])
    else:
        print("\n  no subset survives deletion of its best season; the base model "
              "ships unchanged.")
        keep, chosen = [], base
        m, se, lv = 0.0, 0.0, 0.0
    names = BASE + keep
    chosen_rho = float(chosen.mean())
    base_rho = float(base.mean())
    print(f"\nkeeping: {', '.join(keep) if keep else '(none - base model unchanged)'}")
    print(f"  LOSO Spearman {chosen_rho:.4f} against {base_rho:.4f} for the base "
          f"model ({m:+.4f} +/- {se:.4f}, worst 11/12 {lv:+.4f})")
    for f in CANDS:
        if f not in keep:
            print(f"  [dropped] {f}")

    full, cols, prov = fit_with_h2h(d, feats, results, names, best_k)
    # rescale so win_pct keeps a weight of 10, matching the existing config units
    scale = 10.0 / full.coef_[0]
    w = {n: round(float(cf * scale), 4) for n, cf in zip(cols, full.coef_)}
    print("\nfitted committee weights (win_pct pinned at 10 for comparability):")
    for n in cols:
        print(f"  {n:<10} {w[n]:+.3f}")

    out = {"weights": w, "features": cols,
           "h2h_within": best_k if "h2h" in cols else None,
           # the simulation needs the h2h-free weights too: they are what produces the
           # provisional ranking that decides which meetings count
           "provisional_weights": (
               {n: round(float(cf * scale), 4)
                for n, cf in zip([c for c in cols if c != "h2h"], prov[0].coef_)}
               if prov is not None else None),
           "seasons": sorted(int(s) for s in d.season.unique()),
           "n": int(len(d)),
           "loso_spearman_fitted": round(chosen_rho, 4),
           "loso_spearman_base": round(base_rho, 4),
           "loso_spearman_previous": round(float(np.mean(rhos_old)), 4)}
    (ARTIFACTS / "committee_model.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {ARTIFACTS / 'committee_model.json'}")


if __name__ == "__main__":
    main()
