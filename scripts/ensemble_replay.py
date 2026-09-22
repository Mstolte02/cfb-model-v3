"""Stdlib runtime for the published v5 live ensemble.

The scheduled market capture runs on a bare GitHub runner, so this is a pure-Python
port of ``src/live_ensemble.py`` and ``scripts/prior_decay_backtest.todate_od``.
tests/test_ensemble_replay.py holds the two to the same numbers.

The ensemble block in viz/data/model_v4.json carries each member's fitted pieces:

  * ``initial``  - the week-0 logit strength of every team (the preseason model);
  * ``score_k``  - the step of the robust margin-residual walk;
  * ``current_halflife`` - the EWMA half-life of opponent-adjusted form (null = mean);
  * ``columns``/``scale``/``coef`` - the logistic stack over
    [prior_level, elo_change, dO, dD, hfa].

The published probability is the unweighted mean of the members.  Everything a
prediction for week W reads - ratings and form alike - is built from games strictly
before W, the same contract the backtest was scored under.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OFF = ("off_ppa", "off_pass_ppa", "off_rush_ppa", "off_success_rate",
       "off_explosiveness")
DEF = ("def_ppa", "def_success_rate", "def_explosiveness")
FORM_FIELDS = ("week", "team", "opponent", *OFF, *DEF)
SCORE_CAP = 2.5
EXPECTED_CLIP = (.01, .99)
MIN_FORM_GAMES = 2
# Bowl and playoff slates sort after every regular-season week. CFBD numbers them
# from 1 again, so the raw week would put a bowl game in front of week 2.
POSTSEASON_OFFSET = 100


def expit(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _halflife(member: dict) -> float:
    value = member.get("current_halflife")
    return math.inf if value is None else float(value)


def slate_key(week, season_type=None) -> int:
    week = int(week or 0)
    return week + POSTSEASON_OFFSET if season_type == "postseason" else week


# ---------------------------------------------------------------- current form
def form_rows(payload, teams) -> list[dict]:
    """Regular-season FBS-vs-FBS team-game rows with every input present."""
    if payload is None:
        return []
    fields = payload.get("fields", FORM_FIELDS)
    field = set(teams)
    out = []
    for values in payload.get("rows", []):
        row = dict(zip(fields, values))
        if row.get("team") not in field or row.get("opponent") not in field:
            continue
        if row.get("week") is None or any(row.get(c) is None for c in (*OFF, *DEF)):
            continue
        out.append(row)
    return out


def _z(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return [(v - mean) / sd for v in values] if sd > 1e-12 else [0.0 for _ in values]


def _ewma(values: list[float], halflife: float) -> float:
    if not math.isfinite(halflife):
        return sum(values) / len(values)
    last = len(values) - 1
    weights = [.5 ** ((last - i) / halflife) for i in range(len(values))]
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


def form_before(rows: list[dict], week: int, halflife: float) -> dict | None:
    """Opponent-adjusted EWMA O/D from games before ``week``, as {team: [O, D, n]}.

    Port of prior_decay_backtest.todate_od: every standardisation is recomputed at
    the cutoff, so a later game can never move an earlier prediction.
    """
    past = [row for row in rows if int(row["week"]) < week]
    if not past:
        return None
    zcols = {c: _z([float(row[c]) for row in past]) for c in (*OFF, *DEF)}
    games: dict[str, list[tuple[int, int]]] = {}
    for i, row in enumerate(past):
        games.setdefault(row["team"], []).append((int(row["week"]), i))
    o_raw, d_raw, count, schedule = {}, {}, {}, {}
    for team, entries in games.items():
        entries.sort(key=lambda item: item[0])          # stable, as pandas'
        order = [i for _, i in entries]
        o_game = [sum(zcols[c][i] for c in OFF) / len(OFF) for i in order]
        d_game = [-sum(zcols[c][i] for c in DEF) / len(DEF) for i in order]
        o_raw[team], d_raw[team] = _ewma(o_game, halflife), _ewma(d_game, halflife)
        count[team] = len(order)
        schedule[team] = [past[i]["opponent"] for i in order]
    O, D = o_raw, d_raw
    for _ in range(2):
        O, D = ({t: O[t] + .5 * statistics.fmean([D.get(o, 0.0) for o in schedule[t]])
                 for t in O},
                {t: D[t] + .5 * statistics.fmean([O.get(o, 0.0) for o in schedule[t]])
                 for t in D})
    names = list(O)
    oz, dz = _z([O[t] for t in names]), _z([D[t] for t in names])
    return {t: [oz[i], dz[i], count[t]] for i, t in enumerate(names)}


# ---------------------------------------------------------------- prediction
def _features(member: dict, ratings: dict, form: dict | None,
              home: str, away: str, hfa: float) -> dict:
    initial = member["initial"]
    fh = form.get(home) if form else None
    fa = form.get(away) if form else None
    have = (fh is not None and fa is not None and fh[2] >= MIN_FORM_GAMES
            and fa[2] >= MIN_FORM_GAMES)
    return {"prior_level": initial[home] - initial[away],
            "elo_change": ((ratings[home] - initial[home])
                           - (ratings[away] - initial[away])),
            "dO": fh[0] - fa[0] if have else 0.0,
            "dD": fh[1] - fa[1] if have else 0.0,
            "hfa": float(hfa)}


def member_probability(member: dict, ratings: dict, form: dict | None,
                       home: str, away: str, hfa: float) -> float:
    x = _features(member, ratings, form, home, away, hfa)
    z = sum(c * x[col] / s for col, s, c in
            zip(member["columns"], member["scale"], member["coef"]))
    return expit(z)


def probability(ensemble: dict, state: dict, home: str, away: str,
                hfa: float) -> float | None:
    """Published home win probability: the unweighted member mean."""
    members = ensemble["members"]
    if any(home not in m["initial"] or away not in m["initial"] for m in members):
        return None
    values = [member_probability(m, state["ratings"][m["name"]],
                                 state["form"].get(m["name"]), home, away, hfa)
              for m in members]
    return sum(values) / len(values)


def initial_state(ensemble: dict) -> dict:
    return {"ratings": {m["name"]: dict(m["initial"]) for m in ensemble["members"]},
            "form": {m["name"]: None for m in ensemble["members"]}}


def _forms(ensemble: dict, rows: list[dict], cutoff: int, cache: dict) -> dict:
    out = {}
    for member in ensemble["members"]:
        half = _halflife(member)
        key = (cutoff, half)
        if key not in cache:
            cache[key] = form_before(rows, cutoff, half)
        out[member["name"]] = cache[key]
    return out


def replay(ensemble: dict, finals: list[dict], rows: list[dict],
           stop_before: int | None = None) -> dict:
    """Walk completed games slate by slate from the week-0 ratings.

    ``finals`` rows are {home, away, neutral, home_score, away_score, week, season_type,
    id?}.  Returns the per-game pregame record, the state after each slate, and the
    final state.  With ``stop_before`` the walk ends at the start of that slate key.
    """
    state = initial_state(ensemble)
    by_slate: dict[int, list[dict]] = {}
    for game in finals:
        by_slate.setdefault(slate_key(game["week"], game.get("season_type")),
                            []).append(game)
    cache: dict = {}
    events, snapshots = [], []
    for key in sorted(by_slate):
        if stop_before is not None and key >= stop_before:
            break
        state["form"] = _forms(ensemble, rows, key, cache)
        changes = {m["name"]: {} for m in ensemble["members"]}
        for game in by_slate[key]:
            home, away = game["home"], game["away"]
            hfa = 0.0 if game.get("neutral") else 1.0
            p = probability(ensemble, state, home, away, hfa)
            margin = float(game["home_score"]) - float(game["away_score"])
            deltas = []
            for member in ensemble["members"]:
                r = state["ratings"][member["name"]]
                sigma = float(member["margin_sigma"])
                logit = r[home] - r[away] + float(member["hfa_coef"]) * hfa
                clipped = min(max(expit(logit), EXPECTED_CLIP[0]), EXPECTED_CLIP[1])
                expected = sigma * statistics.NormalDist().inv_cdf(clipped)
                delta = float(member["score_k"]) * min(max(
                    (margin - expected) / sigma, -SCORE_CAP), SCORE_CAP)
                bucket = changes[member["name"]]
                bucket[home] = bucket.get(home, 0.0) + delta
                bucket[away] = bucket.get(away, 0.0) - delta
                deltas.append(delta)
            events.append({**game, "slate": key, "p_home": p,
                           "home_rating_delta": sum(deltas) / len(deltas)})
        for name, bucket in changes.items():
            for team, delta in bucket.items():
                state["ratings"][name][team] += delta
        snapshots.append((key, {"ratings": {n: dict(r) for n, r in
                                            state["ratings"].items()},
                                "form": _forms(ensemble, rows, key + 1, cache)}))
    end = stop_before if stop_before is not None else (
        max(by_slate) + 1 if by_slate else 1)
    state["form"] = _forms(ensemble, rows, end, cache)
    return {"events": events, "snapshots": snapshots, "state": state}


def power_table(ensemble: dict, state: dict, names: list[str]) -> list[dict]:
    """Mean neutral-site win probability against every other rated team.

    Neutral probabilities are exact complements (no intercept, antisymmetric
    features), so each pair is evaluated once.  ``vs_average`` is the probability
    against a synthetic team whose every feature is the field mean.
    """
    total = {t: 0.0 for t in names}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            p = probability(ensemble, state, left, right, 0.0)
            total[left] += p
            total[right] += 1.0 - p
    average = {}
    for member in ensemble["members"]:
        name = member["name"]
        initial, ratings = member["initial"], state["ratings"][name]
        form = state["form"].get(name) or {}
        eligible = [form[t] for t in names if t in form and form[t][2] >= MIN_FORM_GAMES]
        average[name] = {
            "initial": statistics.fmean(initial[t] for t in names),
            "change": statistics.fmean(ratings[t] - initial[t] for t in names),
            "O": statistics.fmean(f[0] for f in eligible) if eligible else 0.0,
            "D": statistics.fmean(f[1] for f in eligible) if eligible else 0.0,
        }
    rows = []
    for team in names:
        values = []
        for member in ensemble["members"]:
            name = member["name"]
            a = average[name]
            f = (state["form"].get(name) or {}).get(team)
            have = f is not None and f[2] >= MIN_FORM_GAMES
            x = {"prior_level": member["initial"][team] - a["initial"],
                 "elo_change": (state["ratings"][name][team]
                                - member["initial"][team]) - a["change"],
                 "dO": f[0] - a["O"] if have else 0.0,
                 "dD": f[1] - a["D"] if have else 0.0, "hfa": 0.0}
            values.append(expit(sum(c * x[col] / s for col, s, c in zip(
                member["columns"], member["scale"], member["coef"]))))
        rows.append({"team": team,
                     "power": total[team] / max(len(names) - 1, 1),
                     "vs_average": sum(values) / len(values)})
    rows.sort(key=lambda row: (-row["power"], row["team"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["power"] = round(row["power"], 6)
        row["vs_average"] = round(row["vs_average"], 6)
    return rows


def compact_state(state: dict, digits: int = 8) -> dict:
    """JSON-ready state: rounded ratings and {team: [O, D, n]} form per member."""
    def form(value):
        if not value:
            return {}
        return {t: [round(v[0], digits), round(v[1], digits), int(v[2])]
                for t, v in value.items()}
    return {"ratings": {n: {t: round(v, digits) for t, v in r.items()}
                        for n, r in state["ratings"].items()},
            "form": {n: form(f) for n, f in state["form"].items()}}


def load_form_payload(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def compact_form_payload(raw: list[dict], season: int) -> dict:
    """Flatten CFBD /stats/game/advanced into the committed regular-season table."""
    rows = []
    for game in raw:
        if game.get("seasonType", "regular") != "regular":
            continue
        off, deff = game.get("offense") or {}, game.get("defense") or {}
        rows.append([
            game.get("week"), game.get("team"), game.get("opponent"),
            off.get("ppa"), (off.get("passingPlays") or {}).get("ppa"),
            (off.get("rushingPlays") or {}).get("ppa"), off.get("successRate"),
            off.get("explosiveness"), deff.get("ppa"), deff.get("successRate"),
            deff.get("explosiveness")])
    rows.sort(key=lambda r: (r[0] or 0, r[1] or "", r[2] or ""))
    return {"season": season,
            "source": "CFBD /stats/game/advanced, excludeGarbageTime=true, regular season",
            "fields": list(FORM_FIELDS), "rows": rows}
