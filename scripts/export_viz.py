"""Export everything the viz app needs into viz/data/:

  ratings.json — power ratings + frame components + projections per team
  model.json   — win-prob + margin + points model coefficients AND per-team
                 feature vectors, so the matchup simulator runs client-side

Run after scripts.rank and scripts.simulate_playoff:
    ./venv/bin/python -m scripts.export_viz
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ROOT, ARTIFACTS, GAME_YEARS, PROJECTION_YEAR
from src import v4 as V4
from src.dynamic import WeeklyRatingState
from src import spread as SP
from scripts.train import load_bundle

VIZ = ROOT / "viz" / "data"


def fit_points_model():
    """Two-sided points model (points ~ O_scorer + D_opp + home) on all seasons,
    entering-season frames — powers projected scores/totals in the matchup sim."""
    from src import oppadj as OA
    from config import OPP_ADJ_ALPHA

    std, talent, ret, games, _ = load_bundle()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)

    Xs, ys = [], []
    for N in GAME_YEARS:
        frame = V4.build_frame(N, std, talent, ret, od)
        if frame is None:
            continue
        X, y = SP.build_points_rows(frame, games[N])
        Xs.append(X); ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    pm, alpha = SP.fit(X, y)
    resid = y - pm.predict(X)
    return {"coef": pm.coef_.tolist(), "intercept": float(pm.intercept_),
            "alpha": alpha, "resid_sd": float(np.std(resid))}


def legacy_cache_compat(model, comp, points):
    """Finite, v4-equivalent model for browsers still caching app.js?v=26.

    That app hard-codes six slots and a cross O-vs-D difference. Two duplicated team
    strengths make its cross difference an ordinary strength difference; separate
    pairs carry the logistic and margin O/D scores. Talent and returning occupy the
    two direct-difference slots the legacy app already understood. The temporary
    client therefore produces the same preseason logit and margin instead of NaN.
    New code loads model_v4.json and never uses this shim.
    """
    wc = dict(zip(model.feature_names, model.coef))
    wm = dict(zip(model.feature_names, model.margin_coef))
    teams = {}
    for team, row in comp.iterrows():
        # Fold every selected team feature (including projected WAR) into the two
        # duplicated strengths. This preserves the old client's cross-difference
        # algebra without assuming the production selector still has four features.
        logit_od = sum(wc.get(name, 0.0) * float(row.get(name, 0.0))
                       for name in model.feature_names)
        margin_od = sum(wm.get(name, 0.0) * float(row.get(name, 0.0))
                        for name in model.feature_names)
        teams[team] = [round(float(v), 4) for v in
                       [logit_od, logit_od, margin_od, margin_od,
                        0.0, 0.0]]
    return {
        "features": ["logit_a", "logit_b", "margin_a", "margin_b",
                     "talent", "returning"],
        "logistic": {"coef": [.5, .5, 0.0, 0.0,
                              0.0, 0.0],
                     "hfa": model.hfa_coef, "intercept": 0.0},
        "margin": {"coef": [0.0, 0.0, .5, .5,
                            0.0, 0.0],
                   "hfa": model.margin_hfa, "intercept": 0.0,
                   "sigma": model.margin_sigma},
        "ens_w": model.ensemble_weight,
        "probability_scale": model.probability_scale,
        "points": {**points, "coef": [0.0, 0.0, 0.0]},
        "teams": teams,
        "whatif": None,
        "derivation": {"compatibility_for": "cached app.js?v=26"},
    }


def main():
    VIZ.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ARTIFACTS / f"{PROJECTION_YEAR}_v4_team_frame.csv",
                        index_col="team")
    model = V4.ReciprocalTeamModel.load()
    teams_meta = {t["school"]: t for t in
                  json.load(open(ROOT / "data" / "raw" / "teams_2026.json"))}

    # Newcomers get the same 5th-percentile fallback the simulation uses.
    comp = frame.copy()
    for t in teams_meta:
        if t not in comp.index:
            comp.loc[t] = frame.quantile(0.05)
    comp = comp.loc[[t for t in teams_meta]]

    power = pd.read_csv(ARTIFACTS / "2026_power_ratings.csv").set_index("team")
    playoff = {r["team"]: r for r in
               json.loads((VIZ / "playoff.json").read_text())["teams"]}

    # Strength of schedule from the real 2026 slate: the mean rating of everyone a
    # team actually plays, so the dashboard can separate a good record from a good
    # team. Non-FBS opponents count at the same fixed rating the simulation uses.
    from scripts.simulate_playoff import FCS_OPP_RATING
    rate = ((comp["O"] + comp["D"]) / 2.0)
    rate = (rate - rate.mean()) / rate.std()
    sos_sum, sos_n = {t: 0.0 for t in comp.index}, {t: 0 for t in comp.index}
    for g in json.load(open(ROOT / "data" / "raw" / "schedule_2026.json")):
        h, a = g["homeTeam"], g["awayTeam"]
        hin, ain = h in rate.index, a in rate.index
        if hin and ain:
            sos_sum[h] += rate[a]; sos_n[h] += 1
            sos_sum[a] += rate[h]; sos_n[a] += 1
        elif hin or ain:
            t = h if hin else a
            sos_sum[t] += FCS_OPP_RATING; sos_n[t] += 1

    ratings = []
    for t in power.index:
        po = playoff.get(t, {})
        ratings.append({
            "rank": int(power.loc[t, "rank"]),
            "team": t,
            "conference": teams_meta.get(t, {}).get("conference", "—"),
            "power": round(float(power.loc[t, "power"]), 4),
            "vs_average": round(float(power.loc[t, "vs_average"]), 4),
            "O": round(float(comp.loc[t, "O"]), 3),
            "D": round(float(comp.loc[t, "D"]), 3),
            # Retired features read from the *_raw copies: the model column is zero
            # by design, and exporting that zero would have shown every team an
            # identical talent of 0.00 (which is what "pythag": 0.0 has been doing
            # since v3.1 - it is retired, so it is no longer exported at all).
            "talent": round(float(comp.loc[t, "talent"]), 3),
            "returning": round(float(comp.loc[t, "returning"]), 3),
            "sos": round(float(sos_sum[t] / max(sos_n[t], 1)), 3),
            "games": int(sos_n[t]),
            "avg_wins": po.get("avg_wins"),
            "avg_losses": po.get("avg_losses"),
            # The dashboard IS the odds table now, so it needs the full run of rounds
            # rather than the three headline numbers it used to show alongside the
            # O/D/talent columns. Teams the simulation never selected are absent from
            # playoff.json entirely and legitimately read 0.
            "conf_champ": po.get("conf_champ", 0.0),
            "playoff": po.get("playoff", 0.0),
            "bye": po.get("bye", 0.0),
            "sf": po.get("sf", 0.0),
            "final": po.get("final", 0.0),
            "champ": po.get("champ", 0.0),
        })
    (VIZ / "ratings.json").write_text(json.dumps(
        {"season": 2026, "teams": ratings}, indent=1))

    # Always fitted now. It used to be reused from the balanced build's model.json
    # when exporting a suffixed variant, because the points model is trained on
    # historical seasons only and is identical across frames; with one build there is
    # nothing to reuse it from.
    print("Fitting points model for projected scores ...")
    points = fit_points_model()

    # Select by name: the frame now carries *_raw companions for retired features,
    # and iterating comp.loc[t] would silently widen the vector past the six the
    # coefficients are indexed against.
    FEATS = model.feature_names
    state_path = ARTIFACTS / f"{PROJECTION_YEAR}_dynamic_state.json"
    state = (WeeklyRatingState.load(state_path) if state_path.exists() else
             WeeklyRatingState.initialize(model, frame, PROJECTION_YEAR))
    for team in comp.index:
        state.ratings.setdefault(team, model.team_logit_strength(comp, team))
    export = {
        "schema_version": 4,
        "features": FEATS,
        "logistic": {"coef": model.coef.tolist(), "hfa": model.hfa_coef,
                     "intercept": 0.0},
        "margin": {"coef": model.margin_coef.tolist(), "hfa": model.margin_hfa,
                   "intercept": 0.0,
                   "sigma": model.margin_sigma},
        "ens_w": model.ensemble_weight,
        "probability_scale": model.probability_scale,
        "architecture": "reciprocal_team_difference_v4",
        "dynamic": {"blend": state.dynamic_blend,
                    "ratings": {t: round(float(state.ratings[t]), 6)
                                for t in comp.index}},
        "points": points,
        "teams": {t: [round(float(comp.loc[t, c]), 4) for c in FEATS]
                  for t in comp.index},
        # Current-roster WAR is still exported in players.json for reporting, but it
        # is not allowed to move win probability until historical dated snapshots
        # support a clean backtest. The UI treats a null block as read-only.
        "whatif": None,
        "derivation": {"features": FEATS,
                       "temporal_contract": "N-1 team performance plus preseason-N inputs"},
    }

    # Key-number probabilities from scripts/score_shape.py. Only the margin PMF ships,
    # and only for display: the empirical margin distribution did NOT beat the normal on
    # win probability (see config.SCORE_SHAPE), so `logistic`, `margin` and `ens_w` above
    # are untouched and the app still computes p from them.
    #
    # `score_table` is deliberately NOT exported any more. It fed a display that snapped
    # each projected score to the nearest real scoreline from 2021-25, which is a better
    # guess at the scoreline and a worse one at everything beside it: snapping moves the
    # implied margin off the model's own spread, so the score and the spread printed next
    # to it disagreed about the same game. The app rounds each side instead, and the table
    # is 15KB of payload with nothing reading it. It is still in artifacts/score_shape.json
    # if the display is ever wanted back.
    from config import SCORE_SHAPE
    shape_path = ARTIFACTS / "score_shape.json"
    if SCORE_SHAPE and shape_path.exists():
        shape = json.loads(shape_path.read_text())
        export["shape"] = {k: v for k, v in shape.items() if k != "score_table"}
        print(f"Score shape: margin PMF only ({shape['n_games']} games); "
              f"score_table withheld (unused by the app)")
    elif SCORE_SHAPE:
        print("  [warn] score_shape.json absent; the matchup page will omit the "
              "key-number row. Run ./venv/bin/python -m scripts.score_shape")
    # New clients load the explicit v4 file. ``model.json`` remains a six-slot shim
    # for already-open/cached v3 clients during the migration.
    (VIZ / "model_v4.json").write_text(json.dumps(export, indent=1))
    compat = legacy_cache_compat(model, comp, points)
    if "shape" in export:
        compat["shape"] = export["shape"]
    (VIZ / "model.json").write_text(json.dumps(compat, indent=1))

    # Schedule (lens-independent) powers the client-side playoff re-simulation.
    export_schedule()
    export_players()

    print(f"exported {len(ratings)} rated teams, {len(export['teams'])} sim teams")
    print(f"-> {VIZ / 'ratings.json'}\n-> {VIZ / 'model_v4.json'}")
    print(f"-> {VIZ / 'model.json'} (cached-v3 compatibility)")


def export_schedule():
    """viz/data/schedule.json: every 2026 game, with week and date for the team page.

    The playoff re-simulation only needs {h, a, n}; the team page also wants to show
    the slate in order, so week/date/conference-game ride along.
    """
    raw = json.load(open(ROOT / "data" / "raw" / "schedule_2026.json"))
    games = [{"h": g["homeTeam"], "a": g["awayTeam"],
              "n": 1 if g.get("neutralSite") else 0,
              "w": g.get("week"), "d": (g.get("startDate") or "")[:10],
              "c": 1 if g.get("conferenceGame") else 0,
              "v": g.get("venue")} for g in raw]
    (VIZ / "schedule.json").write_text(json.dumps(games))
    print(f"-> {VIZ / 'schedule.json'} ({len(games)} games)")


def export_players():
    """viz/data/players.json: each team's 2026 two-deep with projected WAR.

    This is the roster side of the team page - who the model thinks is producing the
    wins. It comes straight from the WAR build's 2026 projection, the same numbers
    that feed the talent input, so the page cannot show a roster that disagrees with
    the rating it sits next to.
    """
    from src.data import war
    p = war.player_contributions()
    if p is None:
        print("  [warn] no WAR projections found; players.json not written")
        return
    p = p[p.proj_war.notna()].copy()

    def clean(v, cast=None):
        """None for anything missing. Bare NaN is legal in Python's json output but
        not in JSON, and the browser rejects the whole file over one of them."""
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            return None
        return cast(v) if cast else v

    # "class" is a Python keyword, so DataFrame.itertuples renames that column to a
    # positional _N and r.class is unreachable. Rename it before iterating.
    p = p.rename(columns={"class": "cls"})

    # ---- put the numbers on a wins scale -----------------------------------
    # Raw WAR does not read as wins, for two compounding reasons. First it is
    # attenuated even historically: regressing actual wins on summed team WAR gives a
    # slope near 1.6, not 1.0, because the Massey rating underneath is a noisy
    # estimate and OLS pulls the fitted spread inward. Second, a projection is a
    # conditional mean, so the 2026 numbers compress further. The league mean is
    # right; the spread is not. That is why the best roster in the country sums to
    # under 6 rather than the ~9 a 11-win team should show.
    #
    # De-attenuation is applied as ONE league-wide factor around the league mean, not
    # per team. Scaling each team to its own projected record was tried and rejected:
    # it hands schedule strength to the players, and Toledo came out needing a x2.96
    # multiplier because the MAC is weak, which says nothing about Toledo's roster.
    #
    # What is left over after de-attenuation - the gap between what the roster is
    # worth and what the team is projected to win - is real, and is reported
    # separately as a schedule and context term rather than buried in player numbers.
    REPL_WIN_PCT = 0.15
    # NO DE-ATTENUATION HERE ANY MORE. This used to multiply every roster by a
    # hardcoded 1.64 so the numbers would "read as wins", which was papering over a
    # defect one stage upstream: build_hybrid fitted the Massey-to-wins slope by OLS on
    # a rating measured with error, so the slope came out attenuated and summed team
    # WAR regressed on actual wins above replacement at 1.308 instead of 1.000. Wins
    # Above Replacement was not in units of wins.
    #
    # It is now fixed where it was broken - build_hybrid solves for the factor that
    # makes that regression exactly 1.0 - so WAR arrives here already in wins and there
    # is nothing to rescale. The identity holds at 1.0000 * WAR - 0.0023.
    #
    # scale stays in the payload at 1.0 rather than being deleted, because the app
    # reads it and a missing key would silently become undefined in the arithmetic.
    sim_path = VIZ / "playoff.json"
    proj_wins, proj_games = {}, {}
    if sim_path.exists():
        for r in json.loads(sim_path.read_text())["teams"]:
            proj_wins[r["team"]] = r["avg_wins"]
            proj_games[r["team"]] = r["avg_wins"] + r["avg_losses"]

    out = {}
    for team, g in p.groupby("team"):
        g = g.sort_values("proj_war", ascending=False)
        raw = float(g.proj_war.sum())
        # roster value on a wins scale, schedule-neutral
        roster_wins = raw
        scale = 1.0
        g = g.assign(wins_added=g.proj_war * scale)

        wins = proj_wins.get(team)
        games = proj_games.get(team) or 12.0
        repl = REPL_WIN_PCT * games
        context = (wins - repl - roster_wins) if wins is not None else None
        out[team] = {
            "total": round(float(g.proj_war.sum()), 3),
            "winsTotal": round(float(g.wins_added.sum()), 2),
            "projWins": round(float(wins), 2) if wins is not None else None,
            "replWins": round(repl, 2),
            "context": round(float(context), 2) if context is not None else None,
            "scale": round(float(scale), 3),
            "byGroup": {k: round(float(v), 3) for k, v in
                        g.groupby("broad_group").wins_added.sum().items()
                        if isinstance(k, str)},
            "players": [{
                "n": clean(r.player, str) or "—",
                "g": clean(r.broad_group, str) or "—",
                "p": clean(getattr(r, "roster_position", None), str),
                "d": clean(getattr(r, "depth", None), int),
                # Who actually starts, which is no longer the same as depth == 1. The
                # quarterback sheet, the availability file and the slot repair all move
                # the starter flag off the listed first-teamer, and the app was drawing
                # its lineup off depth - so Notre Dame's line still showed an injured
                # Charles Jagusah at right guard instead of Sullivan Absher.
                "st": bool(getattr(r, "is_starter", False)),
                "out": not bool(getattr(r, "available", True)),
                "c": clean(getattr(r, "cls", None), str),
                "cs": clean(getattr(r, "class_source", None), str),
                # Redshirt and transfer status, kept as separate flags rather than
                # baked into `c` as "RS JR". The class filter needs to be able to ask
                # for juniors and get redshirt juniors too, which a merged string
                # would not allow without parsing it back apart in the browser.
                "rs": bool(getattr(r, "redshirt", False)),
                "tr": bool(getattr(r, "is_transfer", False)),
                "w": round(float(r.wins_added), 3),
                "raw": round(float(r.proj_war), 3),
                "s": clean(r.stars, int),
                "i": bool(r.imputed),
                "sn": clean(r.snaps_2025, int),
                "w25": clean(r.war_2025, lambda x: round(float(x), 3)),
            } for r in g.itertuples()],
        }
    # allow_nan=False turns a stray NaN into a build failure instead of a file the
    # browser silently refuses to parse
    (VIZ / "players.json").write_text(json.dumps(out, allow_nan=False))
    n = sum(len(v["players"]) for v in out.values())
    print(f"-> {VIZ / 'players.json'} ({len(out)} teams, {n} players)")


if __name__ == "__main__":
    main()
