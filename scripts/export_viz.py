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

from config import ROOT, ARTIFACTS, GAME_YEARS, TEST_GAME_YEAR, PROJECTION_YEAR
from src.model import CFBModel
from src import matchup as MU
from src import spread as SP
from scripts.train import (build_projection_frame, load_bundle,
                           projection_returning_raw)

VIZ = ROOT / "viz" / "data"


def fit_points_model():
    """Two-sided points model (points ~ O_scorer + D_opp + home) on all seasons,
    entering-season frames — powers projected scores/totals in the matchup sim."""
    from scripts.train import raw_returning, blended_talent
    from src.data import load, pff
    from src import oppadj as OA
    from config import UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA

    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    b_o, b_d = MU.fit_talent_od_slopes(
        [g for g in GAME_YEARS if g != TEST_GAME_YEAR], std, talent, od_by_year=od)

    Xs, ys = [], []
    for N in GAME_YEARS:
        u = MU.uncertainty_u(ret_raw[N]) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        frame = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc,
                              od_by_year=od)
        if frame is None:
            continue
        X, y = SP.build_points_rows(frame, games[N])
        Xs.append(X); ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    pm, alpha = SP.fit(X, y)
    resid = y - pm.predict(X)
    return {"coef": pm.coef_.tolist(), "intercept": float(pm.intercept_),
            "alpha": alpha, "resid_sd": float(np.std(resid))}


def whatif_block(comp):
    """What the page needs to re-derive O and D after someone edits a player's WAR.

    Talent is a retired feature - its coefficient is exactly zero - so it does not
    reach a prediction directly. It reaches one through the shrink in
    matchup.team_frame, where a team with little returning production is pulled toward
    its talent-implied rating:

        O_adj = (1 - lam*u)*O + lam*u*(b_o*talent)

    The useful thing about that line is that it is LINEAR in talent, so the page never
    has to reconstruct the pre-shrink O. A talent delta moves the exported vector by

        dO = lam*u*b_o*dt        dD = lam*u*b_d*dt

    and that is exact, not an approximation. So the only per-team quantity that has to
    ship is u; everything else is four scalars. Inverting the shrink to recover O would
    have worked too and is what I tried first, but it divides by (1 - lam*u) and there
    is no reason to introduce a division when the derivative is what is wanted.

    WAR reaches talent through config.WAR_BLEND, and only that share of it moves:
        talent = (1 - WAR_BLEND)*(PFF/CFBD blend) + WAR_BLEND*z(team WAR)
    The z is taken within season over the teams the WAR build covers, with ddof=0, and
    the page has to reproduce that exactly - editing one player shifts the league mean
    and sd, so every other team's talent moves a little too. That is not a rounding
    artefact, it is what the model would actually do.

    u ships per team even though it is currently 1.0 for all of them: matchup.py holds
    it flat while UNCERTAINTY_USE_RETURNING is False, which it is, and which it is for
    a measured reason. Exporting the scalar it happens to equal today would turn that
    config flag into a silent liar the moment anyone set it back to True.
    """
    from config import UNCERTAINTY_LAMBDA, WAR_BLEND
    _, b_o, b_d, lam = build_projection_frame(return_params=True)
    u = MU.uncertainty_u(projection_returning_raw())
    u = u.reindex(comp.index).fillna(0.0).clip(lower=0, upper=1)
    return {
        "lam": float(lam if lam is not None else UNCERTAINTY_LAMBDA),
        "bO": float(b_o), "bD": float(b_d), "warBlend": float(WAR_BLEND),
        "u": {t: round(float(u[t]), 4) for t in comp.index},
    }


def main(variant=""):
    from config import ROSTER_VARIANT
    kw = ROSTER_VARIANT if variant == "roster" else {}
    suffix = "_roster" if variant == "roster" else ""
    VIZ.mkdir(parents=True, exist_ok=True)
    frame = build_projection_frame(**kw)
    model = CFBModel.load()
    teams_meta = {t["school"]: t for t in
                  json.load(open(ROOT / "data" / "raw" / "teams_2026.json"))}

    # Newcomers get the same 5th-percentile fallback the simulation uses.
    comp = frame.copy()
    for t in teams_meta:
        if t not in comp.index:
            comp.loc[t] = frame.quantile(0.05)
    comp = comp.loc[[t for t in teams_meta]]

    power = pd.read_csv(ARTIFACTS / f"2026_power_ratings{suffix}.csv").set_index("team")
    playoff = {r["team"]: r for r in
               json.loads((VIZ / f"playoff{suffix}.json").read_text())["teams"]}

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
            "talent": round(float(comp.loc[t, "talent_raw"]), 3),
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
    (VIZ / f"ratings{suffix}.json").write_text(json.dumps(
        {"season": 2026, "teams": ratings, "variant": variant or "balanced"},
        indent=1))

    # The points model is trained on historical seasons only — identical across
    # frame variants, so reuse the balanced export's fit when it exists.
    base_model = VIZ / "model.json"
    if suffix and base_model.exists():
        points = json.loads(base_model.read_text())["points"]
        print("Reusing points model from model.json")
    else:
        print("Fitting points model for projected scores ...")
        points = fit_points_model()

    # Select by name: the frame now carries *_raw companions for retired features,
    # and iterating comp.loc[t] would silently widen the vector past the six the
    # coefficients are indexed against.
    FEATS = ["O", "D", "fp_margin", "pythag", "talent", "returning"]
    export = {
        "features": FEATS,
        "logistic": {"coef": model.coef.tolist(), "hfa": model.hfa_coef,
                     "intercept": model.intercept},
        "margin": {"coef": model.margin_coef.tolist(), "hfa": model.margin_hfa,
                   "intercept": model.margin_intercept,
                   "sigma": model.margin_sigma},
        "ens_w": model.ens_w,
        "points": points,
        "teams": {t: [round(float(comp.loc[t, c]), 4) for c in FEATS]
                  for t in comp.index},
        "whatif": whatif_block(comp),
    }

    # Realistic scorelines and the key-number probabilities, from scripts/score_shape.py.
    # Only the display half of that work ships: the empirical margin distribution did NOT
    # beat the normal on win probability (see config.SCORE_SHAPE), so `logistic`,
    # `margin` and `ens_w` above are untouched and the app still computes p from them.
    from config import SCORE_SHAPE
    shape_path = ARTIFACTS / "score_shape.json"
    if SCORE_SHAPE and shape_path.exists():
        export["shape"] = json.loads(shape_path.read_text())
        n_pairs = len(export["shape"]["score_table"]["pairs"])
        print(f"Score shape: {n_pairs} scorelines + margin PMF "
              f"({export['shape']['n_games']} games)")
    elif SCORE_SHAPE:
        print("  [warn] score_shape.json absent; the app will round scores as before. "
              "Run ./venv/bin/python -m scripts.score_shape")
    (VIZ / f"model{suffix}.json").write_text(json.dumps(export, indent=1))

    # Schedule (lens-independent) powers the client-side playoff re-simulation.
    export_schedule()
    export_players()

    print(f"exported {len(ratings)} rated teams, {len(export['teams'])} sim teams")
    print(f"-> {VIZ / f'ratings{suffix}.json'}\n-> {VIZ / f'model{suffix}.json'}")


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
    main(sys.argv[1] if len(sys.argv) > 1 else "")
