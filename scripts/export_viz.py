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

from config import ROOT, ARTIFACTS, GAME_YEARS, TEST_GAME_YEAR
from src.model import CFBModel
from src import matchup as MU
from src import spread as SP
from scripts.train import build_projection_frame, load_bundle

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
        u = (1.0 - ret_raw[N]).clip(lower=0, upper=1) if N in ret_raw else None
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
            "avg_wins": po.get("avg_wins"),
            "avg_losses": po.get("avg_losses"),
            "playoff": po.get("playoff", 0.0),
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

    export = {
        "features": ["O", "D", "fp_margin", "pythag", "talent", "returning"],
        "logistic": {"coef": model.coef.tolist(), "hfa": model.hfa_coef,
                     "intercept": model.intercept},
        "margin": {"coef": model.margin_coef.tolist(), "hfa": model.margin_hfa,
                   "intercept": model.margin_intercept,
                   "sigma": model.margin_sigma},
        "ens_w": model.ens_w,
        "points": points,
        "teams": {t: [round(float(v), 4) for v in comp.loc[t]] for t in comp.index},
    }
    (VIZ / f"model{suffix}.json").write_text(json.dumps(export, indent=1))

    # Schedule (lens-independent) powers the client-side playoff re-simulation.
    export_schedule()

    print(f"exported {len(ratings)} rated teams, {len(export['teams'])} sim teams")
    print(f"-> {VIZ / f'ratings{suffix}.json'}\n-> {VIZ / f'model{suffix}.json'}")


def export_schedule():
    """viz/data/schedule.json: compact {h, a, n} records for every 2026 game."""
    raw = json.load(open(ROOT / "data" / "raw" / "schedule_2026.json"))
    games = [{"h": g["homeTeam"], "a": g["awayTeam"],
              "n": 1 if g.get("neutralSite") else 0} for g in raw]
    (VIZ / "schedule.json").write_text(json.dumps(games))
    print(f"-> {VIZ / 'schedule.json'} ({len(games)} games)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
