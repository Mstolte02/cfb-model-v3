"""Everything needed to audit the model from the outside, as one JSON.

The model has accreted a lot of moving parts - three talent sources feeding one
feature, six features of which two are now zeroed, a separate WAR build with twenty
facets behind it, a committee model fitted on twelve seasons - and none of that is
visible from the app. This exports the diagnostics so the Method view can show what
goes in, how correlated the inputs are with each other, what the model learned, and
how well it actually does.

Run: ./venv/bin/python -m scripts.export_diagnostics
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (ROOT, ARTIFACTS, GAME_YEARS, TEST_GAME_YEAR, FEATURES as CFG_FEATURES,
                    UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA, DROPPED_FEATURES,
                    TALENT_BLEND, WAR_BLEND, ENSEMBLE_W)
from src.data import load, pff, war
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from src.model import CFBModel
from scripts.train import load_bundle, raw_returning, blended_talent
from scripts.talent_sweep import sources

VIZ = ROOT / "viz" / "data"
MODEL_FEATURES = ["O", "D", "fp_margin", "pythag", "talent", "returning"]

# What each upstream source actually supplies. Written out because "CFBD" and "PFF"
# appear all over the pipeline doing quite different jobs in each place.
SOURCES = [
    {"name": "CFBD team stats", "provider": "CollegeFootballData",
     "feeds": "O / D composites, field position",
     "detail": "Success rate, rush EPA, havoc, line yards, EPA allowed - "
               "opponent-adjusted, prior season.", "years": "2014-2025"},
    {"name": "CFBD recruiting", "provider": "CollegeFootballData",
     "feeds": "talent (part)",
     "detail": "247 composite team talent. The only signal with no dependence on "
               "how the team actually played.", "years": "2015-2026"},
    {"name": "PFF player grades", "provider": "PFF",
     "feeds": "talent (part), WAR",
     "detail": "Player-season grades. Position-weighted over this year's roster "
               "carrying last year's grades.", "years": "2014-2025"},
    {"name": "CFBD player PPA", "provider": "CollegeFootballData",
     "feeds": "WAR (part)",
     "detail": "EPA per play by player, garbage time excluded. Covers QB/RB/WR/TE "
               "only - no line, no coverage.", "years": "2014-2025"},
    {"name": "Player WAR", "provider": "derived",
     "feeds": "talent (part)",
     "detail": "PFF grades + CFBD play value -> facet weights fitted against the "
               "FOLLOWING season -> Massey ratings -> wins above replacement.",
     "years": "2014-2026"},
    {"name": "Ourlads two-deep", "provider": "Ourlads",
     "feeds": "2026 roster",
     "detail": "Live depth charts, all 136 FBS teams, used to know who is on the "
               "field in 2026.", "years": "2026"},
    {"name": "CFP committee rankings", "provider": "CollegeFootballData",
     "feeds": "playoff selection",
     "detail": "Every published committee ranking; the selection proxy is fitted "
               "on them.", "years": "2014-2025"},
    {"name": "Returning production", "provider": "CollegeFootballData",
     "feeds": "returning, uncertainty",
     "detail": "Share of last year's production back. Drives how far a team "
               "regresses toward its talent baseline.", "years": "2015-2026"},
]


def calibration(p, y, bins=10):
    """Predicted vs observed win rate, for the reliability plot."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() < 15:
            continue
        out.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                    "pred": round(float(p[m].mean()), 4),
                    "actual": round(float(y[m].mean()), 4),
                    "n": int(m.sum())})
    return out


def main():
    load.require_key()
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    b_o, b_d = MU.fit_talent_od_slopes(list(GAME_YEARS), std, talent, od_by_year=od)
    parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                        lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                        ret_raw_by_year=ret_raw, od_by_year=od)

    X = np.vstack([parts[g][0] for g in GAME_YEARS if g in parts])
    y = np.concatenate([parts[g][1] for g in GAME_YEARS if g in parts])
    out = {"generated_for": 2026, "n_games": int(len(X))}

    # ---- feature correlation + VIF -----------------------------------------
    C = pd.DataFrame(np.corrcoef(X, rowvar=False),
                     index=MODEL_FEATURES, columns=MODEL_FEATURES).fillna(0.0)
    vifs = {}
    for j, f in enumerate(MODEL_FEATURES):
        col = X[:, j]
        if col.std() < 1e-12:
            vifs[f] = None                      # a retired feature, zeroed
            continue
        Z = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Z, col, rcond=None)
        ss = ((col - col.mean()) ** 2).sum()
        r2 = 1 - ((col - Z @ beta) ** 2).sum() / ss if ss > 0 else 0.0
        vifs[f] = round(float(1 / max(1e-9, 1 - r2)), 2)
    out["features"] = {
        "names": MODEL_FEATURES,
        "active": [f for f in MODEL_FEATURES if f not in DROPPED_FEATURES],
        "dropped": list(DROPPED_FEATURES),
        "corr": {a: {b: round(float(C.loc[a, b]), 3) for b in MODEL_FEATURES}
                 for a in MODEL_FEATURES},
        "vif": vifs,
    }

    # ---- talent-source correlation -----------------------------------------
    src, _ = sources()
    names = ["PFF", "CFBD", "WAR"]
    # 2021 has no prior-year PFF grades, so PFF falls back to CFBD entirely and the
    # pair correlates at exactly 1.0; including it would overstate the overlap.
    yrs = [yv for yv in sorted(src["PFF"])
           if abs(src["PFF"][yv].corr(src["CFBD"][yv]) - 1.0) > 1e-9]
    st = {n: pd.concat([src[n][yv].rename(yv) for yv in yrs], keys=yrs) for n in names}
    T = pd.DataFrame(st).dropna()
    TC = T.corr()
    out["talent_sources"] = {
        "names": names,
        "seasons_used": [int(v) for v in yrs],
        "n": int(len(T)),
        "corr": {a: {b: round(float(TC.loc[a, b]), 3) for b in names} for a in names},
        "by_season": [
            {"season": int(yv),
             "PFF_WAR": round(float(src["PFF"][yv].corr(src["WAR"][yv])), 3),
             "PFF_CFBD": round(float(src["PFF"][yv].corr(src["CFBD"][yv])), 3),
             "WAR_CFBD": round(float(src["WAR"][yv].corr(src["CFBD"][yv])), 3)}
            for yv in sorted(src["PFF"])],
        "weights": {"PFF": round((1 - WAR_BLEND) * TALENT_BLEND, 3),
                    "CFBD": round((1 - WAR_BLEND) * (1 - TALENT_BLEND), 3),
                    "WAR": WAR_BLEND},
    }

    # ---- what the model learned --------------------------------------------
    mdl = CFBModel.load()
    out["coefficients"] = {
        "logistic": {f: round(float(c), 4) for f, c in zip(MODEL_FEATURES, mdl.coef)},
        "logistic_hfa": round(float(mdl.hfa_coef), 4),
        "margin": {f: round(float(c), 4) for f, c in zip(MODEL_FEATURES, mdl.margin_coef)},
        "margin_hfa": round(float(mdl.margin_hfa), 4),
        "margin_sigma": round(float(mdl.margin_sigma), 3),
        "ensemble_w": ENSEMBLE_W,
    }

    # ---- evaluation: LOSO, per-season, baselines, calibration ---------------
    per_season, preds, actual = [], [], []
    for ty in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != ty and g in parts]
        if ty not in parts or not tr:
            continue
        Xtr = np.vstack([parts[g][0] for g in tr])
        ytr = np.concatenate([parts[g][1] for g in tr])
        hf = np.concatenate([parts[g][2] for g in tr])
        m_, _ = M.train(Xtr, ytr, hf)
        ev = M.evaluate(m_, *parts[ty])
        per_season.append({"season": int(ty), "n": int(len(parts[ty][1])),
                           "brier": round(ev["brier"], 4),
                           "log_loss": round(ev["log_loss"], 4),
                           "accuracy": round(ev["accuracy"], 4)})
        Xt, yt, ht = parts[ty][0], parts[ty][1], parts[ty][2]
        preds.append(np.array([m_.win_prob(Xt[i], ht[i]) for i in range(len(yt))]))
        actual.append(yt)
    p = np.concatenate(preds); a = np.concatenate(actual)
    # Two reference points a reader can hold onto: guessing every game at the base
    # home-win rate, and a coin flip.
    base = float(a.mean())
    out["evaluation"] = {
        "loso": {k: round(float(np.mean([s[k] for s in per_season])), 4)
                 for k in ("brier", "log_loss", "accuracy")},
        "per_season": per_season,
        "baselines": {
            "home_team_always": round(float(np.mean((base - a) ** 2)), 4),
            "coin_flip": round(float(np.mean((0.5 - a) ** 2)), 4),
        },
        "calibration": calibration(p, a),
        "home_win_rate": round(base, 4),
    }

    # ---- the WAR build behind the talent signal ----------------------------
    wpath = Path.home() / "Downloads" / "rb-win-model" / "hybrid_facet_weights.csv"
    if wpath.exists():
        w = pd.read_csv(wpath, index_col=0)["rf"].sort_values(ascending=False)
        out["war_facets"] = [{"facet": str(i), "weight": round(float(v), 4),
                              "source": "CFBD" if str(i).startswith("cfbd_") else "PFF"}
                             for i, v in w.items()]
        cf = sum(v for i, v in w.items() if str(i).startswith("cfbd_"))
        out["war_source_split"] = {"CFBD": round(float(cf), 3),
                                   "PFF": round(float(1 - cf), 3)}

    # ---- decisions taken, with the numbers that drove them -----------------
    sweep = ARTIFACTS / "talent_sweep.json"
    grid = ARTIFACTS / "talent_sweep.csv"
    if sweep.exists() and grid.exists():
        g = pd.read_csv(grid)
        out["talent_sweep"] = json.loads(sweep.read_text())
        out["talent_sweep"]["grid"] = json.loads(
            g.round(4).to_json(orient="records"))
        # the comparisons that answer "should WAR just replace PFF?"
        def best(mask, label):
            d = g[mask].sort_values("brier")
            if not len(d):
                return None
            r0 = d.iloc[0]
            return {"label": label, "pff": float(r0.pff), "cfbd": float(r0.cfbd),
                    "war": float(r0.war), "brier": float(r0.brier),
                    "accuracy": float(r0.accuracy)}
        out["talent_sweep"]["contrasts"] = [c for c in [
            best(g.index == g.brier.idxmin(), "all three (best)"),
            best(g.war == 0, "PFF + CFBD, no WAR"),
            best(g.pff == 0, "WAR + CFBD, WAR replaces PFF"),
            best(g.cfbd == 0, "PFF + WAR, no recruiting"),
            best(g.pff == 1, "PFF alone"),
            best(g.war == 1, "WAR alone"),
            best(g.cfbd == 1, "recruiting alone"),
        ] if c]
    cm = ARTIFACTS / "committee_model.json"
    if cm.exists():
        out["committee"] = json.loads(cm.read_text())

    out["sources"] = SOURCES
    out["decisions"] = [
        {"question": "Can CFBD replace PFF grades in the WAR build?",
         "answer": "No - CFBD has no line data and defense is counting stats only.",
         "evidence": "Facets -> adj win pct: PFF 0.826, CFBD 0.741, both 0.845."},
        {"question": "Should WAR replace PFF as the talent signal?",
         "answer": "No. WAR and PFF are complements, and PFF is the stronger of the "
                   "two. Swapping PFF out for WAR is the single worst two-source "
                   "combination.",
         "evidence": "Best blend without PFF 0.2062, without WAR 0.2054, with all "
                     "three 0.2047. They correlate at only 0.69."},
        {"question": "Are the blend weights right, or just assembled?",
         "answer": "Right - the shipping 38/38/25 split is the joint optimum of all "
                   "45 three-way blends, though the surface is flat.",
         "evidence": "Grid search range 0.2047-0.2088; 7 of 45 blends sit within "
                     "0.0005 of the best."},
        {"question": "Is WAR predictive enough to stand alone?",
         "answer": "No - well behind the O/D composites on its own.",
         "evidence": "As the single feature: WAR 0.2288 vs O/D-only 0.2116."},
        {"question": "Are talent / returning / WAR the same thing?",
         "answer": "No - returning is nearly orthogonal; talent is the most valuable "
                   "single feature.",
         "evidence": "max |r| with anything = 0.21 for returning; dropping talent "
                     "costs +0.0040 Brier."},
        {"question": "Why were corners worth more than edge rushers?",
         "answer": "They were not - the facet weights were fitted against the same "
                   "season's wins, which rewards facets contaminated by the result. "
                   "Refitted against the FOLLOWING season by ridge.",
         "evidence": "Coverage grade: same-season r .55, year-over-year stability "
                     ".21. Pass rush: .38 and .57. After the fix, EDGE and CB are "
                     "both 10.4% of league WAR (was 6.1% vs 15.8%)."},
        {"question": "Is the projection just a copy of last season?",
         "answer": "No. It is about as persistent as the sport itself is.",
         "evidence": "Reality's season-to-season r = .502; the 2026 power rating "
                     "against 2025 is .527. Teams move 29.9 rank places on average, "
                     "74 of 136 by 20 or more."},
        {"question": "Should the three talent sources be separate features?",
         "answer": "Tested; blending is marginally better and simpler. They are too "
                   "collinear for three coefficients to be stable.",
         "evidence": "Blended .2039, separate columns .2046. Left free, the model "
                     "wants PFF 28 / recruiting 36 / WAR 36 - more WAR than the "
                     "blend gives it."},
        {"question": "Is pythag redundant with O and D?",
         "answer": "Yes - retired.",
         "evidence": "r = +0.61 with O, +0.63 with D, highest VIF (3.0); dropping it "
                     "raised accuracy 0.676 -> 0.680."},
        {"question": "Does field-position margin earn its place?",
         "answer": "No - retired.",
         "evidence": "Dropping it improved Brier 0.2054 -> 0.2045."},
    ]

    VIZ.mkdir(parents=True, exist_ok=True)
    (VIZ / "diagnostics.json").write_text(json.dumps(out, indent=1, allow_nan=False))
    print(f"-> {VIZ / 'diagnostics.json'}")
    print(f"   LOSO {out['evaluation']['loso']}")
    print(f"   talent-source corr PFF~WAR "
          f"{out['talent_sources']['corr']['PFF']['WAR']:.3f} "
          f"(seasons {out['talent_sources']['seasons_used']})")


if __name__ == "__main__":
    main()
