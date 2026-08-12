"""Independent empirical audit harness for the team-prediction pipeline.

The harness deliberately evaluates seasons in chronological order.  It does not
modify the shipped model; it writes compact JSON/CSV evidence under audit/output.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "source-data"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PFF_DIR", str(SOURCE / "pff_exports"))
os.environ.setdefault("CFB_GAMES_CSV", str(SOURCE / "CFB_Data" / "data" / "games.csv"))
os.environ.setdefault("CFB_TWODEEP_2026", str(SOURCE / "fbs_2026_two_deep_pfsn_full_position_weights.xlsx"))
os.environ.setdefault("CFB_PFSN_MASTER", str(SOURCE / "cfb_pfsn_all_raw_numbers_master.xlsx"))
os.environ.setdefault("CFB_LOGO_DIR", str(SOURCE / "cfb_logos"))

from config import (GAME_YEARS, OPP_ADJ_ALPHA, UNCERTAINTY_LAMBDA,
                    TALENT_BLEND, WAR_BLEND)  # noqa: E402
from scripts.train import blended_talent, load_bundle, raw_returning  # noqa: E402
from src import matchup as MU, model as M, oppadj as OA  # noqa: E402
from src.data import pff  # noqa: E402


OUT = ROOT / "audit" / "output"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-6


def metrics(y, p, margins=None, pred_margin=None):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    out = {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5) == y)),
    }
    if len(np.unique(y)) == 2:
        z = np.log(p / (1 - p)).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
        out["calibration_intercept"] = float(cal.intercept_[0])
        out["calibration_slope"] = float(cal.coef_[0, 0])
    if margins is not None and pred_margin is not None:
        margins, pred_margin = np.asarray(margins), np.asarray(pred_margin)
        out.update({
            "margin_mae": float(mean_absolute_error(margins, pred_margin)),
            "margin_rmse": float(mean_squared_error(margins, pred_margin) ** 0.5),
            "margin_r2": float(r2_score(margins, pred_margin)),
            "margin_r": float(np.corrcoef(margins, pred_margin)[0, 1]),
            "margin_bias": float(np.mean(pred_margin - margins)),
        })
    return out


def predict_model(model, part):
    X, y, hf, mg = part
    p = np.array([model.win_prob(x, h) for x, h in zip(X, hf)])
    pm = np.array([model.pred_margin(x, h) for x, h in zip(X, hf)])
    return y, p, mg, pm


def fit_eval_parts(parts, variant, transform=lambda x: x):
    """Expanding-window tests: train only on seasons strictly before test."""
    fold_rows, pred_rows = [], []
    for test in [2022, 2023, 2024, 2025]:
        train = [y for y in GAME_YEARS if y < test and y in parts]
        if not train or test not in parts:
            continue
        Xtr = np.vstack([transform(parts[y][0]) for y in train])
        ytr = np.concatenate([parts[y][1] for y in train])
        hftr = np.concatenate([parts[y][2] for y in train])
        mgtr = np.concatenate([parts[y][3] for y in train])
        mdl, _ = M.train(Xtr, ytr, hftr, margins=mgtr)
        Xt = transform(parts[test][0])
        part = (Xt, parts[test][1], parts[test][2], parts[test][3])
        y, p, mg, pm = predict_model(mdl, part)
        met = metrics(y, p, mg, pm)
        met.update({"variant": variant, "season": test})
        fold_rows.append(met)
        for i in range(len(y)):
            pred_rows.append({"variant": variant, "season": test, "y": int(y[i]),
                              "p": p[i], "margin": mg[i], "pred_margin": pm[i],
                              "home": int(parts[test][2][i])})
    pred = pd.DataFrame(pred_rows)
    pooled = metrics(pred.y, pred.p, pred.margin, pred.pred_margin)
    pooled["variant"] = variant
    return pooled, fold_rows, pred


def assemble_variant(std, talent, ret, games, pyth, ret_raw, od, lam, years=GAME_YEARS):
    # Slopes are re-fit inside each expanding fold below, so return fold-specific parts.
    by_test = {}
    for test in [2022, 2023, 2024, 2025]:
        train = [y for y in years if y < test]
        bo, bd = MU.fit_talent_od_slopes(train, std, talent, od_by_year=od)
        parts = MU.assemble(years, std, pyth, talent, ret, games, lam=lam,
                            b_o=bo, b_d=bd, ret_raw_by_year=ret_raw,
                            od_by_year=od)
        by_test[test] = parts
    return by_test


def fit_eval_fold_specific(by_test, variant, transform=lambda x: x):
    fold_rows, all_pred = [], []
    for test, parts in by_test.items():
        train = [y for y in GAME_YEARS if y < test and y in parts]
        Xtr = np.vstack([transform(parts[y][0]) for y in train])
        ytr = np.concatenate([parts[y][1] for y in train])
        hftr = np.concatenate([parts[y][2] for y in train])
        mgtr = np.concatenate([parts[y][3] for y in train])
        mdl, _ = M.train(Xtr, ytr, hftr, margins=mgtr)
        Xt = transform(parts[test][0])
        y, p, mg, pm = predict_model(mdl, (Xt, parts[test][1], parts[test][2], parts[test][3]))
        met = metrics(y, p, mg, pm)
        met.update({"variant": variant, "season": test})
        fold_rows.append(met)
        all_pred.append(pd.DataFrame({"variant": variant, "season": test, "y": y,
                                      "p": p, "margin": mg, "pred_margin": pm,
                                      "home": parts[test][2]}))
    pred = pd.concat(all_pred, ignore_index=True)
    pooled = metrics(pred.y, pred.p, pred.margin, pred.pred_margin)
    pooled["variant"] = variant
    return pooled, fold_rows, pred


def home_baseline(games):
    rows = []
    for test in [2022, 2023, 2024, 2025]:
        tr = pd.concat([games[y] for y in GAME_YEARS if y < test], ignore_index=True)
        tr = tr[tr.home_points != tr.away_points]
        ytr = (tr.home_points > tr.away_points).astype(int).to_numpy()
        hftr = (~tr.neutral_site.astype(bool)).astype(int).to_numpy().reshape(-1, 1)
        clf = LogisticRegression(C=1e6).fit(hftr, ytr)
        te = games[test]
        te = te[te.home_points != te.away_points]
        y = (te.home_points > te.away_points).astype(int).to_numpy()
        hf = (~te.neutral_site.astype(bool)).astype(int).to_numpy().reshape(-1, 1)
        p = clf.predict_proba(hf)[:, 1]
        rows.append(pd.DataFrame({"variant": "home_only_all_fbs", "season": test,
                                  "y": y, "p": p}))
    d = pd.concat(rows, ignore_index=True)
    out = metrics(d.y, d.p)
    out["variant"] = "home_only_all_fbs"
    return out


def home_baseline_aligned(by_test):
    rows = []
    for test, parts in by_test.items():
        train = [y for y in GAME_YEARS if y < test and y in parts]
        ytr = np.concatenate([parts[y][1] for y in train])
        hftr = np.concatenate([parts[y][2] for y in train]).reshape(-1, 1)
        clf = LogisticRegression(C=1e6).fit(hftr, ytr)
        y, hf = parts[test][1], parts[test][2]
        rows.append(pd.DataFrame({"variant": "home_only_aligned", "season": test,
                                  "y": y, "p": clf.predict_proba(hf.reshape(-1, 1))[:, 1]}))
    d = pd.concat(rows, ignore_index=True)
    out = metrics(d.y, d.p); out["variant"] = "home_only_aligned"
    return out


def cfbd_elo_baseline(pred_template):
    rows = []
    for year in [2022, 2023, 2024, 2025]:
        raw = json.loads((ROOT / "data" / "raw" / f"games_{year}.json").read_text())
        lookup = {(g.get("week"), g.get("homeTeam"), g.get("awayTeam")):
                  (g.get("homePregameElo"), g.get("awayPregameElo"), g.get("neutralSite", False))
                  for g in raw if g.get("completed")}
        # The model template has the filtered game order; reconstruct that order.
        # team_frame exclusions are reproduced by joining on y/margin in sequence below.
        # We instead use all raw FBS-v-FBS games with both Elo ratings, a slightly broader benchmark.
        for g in raw:
            if not g.get("completed") or g.get("homePoints") is None:
                continue
            if g.get("homeClassification") != "fbs" or g.get("awayClassification") != "fbs":
                continue
            rh, ra = g.get("homePregameElo"), g.get("awayPregameElo")
            if rh is None or ra is None or g["homePoints"] == g["awayPoints"]:
                continue
            hfa = 0.0 if g.get("neutralSite", False) else 65.0
            p = 1 / (1 + 10 ** (-((rh - ra + hfa) / 400)))
            rows.append({"variant": "cfbd_pregame_elo", "season": year,
                         "y": int(g["homePoints"] > g["awayPoints"]), "p": p})
    d = pd.DataFrame(rows)
    out = metrics(d.y, d.p)
    out["variant"] = "cfbd_pregame_elo"
    return out, d


def frame_for_year(N, train, std, talent, ret, games, pyth, ret_raw, od):
    bo, bd = MU.fit_talent_od_slopes(train, std, talent, od_by_year=od)
    u = MU.uncertainty_u(ret_raw[N])
    return MU.team_frame(N, std, pyth, talent, ret,
                         uncertainty=(UNCERTAINTY_LAMBDA, bo, bd, u), od_by_year=od)


def aligned_elo_and_pythag(std, talent, ret, games, pyth, ret_raw, od):
    elo_rows, py_parts = [], {}
    for test in [2022, 2023, 2024, 2025]:
        train = [y for y in GAME_YEARS if y < test]
        all_years = train + [test]
        fold_parts = {}
        for N in all_years:
            frame = frame_for_year(N, train, std, talent, ret, games, pyth, ret_raw, od)
            teams = set(frame.index)
            X, y, hf, mg = [], [], [], []
            raw = json.loads((ROOT / "data" / "raw" / f"games_{N}.json").read_text())
            elo_lookup = {(g.get("week"), g.get("homeTeam"), g.get("awayTeam")):
                          (g.get("homePregameElo"), g.get("awayPregameElo")) for g in raw}
            for _, g in games[N].iterrows():
                h, a = g.home_team, g.away_team
                if h not in teams or a not in teams or g.home_points == g.away_points:
                    continue
                X.append([pyth[N - 1].get(h, 0) - pyth[N - 1].get(a, 0)])
                y.append(int(g.home_points > g.away_points))
                neutral = bool(g.neutral_site)
                hf.append(0 if neutral else 1)
                mg.append(float(g.home_points - g.away_points))
                if N == test:
                    rh, ra = elo_lookup.get((g.week, h, a), (None, None))
                    if rh is not None and ra is not None:
                        hfa = 0.0 if neutral else 65.0
                        pe = 1 / (1 + 10 ** (-((rh - ra + hfa) / 400)))
                        elo_rows.append({"variant": "cfbd_pregame_elo_aligned", "season": N,
                                         "week": g.week, "y": y[-1], "p": pe})
            fold_parts[N] = (np.asarray(X), np.asarray(y), np.asarray(hf), np.asarray(mg))
        py_parts[test] = fold_parts
    py, pyfold, pypred = fit_eval_fold_specific(py_parts, "prior_pythag_only")
    ed = pd.DataFrame(elo_rows)
    elo = metrics(ed.y, ed.p); elo["variant"] = "cfbd_pregame_elo_aligned"
    elo["week_1_4_brier"] = float(np.mean((ed.loc[ed.week <= 4, "p"] - ed.loc[ed.week <= 4, "y"]) ** 2))
    elo["week_5_plus_brier"] = float(np.mean((ed.loc[ed.week >= 5, "p"] - ed.loc[ed.week >= 5, "y"]) ** 2))
    return elo, py, pyfold, pypred


def war_talent_without_qb(index_by_year):
    w = pd.read_csv(ROOT / "war_model" / "hybrid_player_war.csv")
    prior = w.groupby(["season", "player_id"], as_index=False).war.sum().rename(columns={"war": "prior_war"})
    prior["season"] += 1
    roster = w.loc[w.position != "QB", ["season", "player_id", "team"]].drop_duplicates()
    j = roster.merge(prior, on=["season", "player_id"], how="left").fillna({"prior_war": 0.0})
    raw = {int(y): g.groupby("team").prior_war.sum() for y, g in j.groupby("season")}
    out = {}
    for y, idx in index_by_year.items():
        s = raw[y].reindex(idx)
        out[y] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    return out


def vif_and_condition(X, names):
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    keep = sd > 1e-10
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    corr = np.corrcoef(Z, rowvar=False)
    vifs = {}
    for j, name in enumerate(names):
        oth = np.delete(Z, j, axis=1)
        r2 = r2_score(Z[:, j], np.column_stack([np.ones(len(Z)), oth]) @
                      np.linalg.lstsq(np.column_stack([np.ones(len(Z)), oth]), Z[:, j], rcond=None)[0])
        vifs[name] = float(1 / max(1 - r2, 1e-12))
    return {"names": names, "correlation": corr.tolist(), "vif": vifs,
            "condition_number": float(np.linalg.cond(Z))}


def antisymmetry(model, frame):
    teams = list(frame.index)
    errs_p, errs_m = [], []
    for i in range(min(len(teams), 60)):
        for j in range(i + 1, min(len(teams), 60)):
            a, b = teams[i], teams[j]
            xa, xb = MU.matchup_vector(frame, a, b), MU.matchup_vector(frame, b, a)
            errs_p.append(model.win_prob(xa, 0) + model.win_prob(xb, 0) - 1)
            errs_m.append(model.pred_margin(xa, 0) + model.pred_margin(xb, 0))
    return {"prob_mean_abs": float(np.mean(np.abs(errs_p))),
            "prob_max_abs": float(np.max(np.abs(errs_p))),
            "margin_mean_abs_sum": float(np.mean(np.abs(errs_m))),
            "margin_max_abs_sum": float(np.max(np.abs(errs_m)))}


def detailed_oos(configured_by_test, std, talent, ret, games, pyth, ret_raw, od):
    rows = []
    for test, parts in configured_by_test.items():
        train = [y for y in GAME_YEARS if y < test]
        Xtr = np.vstack([parts[y][0] for y in train])
        ytr = np.concatenate([parts[y][1] for y in train])
        hftr = np.concatenate([parts[y][2] for y in train])
        mgtr = np.concatenate([parts[y][3] for y in train])
        mdl, _ = M.train(Xtr, ytr, hftr, margins=mgtr)
        frame = frame_for_year(test, train, std, talent, ret, games, pyth, ret_raw, od)
        raw = json.loads((ROOT / "data" / "raw" / f"games_{test}.json").read_text())
        meta = {(g.get("week"), g.get("homeTeam"), g.get("awayTeam")): g for g in raw}
        teams = set(frame.index)
        for _, g in games[test].iterrows():
            h, a = g.home_team, g.away_team
            if h not in teams or a not in teams or g.home_points == g.away_points:
                continue
            neutral = bool(g.neutral_site); hf = 0 if neutral else 1
            x = MU.matchup_vector(frame, h, a)
            p = mdl.win_prob(x, hf); pm = mdl.pred_margin(x, hf)
            m = meta.get((g.week, h, a), {})
            rh, ra = m.get("homePregameElo"), m.get("awayPregameElo")
            pe = None if rh is None or ra is None else 1 / (1 + 10 ** (-((rh - ra + (0 if neutral else 65)) / 400)))
            rows.append({"season": test, "week": g.week, "home_team": h, "away_team": a,
                         "home_conference": m.get("homeConference"), "away_conference": m.get("awayConference"),
                         "neutral": neutral, "y": int(g.home_points > g.away_points), "p": p,
                         "p_elo": pe, "actual_margin": float(g.home_points - g.away_points),
                         "pred_margin": pm, "home_O": frame.loc[h, "O"], "away_O": frame.loc[a, "O"],
                         "home_D": frame.loc[h, "D"], "away_D": frame.loc[a, "D"],
                         "home_talent": frame.loc[h].get("talent_raw", np.nan),
                         "away_talent": frame.loc[a].get("talent_raw", np.nan),
                         "home_returning": frame.loc[h, "returning"], "away_returning": frame.loc[a, "returning"]})
    d = pd.DataFrame(rows)
    d["prob_resid"] = d.y - d.p
    d["margin_resid"] = d.actual_margin - d.pred_margin
    d["favorite"] = np.where(d.p >= .5, "home_favorite", "away_favorite")
    d["spread_bucket"] = pd.cut(d.pred_margin.abs(), [-.01, 3, 7, 14, 21, np.inf],
                                 labels=["0-3", "3-7", "7-14", "14-21", "21+"])
    d["week_bucket"] = pd.cut(d.week, [0, 4, 9, 99], labels=["wk1-4", "wk5-9", "wk10+"])
    d.to_csv(OUT / "detailed_predictions.csv", index=False)

    long = pd.concat([
        d[["season", "home_team", "home_conference", "prob_resid", "margin_resid"]].rename(
            columns={"home_team": "team", "home_conference": "conference"}),
        d[["season", "away_team", "away_conference", "prob_resid", "margin_resid"]].rename(
            columns={"away_team": "team", "away_conference": "conference"}).assign(
                prob_resid=lambda x: -x.prob_resid, margin_resid=lambda x: -x.margin_resid),
    ], ignore_index=True)
    team = long.groupby("team").agg(n=("prob_resid", "size"), prob_bias=("prob_resid", "mean"),
                                     margin_bias=("margin_resid", "mean")).reset_index()
    team.to_csv(OUT / "team_residuals.csv", index=False)

    def grouped(cols):
        return d.groupby(cols, observed=True).agg(
            n=("y", "size"), brier=("prob_resid", lambda x: float(np.mean(x*x))),
            prob_bias=("prob_resid", "mean"), margin_bias=("margin_resid", "mean"),
            margin_rmse=("margin_resid", lambda x: float(np.sqrt(np.mean(x*x))))).reset_index().to_dict("records")

    conf = long.dropna(subset=["conference"]).groupby("conference").agg(
        n=("prob_resid", "size"), prob_bias=("prob_resid", "mean"),
        margin_bias=("margin_resid", "mean"),
        margin_rmse=("margin_resid", lambda x: float(np.sqrt(np.mean(x*x))))).reset_index()
    conf.to_csv(OUT / "conference_residuals.csv", index=False)

    # Fraction of residual variance explained by stable team identity, with a label permutation null.
    vals = long.prob_resid.to_numpy(); grand = vals.mean(); total = np.sum((vals - grand) ** 2)
    def team_r2(labels):
        z = pd.DataFrame({"team": labels, "r": vals})
        means = z.groupby("team").r.transform("mean").to_numpy()
        return float(np.sum((means - grand) ** 2) / total)
    observed = team_r2(long.team.to_numpy())
    rng = np.random.default_rng(20260812)
    null = np.array([team_r2(rng.permutation(long.team.to_numpy())) for _ in range(500)])
    return {
        "by_week": grouped(["week_bucket"]), "by_spread": grouped(["spread_bucket"]),
        "by_favorite": grouped(["favorite"]), "conference": conf.to_dict("records"),
        "team_residual_r2": observed, "team_residual_null_mean": float(null.mean()),
        "team_residual_permutation_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "worst_team_margin_bias": team.reindex(team.margin_bias.abs().sort_values(ascending=False).index).head(10).to_dict("records"),
        "model_vs_elo_same_games": {"model_brier": float(np.mean((d.p - d.y) ** 2)),
                                     "elo_brier": float(np.mean((d.p_elo - d.y) ** 2))},
    }


def main():
    std, cfbd_talent, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    group_scores = pff.build_group_scores()
    pff_roster = pff.build_roster_talent(group_scores=group_scores)
    no_qb_weights = dict(pff.PFF_OPT_WEIGHTS, QB=0.0)
    pff_no_qb = pff.build_roster_talent(weights=no_qb_weights, group_scores=group_scores)
    talent_no_war = blended_talent(cfbd_talent, pff_roster, war_w=0.0)
    talent_full = blended_talent(cfbd_talent, pff_roster)
    war_no_qb = war_talent_without_qb({y: s.index for y, s in cfbd_talent.items()})
    talent_no_qb = {}
    for y, cf in cfbd_talent.items():
        pr = pff_no_qb.get(y, cf).reindex(cf.index).fillna(cf)
        base = TALENT_BLEND * pr + (1 - TALENT_BLEND) * cf
        wz = war_no_qb.get(y, base).reindex(cf.index).fillna(base)
        talent_no_qb[y] = (1 - WAR_BLEND) * base + WAR_BLEND * wz
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)

    variants = []
    folds = []
    predictions = []

    configs = [
        ("configured", talent_full, od, UNCERTAINTY_LAMBDA),
        ("no_player_pff_or_war", cfbd_talent, od, UNCERTAINTY_LAMBDA),
        ("no_war", talent_no_war, od, UNCERTAINTY_LAMBDA),
        ("no_qb_player_signal", talent_no_qb, od, UNCERTAINTY_LAMBDA),
        ("no_talent_shrink", talent_full, od, 0.0),
        ("talent_only_od", talent_full, od, 1.0),
        ("no_opponent_adjustment", talent_full, None, UNCERTAINTY_LAMBDA),
    ]
    configured_by_test = None
    for name, talent, od_src, lam in configs:
        by_test = assemble_variant(std, talent, ret, games, pyth, ret_raw, od_src, lam)
        if name == "configured":
            configured_by_test = by_test
        pooled, fr, pr = fit_eval_fold_specific(by_test, name)
        variants.append(pooled); folds.extend(fr); predictions.append(pr)

    # Feature/architecture ablations on the configured fold-specific frames.
    transforms = {
        "team_strength_scalar": lambda X: (X[:, 0] + X[:, 1]).reshape(-1, 1),
        "off_def_edges_only": lambda X: X[:, :2],
        "returning_only": lambda X: X[:, 5:6],
    }
    for name, fn in transforms.items():
        pooled, fr, pr = fit_eval_fold_specific(configured_by_test, name, fn)
        variants.append(pooled); folds.extend(fr); predictions.append(pr)

    variants.append(home_baseline_aligned(configured_by_test))
    variants.append(home_baseline(games))
    elo, elo_pred = cfbd_elo_baseline(None)
    variants.append(elo)
    elo_a, py, pyfold, pypred = aligned_elo_and_pythag(
        std, talent_full, ret, games, pyth, ret_raw, od)
    variants.extend([elo_a, py]); folds.extend(pyfold); predictions.append(pypred)

    # Active design diagnostics on the 2025 configured frame using expanding training.
    parts25 = configured_by_test[2025]
    train = [y for y in GAME_YEARS if y < 2025]
    Xtr = np.vstack([parts25[y][0] for y in train])
    ytr = np.concatenate([parts25[y][1] for y in train])
    hftr = np.concatenate([parts25[y][2] for y in train])
    mgtr = np.concatenate([parts25[y][3] for y in train])
    final, _ = M.train(Xtr, ytr, hftr, margins=mgtr)
    diagX = np.column_stack([Xtr, hftr])
    col = vif_and_condition(diagX, MU.MATCHUP_COLS + ["home_field"])

    # Rebuild the 2025 frame for neutral-site reversal diagnostics.
    bo, bd = MU.fit_talent_od_slopes(train, std, talent_full, od_by_year=od)
    u = MU.uncertainty_u(ret_raw[2025])
    frame25 = MU.team_frame(2025, std, pyth, talent_full, ret,
                            uncertainty=(UNCERTAINTY_LAMBDA, bo, bd, u), od_by_year=od)
    anti = antisymmetry(final, frame25)

    # Residual summaries for configured expanding predictions.
    pred = pd.concat(predictions, ignore_index=True)
    cfg = pred[pred.variant == "configured"].copy()
    cfg["prob_resid"] = cfg.y - cfg.p
    cfg["margin_resid"] = cfg.margin - cfg.pred_margin
    cfg["p_bucket"] = pd.cut(cfg.p, [0, .2, .35, .5, .65, .8, 1], include_lowest=True).astype(str)
    residual = {
        "by_season": cfg.groupby("season").agg(n=("y", "size"), brier=("prob_resid", lambda x: float(np.mean(x*x))),
                                                  prob_bias=("prob_resid", "mean"), margin_bias=("margin_resid", "mean"),
                                                  margin_rmse=("margin_resid", lambda x: float(np.sqrt(np.mean(x*x))))).reset_index().to_dict("records"),
        "by_probability_bucket": cfg.groupby("p_bucket", observed=True).agg(n=("y", "size"), predicted=("p", "mean"),
                                                                               actual=("y", "mean"), margin_bias=("margin_resid", "mean")).reset_index().to_dict("records"),
        "neutral_brier": float(np.mean(cfg.loc[cfg.home == 0, "prob_resid"] ** 2)),
        "home_brier": float(np.mean(cfg.loc[cfg.home == 1, "prob_resid"] ** 2)),
    }
    residual["detail"] = detailed_oos(configured_by_test, std, talent_full, ret, games,
                                       pyth, ret_raw, od)

    pd.DataFrame(variants).to_csv(OUT / "benchmark_ablation.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "fold_metrics.csv", index=False)
    pred.to_csv(OUT / "all_variant_predictions.csv", index=False)
    cfg.to_csv(OUT / "configured_predictions.csv", index=False)
    result = {
        "benchmark_ablation": variants,
        "collinearity": col,
        "antisymmetry": anti,
        "residual": residual,
        "sample": {"configured_games_2022_2025": int(len(cfg)),
                   "all_fbs_games_2022_2025": int(sum(len(games[y]) for y in [2022, 2023, 2024, 2025]))},
    }
    (OUT / "audit_metrics.json").write_text(json.dumps(result, indent=2))
    print(pd.DataFrame(variants).to_string(index=False))
    print(json.dumps({"collinearity": col, "antisymmetry": anti, "residual": residual}, indent=2))


if __name__ == "__main__":
    main()
