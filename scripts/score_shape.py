"""Football scores are not smooth, and this measures what pretending otherwise costs.

The model predicts a continuous margin and a continuous total, then the app rounds them
to integers. That treats every margin as equally reachable, and they are not: points
arrive in 7s, 3s, 6s and 8s, so 3- and 7-point margins are enormously more common than
4- and 5-point ones, and the gap does not shrink with sample size. Two separate things
follow from that, and they are fitted and judged separately here because only one of
them turns out to be worth shipping.

MARGIN PMF - the distribution of the actual margin given the predicted one, estimated
non-parametrically: weight every historical game by a Gaussian kernel on how close its
own predicted margin was to the one being asked about, and read off the distribution of
what actually happened. Non-parametric rather than "normal plus spikes" because the key
numbers sit at FIXED margins - 3 is common whatever the prediction - so a residual
distribution shifted by the prediction gets them in the wrong place. Conditioning on the
prediction and letting the data place the spikes gets them right by construction.

SCORE PAIRS - which actual scoreline to display. The projected total and margin imply a
scoreline, but rounding each side independently produces 28-24 for a game that would far
more often finish 27-24. So real final scores are counted, and the displayed pair is the
most common real scoreline near the predicted total and margin.

Judged by leave-one-season-out over 2021-25, refitting everything inside each fold:
win probability by Brier and log-loss, the key numbers by predicted-against-actual
frequency, and the displayed score by how often it lands exactly right.

Run: ./venv/bin/python -m scripts.score_shape
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (ARTIFACTS, GAME_YEARS, OPP_ADJ_ALPHA, ROOT, TEST_GAME_YEAR,
                    UNCERTAINTY_LAMBDA)
from src import matchup as MU
from src import model as M
from src import oppadj as OA
from src import spread as SP
from src.data import load, pff

# Kernel bandwidth on the predicted margin, in points. 3 is a compromise: narrow enough
# that a 21-point favourite and a pick'em are not pooled, wide enough that each estimate
# still draws on ~1500 games. Swept below and reported.
H_MARGIN = 3.0

# Margins beyond this are lumped into the tails; nothing about a 60-point win needs
# resolving to the point and the cells out there are too thin to estimate.
MARGIN_CAP = 56


def build_rows():
    """Per season: matchup features, home flag, outcome, margin, and both scores."""
    from scripts.train import raw_returning, blended_talent, load_bundle
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    b_o, b_d = MU.fit_talent_od_slopes(
        [g for g in GAME_YEARS if g != TEST_GAME_YEAR], std, talent, od_by_year=od)

    out = {}
    for N in GAME_YEARS:
        u = MU.uncertainty_u(ret_raw[N]) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        frame = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc, od_by_year=od)
        if frame is None:
            continue
        teams = set(frame.index)
        X, hf, y, mg, hp, ap, po = [], [], [], [], [], [], []
        for _, g in games[N].iterrows():
            h, a = g["home_team"], g["away_team"]
            if h not in teams or a not in teams:
                continue
            if pd.isna(g["home_points"]) or g["home_points"] == g["away_points"]:
                continue
            fh, fa = frame.loc[h], frame.loc[a]
            X.append([fh.O - fa.D, fh.D - fa.O, fh.fp_margin - fa.fp_margin,
                      fh.pythag - fa.pythag, fh.talent - fa.talent,
                      fh.returning - fa.returning])
            hf.append(0 if g.get("neutral_site", False) else 1)
            y.append(1 if g["home_points"] > g["away_points"] else 0)
            mg.append(float(g["home_points"] - g["away_points"]))
            hp.append(float(g["home_points"])); ap.append(float(g["away_points"]))
            po.append([fh.O, fa.D, fa.O, fh.D])
        out[N] = dict(X=np.array(X, float), hf=np.array(hf), y=np.array(y),
                      margin=np.array(mg), hp=np.array(hp), ap=np.array(ap),
                      pts=np.array(po, float), frame=frame)
        print(f"  {N}: {len(mg)} games")
    return out


# ---------------------------------------------------------------------------------
# the margin PMF
# ---------------------------------------------------------------------------------
class MarginPMF:
    """P(actual margin = m | predicted margin), by kernel regression over past games.

    Fitted on nothing but pairs of (what we predicted, what happened), so it inherits
    whatever bias the margin model has AND corrects for it: if the model runs three
    points hot at the top of the range, the conditional distribution learned there is
    centred three points low, and the win probabilities come out right anyway.
    """

    def __init__(self, pred, actual, h=H_MARGIN, cap=MARGIN_CAP):
        self.h, self.cap = h, cap
        self.pred = np.asarray(pred, float)
        self.actual = np.clip(np.rint(actual).astype(int), -cap, cap)
        self.grid = np.arange(-cap, cap + 1)
        # one-hot of the realized margin, so a weighted mean over games IS the PMF
        self.onehot = np.zeros((len(self.actual), len(self.grid)))
        self.onehot[np.arange(len(self.actual)), self.actual + cap] = 1.0

    def pmf(self, m):
        """PMF rows for an array of predicted margins."""
        m = np.atleast_1d(np.asarray(m, float))
        w = np.exp(-0.5 * ((m[:, None] - self.pred[None, :]) / self.h) ** 2)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        return w @ self.onehot

    def win_prob(self, m):
        """P(margin > 0). Ties do not exist in college football, so this is the
        complement of the losing mass and needs no half-credit at zero."""
        p = self.pmf(m)
        return p[:, self.grid > 0].sum(1)

    def key_number(self, m, k):
        """P(|margin| = k), the quantity a normal model gets structurally wrong."""
        p = self.pmf(m)
        return p[:, np.abs(self.grid) == k].sum(1)


# ---------------------------------------------------------------------------------
# the score-pair table
# ---------------------------------------------------------------------------------
class ScoreTable:
    """The most common real scoreline near a given total and margin.

    Conditioned on ACTUAL totals and margins rather than predicted ones, which is the
    right object for display: the question is "what does a 51-point game decided by 3
    usually look like", and the answer is 27-24. Weighting by how often each scoreline
    genuinely happens is what stops it from answering 26-23.
    """

    def __init__(self, hp, ap, h_total=2.5, h_margin=2.0):
        pairs, counts = np.unique(np.stack([hp, ap], 1).astype(int), axis=0,
                                  return_counts=True)
        self.pairs, self.counts = pairs, counts.astype(float)
        self.tot = pairs.sum(1).astype(float)
        self.mar = (pairs[:, 0] - pairs[:, 1]).astype(float)
        self.h_total, self.h_margin = h_total, h_margin

    def pick(self, total, margin):
        """(home, away) - the likeliest real scoreline near this total and margin."""
        w = (self.counts
             * np.exp(-0.5 * ((self.tot - total) / self.h_total) ** 2)
             * np.exp(-0.5 * ((self.mar - margin) / self.h_margin) ** 2))
        return tuple(int(v) for v in self.pairs[int(np.argmax(w))])

    def export(self, max_pairs=1200):
        """The table the browser needs, trimmed to the scorelines that ever happen."""
        keep = np.argsort(-self.counts)[:max_pairs]
        return {"h_total": self.h_total, "h_margin": self.h_margin,
                "pairs": [[int(self.pairs[i, 0]), int(self.pairs[i, 1]),
                           int(self.counts[i])] for i in keep]}


def _norm_mass(k, mu, sigma):
    """P(margin rounds to exactly k) under the normal the win model uses.

    The comparison has to be like for like: the empirical PMF puts mass on integers,
    so the normal's claim about an integer is the density integrated over [k-.5, k+.5],
    not the density at k.
    """
    return norm.cdf((k + 0.5 - mu) / sigma) - norm.cdf((k - 0.5 - mu) / sigma)


def round_pair(total, margin):
    """What the app does today: round each side of the implied score independently."""
    a, b = (total + margin) / 2, (total - margin) / 2
    a, b = max(0, round(a)), max(0, round(b))
    if margin > 0 and a <= b:
        a = b + 1
    elif margin < 0 and b <= a:
        b = a + 1
    return int(a), int(b)


# ---------------------------------------------------------------------------------
def main():
    load.require_key()
    print("Building per-season game rows ...")
    data = build_rows()
    years = sorted(data)

    print("\nLeave-one-season-out: refit the win model, the points model, the PMF and "
          "the score table on the other seasons, then score the held-out one.")
    rows = []
    for N in years:
        tr = [z for z in years if z != N]
        Xtr = np.vstack([data[z]["X"] for z in tr])
        ytr = np.concatenate([data[z]["y"] for z in tr])
        hftr = np.concatenate([data[z]["hf"] for z in tr])
        mgtr = np.concatenate([data[z]["margin"] for z in tr])
        model, _ = M.train(Xtr, ytr, hftr, feature_names=MU.MATCHUP_COLS, margins=mgtr)

        # points model, on the same training seasons
        Xp = np.vstack([np.column_stack([data[z]["pts"][:, 0], data[z]["pts"][:, 1],
                                         data[z]["hf"]]) for z in tr])
        Xp2 = np.vstack([np.column_stack([data[z]["pts"][:, 2], data[z]["pts"][:, 3],
                                          np.zeros(len(data[z]["hf"]))]) for z in tr])
        yp = np.concatenate([data[z]["hp"] for z in tr])
        yp2 = np.concatenate([data[z]["ap"] for z in tr])
        pm, _alpha = SP.fit(np.vstack([Xp, Xp2]), np.concatenate([yp, yp2]))

        def pred_margin(d):
            return (model.margin_intercept + d["X"] @ model.margin_coef
                    + model.margin_hfa * d["hf"])

        def pred_total(d):
            h = pm.predict(np.column_stack([d["pts"][:, 0], d["pts"][:, 1], d["hf"]]))
            a = pm.predict(np.column_stack([d["pts"][:, 2], d["pts"][:, 3],
                                            np.zeros(len(d["hf"]))]))
            return h + a

        mtr = np.concatenate([pred_margin(data[z]) for z in tr])
        pmf = MarginPMF(mtr, mgtr)
        tab = ScoreTable(np.concatenate([data[z]["hp"] for z in tr]),
                         np.concatenate([data[z]["ap"] for z in tr]))

        te = data[N]
        m_te = pred_margin(te)
        t_te = pred_total(te)
        z = (model.intercept + te["X"] @ model.coef + model.hfa_coef * te["hf"])
        p_log = 1 / (1 + np.exp(-z))
        p_norm = norm.cdf(m_te / model.margin_sigma)
        p_pmf = pmf.win_prob(m_te)
        w = model.ens_w
        rows.append({
            "year": N, "n": len(te["y"]), "y": te["y"],
            "A_current": w * p_log + (1 - w) * p_norm,
            "B_pmf_in_ensemble": w * p_log + (1 - w) * p_pmf,
            "C_pmf_alone": p_pmf,
            "margin": te["margin"], "pred_margin": m_te, "pred_total": t_te,
            "pmf": pmf, "tab": tab, "hp": te["hp"], "ap": te["ap"],
            "sigma": model.margin_sigma,
        })

    # ---- win probability ---------------------------------------------------------
    print(f"\n{'':16}{'Brier':>10}{'log-loss':>11}{'accuracy':>11}")
    arms = ["A_current", "B_pmf_in_ensemble", "C_pmf_alone"]
    label = {"A_current": "current", "B_pmf_in_ensemble": "PMF in ensemble",
             "C_pmf_alone": "PMF alone"}
    per_year = {a: [] for a in arms}
    for a in arms:
        bs, ls, ac = [], [], []
        for r in rows:
            p = np.clip(r[a], 1e-6, 1 - 1e-6)
            y = r["y"]
            bs.append(np.mean((p - y) ** 2))
            ls.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            ac.append(np.mean((p > 0.5) == y))
            per_year[a].append(np.mean((p - y) ** 2))
        print(f"  {label[a]:<14}{np.mean(bs):>10.4f}{np.mean(ls):>11.4f}"
              f"{np.mean(ac):>11.4f}")
    base = np.array(per_year["A_current"])
    for a in arms[1:]:
        d = np.array(per_year[a]) - base
        print(f"  {label[a]} vs current, per season Brier: "
              f"{d.mean():+.4f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.4f} "
              f"({int((d < 0).sum())} of {len(d)} seasons better)")

    # ---- key numbers -------------------------------------------------------------
    print("\nkey numbers - how often the margin lands exactly on each value, predicted "
          "against actual (out of sample, pooled over the folds):")
    print(f"  {'margin':>7}{'actual':>10}{'normal':>10}{'PMF':>10}")
    allm = np.concatenate([r["margin"] for r in rows])
    for k in (1, 2, 3, 4, 6, 7, 8, 10, 14, 17, 21):
        act = np.mean(np.abs(allm) == k)
        # The normal's mass on an INTEGER margin is the density integrated over the
        # half-point either side, not the density itself - and it has to use the fold's
        # own fitted sigma, which is the one the shipped win model actually uses.
        nrm = np.mean(np.concatenate([
            _norm_mass(k, r["pred_margin"], r["sigma"])
            + _norm_mass(-k, r["pred_margin"], r["sigma"]) for r in rows]))
        pm_ = np.mean(np.concatenate([r["pmf"].key_number(r["pred_margin"], k)
                                      for r in rows]))
        print(f"  {k:>7}{act:>10.4f}{nrm:>10.4f}{pm_:>10.4f}")

    # ---- displayed score ---------------------------------------------------------
    print("\ndisplayed score - the projected final, against what was actually played:")
    ARMS = ("round", "table")
    mae = {a: [] for a in ARMS}
    exact_m = {a: 0 for a in ARMS}
    exact_s = {a: 0 for a in ARMS}
    key_hit = {a: 0 for a in ARMS}
    n_tot = 0
    for r in rows:
        for i in range(r["n"]):
            t, m = r["pred_total"][i], r["pred_margin"][i]
            ha, aa = int(r["hp"][i]), int(r["ap"][i])
            n_tot += 1
            for name, pair in (("round", round_pair(t, m)),
                               ("table", r["tab"].pick(t, m))):
                mae[name].append((abs(pair[0] - ha) + abs(pair[1] - aa)) / 2)
                if pair[0] - pair[1] == ha - aa:
                    exact_m[name] += 1
                if pair == (ha, aa):
                    exact_s[name] += 1
                if abs(pair[0] - pair[1]) in (3, 7):
                    key_hit[name] += 1
    print(f"  {'':8}{'MAE per side':>14}{'exact margin':>14}{'exact score':>13}"
          f"{'lands on 3 or 7':>18}")
    for name in ARMS:
        print(f"  {name:<8}{np.mean(mae[name]):>14.3f}{exact_m[name]/n_tot:>14.4f}"
              f"{exact_s[name]/n_tot:>13.4f}{key_hit[name]/n_tot:>18.4f}")
    # Two reference rows, because the last column is easy to misread. A projected score
    # is a MODAL prediction, and a mode concentrates: the fraction of GAMES that finish
    # on 3 or 7 is not the fraction of PREDICTIONS that should say 3 or 7. The right
    # comparison is the margin PMF's own most likely value, computed independently of
    # the score table. If the two agree, the table is tracking the distribution rather
    # than drifting toward round numbers.
    mode_key = np.concatenate([
        np.isin(np.abs(r["pmf"].grid[np.argmax(r["pmf"].pmf(r["pred_margin"]), 1)]),
                [3, 7]) for r in rows])
    print(f"  {'PMF mode':<8}{'':>14}{'':>14}{'':>13}{mode_key.mean():>18.4f}"
          f"   <- what the distribution itself says")
    print(f"  {'actual':<8}{'':>14}{'':>14}{'':>13}"
          f"{np.mean(np.isin(np.abs(allm), [3, 7])):>18.4f}"
          f"   <- the marginal, not comparable")

    # ---- fit on everything and export --------------------------------------------
    Xa = np.vstack([data[z]["X"] for z in years])
    ya = np.concatenate([data[z]["y"] for z in years])
    hfa = np.concatenate([data[z]["hf"] for z in years])
    mga = np.concatenate([data[z]["margin"] for z in years])
    full, _ = M.train(Xa, ya, hfa, feature_names=MU.MATCHUP_COLS, margins=mga)
    m_all = (full.margin_intercept + Xa @ full.margin_coef + full.margin_hfa * hfa)
    pmf = MarginPMF(m_all, mga)
    tab = ScoreTable(np.concatenate([data[z]["hp"] for z in years]),
                     np.concatenate([data[z]["ap"] for z in years]))

    # The PMF is exported as a lookup over predicted margin rather than as the raw
    # training pairs: the browser cannot carry 17k games, and the conditional
    # distribution is smooth in the predicted margin, so a half-point grid over the
    # range that ever gets predicted reproduces it to well inside rounding.
    grid = np.arange(-45.0, 45.5, 0.5)
    table = pmf.pmf(grid)
    out = {
        "margin_pmf": {
            "pred_lo": float(grid[0]), "pred_step": float(grid[1] - grid[0]),
            "margin_lo": int(pmf.grid[0]),
            "rows": [[round(float(v), 6) for v in row] for row in table],
        },
        "score_table": tab.export(),
        "h_margin": H_MARGIN,
        "n_games": int(len(mga)),
        "seasons": years,
    }
    (ARTIFACTS / "score_shape.json").write_text(json.dumps(out))
    print(f"\n-> {ARTIFACTS / 'score_shape.json'} "
          f"({len(grid)} predicted-margin rows, {len(out['score_table']['pairs'])} "
          f"scorelines)")


if __name__ == "__main__":
    main()
