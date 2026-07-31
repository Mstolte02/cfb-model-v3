"""Facet weights in two levels: concepts fit against wins, facets split by reliability.

The flat fit asks one regression two questions at once - how much do quarterbacks
matter, and what makes a quarterback good - and it can only answer the first.

Bootstrapping the NNLS fit over team-seasons shows exactly where it breaks. The QB
block total is well identified at 15.5% (5-95% band 11.1-19.1). The split INSIDE that
block is not identified at all: QB_twp_rate takes anywhere from 7.6% to 28.1% of it,
QB_pass takes 0-12.6% and is driven to exactly zero in 16% of resamples, and eight
different facets hold the top QB slot at some point. Every other block behaves the
same way - 7 of 9 tight end facets are zeroed in a quarter or more of fits.

Two things cause it. The facets are six views of one throw (mean |r| among the QB
facets is 0.36, up to 0.93; QB_pass correlates 0.82 with QB_qb_rating and 0.79 with
QB_positive_epa_pct), so the regression is genuinely indifferent between loading the
grade and loading six pieces of the grade. And NNLS is a sparse solver: faced with
near-duplicates it does not split the weight, it sends one to exactly zero. The 30
zeros in the flat fit are that, not feature selection.

For predicting team wins none of this matters - the team totals come out the same
either way. For WAR it is fatal, because a player's value is attributed as
w * value / sigma, so which of two near-twins won the coin flip decides who gets
credit. That is how turnover-worthy-play rate ended up carrying three times the weight
of the passing grade itself, and how a tight end's receiving value ended up rooted in
targeted QB rating - a statistic describing the quarterback throwing to him.

So the questions get separated, and each level uses the finest rule the data can
actually support:

  BETWEEN groups, FIT.        "Does pass rush beat coverage" is a question the wins
                              data can answer, and the block totals prove it - the QB
                              block is stable at 15.5% even while its contents are not.
                              Thirteen groups instead of 98 facets is also a far better
                              conditioned problem. Bagged over resamples, which removes
                              the arbitrary zeroing on its own.

  WITHIN a group, UNIVARIATE. A group is a cluster of concepts too collinear to
                              separate - in practice exactly one exists, the passing
                              game, where passing and receiving correlate at 0.83
                              because PFF grades the quarterback and the receiver on
                              the same throw. The joint fit resolves that by starving
                              receiving (2.7% of the model, zeroed in 28% of resamples)
                              despite it having the second-highest univariate
                              correlation with next season's wins of any concept. So
                              the group's weight is split by that univariate strength
                              instead. Splitting by reliability was tried and rejected:
                              the three concepts' mean rho is .385/.349/.367, near
                              enough to identical that the split collapsed onto how
                              many candidate features the catalogue happened to
                              generate per job - 9 passing against 20 receiving - which
                              is the coverage unfairness candidates.py already fixed
                              once.

  WITHIN a concept, DON'T.    Every facet in a concept measures the same job, so the
                              regression has no real basis for preferring one, and the
                              bootstrap confirms it has none. Split by year-over-year
                              reliability: of two measures of the same thing, the one
                              that repeats is the better measure. For a model whose
                              output is a projection that is the right criterion, and
                              it cannot produce a zero by accident.

Final per-facet weight is the product of the three, so the artifact keeps the shape
every downstream stage already reads.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from concepts import CONCEPTS

HERE = os.path.dirname(os.path.abspath(__file__))

# concepts.py covers the 86 PFF candidates only. The CFBD facets play the same jobs;
# havoc splits by unit because a defensive back's havoc is mostly passes defensed
# while a lineman's is mostly pressure.
CFBD_CONCEPT = {
    "cfbd_pass_qb": "passing", "cfbd_run_qb": "qb_rushing", "cfbd_run_rb": "rushing",
    "cfbd_recv_wr": "receiving", "cfbd_recv_te": "receiving", "cfbd_recv_rb": "receiving",
    "cfbd_havoc_dl": "pass_rush", "cfbd_havoc_lb": "pass_rush",
    "cfbd_havoc_db": "coverage", "cfbd_cov_db": "coverage",
    "cfbd_tackle_lb": "tackling", "cfbd_tackle_db": "tackling",
}


RHO_FLOOR, RHO_CEIL, MIN_PAIRS = 0.02, 0.95, 60
RELIABILITY = "facet_reliability_player.csv"


def load_reliability(fv=None, path=None, rebuild=False):
    """Player-specific reliability, computed once and cached beside the artifacts.

    Falls back to uncertainty.py's facet_reliability.csv only if the cache is absent
    and no facet frame was handed in - that file measures raw repeatability, which is
    the right quantity for its own error bars and the wrong one for splitting weight.
    """
    path = path or f"{HERE}/{RELIABILITY}"
    if os.path.exists(path) and not rebuild:
        return pd.read_csv(path).set_index("facet")
    if fv is None:
        return pd.read_csv(f"{HERE}/facet_reliability.csv").set_index("facet")
    rel = player_specific_rho(fv)
    rel.to_csv(path)
    return rel


def player_specific_rho(fv, key=None):
    """Year-over-year repeatability of a player's z with the TEAMMATE component removed.

    rho is the correlation of a player's z with his own next-season z, PARTIALLING OUT
    his team's contemporaneous z over his teammates (leave-one-out, so a player is
    never a control for himself). Where a facet has fewer than two qualifying players
    per team-season there is no teammate mean and the raw correlation stands - the
    normal case at quarterback.

    THIS WAS BUILT TO FIX SOMETHING IT DOES NOT FIX, and the record is worth keeping.
    The suspicion was that the CFBD defensive facets score high (cfbd_tackle_lb .620,
    cfbd_havoc_dl .608, against .10-.55 for a typical PFF facet) because they share a
    team denominator, so team continuity would be doing the repeating. Partialling the
    team out barely moves them - .620 to .605, .608 to .608 - so that was wrong. Their
    denominator is a real problem, but it is a problem with the replacement credit
    rather than with reliability, and build_hybrid.py fixes it there.

    It is kept because it turned out to correct something else, and something that
    matters more for splitting weight inside a concept. WR_positive_epa_pct falls from
    .411 to .106: almost all of what repeats in a receiver's positive-EPA rate is his
    QUARTERBACK, not him. QB_pass falls .385 to .255, WR_targeted_qb_rating .213 to
    .153. Those are facets that look like good measurements of a player and are partly
    measurements of the people around him - exactly the thing that should lose weight
    when the question is which facet better measures THIS player.
    """
    # The frame is called uid inside build_hybrid and player_id once it has been
    # written out, so the column is detected rather than assumed - the same thing
    # uncertainty.py does, and for the same reason.
    key = key or ("player_id" if "player_id" in fv.columns else "uid")
    rows = []
    for f, g in fv.groupby("facet"):
        g = g[g.snaps >= g.snaps.quantile(0.4)]
        n = g.groupby(["season", "team"]).z.transform("size")
        tot = g.groupby(["season", "team"]).z.transform("sum")
        g = g.assign(loo=np.where(n > 1, (tot - g.z) / (n - 1), np.nan))

        nxt = g[["season", key, "z"]].copy()
        nxt["season"] -= 1
        j = g.merge(nxt.rename(columns={"z": "z1"}), on=["season", key])
        raw = (float(np.corrcoef(j.z, j.z1)[0, 1]) if len(j) >= MIN_PAIRS else 0.25)

        jj = j.dropna(subset=["loo"])
        if len(jj) >= MIN_PAIRS and jj.loo.std() > 0:
            r_xy = np.corrcoef(jj.z, jj.z1)[0, 1]
            r_xz = np.corrcoef(jj.z, jj.loo)[0, 1]
            r_yz = np.corrcoef(jj.z1, jj.loo)[0, 1]
            denom = np.sqrt(max((1 - r_xz ** 2) * (1 - r_yz ** 2), 1e-12))
            rho, basis = float((r_xy - r_xz * r_yz) / denom), "partial"
        else:
            rho, basis = raw, "raw"
        rows.append({"facet": f, "rho": float(np.clip(rho, RHO_FLOOR, RHO_CEIL)),
                     "rho_raw": float(np.clip(raw, RHO_FLOOR, RHO_CEIL)),
                     "basis": basis, "n_pairs": len(j)})
    return pd.DataFrame(rows).set_index("facet")


def concept_map(facets):
    """{facet: concept}. Anything unclaimed becomes its own concept rather than being
    dropped, so a new candidate cannot silently fall out of the model."""
    m = {f: c for c, members in CONCEPTS.items() for f in members}
    m.update(CFBD_CONCEPT)
    return {f: m.get(f, f"_solo_{f}") for f in facets}


def within_weights(facets, rel, cmap):
    """Reliability shares within each concept, summing to 1 per concept.

    rho is the year-over-year correlation of a player's z on that facet (uncertainty.py),
    already floored at 0.02, so no facet can be weighted out entirely and none can
    divide by zero.
    """
    rho = rel.rho.reindex(facets).fillna(rel.rho.median())
    v = pd.Series(rho.to_numpy(), index=facets, dtype=float)
    return v.groupby(pd.Series(cmap, dtype=object).reindex(facets)).transform(
        lambda s: s / s.sum())


def concept_scores(Z, v, cmap):
    """Team-season score per concept: the reliability-weighted mean of its facet z's,
    rescaled to unit sd so the ridge penalty means the same thing for every concept.

    Returns (scores, sd) - sd is needed to compose the two levels back into per-facet
    weights, because the rescaling is part of the linear map.
    """
    c = pd.Series(cmap, dtype=object).reindex(Z.columns)
    S = pd.DataFrame({name: Z[list(g.index)] @ v[list(g.index)]
                      for name, g in c.groupby(c)})
    sd = S.std(ddof=0).replace(0.0, 1.0)
    return S / sd, sd


def _nnls_ridge(X, y, lam):
    A = np.vstack([X, np.sqrt(lam) * np.eye(X.shape[1])])
    b = np.concatenate([y - y.mean(), np.zeros(X.shape[1])])
    return nnls(A, b)[0]


def fit_concept_weights(S, y, lam=None, n_boot=200, seed=0):
    """Non-negative ridge of next-season win pct on the concept scores, bagged.

    Bagging matters even here. NNLS still zeroes a concept it is unsure about in any
    single fit; averaging over resamples returns its expected weight instead, which is
    the honest summary of what the data supports.
    """
    X, yv = S.to_numpy(float), np.asarray(y, float)
    if lam is None:
        lam = tune_lambda(X, yv)
    rng = np.random.default_rng(seed)
    W = np.mean([_nnls_ridge(X[i], yv[i], lam)
                 for i in (rng.integers(0, len(X), len(X)) for _ in range(n_boot))],
                axis=0)
    return pd.Series(W, index=S.columns), lam


def tune_lambda(X, y, grid=(1, 3, 10, 30, 100, 300, 1000), folds=5, seed=0):
    """Blocked CV on the concept design. The flat fit's lam=1000 was tuned for 98
    collinear columns and is far too strong for 15 nearly independent ones."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(X))
    best, best_err = grid[0], np.inf
    for lam in grid:
        err = 0.0
        for k in range(folds):
            tr, te = fold != k, fold == k
            w = _nnls_ridge(X[tr], y[tr], lam)
            p = X[te] @ w
            err += float(np.sum(((y[te] - y[te].mean()) - p) ** 2))
        if err < best_err:
            best, best_err = lam, err
    return best


R_GROUP = 0.60      # concepts correlating above this cannot be separated by the fit


def group_map(S, thresh=R_GROUP):
    """Cluster concept scores into groups by single-linkage on |correlation|.

    Derived rather than declared, so a change to the facet set cannot silently leave a
    new collinear pair being resolved by a fit that has no basis for resolving it. At
    0.60 on the current set this finds exactly one non-trivial group - the passing
    game - and leaves the other eleven concepts alone (mean |r| between concepts is
    0.14; the next pair down is pass rush and run defense at 0.55).
    """
    C = S.corr().abs()
    parent = {c: c for c in S.columns}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(S.columns):
        for b in S.columns[i + 1:]:
            if C.loc[a, b] >= thresh:
                parent[find(a)] = find(b)
    roots = {}
    for c in S.columns:
        roots.setdefault(find(c), []).append(c)
    # name a group for its strongest member so the output stays readable
    return {c: (f"{max(g, key=lambda x: len(CONCEPTS.get(x, [])))}_group"
                if len(g) > 1 else c)
            for g in roots.values() for c in g}


def group_shares(S, y, gmap):
    """Each concept's share of its group, by univariate correlation with next-season
    wins. Within a group the concepts are near-duplicates, so this asks the only
    question that is still answerable of each one alone: how much does it know about
    next season? Singleton groups get 1.0 and never touch this path."""
    uni = pd.Series({c: np.corrcoef(S[c], y)[0, 1] for c in S.columns}).clip(lower=0.0)
    g = pd.Series(gmap, dtype=object).reindex(S.columns)
    return uni.groupby(g).transform(lambda s: s / s.sum() if s.sum() else 1.0 / len(s))


def build(Z, y, rel, facets=None, n_boot=200, lam=None, seed=0):
    """Per-facet weights, summing to 1, in the shape build_hybrid already writes.

    Z    within-season standardized team facet totals
    y    next-season adjusted win pct, aligned to Z's rows
    rel  facet_reliability.csv, indexed by facet
    """
    facets = list(facets or Z.columns)
    y = np.asarray(y, float)
    cmap = concept_map(facets)
    v = within_weights(facets, rel, cmap)                       # level 3
    assert not v.isna().any(), "a facet has no reliability and would fall out silently"
    S, sd = concept_scores(Z[facets], v, cmap)

    gmap = group_map(S)                                         # level 2
    share = group_shares(S, y, gmap)
    g = pd.Series(gmap, dtype=object).reindex(S.columns)
    G = pd.DataFrame({name: (S[list(k.index)] * share[list(k.index)]).sum(axis=1)
                      for name, k in g.groupby(g)})
    gsd = G.std(ddof=0).replace(0.0, 1.0)

    W, lam = fit_concept_weights(G / gsd, y, lam=lam, n_boot=n_boot, seed=seed)  # level 1
    W = W / W.sum() if W.sum() else W

    # compose the three linear maps back down onto the facets
    cw = pd.Series({c: W[gmap[c]] * share[c] / gsd[gmap[c]] for c in S.columns})
    c = pd.Series(cmap, dtype=object).reindex(facets)
    w = v * c.map(cw) / c.map(sd)
    w = w / w.sum()
    return w, {"group_weights": W.sort_values(ascending=False),
               "concept_weights": (cw * c.map(sd).groupby(c).first()).pipe(
                   lambda s: s / s.sum()).sort_values(ascending=False),
               "groups": {k: v2 for k, v2 in gmap.items() if v2 != k},
               "within_group": share, "within": v, "lam": lam,
               "n_groups": G.shape[1], "n_concepts": S.shape[1]}
