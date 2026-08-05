"""Collapse the near-duplicate facet clusters onto their first principal component.

collinearity.py has been saying this for as long as it has existed and nothing acted
on it: seven of the correlated clusters carry 82-98% of their variance on one
component AND predict next season's wins as well from that component alone as from
all their members together. They are one football ability measured several ways, and
carrying them separately buys nothing but condition number - 33 components are needed
to reach 80% of the variance of 98 features.

What that costs is interpretability of the weights, which for a model whose OUTPUT IS
AN ATTRIBUTION is the whole product. Six views of one throw compete for the same
weight; the fit is indifferent between loading the grade and loading six pieces of the
grade, and whichever wins decides who gets credited.

THE COMPOSITE HAS TO BE BUILT AT THE PLAYER LEVEL, not the team level, or WAR cannot
be attributed through it. It is, and exactly, because every step between a player and
the rating is linear:

    tot[t,f]  = sum over players of value[p,f]
    Z[t,f]    = (tot[t,f] - mean_f) / sigma_f
    PC1[t]    = sum over f in cluster of L_f * Z[t,f]

so defining the composite's per-player value as

    value_c[p] = sum over f in cluster of L_f * value[p,f] / sigma_f

gives a team total that IS PC1 up to an additive constant, which the pipeline's own
within-season standardization removes. Nothing downstream needs to know it happened:
the composite is a facet like any other, and f_contrib = w * value / sigma still
decomposes a rating into the players who produced it.

Clusters and the combine/keep-separate call use collinearity.py's own rule, computed
here rather than read from collinearity.json, so a change to the candidate set cannot
leave this stage acting on a stale recommendation.
"""
import os

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

CLUSTER_CUT = 0.30      # 1 - |r|, i.e. |r| >= 0.70. collinearity.py's line.
PC1_MIN = 0.65          # one component has to carry this much of the cluster
R_TOL = 0.02            # ...and not cost more than this against the target


def find_clusters(Z, y, cut=CLUSTER_CUT):
    """[{name, members, loadings, pc1_explained, ...}] for the clusters worth merging.

    Z  within-season standardized team facet totals (team-seasons x facets)
    y  the target those facets are being judged against
    """
    names = list(Z.columns)
    X = Z.to_numpy(float)
    C = pd.DataFrame(np.corrcoef(X, rowvar=False), index=names, columns=names)
    D = 1 - C.abs().to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    lab = hierarchy.fcluster(
        hierarchy.linkage(squareform(D, checks=False), method="average"),
        cut, "distance")

    groups = {}
    for f, c in zip(names, lab):
        groups.setdefault(int(c), []).append(f)

    out = []
    for members in sorted(groups.values(), key=lambda m: -len(m)):
        if len(members) < 2:
            continue
        idx = [names.index(m) for m in members]
        S = StandardScaler().fit_transform(X[:, idx])
        p = PCA(n_components=min(len(members), 3)).fit(S)
        ev = p.explained_variance_ratio_
        c1 = S @ p.components_[0]
        r_pc1 = abs(np.corrcoef(c1, y)[0, 1])
        A = np.column_stack([np.ones(len(S)), S])
        B, *_ = np.linalg.lstsq(A, y, rcond=None)
        r_full = abs(np.corrcoef(A @ B, y)[0, 1])
        if not (ev[0] >= PC1_MIN and r_pc1 >= r_full - R_TOL):
            continue

        L = p.components_[0]
        # Sign is arbitrary in PCA; orient the component so that MORE of it means
        # BETTER, or a whole cluster's weight would be applied backwards and the
        # non-negativity constraint downstream would simply zero it.
        if np.corrcoef(c1, y)[0, 1] < 0:
            L = -L
        loadings = {m: float(v) for m, v in zip(members, L)}
        out.append({
            "name": _name_for(members),
            "members": members,
            "loadings": loadings,
            "concept": _concept_for(loadings),
            "pc1_explained": float(ev[0]),
            "r_pc1": float(r_pc1),
            "r_full": float(r_full),
        })
    return out


def _concept_for(loadings):
    """Which concept the composite belongs to: its heaviest-loading member's.

    Without this a composite becomes its OWN concept, and the block it came out of is
    split in two - consolidating the four edge-rush facets left `pass_rush` at 4.0%
    with a separate `ED_core` at 9.1% beside it, which is less interpretable than what
    it replaced, not more. Inheriting the concept keeps the block whole and means the
    printed group weights still answer "how much is pass rush worth".
    """
    from concepts import CONCEPTS
    from two_level_weights import CFBD_CONCEPT
    m = {f: c for c, members in CONCEPTS.items() for f in members}
    m.update(CFBD_CONCEPT)
    ranked = sorted(loadings, key=lambda f: -abs(loadings[f]))
    for f in ranked:
        if f in m:
            return m[f]
    return None


def _name_for(members):
    """A readable name for the composite: the shared position prefix plus the job.

    QB_pass + QB_qb_rating + WR_targeted_qb_rating is the passing game, and calling it
    `QBWR_core` says more than `cluster_57`.
    """
    pref = sorted({m.split("_")[0] for m in members})
    tag = "".join(p.replace("-", "") for p in pref)[:12]
    return f"{tag}_core"


def apply(fv, sigma, clusters):
    """Replace each cluster's member rows in the facet frame with one composite row.

    fv     long facet frame: season, uid/player_id, ..., facet, snaps, z, value
    sigma  DataFrame season x facet of the team-total sd used to standardize
    """
    if not clusters:
        return fv, {}

    member_of = {m: c["name"] for c in clusters for m in c["members"]}
    loading = {(c["name"], m): l for c in clusters for m, l in c["loadings"].items()}

    hit = fv[fv.facet.isin(member_of)].copy()
    rest = fv[~fv.facet.isin(member_of)]
    if hit.empty:
        return fv, {}

    hit["composite"] = hit.facet.map(member_of)
    hit["L"] = [loading[(c, f)] for c, f in zip(hit.composite, hit.facet)]
    hit["sig"] = [sigma.loc[s, f] for s, f in zip(hit.season, hit.facet)]
    # value in units of the standardized team total, which is what makes the summed
    # composite equal PC1
    hit["_v"] = hit.L * hit.value / hit.sig.replace(0.0, np.nan)
    # z is only consumed by the reliability estimate; unit-norm so it stays a z
    norm = {c["name"]: float(np.linalg.norm(list(c["loadings"].values())))
            for c in clusters}
    hit["_z"] = hit.L * hit.z / hit.composite.map(norm)

    key = [c for c in ("season", "uid", "player_id", "player", "position", "team",
                       "source") if c in hit.columns]
    comp = hit.groupby(key + ["composite"], as_index=False).agg(
        value=("_v", "sum"), z=("_z", "sum"),
        # the composite occupies whichever of its members' denominators is the real
        # playing time; every other member's is a subset of it, exactly as
        # build_hybrid argues when it takes a max to report a player's snaps
        snaps=("snaps", "max"))
    comp = comp.rename(columns={"composite": "facet"})

    out = pd.concat([rest, comp], ignore_index=True)
    report = {c["name"]: {"members": c["members"], "n": len(c["members"]),
                          "concept": c["concept"],
                          "pc1_explained": c["pc1_explained"],
                          "r_pc1": c["r_pc1"], "r_full": c["r_full"]}
              for c in clusters}
    return out, report


def condition_number(Z):
    """kappa of the standardized design. The number this stage exists to bring down."""
    X = Z.to_numpy(float)
    X = X - X.mean(0)
    sd = X.std(0)
    X = X / np.where(sd > 0, sd, 1.0)
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else np.inf
