"""Learn the concept groupings instead of typing them.

concepts.py grouped 86 features into 15 concepts using a dict I wrote by hand. That
reintroduced exactly the assumption the refactor was meant to remove - it replaced 20
hand-picked facets with 15 hand-picked concepts one level up - and it means every
concept-level conclusion could be an artefact of my grouping rather than a property of
the data. "Receiving adds nothing given passing" is only a finding if "receiving" was
not a category I invented.

Two ways to derive groupings without football judgment:

  correlation   hierarchical clustering on 1 - |r|. Features that move together get
                grouped, whatever position or metric they came from.
  factors       varimax-rotated factor analysis; each feature joins the factor it
                loads on hardest. Rotation matters - unrotated components spread every
                feature across every factor and the assignment is meaningless.

Then the tests that matter: how much does each learned grouping agree with mine, and
do the ablation conclusions survive being re-run on groupings nobody chose?

Run: ./rbenv/bin/python learned_concepts.py
"""
import json, os
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import adjusted_rand_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from concepts import CONCEPTS, concept_of, cv_rmse, load

HERE = os.path.dirname(os.path.abspath(__file__))
K_SWEEP = [6, 8, 10, 12, 15, 20]


def varimax(L, tol=1e-6, max_iter=200):
    """Kaiser varimax rotation. Without it, factor assignment is arbitrary."""
    L = L.copy()
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lam = L @ R
        u, s, vt = np.linalg.svd(
            L.T @ (Lam ** 3 - Lam @ np.diag(np.sum(Lam ** 2, axis=0)) / p))
        R = u @ vt
        d = float(np.sum(s))
        if d_old and abs(d - d_old) < tol:
            break
    return L @ R


def cluster_groups(Z, names, k):
    """k groups by hierarchical clustering on 1 - |correlation|."""
    C = np.nan_to_num(np.corrcoef(Z[names].to_numpy(float), rowvar=False), nan=0.0)
    D = 1 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    lab = hierarchy.fcluster(
        hierarchy.linkage(squareform(D, checks=False), method="average"), k, "maxclust")
    g = {}
    for n, c in zip(names, lab):
        g.setdefault(f"corr_{c}", []).append(n)
    return g


def factor_groups(Z, names, k):
    """k groups by varimax-rotated factor loadings; a feature joins its top factor."""
    S = StandardScaler().fit_transform(Z[names].to_numpy(float))
    fa = FactorAnalysis(n_components=k, random_state=0).fit(S)
    L = varimax(fa.components_.T)          # (features, factors)
    top = np.argmax(np.abs(L), axis=1)
    g = {}
    for n, t in zip(names, top):
        g.setdefault(f"fa_{t+1}", []).append(n)
    return g


def label_of(members):
    """Describe a learned group by what its members have in common."""
    pos = pd.Series([m.split("_")[0] for m in members]).value_counts()
    concepts = pd.Series([concept_of(m) or "?" for m in members]).value_counts()
    top_pos = ", ".join(pos.index[:2])
    top_con = ", ".join(f"{c} {n}" for c, n in concepts.items() if n > 0)
    return f"[{top_pos}] {top_con}"


def as_labels(groups, names):
    """Group dict -> integer label per feature, for the agreement score."""
    m = {}
    for i, (_, members) in enumerate(groups.items()):
        for n in members:
            m[n] = i
    return [m.get(n, -1) for n in names]


def ablate(Z, names, y, seasons, groups, base):
    """Leave-one-group-out and forward selection on an arbitrary grouping."""
    X = Z[names].to_numpy(float)
    owner = {n: g for g, ms in groups.items() for n in ms}

    loco = []
    for g, ms in groups.items():
        cols = [i for i, n in enumerate(names) if owner.get(n) != g]
        if len(cols) == len(names):
            continue
        r = cv_rmse(X, y, seasons, cols)
        loco.append({"group": g, "n": len(ms), "rmse": r, "delta": r - base,
                     "label": label_of(ms)})
    loco = sorted(loco, key=lambda d: -d["delta"])

    chosen, remaining = [], list(groups)
    best = cv_rmse(X, y, seasons, [])
    fwd = []
    while remaining:
        scored = []
        for g in remaining:
            cols = [i for i, n in enumerate(names)
                    if owner.get(n) in set(chosen) | {g}]
            scored.append((cv_rmse(X, y, seasons, cols), g))
        scored.sort()
        r, g = scored[0]
        fwd.append({"group": g, "step": len(chosen) + 1, "rmse": r,
                    "gain": best - r, "label": label_of(groups[g])})
        chosen.append(g)
        remaining.remove(g)
        best = r
    return loco, fwd


def main():
    fv, df, names = load()
    y = df.y.to_numpy(float)
    seasons = df.index.get_level_values("season").to_numpy()
    Z = df[names]
    X = Z.to_numpy(float)
    base = cv_rmse(X, y, seasons)

    hand = {c: [m for m in ms if m in names] for c, ms in CONCEPTS.items()}
    hand_lab = as_labels(hand, names)
    print(f"features {len(names)}   all-features rmse {base:.4f}")
    print(f"hand-chosen concepts: {len(hand)}\n")

    # ---- 1. how many groups does the data want? -----------------------------
    print("=" * 80)
    print("1. SWEEP THE NUMBER OF GROUPS — agreement with my hand-chosen version")
    print("=" * 80)
    print(f"  {'k':>3}  {'method':<12}{'6-group fwd rmse':>18}{'ARI vs hand':>14}")
    rows = []
    best_by = {}
    for k in K_SWEEP:
        for meth, fn in (("correlation", cluster_groups), ("factor", factor_groups)):
            g = fn(Z, names, k)
            ari = adjusted_rand_score(hand_lab, as_labels(g, names))
            # how good is a model using only the best six of these learned groups?
            _, fwd = ablate(Z, names, y, seasons, g, base)
            six = fwd[min(5, len(fwd) - 1)]["rmse"]
            rows.append({"k": k, "method": meth, "ari_vs_hand": ari,
                         "rmse_top6_groups": six, "n_groups": len(g)})
            best_by[(k, meth)] = g
            print(f"  {k:>3}  {meth:<12}{six:>18.4f}{ari:>14.3f}")

    r = pd.DataFrame(rows)
    print(f"\n  adjusted Rand index: 1.0 = identical grouping, 0.0 = no better than "
          f"chance.\n  best agreement with my hand grouping: "
          f"{r.ari_vs_hand.max():.3f} "
          f"({r.loc[r.ari_vs_hand.idxmax(), 'method']}, k="
          f"{r.loc[r.ari_vs_hand.idxmax(), 'k']})")

    # ---- 2. the learned grouping at the size the data prefers ---------------
    pick = r.loc[r.rmse_top6_groups.idxmin()]
    gk, gm = int(pick.k), pick.method
    learned = best_by[(gk, gm)]
    print("\n" + "=" * 80)
    print(f"2. THE BEST LEARNED GROUPING — {gm} clustering, k={gk}")
    print("=" * 80)
    for g, ms in sorted(learned.items(), key=lambda kv: -len(kv[1])):
        print(f"  {g:<9} {len(ms):>2}  {label_of(ms)}")
        print(f"            {', '.join(ms[:7])}{' ...' if len(ms) > 7 else ''}")

    # ---- 3. do the conclusions survive? ------------------------------------
    print("\n" + "=" * 80)
    print("3. ABLATION ON THE LEARNED GROUPING")
    print("=" * 80)
    loco, fwd = ablate(Z, names, y, seasons, learned, base)
    print("  leave-one-group-out:")
    for d in loco:
        print(f"    {d['group']:<9} n={d['n']:>2}  rmse {d['rmse']:.4f}  "
              f"{d['delta']:+.5f}   {d['label']}")
    print("\n  forward selection:")
    for d in fwd:
        flag = "" if d["gain"] > 0.0005 else "   <- stops paying"
        print(f"    +{d['group']:<9} ({d['step']:>2}) rmse {d['rmse']:.4f}  "
              f"gain {d['gain']:+.4f}{flag}  {d['label']}")

    # ---- 4. hand vs learned, head to head -----------------------------------
    print("\n" + "=" * 80)
    print("4. HAND-CHOSEN vs LEARNED, same test")
    print("=" * 80)
    hl, hf = ablate(Z, names, y, seasons, hand, base)
    n_pay_hand = sum(1 for d in hf if d["gain"] > 0.0005)
    n_pay_lrn = sum(1 for d in fwd if d["gain"] > 0.0005)
    print(f"  hand-chosen  {len(hand):>2} groups, "
          f"{n_pay_hand} pay their way, best-6 rmse "
          f"{hf[min(5, len(hf)-1)]['rmse']:.4f}")
    print(f"  learned      {len(learned):>2} groups, "
          f"{n_pay_lrn} pay their way, best-6 rmse "
          f"{fwd[min(5, len(fwd)-1)]['rmse']:.4f}")
    print(f"  agreement (ARI) {adjusted_rand_score(hand_lab, as_labels(learned, names)):.3f}")

    json.dump({
        "sweep": json.loads(r.round(5).to_json(orient="records")),
        "chosen": {"method": gm, "k": gk},
        "learned_groups": {g: ms for g, ms in learned.items()},
        "learned_labels": {g: label_of(ms) for g, ms in learned.items()},
        "learned_loco": loco, "learned_forward": fwd,
        "hand_loco": hl, "hand_forward": hf,
        "ari_hand_vs_learned": adjusted_rand_score(hand_lab,
                                                   as_labels(learned, names)),
        "all_features_rmse": base,
    }, open(f"{HERE}/learned_concepts.json", "w"), indent=1)
    print("\n-> learned_concepts.json")


if __name__ == "__main__":
    main()
