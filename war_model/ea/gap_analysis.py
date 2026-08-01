"""Where is the PFF projection weak, and could EA plausibly fill that gap?

The hypothesis is that EA should beat us on players with little prior playing time,
because we impute them from a positional prior and EA has an actual opinion on
everyone. Testing it properly needs a preseason EA rating for a season that has been
played, which is exactly what cannot be obtained - CFB 26's player database is behind a
robots.txt that disallows this crawler, and CFB 25 is archived nowhere.

So the question is split into the parts that ARE answerable:

  1. Is there a gap at all? Re-run the projection's 2025 holdout and score it by how
     much the player had played before. If we are already fine on backups there is
     nothing for EA to fix.

  2. Would EA even change the answer? If EA and the projection agree closely on
     low-snap players, EA cannot help there whatever its accuracy.

  3. Is EA's opinion of an unproven player anything more than his recruiting rank?
     This is the crux and it is decidable today. Our projection already feeds on
     `stars` and `rating`, so if EA's view of a no-history player is just the
     composite again, it adds nothing we do not have.

Run: ../../venv/bin/python gap_analysis.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WAR_DIR = os.path.dirname(HERE)
sys.path.insert(0, WAR_DIR)

from build_roster_2026 import norm_name  # noqa: E402

BUCKETS = [-1, 0, 100, 300, 600, 10_000]
LABELS = ["none (0)", "1-100", "100-300", "300-600", "600+"]


def holdout_2025():
    """The projection model's own 2025 holdout, kept per player instead of aggregated.

    Rebuilt here rather than imported because project_2026_v2.main() prints summary
    numbers and throws the per-row predictions away, and the whole question is how the
    error behaves in the tail.
    """
    import project_2026_v2 as P
    from build_recruiting import load_recruits
    import artifacts

    war = pd.read_csv(f"{WAR_DIR}/{artifacts.PLAYER_WAR}")
    ratings = pd.read_csv(f"{WAR_DIR}/{artifacts.TEAM_RATINGS}")
    recs = pd.read_csv(f"{WAR_DIR}/records.csv")
    rec = load_recruits()
    ros26 = pd.read_csv(f"{WAR_DIR}/roster_2026.csv")

    K, S = P.slot_counts(ros26)
    w = P.build_history(war)
    fbs = set(recs.team.unique())
    rosters = P.load_rosters(P.WAR_YEARS, fbs)
    pop = P.build_population(w, rosters, K)
    targets = [y for y in P.WAR_YEARS if (y - 1) in set(P.WAR_YEARS)]
    tr = P.make_training(pop, w, ratings, rec, rosters, S, targets)
    groups = sorted(w.group.dropna().unique())
    tr["group_code"] = tr.group.map({g: i for i, g in enumerate(groups)})
    tr["share_lag1"] = tr.share_lag1.fillna(0.0)

    trn, tst = tr[tr.target_season < 2025], tr[tr.target_season == 2025]
    m = P.fit().fit(trn[P.FEATURES], trn.war)
    out = tst.copy()
    out["pred"] = m.predict(tst[P.FEATURES])
    out["prior_snaps"] = out.snaps_lag1.fillna(0.0)
    # what a positional prior alone would have said - the fallback the imputed
    # players effectively get
    out["baseline"] = out.group_code.map(trn.groupby("group_code").war.mean())
    return out


def gap_table(h):
    h = h.copy()
    h["bucket"] = pd.cut(h.prior_snaps, BUCKETS, labels=LABELS)
    rows = []
    for b, d in h.groupby("bucket", observed=True):
        if len(d) < 30:
            continue
        rows.append({
            "prior snaps": b, "players": len(d),
            "actual mean WAR": d.war.mean(),
            "model r": d.pred.corr(d.war),
            "model MAE": (d.pred - d.war).abs().mean(),
            "positional-prior MAE": (d.baseline - d.war).abs().mean(),
            "model beats prior by": ((d.baseline - d.war).abs().mean()
                                     - (d.pred - d.war).abs().mean()),
        })
    return pd.DataFrame(rows)


def ea_join():
    ea = pd.read_csv(f"{HERE}/ea_player_war_2026.csv")
    pff = pd.read_csv(f"{WAR_DIR}/projections_2026_v2.csv")
    pff["key"] = pff.player.map(norm_name)
    e = ea[["key", "team", "overall", "war"]].rename(
        columns={"war": "ea_war", "overall": "ea_ovr"}).drop_duplicates(["key", "team"])
    d = pff.merge(e, on=["key", "team"], how="left")
    d["prior_snaps"] = d.snaps_2025.fillna(0.0)
    d["bucket"] = pd.cut(d.prior_snaps, BUCKETS, labels=LABELS)
    return d


def disagreement_table(d):
    z = lambda s: (s - s.mean()) / s.std()
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        g = g.dropna(subset=["ea_war"])
        if len(g) < 30:
            continue
        rows.append({
            "prior snaps": b, "players": len(g),
            "EA coverage": d[d.bucket == b].ea_war.notna().mean(),
            "corr(EA, ours)": g.ea_war.corr(g.proj_war),
            "corr of z-scores": z(g.ea_war).corr(z(g.proj_war)),
            "mean |z gap|": (z(g.ea_war) - z(g.proj_war)).abs().mean(),
        })
    return pd.DataFrame(rows)


def freshman_test(d):
    """Is EA's opinion of an unproven player more than his recruiting rank?

    Our projection already has stars and rating as features, so anything EA says that
    is just the composite restated is information the model has already priced.
    """
    n = d[(d.prior_snaps == 0) & d.ea_ovr.notna()].copy()
    rows = [("no-prior-snap players with an EA rating", len(n))]
    for col, lbl in (("stars", "recruiting stars"), ("rating", "247 composite rating")):
        v = n[col].astype(float)
        ok = v.notna()
        if ok.sum() > 30:
            rows.append((f"corr(EA overall, {lbl})", round(float(n.ea_ovr[ok].corr(v[ok])), 3)))
            rows.append((f"corr(our proj_war, {lbl})", round(float(n.proj_war[ok].corr(v[ok])), 3)))
    # partial: does EA say anything about these players our projection does not?
    sub = n.dropna(subset=["ea_ovr", "proj_war", "rating"])
    if len(sub) > 50:
        X = np.column_stack([np.ones(len(sub)), sub.rating.astype(float)])
        b = np.linalg.lstsq(X, sub.ea_ovr, rcond=None)[0]
        ea_resid = sub.ea_ovr - X @ b
        b2 = np.linalg.lstsq(X, sub.proj_war, rcond=None)[0]
        pf_resid = sub.proj_war - X @ b2
        rows += [
            ("", ""),
            ("after removing recruiting from BOTH:", ""),
            ("corr(EA residual, our residual)", round(float(ea_resid.corr(pf_resid)), 3)),
            ("EA variance explained by recruiting",
             round(float(1 - ea_resid.var() / sub.ea_ovr.var()), 3)),
            ("our variance explained by recruiting",
             round(float(1 - pf_resid.var() / sub.proj_war.var()), 3)),
        ]
    return pd.DataFrame(rows, columns=["test", "value"])


def recruiting_worth(h):
    """Does recruiting predict what an unproven player actually does?

    It matters because 19% of EA's opinion of a no-history player IS recruiting, so
    this bounds how much of that share can possibly be useful. And it is a free check
    on our own model: if recruiting predicted these players well and we were ignoring
    it, that would be an improvement available without EA at all.
    """
    n = h[h.prior_snaps == 0].copy()
    n["rating"] = pd.to_numeric(n.rating, errors="coerce")
    sub = n.dropna(subset=["rating"])
    X = np.column_stack([np.ones(len(sub)), sub.pred])
    res = sub.war - X @ np.linalg.lstsq(X, sub.war, rcond=None)[0]
    X2 = np.column_stack([np.ones(len(sub)), sub.pred, sub.rating])
    res2 = sub.war - X2 @ np.linalg.lstsq(X2, sub.war, rcond=None)[0]
    return pd.DataFrame([
        ("no-prior-snap players (2025 holdout)", len(n)),
        ("corr(247 composite, ACTUAL 2025 WAR)", round(float(sub.rating.corr(sub.war)), 3)),
        ("corr(247 composite, our prediction)", round(float(sub.rating.corr(sub.pred)), 3)),
        ("", ""),
        ("corr(recruiting, what our model MISSES)", round(float(sub.rating.corr(res)), 3)),
        ("R2 ours alone", round(float(1 - res.var() / sub.war.var()), 4)),
        ("R2 ours + recruiting", round(float(1 - res2.var() / sub.war.var()), 4)),
        ("", ""),
        ("READ", "Recruiting barely predicts what an unproven player does (r = .09), "
                 "and our model already leans on it slightly HARDER than reality "
                 "warrants (.19 vs .09). So the ~19% of EA's unproven-player opinion "
                 "that is recruiting is unlikely to be the useful part - and there is "
                 "no free win here without EA either."),
    ], columns=["test", "value"])


def position_table(d):
    g = d.dropna(subset=["ea_war"]).groupby("broad_group")
    z = lambda s: (s - s.mean()) / s.std()
    out = g.apply(lambda x: pd.Series({
        "players": len(x),
        "corr(EA, ours)": x.ea_war.corr(x.proj_war),
        "mean |z gap|": (z(x.ea_war) - z(x.proj_war)).abs().mean(),
    }), include_groups=False).reset_index()
    return out.sort_values("corr(EA, ours)")


def main():
    print("1. IS THERE A GAP? projection's own 2025 holdout, split by prior playing time")
    h = holdout_2025()
    gt = gap_table(h)
    print(gt.round(4).to_string(index=False))
    print()

    d = ea_join()
    print("2. WOULD EA EVEN CHANGE THE ANSWER? agreement by prior playing time (2026)")
    dt = disagreement_table(d)
    print(dt.round(3).to_string(index=False))
    print()

    print("3. IS EA'S VIEW OF AN UNPROVEN PLAYER JUST RECRUITING?")
    ft = freshman_test(d)
    print(ft.to_string(index=False))
    print()

    print("4. IS RECRUITING WORTH ANYTHING FOR THESE PLAYERS?")
    rw = recruiting_worth(h)
    print(rw.to_string(index=False))
    print()

    print("5. BY POSITION (2026, where both exist)")
    pt = position_table(d)
    print(pt.round(3).to_string(index=False))

    with pd.ExcelWriter(f"{HERE}/ea_gap_analysis.xlsx", engine="openpyxl") as xl:
        gt.to_excel(xl, sheet_name="1 where we are weak", index=False)
        dt.to_excel(xl, sheet_name="2 agreement by playing time", index=False)
        ft.to_excel(xl, sheet_name="3 is EA just recruiting", index=False)
        rw.to_excel(xl, sheet_name="4 is recruiting worth it", index=False)
        pt.to_excel(xl, sheet_name="5 by position", index=False)
        big = d.dropna(subset=["ea_war"]).copy()
        zz = lambda s: (s - s.mean()) / s.std()
        big["z_gap"] = zz(big.ea_war) - zz(big.proj_war)
        big = big.reindex(big.z_gap.abs().sort_values(ascending=False).index)
        big[["player", "team", "broad_group", "class", "depth", "snaps_2025",
             "war_2025", "proj_war", "ea_ovr", "ea_war", "z_gap"]].head(400).to_excel(
            xl, sheet_name="6 biggest disagreements", index=False)
    print(f"\n-> {HERE}/ea_gap_analysis.xlsx")


if __name__ == "__main__":
    main()
