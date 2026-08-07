"""Does this measure football ability, or PFF's grading model plus playing time?

Nothing in this project has ever been checked against a judgement made outside it.
Every validation is internal - WAR predicts wins, WAR predicts next season's WAR - and
all of those are satisfied by a number that faithfully reproduces PFF's opinion and a
snap count. That is the one hypothesis the internal tests cannot rule out, because PFF
grades and playing time are the inputs.

EA's CFB 27 ratings are an outside opinion. Whatever else is true of them, they were
not produced from PFF's grades: they come from a different organisation watching the
same football and assigning an `overall` per player. They are not ground truth - they
carry their own recruiting-pedigree and name-recognition biases, and a video game has
reasons to rate famous players highly - so this is a CONVERGENCE test, not an accuracy
one. Two estimates of the same thing built from different inputs should agree more than
either agrees with noise.

THE COMPARISON THAT MATTERS IS AGAINST SNAPS. A WAR that correlates .55 with EA is
impressive only if playing time alone does not, and playing time alone gets a long way:
EA rates starters above backups, and so does any volume-weighted statistic. So every
correlation here is reported beside the correlation of raw snaps with the same target,
and beside the partial correlation of WAR with EA holding snaps fixed. That last column
is the one that answers the question in the title.

Run: ../venv/bin/python external_validation.py
"""
import json
import os

import numpy as np
import pandas as pd

import artifacts
from build_roster_2026 import norm_name, PFF_TO_GROUP

HERE = os.path.dirname(os.path.abspath(__file__))
EA = f"{HERE}/ea/ea_cfb27.csv"

EA_TO_GROUP = {
    "QB": "QB", "HB": "RB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "LT": "OT", "RT": "OT", "OT": "OT",
    "LG": "IOL", "C": "IOL", "RG": "IOL",
    "LE": "EDGE", "RE": "EDGE", "DE": "EDGE", "EDGE": "EDGE",
    "DT": "DT", "NT": "DT",
    "LOLB": "LB", "MLB": "LB", "ROLB": "LB", "LB": "LB",
    "CB": "CB", "FS": "SAF", "SS": "SAF", "S": "SAF",
}
MIN_N = 40


def load_ea():
    d = pd.read_csv(EA)
    d = d[["player", "team", "position", "overall"]].dropna()
    d["group"] = d.position.map(EA_TO_GROUP)
    d["key"] = d.player.map(norm_name)
    # a name that is not unique within its team cannot be resolved, so it is refused
    n = d.groupby(["team", "key"]).overall.transform("size")
    return d[n == 1].copy()


def partial(x, y, z):
    """corr(x, y) holding z fixed."""
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 10 or z.std() == 0:
        return np.nan
    rxy, rxz, ryz = (np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1],
                     np.corrcoef(y, z)[0, 1])
    den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    return float((rxy - rxz * ryz) / den)


def table(df, value, volume, target, label):
    """Convergence with `target`, per position group and pooled."""
    rows = []
    for grp, g in df.groupby("group"):
        if len(g) < MIN_N:
            continue
        rows.append({
            "group": grp, "n": len(g),
            "r_value": float(np.corrcoef(g[value], g[target])[0, 1]),
            "r_volume": float(np.corrcoef(g[volume], g[target])[0, 1]),
            "partial": partial(g[value], g[target], g[volume]),
        })
    rows.append({
        "group": "ALL", "n": len(df),
        "r_value": float(np.corrcoef(df[value], df[target])[0, 1]),
        "r_volume": float(np.corrcoef(df[volume], df[target])[0, 1]),
        "partial": partial(df[value], df[target], df[volume]),
    })
    out = pd.DataFrame(rows).set_index("group")
    print(f"\n{label}")
    print(f"  {'group':<8}{'n':>6}{'r(WAR,EA)':>11}{'r(snaps,EA)':>13}"
          f"{'partial':>10}")
    for grp, r in out.iterrows():
        print(f"  {grp:<8}{int(r.n):>6}{r.r_value:>11.3f}{r.r_volume:>13.3f}"
              f"{r.partial:>10.3f}")
    return out


def main():
    ea = load_ea()
    print(f"EA CFB 27: {len(ea)} players with a resolvable name within their team")

    # ---- historical: 2025 WAR against an evaluation made after 2025 ----------
    war = pd.read_csv(f"{HERE}/{artifacts.PLAYER_WAR}")
    w25 = war[war.season == 2025].copy()
    w25["group"] = w25.position.map(PFF_TO_GROUP)
    w25["key"] = w25.player.map(norm_name)
    n = w25.groupby(["team", "key"]).war.transform("size")
    w25 = w25[n == 1]
    h = w25.merge(ea[["team", "key", "overall"]], on=["team", "key"], how="inner")
    h = h[h.group.notna() & (h.snaps > 0)]
    print(f"  matched to 2025 player-seasons: {len(h)}")
    hist = table(h, "war", "snaps", "overall",
                 "2025 WAR vs EA CFB 27 overall (EA's raters watched this season)")

    # ---- the projection, against the same outside opinion --------------------
    proj_path = f"{HERE}/projections_2026_final.csv"
    if not os.path.exists(proj_path):
        proj_path = f"{HERE}/projections_2026_v2.csv"
    p = pd.read_csv(proj_path)
    p["key"] = p.player.map(norm_name)
    p = p.rename(columns={"broad_group": "group"})
    p["prior_snaps"] = pd.to_numeric(p.get("snaps_2025"), errors="coerce").fillna(0.0)
    n = p.groupby(["team", "key"]).proj_war.transform("size")
    p = p[n == 1]
    j = p.merge(ea[["team", "key", "overall"]], on=["team", "key"], how="inner")
    j = j[j.group.notna()]
    print(f"\n2026 projection matched to EA: {len(j)} of {len(p)} slots")
    proj = table(j, "proj_war", "prior_snaps", "overall",
                 "2026 projected WAR vs EA CFB 27 overall (both preseason)")

    out = {
        "ea_players": int(len(ea)),
        "hist_matched": int(len(h)),
        "hist": hist.reset_index().to_dict("records"),
        "proj_matched": int(len(j)),
        "proj": proj.reset_index().to_dict("records"),
        "note": ("EA is an independent evaluation, not ground truth. The partial "
                 "column is WAR against EA holding playing time fixed, which is the "
                 "only column that separates measuring the player from measuring "
                 "his workload."),
    }
    json.dump(out, open(f"{HERE}/external_validation.json", "w"), indent=1)
    print("\n-> external_validation.json")


if __name__ == "__main__":
    main()
