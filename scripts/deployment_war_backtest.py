"""Does where and how a player is deployed change what his performance is worth?

Mark's rule (5 October 2026): observed role over assumed role, performance within a
deployment over alignment alone, scarcity learned not declared, versatility only if
the data says so. This tests, position by position, whether a deployment measure
changes the value of the production WAR a player already earns.

For a position group g and a deployment measure x (wide share, slot share, gap
alignment, box share, ...), the team-season term is

    T = sum over the team's g players of WAR_i * (x_i - mean x)

so a team gets credit only for WAR earned in a deployment, never for the deployment
itself: a mediocre outside receiver adds nothing because his WAR is small. Model A
explains a side's efficiency from the production WAR totals of that side's groups;
model B adds T. If B predicts better out of sample, the value of WAR depends on the
deployment; the ratio of the two coefficients says by how much.

Outcome: that side's season PPA from CFBD (offence for offensive groups, minus PPA
allowed for defensive groups), seasons with both PFF and CFBD season stats. Every
comparison is leave-one-season-out. A second outcome, schedule-adjusted win
percentage, is reported as a check.

    python -m scripts.deployment_war_backtest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import ARTIFACTS, PFF_DIR  # noqa: E402
from src.war_window import GROUP  # noqa: E402

OUT = ARTIFACTS / "deployment_war_backtest.json"
OFF = ["QB", "RB", "WR", "TE", "OT", "IOL"]
DEF = ["DT", "EDGE", "LB", "CB", "SAF"]


def _share(num, den):
    return (num / den.replace(0, np.nan)).clip(0, 1)


def _entropy(frame):
    p = frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log(p)).sum(axis=1, min_count=1)
    return h / np.log(frame.shape[1])


def deployment(season: int) -> pd.DataFrame:
    """Per (season, player_id): every deployment measure, from the PFF season files."""
    rc = pd.read_csv(Path(PFF_DIR) / f"receiving_{season}.csv", low_memory=False)
    df = pd.read_csv(Path(PFF_DIR) / f"defense_{season}.csv", low_memory=False)
    bk = pd.read_csv(Path(PFF_DIR) / f"blocking_{season}.csv", low_memory=False)
    out = []
    a = rc[["player_id", "wide_snaps", "slot_snaps", "inline_snaps"]].fillna(0)
    tot = a[["wide_snaps", "slot_snaps", "inline_snaps"]].sum(axis=1)
    out.append(pd.DataFrame({
        "player_id": rc.player_id, "wide": _share(a.wide_snaps, tot),
        "slot": _share(a.slot_snaps, tot), "inline": _share(a.inline_snaps, tot),
        "route_rate": rc.route_rate / 100, "rcv_block_rate": rc.pass_block_rate / 100,
        "rcv_versatility": _entropy(a[["wide_snaps", "slot_snaps", "inline_snaps"]])}))
    d = df.fillna(0)
    dl = d.snap_counts_dl
    out.append(pd.DataFrame({
        "player_id": df.player_id,
        "a_gap": _share(d.snap_counts_dl_a_gap, dl), "b_gap": _share(d.snap_counts_dl_b_gap, dl),
        "over_t": _share(d.snap_counts_dl_over_t, dl), "outside_t": _share(d.snap_counts_dl_outside_t, dl),
        "box": _share(d.snap_counts_box, d.snap_counts_defense),
        "dslot": _share(d.snap_counts_slot, d.snap_counts_defense),
        "deep": _share(d.snap_counts_fs, d.snap_counts_defense),
        "on_line": _share(dl, d.snap_counts_defense),
        "cb_slot": _share(d.snap_counts_slot, d.snap_counts_slot + d.snap_counts_corner),
        "rush_share": _share(d.snap_counts_pass_rush, d.snap_counts_pass_rush + d.snap_counts_coverage),
        "def_versatility": _entropy(d[["snap_counts_box", "snap_counts_slot", "snap_counts_fs",
                                       "snap_counts_corner", "snap_counts_dl"]])}))
    b = bk.fillna(0)
    spots = b[["snap_counts_lt", "snap_counts_lg", "snap_counts_ce", "snap_counts_rg",
               "snap_counts_rt"]]
    out.append(pd.DataFrame({
        "player_id": bk.player_id,
        "left_tackle": _share(b.snap_counts_lt, b.snap_counts_lt + b.snap_counts_rt),
        "center": _share(b.snap_counts_ce, b.snap_counts_lg + b.snap_counts_ce + b.snap_counts_rg),
        "pass_block_share": _share(b.snap_counts_pass_block,
                                   b.snap_counts_pass_block + b.snap_counts_run_block),
        "ol_versatility": _entropy(spots)}))
    m = out[0]
    for o in out[1:]:
        m = m.merge(o, on="player_id", how="outer")
    m = m.groupby("player_id", as_index=False).first()
    m["season"] = season
    return m


# Measures tested per group. Versatility is tested like any other measure.
TESTS = {
    "WR": ["wide", "slot", "rcv_versatility"],
    "TE": ["inline", "slot", "wide", "route_rate", "rcv_block_rate", "rcv_versatility"],
    "RB": ["route_rate", "rcv_block_rate", "slot", "wide"],
    "OT": ["left_tackle", "pass_block_share", "ol_versatility"],
    "IOL": ["center", "pass_block_share", "ol_versatility"],
    "DT": ["a_gap", "b_gap", "over_t", "rush_share"],
    "EDGE": ["outside_t", "over_t", "rush_share", "on_line", "def_versatility"],
    "LB": ["box", "dslot", "on_line", "rush_share", "def_versatility"],
    "CB": ["cb_slot", "box", "def_versatility"],
    "SAF": ["deep", "box", "dslot", "def_versatility"],
}


def outcomes() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data" / "raw").glob("advanced_*.json")):
        season = int(f.stem.split("_")[1])
        for r in json.loads(f.read_text()):
            rows.append({"season": season, "team": r["team"],
                         "off": (r.get("offense") or {}).get("ppa"),
                         "def": -((r.get("defense") or {}).get("ppa") or np.nan)})
    adv = pd.DataFrame(rows)
    rec = pd.read_csv(ROOT / "war_model" / "records.csv")[["season", "team", "adj_win_pct"]]
    return adv.merge(rec, on=["season", "team"], how="outer")


def environment(seasons) -> pd.DataFrame:
    """Team pass rate: the offensive environment, kept apart from player deployment.

    Pass-block share and route rate are mostly a team's play-calling. Without this
    control they look like player effects (OT pass-block share -1.1% before it)."""
    from src.inseason_war import canonical_team
    tm = json.load(open(ROOT / "war_model" / "team_map.json"))
    rows = []
    for s in seasons:
        b = pd.read_csv(Path(PFF_DIR) / f"blocking_{s}.csv", low_memory=False)
        b = b[b.position.isin(["T", "G", "C"])]
        g = b.groupby("team_name")[["snap_counts_pass_block", "snap_counts_run_block"]].sum()
        for t, r in g.iterrows():
            tot = r.snap_counts_pass_block + r.snap_counts_run_block
            if tot:
                rows.append({"season": s, "team": canonical_team([t], tm),
                             "pass_rate": r.snap_counts_pass_block / tot})
    return pd.DataFrame(rows).dropna().groupby(["season", "team"]).pass_rate.mean()


def loso(X: np.ndarray, y: np.ndarray, seasons: np.ndarray) -> float:
    err = []
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        A = np.c_[np.ones(tr.sum()), X[tr]]
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        err.append((np.c_[np.ones(te.sum()), X[te]] @ b - y[te]) ** 2)
    return float(np.sqrt(np.concatenate(err).mean()))


def main():
    war = pd.read_csv(ROOT / "war_model" / "hybrid_player_war.csv", dtype={"player_id": str})
    war["group"] = war.position.map(GROUP)
    war = war[war.group.notna()]
    seasons = sorted(war.season.unique())
    dep = pd.concat([deployment(s) for s in seasons], ignore_index=True)
    dep["player_id"] = dep.player_id.astype(str)
    w = war.merge(dep, on=["season", "player_id"], how="left")
    totals = w.groupby(["season", "team", "group"]).war.sum().unstack(fill_value=0.0)
    y = outcomes().set_index(["season", "team"]).join(environment(seasons), how="left")
    y["pass_rate"] = y.pass_rate.fillna(y.pass_rate.mean())
    report = {"seasons": [int(s) for s in seasons], "tests": {}}
    for g, xs in TESTS.items():
        side = "off" if g in OFF else "def"
        groups = OFF if side == "off" else DEF
        for x in xs:
            d = w[(w.group == g) & w[x].notna()]
            mean = np.average(d[x], weights=d.snaps.clip(lower=1))
            T = (d.war * (d[x] - mean)).groupby([d.season, d.team]).sum().rename("T")
            frame = totals[groups].join(T, how="left").fillna({"T": 0.0}).join(y, how="inner")
            res = {}
            for target in (side, "adj_win_pct"):
                f = frame.dropna(subset=[target])
                Xa = f[groups + ["pass_rate"]].to_numpy()
                Xb = np.c_[Xa, f["T"].to_numpy()]
                ss = f.index.get_level_values(0).to_numpy()
                ra, rb = loso(Xa, f[target].to_numpy(), ss), loso(Xb, f[target].to_numpy(), ss)
                A = np.c_[np.ones(len(f)), Xb]
                coef, *_ = np.linalg.lstsq(A, f[target].to_numpy(), rcond=None)
                resid = f[target].to_numpy() - A @ coef
                cov = np.linalg.pinv(A.T @ A) * resid.var(ddof=A.shape[1])
                gi = 1 + groups.index(g)
                res[target] = {"rmse_change_pct": round(100 * (rb / ra - 1), 3),
                               "n": int(len(f)),
                               "t_stat": round(float(coef[-1] / np.sqrt(cov[-1, -1])), 2),
                               # value of one WAR at x one unit above the group mean,
                               # relative to the group's average WAR
                               "relative_slope": round(float(coef[-1] / coef[gi]), 3)
                               if coef[gi] else None}
            res["mean_x"] = round(float(mean), 3)
            report["tests"][f"{g}:{x}"] = res
            print(f"{g:5}{x:18} side {res[side]['rmse_change_pct']:+7.3f}% t={res[side]['t_stat']:+5.2f}"
                  f" rel={res[side]['relative_slope']}   wins {res['adj_win_pct']['rmse_change_pct']:+7.3f}%"
                  f" t={res['adj_win_pct']['t_stat']:+5.2f}", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
