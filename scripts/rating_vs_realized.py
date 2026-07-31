"""Do the entering-season O/D ratings survive contact with the season?

`scripts/roster_vs_results.py` already sweeps the same two knobs - the WAR share of
the talent blend, and how far a team regresses off last season - but it scores them
on game outcomes, and that surface is flat: every combination lands within .001
Brier of every other. Flat because a game's win probability is dominated by the
talent coefficient, which barely moves as weight shifts between the O/D composites
and the roster baseline underneath them.

The complaint that prompted this is not about win probability. It is that a team can
carry a top-10 DEFENCE RATING off one strong season while its roster says average -
Wake Forest entering 2026 is 7th on defence with the 83rd offence and a talent z of
+0.10. That is a claim about the rating, so score the rating:

    entering-N D  (prior season, opponent-adjusted, regressed toward the roster)
        vs
    realised-N D  (what the defence actually produced in season N, opponent-adjusted)

Both sides are the same quantity on the same scale, one season apart, so a straight
correlation answers "how much of this rating is real". Sweeping (WAR share, lambda)
against THAT tells us how far to trust last season - and unlike Brier it has room to
discriminate, because moving weight onto the roster changes the rating directly.

Leakage: b_o/b_d are refit per fold on the other seasons, exactly as in the LOSO
scripts. Season N's own results appear only on the right-hand side.

Run: ./venv/bin/python -m scripts.rating_vs_realized
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (GAME_YEARS, OPP_ADJ_ALPHA, TALENT_BLEND, WAR_BLEND,
                    UNCERTAINTY_LAMBDA, ARTIFACTS)
from src.data import load
from src import matchup as MU
from src import oppadj as OA
from scripts.train import load_bundle, raw_returning
from scripts.talent_sweep import sources

WAR_GRID = [0.0, 0.25, 0.40, 0.55, 0.70, 1.0]
LAM_GRID = [0.0, 0.35, 0.55, 0.70, 0.85, 1.00]
SHIP_LAM = UNCERTAINTY_LAMBDA


def entering_frames(talent, lam, std, ret, games, pyth, ret_raw, od,
                    use_returning=None):
    """{N: entering-N team frame}, slopes refit per fold on the other seasons.

    use_returning=True forces the old u = (1 - returning production) weighting
    regardless of config, so the two can be compared at matched shrinkage.
    """
    out = {}
    for N in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != N]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        if N not in ret_raw:
            continue
        u = ((1.0 - ret_raw[N]).clip(lower=0, upper=1) if use_returning
             else MU.uncertainty_u(ret_raw[N]))
        f = MU.team_frame(N, std, pyth, talent, ret,
                          uncertainty=(lam, b_o, b_d, u), od_by_year=od)
        if f is not None:
            out[N] = f
    return out


def score(frames, od):
    """Correlation and RMSE of entering-N O/D against realised-N O/D."""
    res = {"O_r": [], "D_r": [], "O_rmse": [], "D_rmse": [], "n": []}
    for N, f in frames.items():
        if N not in od:
            continue
        real = od[N]
        idx = f.index.intersection(real.index)
        if len(idx) < 30:
            continue
        for side in ("O", "D"):
            a, b = f.loc[idx, side].astype(float), real.loc[idx, side].astype(float)
            res[f"{side}_r"].append(float(np.corrcoef(a, b)[0, 1]))
            res[f"{side}_rmse"].append(float(np.sqrt(np.mean((a - b) ** 2))))
        res["n"].append(len(idx))
    out = {k: float(np.mean(v)) for k, v in res.items() if k != "n"}
    out["n"] = int(np.sum(res["n"]))
    # One headline number: the mean of the two correlations, so a setting cannot
    # win by trading defence accuracy for offence accuracy.
    out["mean_r"] = (out["O_r"] + out["D_r"]) / 2.0
    return out


def talent_at(src, war_share):
    """PFF and recruiting keep their 50/50 split of whatever WAR leaves."""
    rest = 1.0 - war_share
    wp, wc = rest * TALENT_BLEND, rest * (1 - TALENT_BLEND)
    return {N: wp * src["PFF"][N] + wc * src["CFBD"][N] + war_share * src["WAR"][N]
            for N in src["CFBD"]}


def misses(frames, od, k=12):
    """Teams whose entering defence rating most overstated the season that followed."""
    rows = []
    for N, f in frames.items():
        if N not in od:
            continue
        idx = f.index.intersection(od[N].index)
        for t in idx:
            rows.append({"season": N, "team": t,
                         "D_entering": float(f.loc[t, "D"]),
                         "D_realised": float(od[N].loc[t, "D"]),
                         "talent": float(f.loc[t, "talent"])})
    d = pd.DataFrame(rows)
    d["error"] = d.D_entering - d.D_realised
    return d


def main():
    load.require_key()
    src, (std, cfbd_tal, ret, games, pyth) = sources()
    ret_raw = raw_returning()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    args = (std, ret, games, pyth, ret_raw, od)

    print("=" * 78)
    print("ENTERING-SEASON RATING vs WHAT THE UNIT ACTUALLY PRODUCED")
    print(f"folds: {GAME_YEARS}   both sides opponent-adjusted (alpha={OPP_ADJ_ALPHA})")
    print("=" * 78)

    rows, best = [], None
    header = "  WAR   " + "".join(f"  lam={l:.2f}" for l in LAM_GRID)
    print("\nmean of (offence r, defence r) - higher is better")
    print(header)
    print("-" * len(header))
    for ws in WAR_GRID:
        talent = talent_at(src, ws)
        line = f"  {ws:.2f}  "
        for lam in LAM_GRID:
            m = score(entering_frames(talent, lam, *args), od)
            rows.append({"war": ws, "lam": lam, **m})
            line += f"  {m['mean_r']:>7.4f}"
            if best is None or m["mean_r"] > best["mean_r"]:
                best = {"war": ws, "lam": lam, **m}
        print(line, flush=True)

    r = pd.DataFrame(rows)
    ship = r[(r.war == WAR_BLEND) & (r.lam == SHIP_LAM)].iloc[0]
    print(f"\nships today:  WAR {WAR_BLEND:.2f}, lambda {SHIP_LAM:.2f} -> "
          f"mean r {ship.mean_r:.4f}  (O {ship.O_r:.4f}, D {ship.D_r:.4f})")
    print(f"best of grid: WAR {best['war']:.2f}, lambda {best['lam']:.2f} -> "
          f"mean r {best['mean_r']:.4f}  (O {best['O_r']:.4f}, D {best['D_r']:.4f})")
    print(f"difference:   {best['mean_r'] - ship.mean_r:+.4f}")

    print("\ndefence only (the side the complaint is about):")
    dbest = r.sort_values("D_r", ascending=False).iloc[0]
    print(f"  best D r: WAR {dbest.war:.2f}, lambda {dbest.lam:.2f} -> {dbest.D_r:.4f} "
          f"(ships {ship.D_r:.4f})")
    print("  marginal effect of lambda at that WAR share:")
    for _, x in r[r.war == dbest.war].sort_values("lam").iterrows():
        print(f"    lambda {x.lam:.2f} -> D r {x.D_r:.4f}  O r {x.O_r:.4f}")

    print("\ntop 8 combinations by mean r:")
    print(r.sort_values("mean_r", ascending=False).head(8)
           .round(4).to_string(index=False))

    # ---- does returning production identify WHICH teams should regress? ------
    # The two are compared at the SAME lambda, so the returning-weighted column
    # carries a mean shrink of lambda x 0.451 against the flat column's lambda.
    # That handicaps the flat side at every row and it still wins, which is the
    # point: the variation returning production adds is noise, not signal.
    talent = talent_at(src, WAR_BLEND)
    print(f"\n{'=' * 78}")
    print("IS THE RETURNING-PRODUCTION WEIGHTING EARNING ITS PLACE?")
    print("=" * 78)
    print(f"{'lambda':>7} {'flat D r':>10} {'flat O r':>10} "
          f"{'ret-wtd D r':>12} {'ret-wtd O r':>12}")
    for lam in [0.25, 0.50, 0.75, 1.00]:
        a = score(entering_frames(talent, lam, *args, use_returning=False), od)
        b = score(entering_frames(talent, lam, *args, use_returning=True), od)
        print(f"{lam:>7.2f} {a['D_r']:>10.4f} {a['O_r']:>10.4f} "
              f"{b['D_r']:>12.4f} {b['O_r']:>12.4f}")

    # ---- who does the shipping setting get wrong, and in which direction? ----
    talent = talent_at(src, WAR_BLEND)
    d = misses(entering_frames(talent, SHIP_LAM, *args), od)
    print(f"\n{'=' * 78}\nWHERE THE SHIPPING RATING MISSES ON DEFENCE  (n={len(d)} team-seasons)")
    print("=" * 78)
    print(f"  bias {d.error.mean():+.3f}   mean |error| {d.error.abs().mean():.3f}")
    print("\n  most overstated (rated high, produced low):")
    print(d.nlargest(10, "error").round(2).to_string(index=False))
    print("\n  is a high entering rating systematically overstated?")
    top = d[d.D_entering > 1.0]
    print(f"    entering D > 1.0  (n={len(top)}): mean error {top.error.mean():+.3f}")
    lo = d[d.D_entering < -1.0]
    print(f"    entering D < -1.0 (n={len(lo)}): mean error {lo.error.mean():+.3f}")

    r.to_csv(ARTIFACTS / "rating_vs_realized.csv", index=False)
    d.to_csv(ARTIFACTS / "rating_vs_realized_errors.csv", index=False)
    json.dump(best, open(ARTIFACTS / "rating_vs_realized.json", "w"), indent=1)
    print(f"\n-> {ARTIFACTS / 'rating_vs_realized.csv'}")


if __name__ == "__main__":
    main()
