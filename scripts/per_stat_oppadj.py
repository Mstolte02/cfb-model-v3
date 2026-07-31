"""Should the opponent adjustment work per stat instead of per composite?

src/oppadj.adjust() corrects each team's OFFENSIVE composite against the opposing
DEFENSIVE composite. That is one correction covering five different measurements, so
a team's red-zone defence gets credited for the pass rush of the offences it faced.

Three of the ten features are true pairs - the same event scored from both sides -
and could instead be corrected against their own counterpart:

    off_havoc         <-> def_havoc
    off_rz_td         <-> def_rz_td
    off_press_allowed <-> def_press

Two more can be paired approximately (off_rush_ppa with def_line_yds, both run game;
off_success_rate with def_ppa, both overall efficiency). That is a guess, so it is
tested as its own arm rather than assumed.

SCORING. The primary target is LOSO game prediction, because it is neutral between
the methods: the games are the same games whatever the ratings are built from. The
rating-vs-realised correlation is reported too, but as a SECONDARY number and with a
caveat - it scores a method's entering rating against a realised rating built the
same way, so a method that merely produces smoother ratings can self-correlate better
without being better. Where the two disagree, the game target wins.

Run: ./venv/bin/python -m scripts.per_stat_oppadj
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import GAME_YEARS, OPP_ADJ_ALPHA, UNCERTAINTY_LAMBDA, WAR_BLEND
from src.data import load
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import raw_returning
from scripts.talent_sweep import sources
from scripts.rating_vs_realized import score, talent_at


def loso(std, talent, ret, games, pyth, ret_raw, od):
    res = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != ty]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                            lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                            ret_raw_by_year=ret_raw, od_by_year=od)
        if ty not in parts:
            continue
        av = [g for g in tr if g in parts]
        mdl, _ = M.train(np.vstack([parts[g][0] for g in av]),
                         np.concatenate([parts[g][1] for g in av]),
                         np.concatenate([parts[g][2] for g in av]))
        ev = M.evaluate(mdl, *parts[ty])
        for k in res:
            res[k].append(ev[k])
    return {k: float(np.mean(v)) for k, v in res.items()}


def rating_r(std, talent, ret, pyth, ret_raw, od):
    frames = {}
    for N in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != N]
        b_o, b_d = MU.fit_talent_od_slopes(tr, std, talent, od_by_year=od)
        if N not in ret_raw:
            continue
        f = MU.team_frame(N, std, pyth, talent, ret, od_by_year=od,
                          uncertainty=(UNCERTAINTY_LAMBDA, b_o, b_d,
                                       MU.uncertainty_u(ret_raw[N])))
        if f is not None:
            frames[N] = f
    return score(frames, od)


def main():
    load.require_key()
    src, (std, cfbd_tal, ret, games, pyth) = sources()
    ret_raw = raw_returning()
    talent = talent_at(src, WAR_BLEND)
    a = OPP_ADJ_ALPHA

    arms = {
        "composite (ships)": OA.build_od_by_year(std, games, a),
        "per-stat, 3 exact pairs": OA.build_od_by_year_per_stat(
            std, games, a, pairs=OA.EXACT_PAIRS),
        "per-stat, +2 loose pairs": OA.build_od_by_year_per_stat(
            std, games, a, pairs=OA.LOOSE_PAIRS),
    }

    print("=" * 84)
    print("PER-STAT vs PER-COMPOSITE OPPONENT ADJUSTMENT")
    print(f"alpha={a}, lambda={UNCERTAINTY_LAMBDA}, WAR share={WAR_BLEND}, "
          f"LOSO folds {GAME_YEARS}")
    print("=" * 84)
    print(f"{'arm':>26} {'Brier':>8} {'LogLoss':>9} {'Acc':>7} | "
          f"{'D r':>7} {'O r':>7}  (r is secondary - see docstring)")
    print("-" * 84)
    rows = []
    for name, od in arms.items():
        g = loso(std, talent, ret, games, pyth, ret_raw, od)
        r = rating_r(std, talent, ret, pyth, ret_raw, od)
        rows.append({"arm": name, **g, "D_r": r["D_r"], "O_r": r["O_r"]})
        print(f"{name:>26} {g['brier']:>8.4f} {g['log_loss']:>9.4f} "
              f"{g['accuracy']:>7.3f} | {r['D_r']:>7.4f} {r['O_r']:>7.4f}", flush=True)

    d = pd.DataFrame(rows)
    base = d[d.arm == "composite (ships)"].iloc[0]
    print()
    for _, x in d[d.arm != "composite (ships)"].iterrows():
        db = x.brier - base.brier
        print(f"  {x.arm:>26}: Brier {db:+.4f}  "
              + ("BETTER" if db < -0.0005 else
                 "worse" if db > 0.0005 else "within noise"))

    # How much do the two methods even disagree? If they produce nearly the same
    # ratings there is nothing to choose between them and the composite stays on
    # grounds of simplicity.
    print("\nhow different are the ratings, really? (2025, per-stat exact vs composite)")
    y = max(y for y in arms["composite (ships)"] if y in arms["per-stat, 3 exact pairs"])
    c, p = arms["composite (ships)"][y], arms["per-stat, 3 exact pairs"][y]
    idx = c.index.intersection(p.index)
    for side in ("O", "D"):
        rr = np.corrcoef(c.loc[idx, side], p.loc[idx, side])[0, 1]
        mv = (c.loc[idx, side].rank(ascending=False)
              - p.loc[idx, side].rank(ascending=False)).abs()
        print(f"  {side}: r = {rr:.4f}   mean |rank move| = {mv.mean():.1f}   "
              f"max = {int(mv.max())}")

    home_away(std, games, y)


def home_away(std, games, y):
    """The other proposed refinement: should the adjustment split home from away?

    The home effect on the composite inputs is real and large - measured on 13,726
    team-games of CFBD per-game advanced stats, playing at home is worth about half a
    team standard deviation. But a SEASON rating only inherits that bias to the extent
    that teams differ in how many home games they play, and in college football they
    barely do.
    """
    g = games[y]
    O0, D0 = MU.od_ratings(std[y])
    tset = set(O0.index)
    h = {t: 0 for t in tset}
    n = {t: 0 for t in tset}
    for r in g.itertuples():
        neutral = bool(getattr(r, "neutral_site", False))
        for t, is_home in ((r.home_team, not neutral), (r.away_team, False)):
            if t in tset:
                n[t] += 1
                h[t] += 1 if is_home else 0
    share = pd.Series({t: h[t] / max(n[t], 1) for t in tset})

    # Measured by scripts/per_stat_oppadj (see docstring): within-team home-minus-away
    # is +0.086 off_ppa against a between-team sd of 0.158 -> 0.51 team-sd.
    HOME_EDGE_SD = 0.51
    bias = (share - share.mean()) * HOME_EDGE_SD
    adj = OA.adjust(std[y], OA.build_schedule(g), OPP_ADJ_ALPHA)
    opp_move = (adj.D - D0.reindex(adj.index)).std()

    print(f"\nhome/away split ({y})")
    print(f"  home-game share: mean {share.mean():.3f}, sd {share.std():.3f}")
    print(f"  implied rating bias: sd {bias.std():.4f} team-sd")
    print(f"  the OPPONENT adjustment, for scale: sd {opp_move:.3f} team-sd "
          f"({opp_move / bias.std():.0f}x larger)")
    new_d = adj.D - bias.reindex(adj.index)
    mv = (adj.D.rank(ascending=False) - new_d.rank(ascending=False)).abs()
    print(f"  correcting for it: r = {np.corrcoef(adj.D, new_d)[0, 1]:.4f}, "
          f"mean |rank move| {mv.mean():.1f}")
    print("  -> everyone plays ~6 home and ~6 away, so the effect cancels in a "
          "season aggregate.")


if __name__ == "__main__":
    main()
