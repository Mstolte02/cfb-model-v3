"""Should PFF, recruiting and WAR be one blended feature or three separate ones?

The model currently collapses them into a single `talent` column at fixed 38/38/25
weights, then hands that to the logistic alongside O and D. That imposes the weighting
instead of letting the model learn it, and it forbids any interaction - the model
cannot say "recruiting matters more when returning production is low", which is
exactly the kind of thing it should be able to discover.

The argument for blending is collinearity: the three correlate .54 to .68, so
separating them spends three coefficients on largely shared variance and the fitted
weights get unstable. The argument against is that a blend is a constraint the data
never asked for.

This tests it. Same harness, same folds, same everything downstream; the only change
is whether talent enters as one column or three.

Run: ./venv/bin/python -m scripts.talent_separate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import GAME_YEARS, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA, TALENT_BLEND, WAR_BLEND
from src.data import load, pff
from src import matchup as MU
from src import oppadj as OA
from src import model as M
from scripts.train import load_bundle, raw_returning
from scripts.talent_sweep import sources

BASE = ["O", "D", "fp_margin", "pythag", "talent", "returning"]


def assemble_with(talent, std, ret, games, pyth, ret_raw, od, extra=None):
    """Build the design, optionally appending extra per-team columns as features.

    `extra` is {name: {season: Series}}; each becomes a home-minus-away difference
    column exactly like the built-in features, so nothing else about the model
    changes.
    """
    b_o, b_d = MU.fit_talent_od_slopes(list(GAME_YEARS), std, talent, od_by_year=od)
    parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                        lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                        ret_raw_by_year=ret_raw, od_by_year=od)
    if not extra:
        return parts
    # The extra columns have to line up row-for-row with the design MU.build_year
    # produced, and that drops any game with a team missing from the frame or a tie.
    # Rebuilding the frame and reapplying the same filter is the only way to stay in
    # step; iterating the raw game list silently goes off by the dropped rows.
    out = {}
    for N, tup in parts.items():
        X, y, hf = tup[0], tup[1], tup[2]
        u = MU.uncertainty_u(ret_raw[N]) if N in ret_raw else None
        unc = (UNCERTAINTY_LAMBDA, b_o, b_d, u) if u is not None else None
        frame = MU.team_frame(N, std, pyth, talent, ret, uncertainty=unc, od_by_year=od)
        teams = set(frame.index)
        rows = [(g["home_team"], g["away_team"]) for _, g in games[N].iterrows()
                if g["home_team"] in teams and g["away_team"] in teams
                and g["home_points"] != g["away_points"]]
        assert len(rows) == len(y), f"{N}: {len(rows)} kept games vs {len(y)} design rows"
        cols = [X]
        for name, by_year in extra.items():
            s = by_year.get(N)
            v = np.zeros(len(rows))
            if s is not None:
                for i, (h, a) in enumerate(rows):
                    hv, av = s.get(h, np.nan), s.get(a, np.nan)
                    v[i] = 0.0 if (np.isnan(hv) or np.isnan(av)) else (hv - av)
            cols.append(v[:, None])
        out[N] = (np.hstack(cols), y, hf) + tuple(tup[3:])
    return out


def loso(parts, keep_idx):
    res = {"brier": [], "log_loss": [], "accuracy": []}
    for ty in GAME_YEARS:
        tr = [g for g in GAME_YEARS if g != ty and g in parts]
        if ty not in parts or not tr:
            continue
        Xtr = np.vstack([parts[g][0][:, keep_idx] for g in tr])
        ytr = np.concatenate([parts[g][1] for g in tr])
        hf = np.concatenate([parts[g][2] for g in tr])
        mdl, _ = M.train(Xtr, ytr, hf)
        ev = M.evaluate(mdl, parts[ty][0][:, keep_idx], parts[ty][1], parts[ty][2])
        for k in res:
            res[k].append(ev[k])
    return {k: float(np.mean(v)) for k, v in res.items()}


def main():
    load.require_key()
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    src, _ = sources()

    wp = (1 - WAR_BLEND) * TALENT_BLEND
    wc = (1 - WAR_BLEND) * (1 - TALENT_BLEND)
    blended = {N: wp * src["PFF"][N] + wc * src["CFBD"][N] + WAR_BLEND * src["WAR"][N]
               for N in src["CFBD"]}

    # active columns today: O, D, talent, returning (fp_margin and pythag retired)
    active = [BASE.index(c) for c in ("O", "D", "talent", "returning")]

    print("=" * 72)
    print("BLENDED vs SEPARATE talent signals, LOSO 2021-25")
    print("=" * 72)
    print(f"  {'design':<40}{'feat':>5}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}")

    p_blend = assemble_with(blended, std, ret, games, pyth, ret_raw, od)
    m = loso(p_blend, active)
    base = m["brier"]
    print(f"  {'blended talent (ships)':<40}{len(active):>5}{m['brier']:>9.4f}"
          f"{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}")

    # separate: keep `talent` as the PFF signal, add CFBD and WAR as their own columns
    extra = {"cfbd": src["CFBD"], "war": src["WAR"]}
    p_sep = assemble_with(src["PFF"], std, ret, games, pyth, ret_raw, od, extra=extra)
    idx_sep = active + [len(BASE), len(BASE) + 1]
    m = loso(p_sep, idx_sep)
    print(f"  {'separate: PFF + CFBD + WAR columns':<40}{len(idx_sep):>5}{m['brier']:>9.4f}"
          f"{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}   "
          f"{m['brier']-base:+.4f}")

    # blended talent PLUS the two others, so the model can adjust the blend it is given
    p_both = assemble_with(blended, std, ret, games, pyth, ret_raw, od, extra=extra)
    m = loso(p_both, idx_sep)
    print(f"  {'blend + CFBD + WAR as corrections':<40}{len(idx_sep):>5}{m['brier']:>9.4f}"
          f"{m['log_loss']:>10.4f}{m['accuracy']:>8.3f}   "
          f"{m['brier']-base:+.4f}")

    # and the coefficients the separate version actually learns
    tr = [g for g in GAME_YEARS if g in p_sep]
    Xtr = np.vstack([p_sep[g][0][:, idx_sep] for g in tr])
    ytr = np.concatenate([p_sep[g][1] for g in tr])
    hf = np.concatenate([p_sep[g][2] for g in tr])
    mdl, _ = M.train(Xtr, ytr, hf)
    print("\n  coefficients when the three are separate:")
    for nm, c in zip(["O", "D", "PFF talent", "returning", "CFBD recruiting", "WAR"],
                     mdl.coef):
        print(f"    {nm:<18}{c:+.4f}")
    tot = sum(abs(c) for nm, c in zip(
        ["O", "D", "PFF talent", "returning", "CFBD recruiting", "WAR"], mdl.coef)
        if nm in ("PFF talent", "CFBD recruiting", "WAR"))
    if tot:
        print("\n  implied talent split from the fitted coefficients:")
        for nm, c in zip(["O", "D", "PFF talent", "returning", "CFBD recruiting", "WAR"],
                         mdl.coef):
            if nm in ("PFF talent", "CFBD recruiting", "WAR"):
                print(f"    {nm:<18}{abs(c)/tot*100:5.1f}%   (blend uses "
                      f"{ {'PFF talent': wp, 'CFBD recruiting': wc, 'WAR': WAR_BLEND}[nm]*100:.0f}%)")


if __name__ == "__main__":
    main()
