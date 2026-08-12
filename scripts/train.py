"""Shared input builders plus the default v4 training entry point.

The legacy v3 trainer below is retained as ``legacy_main`` for reproducibility. The
normal ``python -m scripts.train`` command now publishes the strict reciprocal v4
model.

Legacy model: train the win-probability model using the MATCHUP-adjusted, team-level
ratings (offense vs opponent defense), with talent / returning production /
Pythagorean priors blended by the L2 logistic on game outcomes (target B).

Forward-looking + leakage-free; TEST_GAME_YEAR held out.
Run:  ./venv/bin/python -m scripts.train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import (GAME_YEARS, STAT_YEARS, TALENT_YEARS, RETURNING_YEARS,
                    PYTHAG_YEARS, TEST_GAME_YEAR, UNCERTAINTY_LAMBDA, OPP_ADJ_ALPHA,
                    TALENT_BLEND)
from src.data import load, pff
from src import strength as S
from src import projection as P
from src import matchup as MU
from src import oppadj as OA
from src import model as M


def blended_talent(cfbd_tal, pff_roster, w=TALENT_BLEND, war_w=None):
    """talent[N] = the PFF/CFBD blend, then WAR mixed in on top of it.

    WAR is a third axis rather than a substitute: it is only worth carrying because
    it adds to the PFF+CFBD mix, not because it beats either (see config.WAR_BLEND).
    Any team the WAR build has no roster for falls back to the blend, exactly as a
    team without PFF talent falls back to CFBD.
    """
    from config import WAR_BLEND
    war_w = WAR_BLEND if war_w is None else war_w

    out = {}
    for N, base in cfbd_tal.items():
        r = pff_roster.get(N)
        out[N] = base if r is None else w * r.reindex(base.index).fillna(base) + (1 - w) * base

    if war_w <= 0:
        return out

    # This used to be wrapped in `except Exception: print(warn); return out`, which
    # meant that when WAR_BLEND said a quarter of talent comes from the WAR build and
    # the build was absent, the run FINISHED - on PFF+CFBD only - and printed an
    # accuracy figure describing a model nobody had configured. A warning in a log is
    # not a substitute for the number being wrong. If the config asks for WAR talent,
    # a run without it is a failure.
    from src.data import war as warmod
    if not warmod.available():
        raise FileNotFoundError(
            f"WAR_BLEND={war_w} but the WAR build is absent ({warmod.PLAYER_WAR}). "
            f"Run war_model/build_hybrid.py, point WAR_DIR at an existing build, "
            f"or set WAR_BLEND=0 to train deliberately without it.")
    wt = warmod.talent_by_year({N: s.index for N, s in out.items()})
    n = 0
    for N, blend in out.items():
        v = wt.get(N)
        if v is None:
            continue
        out[N] = (1 - war_w) * blend + war_w * v.reindex(blend.index).fillna(blend)
        n += 1
    print(f"  [info] WAR talent blended at {war_w:.2f} for {n} seasons.")
    return out


def raw_returning():
    """Raw returning-production fractions (not z-scored) for the uncertainty index."""
    return {y: load.returning_production(y).set_index("team")["rp"]
            for y in RETURNING_YEARS}


def projection_returning_raw(ret_raw=None):
    """Raw returning production entering PROJECTION_YEAR, curated CSV then proxy.

    Pulled out of build_projection_frame because export_viz needs the same series to
    build the per-team shrink u for the what-if block, and a second copy of this
    fallback chain would be free to drift out of step with this one - which would
    show the page a shrink the model never applied.
    """
    import pandas as pd
    from config import PROJECTION_YEAR, PROJECTION_RETURNING_FALLBACK_YEAR, ROOT
    rp_csv = ROOT / "data" / f"returning_{PROJECTION_YEAR}.csv"
    if rp_csv.exists():
        rp = pd.read_csv(rp_csv).set_index("team")["ret_prod"]
        print(f"  [info] {PROJECTION_YEAR} returning production from {rp_csv.name} "
              f"({len(rp)} teams).")
        return rp
    ret_raw = raw_returning() if ret_raw is None else ret_raw
    print(f"  [warn] {PROJECTION_YEAR} returning unavailable; using "
          f"{PROJECTION_RETURNING_FALLBACK_YEAR} proxy.")
    return ret_raw[PROJECTION_RETURNING_FALLBACK_YEAR]


def build_projection_frame(talent_blend=None, unc_lambda=None, return_params=False,
                           return_parts=False):
    """Entering-PROJECTION_YEAR team frame (O/D/pythag/talent/returning, uncertainty
    applied), with 2026 returning from the curated CSV and talent proxied from the
    latest composite. Shared by scripts/rank.py and scripts/spreads.py.

    return_parts hands back every INTERMEDIATE the frame was assembled from - the
    CFBD composite before the PFF roster is blended in, the roster talent itself, the
    WAR z, and the opponent-adjusted O/D before the uncertainty shrink. The testing
    tab's calculator shows the derivation from a roster's WAR to a power rating, and
    it has to show the numbers the model actually used. Recomputing them in export_viz
    from the same inputs would work right up until one of these steps changed and the
    other copy did not, so the parts come out of the function that does the work.

    talent_blend / unc_lambda override the config defaults — used by the
    ROSTER-WEIGHTED variant (blend=0.7, lam=1.0), which leans harder on the 2026
    two-deep PFF talent and regresses low-continuity teams fully toward it
    (LOSO cost ~0.4% Brier, same accuracy; see README v2 notes)."""
    import pandas as pd
    from config import (PROJECTION_YEAR, PROJECTION_TALENT_FALLBACK_YEAR,
                        PROJECTION_RETURNING_FALLBACK_YEAR, ROOT, UNCERTAINTY_LAMBDA,
                        OPP_ADJ_ALPHA)
    from src.projection import _z
    if talent_blend is None:
        talent_blend = TALENT_BLEND
    if unc_lambda is None:
        unc_lambda = UNCERTAINTY_LAMBDA

    std, talent, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()

    if PROJECTION_YEAR not in ret:
        rp = projection_returning_raw(ret_raw)
        ret[PROJECTION_YEAR], ret_raw[PROJECTION_YEAR] = _z(rp), rp

    tal_csv = ROOT / "data" / f"talent_{PROJECTION_YEAR}.csv"
    if PROJECTION_YEAR not in talent and tal_csv.exists():
        td = pd.read_csv(tal_csv).set_index("team")["talent"]
        talent[PROJECTION_YEAR] = _z(td)
        print(f"  [info] {PROJECTION_YEAR} talent from {tal_csv.name} ({len(td)} teams).")
    elif PROJECTION_YEAR not in talent:
        talent[PROJECTION_YEAR] = talent[PROJECTION_TALENT_FALLBACK_YEAR]
        print(f"  [warn] {PROJECTION_YEAR} talent unavailable; using "
              f"{PROJECTION_TALENT_FALLBACK_YEAR} composite proxy (r~0.97 stable).")

    # Blend in roster-aware PFF talent (historical + 2026 from the Ourlads two-deep).
    # QBs use opponent-adjusted CFBD WAR (rescaled to PFF scale) instead of the PFF
    # grade; all other positions keep PFF (CFBD can't value OL/coverage).
    from src import qbwar
    from config import ARTIFACTS as _ART
    # The QB talent override now comes from the WAR build rather than the parallel
    # CFBD-only regression in artifacts/qb_values.csv - one valuation of one thing.
    # See qbwar.build_qb_grades; QB_GRADES=cfbd restores the old source.
    qb_grades = qbwar.build_qb_grades(2025)
    print(f"  [info] QB grades for {len(qb_grades)} QBs from the WAR build "
          f"(replacing PFF at QB).")

    # The projection year's roster talent is NOT optional. Without it PROJECTION_YEAR
    # silently kept whatever build_roster_talent() happened to leave in the dict -
    # in practice the prior season's roster - and the projection was then published as
    # if it described next year's two-deep.
    pff_roster = pff.build_roster_talent()
    pff_roster[PROJECTION_YEAR] = pff.build_2026_roster_talent(
        qb_grades=qb_grades)[PROJECTION_YEAR]
    print(f"  [info] {PROJECTION_YEAR} roster talent from two-deep "
          f"({len(pff_roster[PROJECTION_YEAR])} teams).")
    # captured before the blend overwrites it - this is the CFBD recruiting composite
    # on its own, which is one of the three things that make up talent
    cfbd_only = talent[PROJECTION_YEAR].copy()
    talent = blended_talent(talent, pff_roster, w=talent_blend)

    # Service academies (Air Force, Navy) have no 247 recruiting composite and
    # were silently dropped from the frame. Give any team with returning
    # production but no talent a low-percentile composite (blended with its PFF
    # roster talent when available) so it keeps a rating.
    tal = talent[PROJECTION_YEAR]
    missing = ret[PROJECTION_YEAR].index.difference(tal.index)
    talent_floor, fallback_teams = None, []
    if len(missing):
        pr = pff_roster.get(PROJECTION_YEAR)
        floor = float(tal.quantile(0.10))
        vals = {t: (talent_blend * float(pr[t]) + (1 - talent_blend) * floor
                    if pr is not None and t in pr.index else floor)
                for t in missing}
        talent[PROJECTION_YEAR] = pd.concat([tal, pd.Series(vals)])
        talent_floor, fallback_teams = floor, sorted(missing)
        print(f"  [info] talent fallback applied for {sorted(missing)}")

    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)
    train_years = [g for g in GAME_YEARS if g != TEST_GAME_YEAR]
    b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent, od_by_year=od)
    unc = (unc_lambda, b_o, b_d, MU.uncertainty_u(ret_raw[PROJECTION_YEAR]))
    frame = MU.team_frame(PROJECTION_YEAR, std, pyth, talent, ret,
                          uncertainty=unc, od_by_year=od)
    if return_parts:
        prior = PROJECTION_YEAR - 1
        # blended_talent() has already raised if WAR_BLEND wants a build that is not
        # there, so by here `available()` is either true or WAR_BLEND is 0 and the
        # derivation genuinely has no WAR column to show.
        from src.data import war as warmod
        war_z = (warmod.talent_by_year({PROJECTION_YEAR: cfbd_only.index})
                 .get(PROJECTION_YEAR) if warmod.available() else None)
        parts = {
            "cfbd_talent": cfbd_only,
            "pff_roster": pff_roster.get(PROJECTION_YEAR),
            "war_z": war_z,
            "talent": talent[PROJECTION_YEAR],
            "O_raw": od[prior]["O"], "D_raw": od[prior]["D"],
            "u": unc[3], "b_o": b_o, "b_d": b_d, "lam": unc_lambda,
            "talent_blend": talent_blend,
            # The service-academy path. These teams have no recruiting composite, so
            # they never reach blended_talent at all: their talent is
            # talent_blend*roster + (1-talent_blend)*floor, with NO WAR term. A
            # calculator that showed them going through the normal three-source blend
            # would be showing a derivation the model did not perform.
            "talent_floor": talent_floor, "fallback_teams": fallback_teams,
        }
        return frame, parts
    # return_params: the talent slopes and the shrinkage, for callers that need to
    # know how talent reaches O and D. The playoff simulator does - talent is a
    # retired column now, so perturbing it has to go through the O/D coefficients.
    return (frame, b_o, b_d, unc_lambda) if return_params else frame


def load_bundle():
    """Load every per-year input dict the pipeline needs (all standardized)."""
    std_by_year, talent_by_year = S.load_seasons(load, STAT_YEARS, TALENT_YEARS)
    returning_by_year = {y: P._z(load.returning_production(y).set_index("team")["rp"])
                         for y in RETURNING_YEARS}
    games_by_year = {y: load.games(y) for y in set(GAME_YEARS) | set(PYTHAG_YEARS)}
    pythag_by_year = P.build_pythag(games_by_year)
    return std_by_year, talent_by_year, returning_by_year, games_by_year, pythag_by_year


def legacy_main():
    load.require_key()
    print("Pulling stats, talent, returning production, games from CFBD ...")
    std, cfbd_tal, ret, games, pyth = load_bundle()
    ret_raw = raw_returning()
    talent = blended_talent(cfbd_tal, pff.build_roster_talent())  # PFF roster + CFBD
    od = OA.build_od_by_year(std, games, OPP_ADJ_ALPHA)   # SOS-adjusted O/D
    train_years = [g for g in GAME_YEARS if g != TEST_GAME_YEAR]
    b_o, b_d = MU.fit_talent_od_slopes(train_years, std, talent, od_by_year=od)

    parts = MU.assemble(GAME_YEARS, std, pyth, talent, ret, games,
                        lam=UNCERTAINTY_LAMBDA, b_o=b_o, b_d=b_d,
                        ret_raw_by_year=ret_raw, od_by_year=od)
    Xtr = np.vstack([parts[g][0] for g in train_years])
    ytr = np.concatenate([parts[g][1] for g in train_years])
    hftr = np.concatenate([parts[g][2] for g in train_years])
    mgtr = np.concatenate([parts[g][3] for g in train_years])

    print(f"Training on {train_years} ({len(ytr)} games), "
          f"testing on {TEST_GAME_YEAR} ({len(parts[TEST_GAME_YEAR][1])} games) ...")
    cfb_model, _ = M.train(Xtr, ytr, hftr, feature_names=MU.MATCHUP_COLS,
                           margins=mgtr)
    print(f"Chosen L2 C = {cfb_model.C}")

    print("\n=== TRAIN (in-sample) ===")
    _pm(M.evaluate(cfb_model, Xtr, ytr, hftr))
    print("=== TEST (held-out season) ===")
    _pm(M.evaluate(cfb_model, *parts[TEST_GAME_YEAR]))

    print("\n=== Learned weights (logit scale) ===")
    for name, w in zip(MU.MATCHUP_COLS, cfb_model.coef):
        print(f"  {name:12} {w:+.3f}")
    print(f"  {'home_field':12} {cfb_model.hfa_coef:+.3f}")

    cfb_model.save()
    print("\nSaved model -> artifacts/model.json")


def _pm(m):
    print(f"  games={m['n_games']}  accuracy={m['accuracy']}  "
          f"Brier={m['brier']}  log_loss={m['log_loss']}")


def main():
    """Publish v4; retained here so the repository's established command improves."""
    from scripts.train_v4 import main as train_v4_main
    return train_v4_main()


if __name__ == "__main__":
    main()
