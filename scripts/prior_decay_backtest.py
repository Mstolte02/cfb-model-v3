"""Forward test of expanded WAR, decaying priors, and EWMA in-season form.

This experiment deliberately does not use the repository's fixed 0.001 adoption
threshold.  Every arm is fixed in advance, regularized, and evaluated in outer
season folds.  Uncertainty is reported after the point estimates; it is not a veto.

The current-season advanced-stat standardization is recomputed using games strictly
before the prediction week.  This is stricter than the older refit experiment, which
cut the rows by week but standardized them using the completed season distribution.

Run after building an expanded WAR tree and setting PFF_WAR_DIR, or pass:

    python -m scripts.prior_decay_backtest --new-war-dir <directory>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS, GAME_YEARS, PFF_API_DIR
from scripts import v4_backtest as BT
from scripts.train_v4 import build_frames, build_inputs
from src import talent_sources as TS
from src import v4 as V4
from src.data import load, player_production


OUT_JSON = ARTIFACTS / "prior_decay_backtest.json"
OUT_CSV = ARTIFACTS / "prior_decay_backtest_predictions.csv"
OUT_ENSEMBLE_CSV = ARTIFACTS / "prior_decay_ensemble_predictions.csv"
OFF = ["off_ppa", "off_pass_ppa", "off_rush_ppa", "off_success_rate",
       "off_explosiveness"]
DEF = ["def_ppa", "def_success_rate", "def_explosiveness"]
SPEC_FEATURES = {
    "war_only_new": ["war_projected"],
    "full_current_war": TS.REDUCED,
    "full_new_war": TS.REDUCED,
    "full_new_war_curvature": [
        *TS.REDUCED, "talent_sq", "returning_sq", "talent_x_returning"],
    "full_new_war_production": [*TS.REDUCED, "player_prod_war"],
    "full_new_war_curvature_production": [
        *TS.REDUCED, "player_prod_war", "talent_sq", "returning_sq",
        "talent_x_returning"],
}
CURRENT_HALFLIVES = (2.0, 4.0, 8.0, math.inf)
PRIOR_HALFLIVES = (2.0, 4.0, 6.0, 8.0, 12.0, 24.0)
C_GRID = (.01, .03, .1, .3, 1.0)
K_GRID = (.05, .10, .15, .20, .25, .30)
DEFAULT_CURRENT_HALFLIFE = 4.0
DEFAULT_PRIOR_HALFLIFE = 6.0
DEFAULT_C = .1


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = float(values.std(ddof=0))
    return (values - values.mean()) / sd if sd > 1e-12 else values * 0.0


def load_expanded_war(directory: Path, indices: dict[int, pd.Index]) -> dict[int, pd.Series]:
    path = Path(directory) / "preseason_team_war.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"expanded WAR projection not found at {path}; run "
            "war_model/preseason_team_projection.py in the expanded WAR build")
    raw = pd.read_csv(path)
    required = {"season", "team", "projected_war"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    out = {}
    for year, index in indices.items():
        rows = raw[raw.season == int(year)].drop_duplicates("team")
        if rows.empty:
            continue
        values = rows.set_index("team").projected_war.reindex(index)
        out[int(year)] = _z(values)
    return out


def raw_game_stats(year: int) -> pd.DataFrame:
    """Unstandardized team-game inputs; all transforms happen at the week cutoff."""
    frame = load.game_advanced(year)
    fbs = set(load.team_stats(year)["team"])
    # CFBD numbers every bowl and playoff game "week 1" of the postseason, so an
    # unfiltered week cutoff would hand a week-2 prediction that team's bowl game
    # from four months later.  Only regular-season rows are pregame information.
    keep = (frame.season_type.eq("regular") &
            frame.team.isin(fbs) & frame.opponent.isin(fbs))
    frame = frame.loc[keep].dropna(subset=OFF + DEF).copy()
    return frame[["week", "team", "opponent", *OFF, *DEF]]


def _ewma(values: np.ndarray, halflife: float) -> float:
    if len(values) == 0:
        return float("nan")
    if not np.isfinite(halflife):
        return float(np.mean(values))
    ages = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.power(.5, ages / float(halflife))
    return float(np.average(values, weights=weights))


def todate_od(raw: pd.DataFrame, week: float, halflife: float) -> pd.DataFrame | None:
    """EWMA O/D built only from games before ``week``, including its scaling."""
    past = raw[raw.week < week].copy()
    if past.empty:
        return None
    for column in [*OFF, *DEF]:
        past[column] = _z(past[column])
    past["O_game"] = past[OFF].mean(axis=1)
    past["D_game"] = -past[DEF].mean(axis=1)
    past = past.sort_values(["team", "week"], kind="stable")
    rows = []
    for team, games in past.groupby("team", sort=False):
        rows.append({"team": team, "n": len(games),
                     "O": _ewma(games.O_game.to_numpy(float), halflife),
                     "D": _ewma(games.D_game.to_numpy(float), halflife)})
    summary = pd.DataFrame(rows).set_index("team")

    # Same schedule correction as the production O/D builder.  It is intentionally
    # recomputed at the cutoff and therefore cannot see future opponents or results.
    schedule = past.groupby("team").opponent.apply(list).to_dict()
    O, D = summary.O.to_dict(), summary.D.to_dict()
    for _ in range(2):
        new_o = {team: O[team] + .5 * np.mean([D.get(opp, 0.0)
                                               for opp in schedule[team]])
                 for team in O}
        new_d = {team: D[team] + .5 * np.mean([O.get(opp, 0.0)
                                               for opp in schedule[team]])
                 for team in D}
        O, D = new_o, new_d
    summary["O"] = _z(pd.Series(O)).reindex(summary.index)
    summary["D"] = _z(pd.Series(D)).reindex(summary.index)
    return summary


def week_states(raw: pd.DataFrame, weeks, halflife: float) -> dict[float, pd.DataFrame | None]:
    return {float(week): todate_od(raw, float(week), halflife)
            for week in sorted(pd.Series(weeks).dropna().unique())}


def season_design(model: V4.ReciprocalTeamModel, frame: pd.DataFrame, part,
                  raw: pd.DataFrame, current_halflife: float,
                  prior_halflife: float, k: float) -> pd.DataFrame:
    """Pregame rows for one season, with a smooth prior and weekly score walk."""
    X, y, home_flag, margins, meta = part
    initial = {team: model.team_logit_strength(frame, team) for team in frame.index}
    ratings = dict(initial)
    states = week_states(raw, meta.week, current_halflife)
    order = meta.assign(_row=np.arange(len(meta))).sort_values(["week", "_row"])
    rows = [None] * len(meta)
    games_seen = {team: 0 for team in frame.index}
    sigma = float(model.margin_sigma)

    for week, slate in order.groupby("week", sort=True, dropna=False):
        od = states.get(float(week))
        changes = {}
        for _, game in slate.iterrows():
            i = int(game._row)
            home, away = game.home_team, game.away_team
            hfa = float(home_flag[i])
            elo_logit = ratings[home] - ratings[away] + model.hfa_coef * hfa
            # Separate new score evidence from the preseason level.  Feeding the
            # full Elo level beside prior_decay duplicates the week-0 prior and
            # creates avoidable multicollinearity.
            elo_change = ((ratings[home] - initial[home]) -
                          (ratings[away] - initial[away]))
            wh = .5 ** (games_seen[home] / prior_halflife)
            wa = .5 ** (games_seen[away] / prior_halflife)
            prior_decay = wh * initial[home] - wa * initial[away]
            have = (od is not None and home in od.index and away in od.index and
                    od.at[home, "n"] >= 2 and od.at[away, "n"] >= 2)
            d_o = float(od.at[home, "O"] - od.at[away, "O"]) if have else 0.0
            d_d = float(od.at[home, "D"] - od.at[away, "D"]) if have else 0.0
            static_p = float(model.win_prob(X[i], hfa))
            rows[i] = {"prior_level": initial[home] - initial[away],
                       "prior_decay": prior_decay, "elo": elo_logit,
                       "elo_change": elo_change,
                       "dO": d_o, "dD": d_d, "hfa": hfa,
                       "have_current": int(have), "preseason": static_p,
                       "elo_probability": float(expit(elo_logit)),
                       "y": float(y[i]), "week": game.week,
                       "home_team": home, "away_team": away,
                       "neutral_site": bool(game.neutral_site)}

            expected = sigma * float(ndtri(np.clip(expit(elo_logit), .01, .99)))
            score = np.clip((float(margins[i]) - expected) / sigma, -2.5, 2.5)
            delta = float(k * score)
            changes[home] = changes.get(home, 0.0) + delta
            changes[away] = changes.get(away, 0.0) - delta
        for team, delta in changes.items():
            ratings[team] += delta
        for _, game in slate.iterrows():
            games_seen[game.home_team] += 1
            games_seen[game.away_team] += 1
    return pd.DataFrame(rows)


ARM_COLUMNS = {
    "ewma": ["prior_decay", "dO", "dD", "hfa"],
    "ewma_elo": ["prior_decay", "elo_change", "dO", "dD", "hfa"],
    # Control for the decay hypothesis: preserve the preseason level, add only
    # orthogonal score innovation and current EWMA form.  This avoids duplicating
    # the prior inside the full Elo level while asking whether decay itself helps.
    "ewma_elo_nodecay": ["prior_level", "elo_change", "dO", "dD", "hfa"],
}
REPORT_ARMS = ("preseason", "elo", *ARM_COLUMNS)


def fit_stack(train: pd.DataFrame, columns: list[str], c: float):
    scaler = StandardScaler(with_mean=False)
    x = scaler.fit_transform(train[columns].to_numpy(float))
    model = LogisticRegression(C=float(c), fit_intercept=False, max_iter=3000)
    model.fit(x, train.y.to_numpy(float))
    return scaler, model


def stack_predict(fitted, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    scaler, model = fitted
    return model.predict_proba(
        scaler.transform(frame[columns].to_numpy(float)))[:, 1]


def matrix_diagnostics(x: np.ndarray, columns: list[str]) -> dict:
    x = np.asarray(x, float)
    scale = x.std(axis=0, ddof=0)
    z = x / np.where(scale > 1e-10, scale, 1.0)
    singular = np.linalg.svd(z, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-10 else math.inf
    corr = np.corrcoef(z, rowvar=False)
    try:
        vif = np.diag(np.linalg.pinv(corr))
        max_vif = float(np.max(vif))
    except np.linalg.LinAlgError:
        max_vif = math.inf
    pairs = {}
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            pairs[f"{left}__{columns[j]}"] = float(corr[i, j])
    return {"condition_number": condition, "max_vif": max_vif,
            "pairwise_correlations": pairs}


def design_diagnostics(frame: pd.DataFrame, columns: list[str]) -> dict:
    return matrix_diagnostics(frame[columns].to_numpy(float), columns)


def tune_k(parts, frames, pool, names, knobs) -> tuple[float, dict]:
    if len(pool) < 2:
        return .20, {}
    losses = {k: [] for k in K_GRID}
    for i in range(1, len(pool)):
        test, train = pool[i], pool[:i]
        model, _, _ = BT.fit_predict(parts, train, test, names, knobs)
        for k in K_GRID:
            pred, _ = BT.dynamic_predictions(model, frames[test], parts[test], k, 1.0)
            losses[k].extend((pred - parts[test][1]) ** 2)
    scores = {str(k): float(np.mean(value)) for k, value in losses.items()}
    return min(losses, key=lambda value: np.mean(losses[value])), scores


def meta_forward_score(designs: dict[int, pd.DataFrame], columns: list[str],
                       c: float) -> float:
    years = sorted(designs)
    losses = []
    for index in range(1, len(years)):
        train = pd.concat([designs[y] for y in years[:index]], ignore_index=True)
        test = designs[years[index]]
        pred = stack_predict(fit_stack(train, columns, c), test, columns)
        losses.extend((pred - test.y.to_numpy(float)) ** 2)
    return float(np.mean(losses)) if losses else math.inf


def select_decay(model_contexts: dict[int, tuple], raw_stats: dict[int, pd.DataFrame],
                 arm: str, k: float) -> tuple[tuple[float, float, float], dict]:
    columns = ARM_COLUMNS[arm]
    years = sorted(model_contexts)
    if len(years) < 2:
        return (DEFAULT_CURRENT_HALFLIFE, DEFAULT_PRIOR_HALFLIFE, DEFAULT_C), {}
    trace = {}
    best = None
    best_score = math.inf
    prior_grid = (DEFAULT_PRIOR_HALFLIFE,) if "prior_decay" not in columns \
        else PRIOR_HALFLIVES
    for current_half in CURRENT_HALFLIVES:
        for prior_half in prior_grid:
            designs = {
                year: season_design(model, frame, part, raw_stats[year],
                                    current_half, prior_half, k)
                for year, (model, frame, part) in model_contexts.items()
            }
            for c in C_GRID:
                score = meta_forward_score(designs, columns, c)
                key = (current_half, prior_half, c)
                trace[",".join("inf" if not np.isfinite(v) else str(v)
                               for v in key)] = score
                if score < best_score:
                    best, best_score = key, score
    assert best is not None
    return best, trace


def bootstrap(frame: pd.DataFrame, left: str, right: str,
              draws: int = 5000, seed: int = 20260922) -> dict:
    data = frame.dropna(subset=[left, right]).copy()
    data["loss"] = (data[left] - data.y) ** 2 - (data[right] - data.y) ** 2
    blocks = [group.loss.to_numpy(float) for _, group in
              data.groupby(["season", "week"], sort=True, dropna=False)]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, len(blocks), len(blocks))
        samples[draw] = np.concatenate([blocks[index] for index in chosen]).mean()
    return {"difference": float(data.loss.mean()),
            "ci95": [float(v) for v in np.quantile(samples, [.025, .975])],
            "probability_left_better": float(np.mean(samples < 0)),
            "blocks": len(blocks), "draws": draws}


def main(new_war_dir: Path) -> dict:
    std, talent, returning, games, od, pff_lag, war_lag, _ = build_inputs(
        include_projection=False)
    current_frames = build_frames(std, talent, returning, od, pff_lag, war_lag,
                                  GAME_YEARS)
    indices = {year: frame.index for year, frame in current_frames.items()}
    expanded = load_expanded_war(new_war_dir, indices)
    new_frames = {year: frame.copy() for year, frame in current_frames.items()}
    production = player_production.by_year(indices)
    for year, frame in new_frames.items():
        values = expanded.get(year, pd.Series(dtype=float)).reindex(frame.index)
        frame["war_projected"] = values.fillna(0.0)
        frame.attrs.update(current_frames[year].attrs)
        frame.attrs["expanded_war_coverage"] = float(values.notna().mean())
        components = production.get(year, pd.DataFrame(index=frame.index)).reindex(
            frame.index)
        frame["player_prod_war"] = components.get(
            "player_prod_war", pd.Series(index=frame.index, dtype=float)).fillna(0.0)
        frame["talent_sq"] = frame.talent ** 2
        frame["returning_sq"] = frame.returning ** 2
        frame["talent_x_returning"] = frame.talent * frame.returning

    frames_by_spec = {
        "war_only_new": new_frames,
        "full_current_war": current_frames,
        "full_new_war": new_frames,
        "full_new_war_curvature": new_frames,
        "full_new_war_production": new_frames,
        "full_new_war_curvature_production": new_frames,
    }
    parts_by_spec = {
        spec: V4.assemble(GAME_YEARS, frames_by_spec[spec], games, features)
        for spec, features in SPEC_FEATURES.items()
    }
    raw_stats = {year: raw_game_stats(year) for year in GAME_YEARS}

    folds, output_rows = [], []
    for test in (2023, 2024, 2025):
        pool = [year for year in GAME_YEARS if year < test]
        for spec, names in SPEC_FEATURES.items():
            frames = frames_by_spec[spec]
            parts = parts_by_spec[spec]
            knobs, knob_trace = BT.tune(parts, pool, names)
            k, k_trace = tune_k(parts, frames, pool, names, knobs)

            # Cross-fitted earlier-season rows train and tune the in-season stack.
            contexts = {}
            for index in range(1, len(pool)):
                validation, train = pool[index], pool[:index]
                model, _, _ = BT.fit_predict(parts, train, validation, names, knobs)
                contexts[validation] = (model, frames[validation], parts[validation])

            selected = {}
            traces = {}
            for arm in ARM_COLUMNS:
                selected[arm], traces[arm] = select_decay(
                    contexts, raw_stats, arm, k)

            model, static, _ = BT.fit_predict(parts, pool, test, names, knobs)
            test_designs = {}
            train_designs = {}
            stack_coefficients = {}
            predictions = {"preseason": static}
            for arm, (current_half, prior_half, c) in selected.items():
                train_designs[arm] = {
                    year: season_design(old_model, old_frame, old_part,
                                        raw_stats[year], current_half,
                                        prior_half, k)
                    for year, (old_model, old_frame, old_part) in contexts.items()
                }
                train = pd.concat(list(train_designs[arm].values()), ignore_index=True)
                test_design = season_design(model, frames[test], parts[test],
                                            raw_stats[test], current_half,
                                            prior_half, k)
                test_designs[arm] = test_design
                fitted = fit_stack(train, ARM_COLUMNS[arm], c)
                predictions[arm] = stack_predict(fitted, test_design,
                                                 ARM_COLUMNS[arm])
                stack_coefficients[arm] = {
                    column: float(value) for column, value in zip(
                        ARM_COLUMNS[arm], fitted[1].coef_[0])}
            # Elo is identical across the two selected designs because only the
            # EWMA/prior half-lives differ; take it from either one.
            predictions["elo"] = test_designs["ewma"].elo_probability.to_numpy()

            meta = parts[test][4].copy()
            meta["season"], meta["y"], meta["spec"] = test, parts[test][1], spec
            for arm, values in predictions.items():
                meta[arm] = values
            output_rows.append(meta)
            fold = {"season": test, "spec": spec, "features": names,
                    "model_knobs": knobs, "model_tuning": knob_trace,
                    "elo_k": k, "elo_k_tuning": k_trace,
                    "decay_selection": {
                        arm: {"current_halflife": value[0],
                              "prior_halflife": value[1], "C": value[2],
                              "trace": traces[arm]}
                        for arm, value in selected.items()},
                    "metrics": {arm: BT.metric(parts[test][1], values)
                                for arm, values in predictions.items()},
                    "preseason_multicollinearity": matrix_diagnostics(
                        BT.stack(parts, pool)[0], names),
                    "stack_coefficients_standardized": stack_coefficients,
                    "multicollinearity": {
                        arm: design_diagnostics(
                            pd.concat(list(train_designs[arm].values()),
                                      ignore_index=True), ARM_COLUMNS[arm])
                        for arm in ARM_COLUMNS}}
            folds.append(fold)
            print(f"{test} {spec:<17} " + "  ".join(
                f"{arm}={fold['metrics'][arm]['brier']:.5f}"
                for arm in REPORT_ARMS),
                flush=True)

    predictions = pd.concat(output_rows, ignore_index=True)
    # Written before any summary, so a failure below cannot cost the folds.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT_CSV, index=False)
    pooled = {}
    for spec, rows in predictions.groupby("spec"):
        pooled[spec] = {arm: BT.metric(rows.y, rows[arm])
                        for arm in REPORT_ARMS}
        pooled[spec]["bootstrap_ewma_vs_elo"] = bootstrap(rows, "ewma", "elo")
        pooled[spec]["bootstrap_ewma_elo_vs_elo"] = bootstrap(
            rows, "ewma_elo", "elo")
        pooled[spec]["bootstrap_ewma_elo_nodecay_vs_elo"] = bootstrap(
            rows, "ewma_elo_nodecay", "elo")
        pooled[spec]["by_period"] = {
            label: {arm: BT.metric(period.y, period[arm])
                    for arm in REPORT_ARMS}
            for label, period in {
                "weeks_1_4": rows[rows.week <= 4],
                "weeks_5_9": rows[(rows.week >= 5) & (rows.week <= 9)],
                "weeks_10_plus": rows[rows.week >= 10],
            }.items() if len(period)
        }

    wide = predictions.pivot_table(
        index=["season", "week", "home_team", "away_team", "neutral_site", "y"],
        columns="spec", values=list(REPORT_ARMS)
    ).reset_index()
    wide.columns = ["_".join(str(v) for v in column if str(v))
                    if isinstance(column, tuple) else str(column)
                    for column in wide.columns]
    ensemble_members = [
        "ewma_elo_nodecay_full_new_war",
        "ewma_elo_nodecay_full_new_war_curvature",
        "ewma_elo_nodecay_full_new_war_production",
        "ewma_elo_nodecay_full_new_war_curvature_production",
    ]
    # Equal weighting is fixed, needs no additional fitted parameter, and prevents
    # a noisy outer fold from forcing a winner-take-all choice among near-ties.
    wide["expanded_equal_ensemble"] = wide[ensemble_members].mean(axis=1)
    compare = {}
    for arm in REPORT_ARMS:
        new = f"{arm}_full_new_war"
        current = f"{arm}_full_current_war"
        alone = f"{arm}_war_only_new"
        compare[arm] = {
            "new_full_vs_current_full": bootstrap(wide, new, current),
            "new_full_vs_new_war_only": bootstrap(wide, new, alone),
        }

    primary_comparisons = {
        "expanded_full_nodecay_vs_current_full_elo": bootstrap(
            wide, "ewma_elo_nodecay_full_new_war", "elo_full_current_war"),
        "expanded_full_nodecay_vs_expanded_full_elo": bootstrap(
            wide, "ewma_elo_nodecay_full_new_war", "elo_full_new_war"),
        "expanded_full_nodecay_vs_expanded_war_only_nodecay": bootstrap(
            wide, "ewma_elo_nodecay_full_new_war",
            "ewma_elo_nodecay_war_only_new"),
        "expanded_equal_ensemble_vs_current_full_elo": bootstrap(
            wide, "expanded_equal_ensemble", "elo_full_current_war"),
    }

    cfbd = BT.attach_cfbd_elo(wide.copy())
    aligned = cfbd.dropna(subset=["p_cfbd_elo"])
    external_benchmark = {
        "expanded_equal_ensemble": BT.metric(
            aligned.y, aligned.expanded_equal_ensemble),
        "cfbd_pregame_elo": BT.metric(aligned.y, aligned.p_cfbd_elo),
        "paired": bootstrap(
            aligned, "expanded_equal_ensemble", "p_cfbd_elo"),
    }

    result = {
        "contract": "outer season forward test; all current-season transforms use only prior weeks",
        "selection_policy": "minimum forward loss among regularized arms; no p-value or fixed materiality veto",
        "new_war_dir": str(Path(new_war_dir)),
        "spec_features": SPEC_FEATURES, "folds": folds,
        "pooled_2023_2025": pooled, "cross_spec_comparisons": compare,
        "primary_comparisons": primary_comparisons,
        "expanded_equal_ensemble": {
            "members": ensemble_members,
            "metrics": BT.metric(wide.y, wide.expanded_equal_ensemble),
            "by_season": {
                str(int(year)): BT.metric(rows.y, rows.expanded_equal_ensemble)
                for year, rows in wide.groupby("season")},
        },
        "external_benchmark": external_benchmark,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    wide.to_csv(OUT_ENSEMBLE_CSV, index=False)
    print("\nPooled 2023-25:")
    for spec, arms in pooled.items():
        print(f"  {spec}")
        for arm in REPORT_ARMS:
            print(f"    {arm:<10} {arms[arm]['brier']:.6f}")
    print(f"-> {OUT_JSON}\n-> {OUT_CSV}\n-> {OUT_ENSEMBLE_CSV}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-war-dir", type=Path,
                        default=PFF_API_DIR / "war_all_facets")
    args = parser.parse_args()
    main(args.new_war_dir)
