"""Leakage-safe CFB player production forecasts and team components.

This adapts the modelling contract from ``Mstolte02/nfl-prop-models`` to the
season-level data available in this repository.  Six familiar player markets are
forecast from information available before the target season.  Candidate models are
selected on the immediately preceding usable season and the selected model is refit
on all earlier seasons before each forward holdout.

The forecasts are evaluated in two downstream channels:

* production features are added to the existing ex-ante player WAR projection;
* projected production is aggregated directly into offense/defense team components.

The script intentionally does not claim a betting backtest.  The repository has
season totals, not timestamped historical CFB player-prop lines and prices.

Run: ``python -m war_model.player_production_forecast``
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import PFF_DIR, require

import artifacts
from build_recruiting import load_recruits
from build_roster_2026 import PFF_TO_GROUP, norm_name
from facets import YEARS as WAR_YEARS
from project_2026_v2 import (
    FEATURES as WAR_FEATURES,
    build_history,
    build_population,
    fit as fit_war,
    load_rosters,
    make_training,
    slot_counts,
)


OUT_COMPONENTS = HERE / "preseason_player_components.csv"
OUT_METRICS = HERE / "player_production_metrics.json"
# 2020 is intentionally absent from the WAR build because its conference-only
# schedules are incomparable. A season-N player frame needs N-1 history, so 2021 is
# absent as well. Earlier cross-fitted forecasts are retained to give the downstream
# WAR channel clean training rows before its common 2022-25 evaluation window.
FORECAST_YEARS = (2017, 2018, 2019, 2022, 2023, 2024, 2025)


@dataclass(frozen=True)
class Market:
    source: str
    column: str
    groups: tuple[str, ...]
    thresholds: tuple[float, ...]
    kind: str = "continuous"
    target_scale: int = 1


MARKETS: dict[str, Market] = {
    "passing_yards": Market("passing", "yards", ("QB",),
                            (1000.5, 2000.5, 3000.5)),
    "passing_touchdowns": Market("passing", "touchdowns", ("QB",),
                                 (5.5, 15.5, 25.5), "count"),
    "rushing_yards": Market("rushing", "yards", ("QB", "RB", "WR"),
                            (250.5, 500.5, 1000.5)),
    "receiving_yards": Market("receiving", "yards", ("RB", "WR", "TE"),
                              (250.5, 500.5, 1000.5)),
    "receptions": Market("receiving", "receptions", ("RB", "WR", "TE"),
                         (20.5, 40.5, 60.5), "count"),
    "defensive_sacks": Market("defense", "sacks",
                              ("DT", "EDGE", "LB", "CB", "SAF"),
                              (1.25, 3.25, 6.25), "count", 2),
}


def _z(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    return (series - series.mean()) / sd if sd > 1e-12 else series * 0.0


def _corr(left, right) -> float:
    x, y = np.asarray(left, float), np.asarray(right, float)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_market_history() -> pd.DataFrame:
    """One row per PFF player/team/season with the six realized targets."""
    require(PFF_DIR, "the PFF exports", "PFF_DIR")
    team_map = json.loads((HERE / "team_map.json").read_text())
    rows: list[pd.DataFrame] = []
    metadata: list[pd.DataFrame] = []
    years = sorted(set(WAR_YEARS))
    for name, spec in MARKETS.items():
        pieces = []
        for season in years:
            path = PFF_DIR / f"{spec.source}_{season}.csv"
            if not path.exists():
                continue
            raw = pd.read_csv(path, usecols=lambda c: c in {
                "player_id", "player", "position", "team_name", spec.column})
            if spec.column not in raw:
                continue
            part = raw.rename(columns={spec.column: name, "team_name": "team_raw"})
            part["season"] = season
            part["team"] = part.team_raw.map(team_map)
            part["group"] = part.position.map(PFF_TO_GROUP)
            part["key"] = part.player.map(norm_name)
            part[name] = pd.to_numeric(part[name], errors="coerce").fillna(0.0)
            pieces.append(part.dropna(subset=["team", "group"]))
        if not pieces:
            raise FileNotFoundError(f"no PFF rows found for {name}")
        d = pd.concat(pieces, ignore_index=True)
        # PFF can emit duplicate category rows for a player/team. Season totals add
        # across legitimate multi-team spells but not duplicate export records.
        keys = ["season", "player_id", "team", "group"]
        metadata.append(d[keys + ["player"]])
        d = d.groupby(keys, as_index=False)[name].max()
        rows.append(d)

    keys = ["season", "player_id", "team", "group"]
    out = rows[0]
    for frame in rows[1:]:
        out = out.merge(frame, on=keys, how="outer")
    names = (pd.concat(metadata, ignore_index=True)
               .sort_values("player").drop_duplicates(keys))
    out = out.merge(names, on=keys, how="left")
    out["key"] = out.player.map(norm_name)
    for name in MARKETS:
        out[name] = out[name].fillna(0.0)
    return out


def build_base_training() -> tuple[pd.DataFrame, dict[str, int], list[int]]:
    """Reuse the existing roster/identity/transfer path for the player population."""
    player_war = pd.read_csv(HERE / artifacts.PLAYER_WAR)
    ratings = pd.read_csv(HERE / artifacts.TEAM_RATINGS)
    records = pd.read_csv(HERE / "records.csv")
    recruits = load_recruits()
    roster_2026 = pd.read_csv(HERE / "roster_2026.csv")
    slots, starter_slots = slot_counts(roster_2026)

    history = build_history(player_war)
    fbs = set(records.team.unique())
    rosters = load_rosters(sorted(set(WAR_YEARS) | set(FORECAST_YEARS)), fbs)
    population = build_population(history, rosters)
    valid_targets = [year for year in WAR_YEARS
                     if year - 1 in set(WAR_YEARS)]
    training = make_training(population, history, ratings, recruits, rosters,
                             starter_slots, valid_targets)
    groups = sorted(history.group.dropna().unique())
    training["group_code"] = training.group.map(
        {group: index for index, group in enumerate(groups)})
    training["share_lag1"] = training.share_lag1.fillna(0.0)
    return training, slots, valid_targets


def attach_market_targets(base: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach current targets and three strictly prior-season market lags."""
    out = base.copy()
    # CFBD roster JSON and PFF CSV parsing disagree on the physical dtype of the
    # same identifier (string versus integer). Normalizing the representation is
    # not identity matching; it simply preserves the stable PFF id across sources.
    out["player_id"] = pd.to_numeric(out.player_id, errors="coerce").astype(
        "Int64").astype("string")
    history = history.copy()
    history["player_id"] = pd.to_numeric(
        history.player_id, errors="coerce").astype("Int64").astype("string")
    current = history.groupby(["season", "player_id", "team"], as_index=False)[
        list(MARKETS)].sum()
    current_cols = ["season", "player_id", "team", *MARKETS]
    out = out.merge(current[current_cols], on=["season", "player_id", "team"],
                    how="left")
    for name in MARKETS:
        out[name] = out[name].fillna(0.0)

    # Name fallback is allowed only when the normalized name resolves to one stable
    # PFF id over the entire history. Aggregate first so transfers and multi-team
    # spells carry the player's full prior-season production instead of whichever
    # team row happened to sort first.
    ids_per_name = history.groupby("key").player_id.nunique(dropna=True)
    safe_names = set(ids_per_name[ids_per_name == 1].index)
    refused = int((~history.key.isin(safe_names)).sum())
    by_id_history = history.groupby(["season", "player_id"], as_index=False)[
        list(MARKETS)].sum()
    by_name_history = (history[history.key.isin(safe_names)]
                       .groupby(["season", "key"], as_index=False)[
                           list(MARKETS)].sum())
    for lag in (1, 2, 3):
        prior_id = by_id_history.copy()
        prior_id["season"] += lag
        prior_name = by_name_history.copy()
        prior_name["season"] += lag
        for name in MARKETS:
            dest = f"{name}_lag{lag}"
            by_id = prior_id[["season", "player_id", name]]
            by_id = by_id.rename(columns={name: dest})
            out = out.merge(by_id, on=["season", "player_id"], how="left")

            need = out[dest].isna()
            if need.any():
                fallback = out.loc[need, ["key", "season"]].merge(
                    prior_name[["key", "season", name]].rename(
                        columns={name: dest}),
                    on=["key", "season"], how="left")
                fallback.index = out.index[need]
                out.loc[need, dest] = fallback[dest]
            out[dest] = out[dest].fillna(0.0)
    print(f"  market lag ambiguous name rows refused: {refused}")
    return out


def _estimator(name: str, kind: str):
    if name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=40.0)),
        ])
    if name in ("hist_gradient_boosting", "interaction_hgb"):
        return Pipeline([
            # The earliest fold predates some three-year history columns. HGB accepts
            # missing cells but not a feature that is entirely missing in the fit.
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0,
                                      keep_empty_features=True)),
            ("regressor", HistGradientBoostingRegressor(
                loss="poisson" if kind == "count" else "squared_error",
                learning_rate=0.05, max_iter=200,
                max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
                random_state=20260825,
            )),
        ])
    raise ValueError(name)


def _market_design(frame: pd.DataFrame, market: str,
                   interactions: bool = False) -> pd.DataFrame:
    features = [*WAR_FEATURES, *(f"{market}_lag{lag}" for lag in (1, 2, 3))]
    out = frame[features].copy()
    if interactions:
        lag = out[f"{market}_lag1"]
        out["market_lag1_x_war_lag1"] = lag * out.war_lag1
        out["market_lag1_x_share"] = lag * out.share_lag1
        out["market_lag1_x_prior_rank"] = lag * out.prior_rank
        out["market_lag1_x_team"] = lag * out.team_massey
        out["war_lag1_x_share"] = out.war_lag1 * out.share_lag1
    return out


def _fit_predict(name: str, train: pd.DataFrame, target: pd.DataFrame,
                 market: str) -> np.ndarray:
    if name == "carry_forward":
        prediction = target[f"{market}_lag1"].to_numpy(float)
    elif name == "hurdle_hgb":
        x_train = _market_design(train, market, interactions=True).fillna(0.0)
        x_target = _market_design(target, market, interactions=True).fillna(0.0)
        active = train[market].to_numpy(float) > 0
        classifier = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=160, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0,
            random_state=20260825).fit(x_train, active)
        regressor = HistGradientBoostingRegressor(
            loss="poisson" if MARKETS[market].kind == "count" else "squared_error",
            learning_rate=.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0,
            random_state=20260825).fit(x_train.loc[active], train.loc[active, market])
        prediction = (classifier.predict_proba(x_target)[:, 1] *
                      regressor.predict(x_target))
    else:
        estimator = _estimator(name, MARKETS[market].kind)
        interactions = name == "interaction_hgb"
        estimator.fit(_market_design(train, market, interactions), train[market])
        prediction = estimator.predict(_market_design(target, market, interactions))
    return np.clip(np.asarray(prediction, float), 0.0, None)


def _continuous_distribution(residuals: np.ndarray):
    candidates = {
        "gaussian": stats.norm, "student_t": stats.t,
        "laplace": stats.laplace, "logistic": stats.logistic,
        "skew_normal": stats.skewnorm,
    }
    fitted = []
    for name, family in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = tuple(float(value) for value in family.fit(residuals))
        # Point masses at zero can drive a continuous fit to an effectively zero
        # scale and a spuriously enormous likelihood. A predictive residual density
        # must retain a scale commensurate with its nonzero errors.
        floor = max(float(np.std(residuals)) * .10, 1e-3)
        if params[-1] < floor:
            params = (*params[:-1], floor)
        ll = float(np.sum(family.logpdf(residuals, *params)))
        if np.isfinite(ll):
            fitted.append((2 * len(params) - 2 * ll, name, family, params))
    if not fitted:
        raise RuntimeError("no residual distribution fitted")
    return min(fitted, key=lambda value: value[0])[1:]


def _count_alpha(outcomes: np.ndarray, means: np.ndarray) -> tuple[str, float]:
    means = np.clip(means, 1e-8, None)
    numerator = float(np.sum((outcomes - means) ** 2 - outcomes))
    denominator = float(np.sum(means ** 2))
    alpha = max(numerator / denominator, 1e-6) if denominator else 1e-6
    pois = float(-np.mean(stats.poisson.logpmf(outcomes, means)))
    size, probability = 1.0 / alpha, (1.0 / alpha) / (1.0 / alpha + means)
    nb = float(-np.mean(stats.nbinom.logpmf(outcomes, size, probability)))
    return ("negative_binomial", alpha) if nb < pois else ("poisson", 0.0)


def _nll(outcomes: np.ndarray, predictions: np.ndarray, spec: Market,
         distribution=None) -> tuple[float, object]:
    if spec.kind == "continuous":
        fitted = distribution or _continuous_distribution(outcomes - predictions)
        _, family, params = fitted
        value = -float(np.mean(family.logpdf(outcomes - predictions, *params)))
        return value, fitted
    scaled_y = np.rint(outcomes * spec.target_scale).astype(int)
    scaled_p = np.clip(predictions * spec.target_scale, 1e-8, None)
    fitted = distribution or _count_alpha(scaled_y, scaled_p)
    name, alpha = fitted
    if name == "poisson":
        logp = stats.poisson.logpmf(scaled_y, scaled_p)
    else:
        size = 1.0 / alpha
        probability = size / (size + scaled_p)
        logp = stats.nbinom.logpmf(scaled_y, size, probability)
    return -float(np.mean(logp)), fitted


def _threshold_brier(outcomes: np.ndarray, predictions: np.ndarray,
                     spec: Market, distribution) -> float:
    probabilities, binaries = [], []
    for threshold in spec.thresholds:
        if spec.kind == "continuous":
            _, family, params = distribution
            p = 1.0 - family.cdf(threshold - predictions, *params)
        else:
            name, alpha = distribution
            means = np.clip(predictions * spec.target_scale, 1e-8, None)
            cutoff = math.floor(threshold * spec.target_scale)
            if name == "poisson":
                p = 1.0 - stats.poisson.cdf(cutoff, means)
            else:
                size = 1.0 / alpha
                probability = size / (size + means)
                p = 1.0 - stats.nbinom.cdf(cutoff, size, probability)
        probabilities.extend(np.asarray(p, float).tolist())
        binaries.extend((outcomes > threshold).astype(int).tolist())
    return float(np.mean((np.asarray(probabilities) - np.asarray(binaries)) ** 2))


def forecast_markets(frame: pd.DataFrame, valid_targets: list[int]):
    """Strict forward forecasts plus per-market fold diagnostics."""
    predictions = []
    metrics: dict[str, list[dict]] = {name: [] for name in MARKETS}
    candidates = ("carry_forward", "ridge", "hist_gradient_boosting",
                  "interaction_hgb", "hurdle_hgb")
    for test_year in FORECAST_YEARS:
        prior_years = [year for year in valid_targets if year < test_year]
        if len(prior_years) < 2:
            continue
        calibration_year = max(prior_years)
        fit_years = [year for year in prior_years if year != calibration_year]
        fit = frame[frame.target_season.isin(fit_years)]
        calibration = frame[frame.target_season == calibration_year]
        train = frame[frame.target_season.isin(prior_years)]
        test = frame[frame.target_season == test_year].copy()
        out = test[["target_season", "team", "group", "player_id", "player",
                    "key", "war", "snaps"]].copy()

        for market, spec in MARKETS.items():
            eligible_fit = fit[fit.group.isin(spec.groups)]
            eligible_cal = calibration[calibration.group.isin(spec.groups)]
            eligible_train = train[train.group.isin(spec.groups)]
            eligible_test = test[test.group.isin(spec.groups)]
            calibration_scores = {}
            fitted_distributions = {}
            for candidate in candidates:
                cp = _fit_predict(candidate, eligible_fit, eligible_cal, market)
                cy = eligible_cal[market].to_numpy(float)
                score, distribution = _nll(cy, cp, spec)
                calibration_scores[candidate] = score
                fitted_distributions[candidate] = distribution
            selected = min(calibration_scores, key=calibration_scores.get)
            pred = _fit_predict(selected, eligible_train, eligible_test, market)
            actual = eligible_test[market].to_numpy(float)
            nll, _ = _nll(actual, pred, spec, fitted_distributions[selected])
            active = actual > 0
            row = {
                "season": test_year,
                "calibration_season": calibration_year,
                "train_target_seasons": prior_years,
                "selected": selected,
                "calibration_nll": calibration_scores,
                "n": int(len(actual)),
                "active_n": int(active.sum()),
                "mae": float(mean_absolute_error(actual, pred)),
                "active_mae": (float(mean_absolute_error(actual[active], pred[active]))
                               if active.any() else None),
                "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
                "pearson_r": _corr(actual, pred),
                "nll": nll,
                "threshold_brier": _threshold_brier(
                    actual, pred, spec, fitted_distributions[selected]),
            }
            metrics[market].append(row)
            out[f"pred_{market}"] = 0.0
            out.loc[eligible_test.index, f"pred_{market}"] = pred
            print(f"{test_year} {market:<22} {selected:<22} "
                  f"r={row['pearson_r']:.3f} MAE={row['mae']:.2f}")
        predictions.append(out)
    return pd.concat(predictions, ignore_index=True), metrics


def aggregate_market_components(predictions: pd.DataFrame,
                                slots: dict[str, int]) -> pd.DataFrame:
    rows = []
    for season, year_rows in predictions.groupby("target_season"):
        team = pd.DataFrame(index=sorted(year_rows.team.unique()))
        for market, spec in MARKETS.items():
            d = year_rows[year_rows.group.isin(spec.groups)].copy()
            column = f"pred_{market}"
            d["rank"] = d.groupby(["team", "group"])[column].rank(
                ascending=False, method="first")
            d = d[d["rank"] <= d.group.map(slots).fillna(2)]
            team[market] = d.groupby("team")[column].sum().reindex(team.index).fillna(0.0)
            team[f"{market}_z"] = _z(team[market])
        offense = [f"{name}_z" for name in MARKETS if name != "defensive_sacks"]
        team["player_prod_off"] = team[offense].mean(axis=1)
        team["player_prod_off"] = _z(team.player_prod_off)
        team["player_prod_def"] = team.defensive_sacks_z
        team = team.reset_index(names="team")
        team.insert(0, "season", int(season))
        rows.append(team)
    return pd.concat(rows, ignore_index=True)


WAR_PRODUCTION = [f"pred_{name}" for name in MARKETS]


def _war_design(frame: pd.DataFrame, production: bool,
                interactions: bool = False) -> pd.DataFrame:
    columns = [*WAR_FEATURES, *(WAR_PRODUCTION if production else [])]
    out = frame[columns].copy()
    if interactions:
        total = out[WAR_PRODUCTION].sum(axis=1)
        out["production_x_war_lag1"] = total * out.war_lag1
        out["production_x_share"] = total * out.share_lag1
        out["production_x_prior_rank"] = total * out.prior_rank
        out["production_x_team"] = total * out.team_massey
        out["war_lag1_x_share"] = out.war_lag1 * out.share_lag1
        for column in WAR_PRODUCTION:
            out[f"{column}_x_share"] = out[column] * out.share_lag1
    return out


def _war_candidate_predict(name: str, train: pd.DataFrame,
                           target: pd.DataFrame, seed: int) -> np.ndarray:
    if name == "baseline_hgb":
        model = fit_war(seed=seed).fit(_war_design(train, False), train.war)
        return model.predict(_war_design(target, False))
    if name in ("production_hgb", "production_interactions_hgb",
                "production_laplace"):
        interactions = name == "production_interactions_hgb"
        loss = "absolute_error" if name == "production_laplace" else "squared_error"
        model = HistGradientBoostingRegressor(
            loss=loss, max_iter=500, learning_rate=.05, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            random_state=seed).fit(_war_design(train, True, interactions), train.war)
        return model.predict(_war_design(target, True, interactions))
    if name == "production_hurdle":
        x = _war_design(train, True, True).fillna(0.0)
        xt = _war_design(target, True, True).fillna(0.0)
        played = train.snaps.to_numpy(float) > 0
        classifier = HistGradientBoostingClassifier(
            learning_rate=.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0,
            random_state=seed).fit(x, played)
        regressor = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=500, learning_rate=.05,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            random_state=seed).fit(x.loc[played], train.loc[played, "war"])
        return classifier.predict_proba(xt)[:, 1] * regressor.predict(xt)
    raise ValueError(name)


def evaluate_war_channel(frame: pd.DataFrame, predictions: pd.DataFrame,
                         slots: dict[str, int]):
    """Compare existing WAR features with WAR + cross-fitted production forecasts."""
    joined = frame.merge(
        predictions[["target_season", "team", "group", "player_id", "key",
                     *WAR_PRODUCTION]],
        on=["target_season", "team", "group", "player_id", "key"], how="inner")
    folds, team_rows = [], []
    years = sorted(joined.target_season.unique())
    candidates = ("baseline_hgb", "production_hgb",
                  "production_interactions_hgb", "production_laplace",
                  "production_hurdle", "ensemble_50")

    def team_sum(data, values):
        d = data[["team", "group"]].copy()
        d["value"] = values
        d["rank"] = d.groupby(["team", "group"]).value.rank(
            ascending=False, method="first")
        d = d[d["rank"] <= d.group.map(slots).fillna(2)]
        return d.groupby("team").value.sum()

    for test_year in years[2:]:
        train_years = [year for year in years if year < test_year]
        train = joined[joined.target_season.isin(train_years)]
        test = joined[joined.target_season == test_year].copy()
        if train.empty or test.empty:
            continue
        calibration_year = max(train_years)
        fit = train[train.target_season < calibration_year]
        calibration = train[train.target_season == calibration_year]
        calibration_metrics = {}
        if not fit.empty:
            calibration_predictions = {}
            for name in candidates[:-1]:
                calibration_predictions[name] = _war_candidate_predict(
                    name, fit, calibration, int(test_year))
            calibration_predictions["ensemble_50"] = .5 * (
                calibration_predictions["baseline_hgb"] +
                calibration_predictions["production_interactions_hgb"])
            actual_cal_team = calibration.groupby("team").war.sum()
            for name, values in calibration_predictions.items():
                predicted_team = team_sum(calibration, values)
                common_cal = actual_cal_team.index.intersection(predicted_team.index)
                player_r = _corr(calibration.war, values)
                team_r = _corr(actual_cal_team.loc[common_cal],
                               predicted_team.loc[common_cal])
                calibration_metrics[name] = {
                    "mae": float(mean_absolute_error(calibration.war, values)),
                    "player_r": player_r, "team_r": team_r,
                    "stability_score": (player_r + team_r) / 2,
                }
        selected = (max(calibration_metrics,
                        key=lambda name: calibration_metrics[name]["stability_score"])
                    if calibration_metrics else "baseline_hgb")
        candidate_predictions = {
            name: _war_candidate_predict(name, train, test, int(test_year))
            for name in candidates[:-1]}
        candidate_predictions["ensemble_50"] = .5 * (
            candidate_predictions["baseline_hgb"] +
            candidate_predictions["production_interactions_hgb"])

        actual_team = test.groupby("team").war.sum()
        team_predictions = {name: team_sum(test, values)
                            for name, values in candidate_predictions.items()}
        common = actual_team.index
        for values in team_predictions.values():
            common = common.intersection(values.index)
        candidate_metrics = {}
        for name, values in candidate_predictions.items():
            candidate_metrics[name] = {
                "mae": float(mean_absolute_error(test.war, values)),
                "r": _corr(test.war, values),
                "team_r": _corr(actual_team.loc[common],
                                team_predictions[name].loc[common]),
            }
        baseline_metrics = candidate_metrics["baseline_hgb"]
        selected_metrics = candidate_metrics[selected]
        folds.append({
            "season": int(test_year), "train_seasons": train_years,
            "calibration_season": int(calibration_year),
            "calibration_metrics": calibration_metrics, "selected": selected,
            "n_players": int(len(test)),
            "baseline_mae": baseline_metrics["mae"],
            "extended_mae": selected_metrics["mae"],
            "baseline_r": baseline_metrics["r"],
            "extended_r": selected_metrics["r"],
            "team_baseline_r": baseline_metrics["team_r"],
            "team_extended_r": selected_metrics["team_r"],
            "candidate_metrics": candidate_metrics,
        })
        selected_team = team_predictions[selected]
        for team in common:
            team_rows.append({"season": int(test_year), "team": team,
                              "player_prod_war": float(selected_team[team])})
        print(f"WAR {test_year}: player r {folds[-1]['baseline_r']:.3f} -> "
              f"{folds[-1]['extended_r']:.3f}; team r "
              f"{folds[-1]['team_baseline_r']:.3f} -> "
              f"{folds[-1]['team_extended_r']:.3f}; selected {selected}")
    team = pd.DataFrame(team_rows)
    if not team.empty:
        team["player_prod_war"] = team.groupby("season").player_prod_war.transform(_z)
    return folds, team


def main():
    base, slots, valid_targets = build_base_training()
    market_history = load_market_history()
    frame = attach_market_targets(base, market_history)
    predictions, market_metrics = forecast_markets(frame, valid_targets)
    components = aggregate_market_components(predictions, slots)
    war_folds, war_component = evaluate_war_channel(frame, predictions, slots)
    if not war_component.empty:
        components = components.merge(war_component, on=["season", "team"], how="left")
    components.to_csv(OUT_COMPONENTS, index=False)

    payload = {
        "source_framework": "Mstolte02/nfl-prop-models",
        "adaptation": "season-level CFB player production; no sportsbook lines used",
        "temporal_contract": (
            "season N roster population and identity use the existing CFBD roster path; "
            "every model feature and fitted parameter uses completed seasons <= N-1"
        ),
        "markets": {name: spec.__dict__ for name, spec in MARKETS.items()},
        "candidate_models": ["carry_forward", "ridge", "hist_gradient_boosting",
                             "interaction_hgb", "hurdle_hgb"],
        "war_stability_candidates": [
            "baseline_hgb", "production_hgb", "production_interactions_hgb",
            "production_laplace", "production_hurdle", "ensemble_50"],
        "selection": "lowest calibration-season predictive NLL",
        "market_folds": market_metrics,
        "war_channel_folds": war_folds,
        "team_component_rows": int(len(components)),
        "important_limitation": (
            "These are football forecasts, not a betting backtest. Historical CFB "
            "player-prop lines, prices, and quote timestamps are unavailable."
        ),
    }
    OUT_METRICS.write_text(json.dumps(
        payload, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ))
    print(f"-> {OUT_COMPONENTS}\n-> {OUT_METRICS}")


if __name__ == "__main__":
    main()
