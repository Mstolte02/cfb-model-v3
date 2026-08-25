"""Coach mover diagnostic and descriptive variance decomposition.

The decomposition is deliberately separate from the leakage-safe predictive path.
Run ``python -m scripts.coach_effects --phase2`` only after the Phase 0 graph gate
has been reviewed.
"""
from __future__ import annotations

import json
import sys
import argparse
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import OPP_ADJ_ALPHA
from scripts.train import load_bundle
from src import oppadj as OA
from src.data import cfbd_client
from src.data.coaches import (DEFAULT_MAX_YEAR, DEFAULT_MIN_YEAR,
                              dominant_head_coaches)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit" / "COACH_EFFECTS_EXPERIMENTS.md"
JSON_OUT = ROOT / "audit" / "coach_mover_graph.json"
PHASE2_JSON = ROOT / "audit" / "coach_variance_decomposition.json"
TOP_EFFECTS = ROOT / "audit" / "coach_effect_estimates.csv"
OUTCOMES = ("rating_overall", "rating_offense", "rating_defense")
BASE_X = ("prior_offense", "prior_defense", "talent", "returning")


def expected_team_seasons(min_year: int, max_year: int) -> pd.DataFrame:
    rows = []
    for year in range(min_year, max_year + 1):
        for team in cfbd_client.fbs_teams(year):
            rows.append({"season": year, "team_id": int(team["id"]),
                         "team": team["school"]})
    return pd.DataFrame(rows).drop_duplicates(["season", "team_id"])


def coach_spells(assignments: pd.DataFrame) -> pd.DataFrame:
    """Split coach-school histories whenever consecutive observed years break."""
    rows = []
    unique = assignments[["coach_id", "coach_name", "team_id", "team",
                          "season"]].drop_duplicates()
    for keys, group in unique.groupby(["coach_id", "coach_name", "team_id", "team"]):
        years = sorted(int(year) for year in group.season.unique())
        spell = [years[0]]
        spells = []
        for year in years[1:]:
            if year == spell[-1] + 1:
                spell.append(year)
            else:
                spells.append(spell)
                spell = [year]
        spells.append(spell)
        for index, years_in_spell in enumerate(spells, start=1):
            rows.append({
                "coach_id": int(keys[0]), "coach_name": keys[1],
                "team_id": int(keys[2]), "team": keys[3], "spell": index,
                "start": years_in_spell[0], "end": years_in_spell[-1],
                "seasons": len(years_in_spell),
            })
    return pd.DataFrame(rows)


def graph_components(assignments: pd.DataFrame):
    adjacency = defaultdict(set)
    for row in assignments.itertuples():
        coach = ("coach", int(row.coach_id))
        school = ("school", int(row.team_id))
        adjacency[coach].add(school)
        adjacency[school].add(coach)
    unseen, components = set(adjacency), []
    while unseen:
        root = next(iter(unseen))
        stack, component = [root], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(adjacency[node] - component)
        components.append(component)
    return components, adjacency


def mover_count(assignments: pd.DataFrame, minimum_seasons: int) -> int:
    counts = assignments.groupby(["coach_id", "team_id"]).season.nunique()
    qualified = counts[counts >= minimum_seasons].reset_index()
    schools = qualified.groupby("coach_id").team_id.nunique()
    return int((schools >= 2).sum())


def phase0(min_year: int = DEFAULT_MIN_YEAR,
           max_year: int = DEFAULT_MAX_YEAR) -> dict:
    expected = expected_team_seasons(min_year, max_year)
    assigned = dominant_head_coaches(min_year, max_year)
    joined_keys = assigned[["season", "team_id"]].drop_duplicates()
    coverage = expected.merge(joined_keys, on=["season", "team_id"], how="left",
                              indicator=True)
    unmatched = coverage[coverage._merge == "left_only"]

    components, adjacency = graph_components(assigned)
    component_rows = []
    for component in components:
        coaches = {node[1] for node in component if node[0] == "coach"}
        schools = {node[1] for node in component if node[0] == "school"}
        seasons = assigned[
            assigned.coach_id.isin(coaches) & assigned.team_id.isin(schools)]
        component_rows.append({"nodes": component, "coaches": coaches,
                               "schools": schools, "team_seasons": len(seasons)})
    largest = max(component_rows, key=lambda row: (len(row["nodes"]),
                                                    row["team_seasons"]))
    mover_components = [row for row in component_rows
                        if len(row["schools"]) >= 2 and
                        any(len(adjacency[("coach", coach)]) >= 2
                            for coach in row["coaches"])]
    mover_covered = sum(row["team_seasons"] for row in mover_components)

    spells = coach_spells(assigned)
    histogram = {str(int(length)): int(count)
                 for length, count in sorted(Counter(spells.seasons).items())}
    quantiles = spells.seasons.quantile([0, .25, .5, .75, .9, .95, 1]).to_dict()
    movers_2, movers_3 = mover_count(assigned, 2), mover_count(assigned, 3)
    lcc_all_fraction = largest["team_seasons"] / len(expected)
    gate_pass = lcc_all_fraction >= .60 and movers_2 >= 40

    return {
        "period": [min_year, max_year],
        "expected_fbs_team_seasons": int(len(expected)),
        "matched_team_seasons": int(len(assigned)),
        "join_coverage": float(len(assigned) / len(expected)),
        "unmatched_team_seasons": int(len(unmatched)),
        "unmatched_examples": unmatched[["season", "team"]].head(20).to_dict("records"),
        "multi_coach_team_seasons": int(assigned.midseason_change.sum()),
        "coach_share": {
            "median": float(assigned.coach_share.median()),
            "p10": float(assigned.coach_share.quantile(.10)),
            "minimum": float(assigned.coach_share.min()),
        },
        "qualifying_movers": {"2_seasons_each_school": movers_2,
                              "3_seasons_each_school": movers_3},
        "components": int(len(components)),
        "largest_component": {
            "schools": int(len(largest["schools"])),
            "coaches": int(len(largest["coaches"])),
            "team_seasons": int(largest["team_seasons"]),
            "fraction_of_matched": float(largest["team_seasons"] / len(assigned)),
            "fraction_of_all_fbs": float(lcc_all_fraction),
        },
        "team_seasons_outside_largest_component":
            int(len(assigned) - largest["team_seasons"] + len(unmatched)),
        "team_seasons_outside_any_mover_component":
            int(len(assigned) - mover_covered + len(unmatched)),
        "spells": {
            "count": int(len(spells)), "histogram": histogram,
            "mean": float(spells.seasons.mean()),
            "quantiles": {str(key): float(value) for key, value in quantiles.items()},
        },
        "gate": {
            "largest_component_min_fraction": .60,
            "minimum_qualifying_movers": 40,
            "pass": bool(gate_pass),
            "decision": ("eligible for Phase 2 after review" if gate_pass else
                         "skip Phase 2; proceed to tenure/change features after review"),
        },
    }


def decomposition_frame() -> pd.DataFrame:
    """Build the model-aligned team-season frame with preseason X in the fit."""
    standardized, talent, returning, games, _ = load_bundle()
    od_by_year = OA.build_od_by_year(standardized, games, OPP_ADJ_ALPHA)
    coaches = dominant_head_coaches(DEFAULT_MIN_YEAR, DEFAULT_MAX_YEAR)
    coach_lookup = coaches.set_index(["season", "team"])
    rows = []
    common_years = sorted(set(od_by_year) & set(talent) & set(returning))
    for season in common_years:
        if season - 1 not in od_by_year:
            continue
        current, prior = od_by_year[season], od_by_year[season - 1]
        teams = (set(current.index) & set(prior.index) & set(talent[season].index) &
                 set(returning[season].index))
        for team in sorted(teams):
            key = (season, team)
            if key not in coach_lookup.index:
                continue
            coach = coach_lookup.loc[key]
            if isinstance(coach, pd.DataFrame):
                coach = coach.iloc[0]
            rows.append({
                "team": team,
                "rating_offense": float(current.loc[team, "O"]),
                "rating_defense": float(current.loc[team, "D"]),
                "prior_offense": float(prior.loc[team, "O"]),
                "prior_defense": float(prior.loc[team, "D"]),
                "talent": float(talent[season][team]),
                "returning": float(returning[season][team]),
                "season": int(season), "team_id": int(coach.team_id),
                "coach_id": int(coach.coach_id), "coach_name": coach.coach_name,
                "coach_share": float(coach.coach_share),
                "midseason_change": int(coach.midseason_change),
            })
    frame = pd.DataFrame(rows)
    frame["rating_overall"] = frame.rating_offense + frame.rating_defense

    # Current-year outcomes can use current coach identity descriptively. Change and
    # tenure are derived only from assignment history, never from same-year SP+/SRS.
    prior_assignment = coaches[["season", "team_id", "coach_id"]].copy()
    prior_assignment["season"] += 1
    prior_assignment = prior_assignment.rename(columns={"coach_id": "prior_coach_id"})
    frame = frame.merge(prior_assignment, on=["season", "team_id"], how="left")
    frame["hc_change"] = (frame.prior_coach_id.notna() &
                          frame.coach_id.ne(frame.prior_coach_id)).astype(int)
    frame["program_fe"] = frame.team_id.astype(str)
    frame["coach_fe"] = frame.coach_id.astype(str)
    return frame


def _variance_terms(y: np.ndarray, xb: np.ndarray, program: np.ndarray,
                    coach: np.ndarray, residual: np.ndarray) -> dict:
    """Exact observation-weighted decomposition, including all cross terms."""
    components = {"xb": xb, "program": program, "coach": coach,
                  "residual": residual}
    result = {f"var_{name}": float(np.var(value))
              for name, value in components.items()}
    names = list(components)
    for i, left in enumerate(names):
        a = components[left] - np.mean(components[left])
        for right in names[i + 1:]:
            b = components[right] - np.mean(components[right])
            result[f"twice_cov_{left}_{right}"] = float(2 * np.mean(a * b))
    result["var_outcome"] = float(np.var(y))
    result["reconstructed"] = float(sum(value for key, value in result.items()
                                               if key.startswith("var_") and
                                               key != "var_outcome") +
                                    sum(value for key, value in result.items()
                                        if key.startswith("twice_cov_")))
    result["reconstruction_error"] = result["var_outcome"] - result["reconstructed"]
    return result


def fit_naive_hdfe(frame: pd.DataFrame, outcome: str) -> dict:
    """Fit the requested two-way HDFE plug-in decomposition with pyfixest."""
    import pyfixest as pf

    formula = (f"{outcome} ~ {' + '.join(BASE_X)} | "
               "program_fe + coach_fe")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = pf.feols(formula, data=frame)
    used = model._data.copy()  # pyfixest's post-singleton estimation sample
    effects = model.fixef()
    # pyfixest omits one reference level from a multi-way FE dictionary; its effect
    # is zero under the package's normalization, not missing data.
    program = used.program_fe.map(effects["C(program_fe)"]).fillna(0).to_numpy(float)
    coach = used.coach_fe.map(effects["C(coach_fe)"]).fillna(0).to_numpy(float)
    y = used[outcome].to_numpy(float)
    residual = np.asarray(model.resid(), dtype=float)
    fitted = np.asarray(model.predict(), dtype=float)
    xb = fitted - program - coach
    terms = _variance_terms(y, xb, program, coach, residual)
    terms.update({
        "n_input": int(len(frame)), "n_used": int(len(used)),
        "singletons_dropped": int(len(frame) - len(used)),
        "coefficients": {key: float(value) for key, value in model.coef().items()},
        "coach_program_correlation": float(np.corrcoef(coach, program)[0, 1]),
        "warnings": [str(item.message) for item in caught],
    })
    return terms


def _posterior_array(idata, name: str) -> np.ndarray:
    value = np.asarray(idata.posterior[name])
    return value.reshape((-1,) + value.shape[2:])


def fit_partial_pool(frame: pd.DataFrame, outcome: str, likelihood: str,
                     interactions: bool, draws: int, tune: int,
                     chains: int, seed: int) -> tuple[dict, pd.DataFrame]:
    """Crossed Bayesian model; optional coach slope and cross-level interactions."""
    import arviz as az
    import pymc as pm

    work = frame.copy()
    predictors = list(BASE_X)
    means = work[predictors].mean()
    scales = work[predictors].std(ddof=0).replace(0, 1)
    x = ((work[predictors] - means) / scales).to_numpy(float)
    predictor_names = predictors.copy()
    returning_z = x[:, predictors.index("returning")]
    if interactions:
        x = np.column_stack([
            x,
            x[:, predictors.index("talent")] * returning_z,
            work.hc_change.to_numpy(float) * returning_z,
            work.midseason_change.to_numpy(float) * returning_z,
        ])
        predictor_names += ["talent_x_returning", "hc_change_x_returning",
                            "midseason_x_returning"]

    y_raw = work[outcome].to_numpy(float)
    y_mean, y_scale = float(y_raw.mean()), float(y_raw.std(ddof=0))
    y = (y_raw - y_mean) / y_scale
    coach_codes, coaches = pd.factorize(work.coach_id, sort=True)
    program_codes, programs = pd.factorize(work.team_id, sort=True)
    coords = {"observation": np.arange(len(work)),
              "predictor": predictor_names,
              "coach": coaches.astype(str), "program": programs.astype(str)}

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", 0, 1)
        beta = pm.Normal("beta", 0, 1, dims="predictor")
        sigma_program = pm.HalfNormal("sigma_program", 1)
        sigma_coach = pm.HalfNormal("sigma_coach", 1)
        program_raw = pm.Normal("program_raw", 0, 1, dims="program")
        coach_raw = pm.Normal("coach_raw", 0, 1, dims="coach")
        program_effect = pm.Deterministic("program_effect",
                                          program_raw * sigma_program,
                                          dims="program")
        coach_effect = pm.Deterministic("coach_effect", coach_raw * sigma_coach,
                                        dims="coach")
        eta = (intercept + x @ beta + program_effect[program_codes] +
               coach_effect[coach_codes])
        if interactions:
            sigma_slope = pm.HalfNormal("sigma_coach_returning", .5)
            slope_raw = pm.Normal("coach_returning_raw", 0, 1, dims="coach")
            coach_slope = pm.Deterministic("coach_returning_slope",
                                           slope_raw * sigma_slope, dims="coach")
            eta = eta + coach_slope[coach_codes] * returning_z
        sigma = pm.HalfNormal("sigma", 1)
        if likelihood == "student_t":
            nu_minus_two = pm.Exponential("nu_minus_two", 1 / 10)
            pm.StudentT("rating", nu=nu_minus_two + 2, mu=eta, sigma=sigma,
                        observed=y, dims="observation")
        else:
            pm.Normal("rating", mu=eta, sigma=sigma, observed=y,
                      dims="observation")
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1, target_accept=.97,
            random_seed=[seed + index for index in range(chains)], progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    loo = az.loo(idata)
    beta_draw = _posterior_array(idata, "beta")
    program_draw = _posterior_array(idata, "program_effect")
    coach_draw = _posterior_array(idata, "coach_effect")
    intercept_draw = _posterior_array(idata, "intercept").reshape(-1)
    xb = beta_draw @ x.T
    program_component = program_draw[:, program_codes]
    coach_component = coach_draw[:, coach_codes]
    if interactions:
        slope_draw = _posterior_array(idata, "coach_returning_slope")
        coach_component = (coach_component +
                           slope_draw[:, coach_codes] * returning_z[None, :])
    fitted = (intercept_draw[:, None] + xb + program_component + coach_component)
    residual = y[None, :] - fitted

    decomp_draws = []
    for index in range(fitted.shape[0]):
        decomp_draws.append(_variance_terms(
            y, xb[index], program_component[index], coach_component[index],
            residual[index]))
    decomp = pd.DataFrame(decomp_draws)
    summary = {}
    for column in decomp:
        summary[column] = {
            "median": float(decomp[column].median()),
            "q05": float(decomp[column].quantile(.05)),
            "q95": float(decomp[column].quantile(.95)),
        }
    coach_mean = coach_draw.mean(axis=0) * y_scale
    coach_sd = coach_draw.std(axis=0, ddof=1) * y_scale
    season_counts = work.groupby("coach_id").season.nunique()
    coach_names = work.groupby("coach_id").coach_name.first()
    effect_rows = pd.DataFrame({
        "outcome": outcome, "likelihood": likelihood,
        "interactions": interactions, "coach_id": coaches.astype(int),
        "coach_name": [coach_names.get(int(value), "") for value in coaches],
        "effect": coach_mean, "posterior_sd": coach_sd,
        "seasons": [int(season_counts.get(int(value), 0)) for value in coaches],
    })
    effect_rows["rank_high"] = effect_rows.effect.rank(method="first",
                                                        ascending=False).astype(int)
    effect_rows["rank_low"] = effect_rows.effect.rank(method="first").astype(int)

    diagnostics = az.summary(idata, kind="diagnostics")
    result = {
        "likelihood": likelihood, "interactions": interactions,
        "n": int(len(work)), "chains": chains, "draws_per_chain": draws,
        "tune": tune,
        "elpd_loo": float(loo.elpd), "se_elpd_loo": float(loo.se),
        "p_loo": float(loo.p), "loo_warning": bool(loo.warning),
        "max_rhat": float(diagnostics["r_hat"].max()),
        "min_bulk_ess": float(diagnostics["ess_bulk"].min()),
        "divergences": int(np.asarray(idata.sample_stats["diverging"]).sum()),
        "coach_program_correlation": float(np.corrcoef(
            coach_component.mean(axis=0), program_component.mean(axis=0))[0, 1]),
        "variance_decomposition_standardized": summary,
        "outcome_scale": y_scale,
    }
    if likelihood == "student_t":
        nu = _posterior_array(idata, "nu_minus_two").reshape(-1) + 2
        result["student_t_nu"] = {
            "median": float(np.median(nu)), "q05": float(np.quantile(nu, .05)),
            "q95": float(np.quantile(nu, .95)),
        }
    return result, effect_rows


def phase2(draws: int = 750, tune: int = 750, chains: int = 4) -> dict:
    """Run naive and bias-corrected models for overall, offense, and defense."""
    frame = decomposition_frame()
    result = {
        "frame": {"seasons": [int(frame.season.min()), int(frame.season.max())],
                  "team_seasons": int(len(frame)),
                  "programs": int(frame.team_id.nunique()),
                  "coaches": int(frame.coach_id.nunique()),
                  "hc_changes": int(frame.hc_change.sum()),
                  "midseason_changes": int(frame.midseason_change.sum())},
        "models": {},
    }
    all_effects = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        print(f"Fitting {outcome}: naive HDFE", flush=True)
        naive = fit_naive_hdfe(frame, outcome)
        candidates = []
        specs = [("gaussian", False), ("student_t", False),
                 ("student_t", True)]
        for spec_index, (likelihood, interactions) in enumerate(specs):
            label = likelihood + ("_interactions" if interactions else "")
            print(f"Fitting {outcome}: {label}", flush=True)
            fitted, effects = fit_partial_pool(
                frame, outcome, likelihood, interactions, draws, tune, chains,
                seed=1701 + 100 * outcome_index + 10 * spec_index)
            candidates.append(fitted)
            all_effects.append(effects)
        best = int(np.argmax([candidate["elpd_loo"] for candidate in candidates]))
        result["models"][outcome] = {
            "naive_hdfe": naive, "partial_pool_candidates": candidates,
            "selected_candidate": best,
            "selection_rule": "largest PSIS-LOO expected log predictive density",
        }
    effects = pd.concat(all_effects, ignore_index=True)
    effects.to_csv(TOP_EFFECTS, index=False)
    result["top_bottom"] = {}
    for outcome, model_result in result["models"].items():
        chosen = model_result["partial_pool_candidates"][
            model_result["selected_candidate"]]
        selected = effects[(effects.outcome == outcome) &
                           (effects.likelihood == chosen["likelihood"]) &
                           (effects.interactions == chosen["interactions"])]
        columns = ["coach_id", "coach_name", "effect", "posterior_sd", "seasons"]
        result["top_bottom"][outcome] = {
            "highest": selected.nlargest(10, "effect")[columns].to_dict("records"),
            "lowest": selected.nsmallest(10, "effect")[columns].to_dict("records"),
        }
    return result


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _phase2_markdown(decomposition: dict) -> str:
    frame = decomposition["frame"]
    lines = [
        "\n## Phase 2 — descriptive variance decomposition\n",
        (f"The reviewed graph gate was followed by a {frame['seasons'][0]}–"
         f"{frame['seasons'][1]} model-aligned frame: **{frame['team_seasons']} "
         f"team-seasons, {frame['programs']} programs, and {frame['coaches']} "
         f"head coaches**. It contains {frame['hc_changes']} between-season head-"
         f"coach changes and {frame['midseason_changes']} dominant-coach rows with "
         "a midseason change. The shorter period is imposed by availability of all "
         "four existing preseason covariates in the same fit.\n"),
        "### Model comparison and bias handling\n",
        ("The naive column is the two-way HDFE plug-in estimate from `pyfixest`. "
         "The corrected candidates are crossed program/coach Bayesian models in "
         "PyMC. Gaussian and Student-t observation models were fit; the interaction "
         "candidate adds talent×returning, head-coach-change×returning, midseason-"
         "change×returning, and a partially pooled coach-specific returning-"
         "production slope. The corrected headline model is selected by the largest "
         "PSIS-LOO expected log predictive density separately for each outcome. "
         "This is descriptive model comparison, not authorization to ship a feature.\n"),
        "| outcome | naive coach share | naive program share | naive 2cov(C,P) | "
        "singletons dropped | corrected candidate | ELPD-LOO | corrected coach share "
        "(90% interval) | corrected corr(C,P) |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for outcome in OUTCOMES:
        models = decomposition["models"][outcome]
        naive = models["naive_hdfe"]
        chosen = models["partial_pool_candidates"][models["selected_candidate"]]
        terms = chosen["variance_decomposition_standardized"]
        naive_total = naive["var_outcome"]
        coach_share = naive["var_coach"] / naive_total
        program_share = naive["var_program"] / naive_total
        cp_share = naive["twice_cov_program_coach"] / naive_total
        corrected = terms["var_coach"]
        total = terms["var_outcome"]
        med = corrected["median"] / total["median"]
        lo = corrected["q05"] / total["q95"]
        hi = corrected["q95"] / total["q05"]
        label = chosen["likelihood"] + (" + interactions" if chosen["interactions"] else "")
        lines.append(
            f"| {outcome.replace('rating_', '')} | {coach_share:.1%} | "
            f"{program_share:.1%} | {cp_share:.1%} | {naive['singletons_dropped']} | "
            f"{label} | {chosen['elpd_loo']:.1f} | {med:.1%} "
            f"({lo:.1%}, {hi:.1%}) | {chosen['coach_program_correlation']:.3f} |")
    lines.extend([
        "\nThe displayed shares are observation-weighted. The machine-readable "
        "artifact reports every variance and twice-covariance term, including X–FE "
        "cross terms that are omitted from the abbreviated identity in the brief, "
        "and verifies exact reconstruction of outcome variance. Bayesian intervals "
        "are posterior 5th–95th percentiles. HDFE singleton removal is recursive; "
        "partial pooling retains all rows.\n",
        "### Distribution and interaction candidates\n",
        "| outcome | candidate | ELPD-LOO | SE | pLOO | warning | max R-hat | min bulk ESS | divergences |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ])
    for outcome in OUTCOMES:
        for candidate in decomposition["models"][outcome]["partial_pool_candidates"]:
            label = candidate["likelihood"] + (" + interactions" if candidate["interactions"] else "")
            lines.append(
                f"| {outcome.replace('rating_', '')} | {label} | "
                f"{candidate['elpd_loo']:.1f} | {candidate['se_elpd_loo']:.1f} | "
                f"{candidate['p_loo']:.1f} | {candidate['loo_warning']} | "
                f"{candidate['max_rhat']:.3f} | {candidate['min_bulk_ess']:.0f} | "
                f"{candidate['divergences']} |")
    lines.append("\n### Highest and lowest shrunken coach intercepts\n")
    lines.append(
        "These are posterior means from the selected corrected model, in the original "
        "rating scale. `posterior SD` is the uncertainty analogue of a standard error. "
        "For an interaction-selected model, the list ranks the intercept at average "
        "returning production; the decomposition itself uses each row's combined "
        "intercept-plus-slope coach component.\n")
    for outcome in OUTCOMES:
        lines.extend([
            f"#### {outcome.replace('rating_', '').title()} rating",
            "| side | coach | effect | posterior SD | seasons |",
            "|---|---|---:|---:|---:|",
        ])
        ranked = decomposition["top_bottom"][outcome]
        for side, rows in (("high", ranked["highest"]), ("low", ranked["lowest"])):
            for row in rows:
                lines.append(f"| {side} | {row['coach_name']} | {row['effect']:.3f} | "
                             f"{row['posterior_sd']:.3f} | {row['seasons']} |")
        lines.append("")
    lines.extend([
        "### Interpretation boundary\n",
        ("This phase estimates association after observed roster covariates and "
         "partial pooling. It does not establish a causal coach effect and does not "
         "satisfy the temporal prediction contract. The next phase must refit every "
         "coach estimate inside expanding folds using seasons no later than N−1. "
         "CFBD has no coordinator history, so unit-level OC/DC changes remain "
         "untested.\n"),
        "Machine-readable results are in `audit/coach_variance_decomposition.json`; "
        "all posterior coach estimates are in `audit/coach_effect_estimates.csv`.\n",
    ])
    return "\n".join(lines)


def write_audit(result: dict, decomposition: dict | None = None) -> None:
    existing = AUDIT.read_text(encoding="utf-8") if AUDIT.exists() else ""
    preserved_phase2 = ""
    phase2_marker = "## Phase 2 — descriptive variance decomposition"
    if decomposition is None and phase2_marker in existing:
        preserved_phase2 = "\n" + phase2_marker + existing.split(
            phase2_marker, 1)[1]
    predictive_tail = ""
    marker = "## Phases 3–4 — leakage-safe mean features"
    has_predictive = marker in existing
    if marker in existing and not preserved_phase2:
        predictive_tail = "\n" + marker + existing.split(marker, 1)[1]
    status = (
        "Phases 0–5 complete. No coach feature clears the predictive adoption bar; nothing ships."
        if has_predictive else
        ("Phase 2 descriptive decomposition complete; predictive phases not yet run."
         if decomposition or preserved_phase2 else
         "Phase 0 complete; stopped for review. No fixed-effect, random-effect, or predictive coach model has been fit."))
    lcc, movers, spells, gate = (result["largest_component"],
                                 result["qualifying_movers"],
                                 result["spells"], result["gate"])
    hist = ", ".join(f"{length}y: {count}" for length, count in
                     spells["histogram"].items())
    examples = ", ".join(f"{row['team']} {row['season']}"
                         for row in result["unmatched_examples"]) or "none"
    verdict = "PASS" if gate["pass"] else "FAIL"
    text = f"""# Coach effects experiments

## Prior and phase status

The prior is that head-coach mean effects will fail the model's predictive adoption
bar because much of the signal is mediated by recruiting talent and returning
production. A coaching-change contribution to uncertainty is more plausible. The
numbers below are descriptive identification diagnostics only; they are not evidence
that a coach feature predicts games.

**Status: {status}**

## Phase 0 — mover graph

Scope is {result['period'][0]}-{result['period'][1]}, the completed FBS seasons shared
with the available player/model history. CFBD stable coach ids and team ids define the
bipartite graph. For the {result['multi_coach_team_seasons']} team-seasons with more
than one returned head coach, the row is attributed to the coach with the most games;
ties break on coach id. The selected coach's game share is retained (median
{result['coach_share']['median']:.3f}, p10 {result['coach_share']['p10']:.3f}, minimum
{result['coach_share']['minimum']:.3f}).

| diagnostic | result |
|---|---:|
| expected FBS team-seasons | {result['expected_fbs_team_seasons']:,} |
| matched coach team-seasons | {result['matched_team_seasons']:,} ({_pct(result['join_coverage'])}) |
| unmatched / outside the graph | {result['unmatched_team_seasons']:,} |
| movers with >=2 seasons at each of >=2 schools | **{movers['2_seasons_each_school']}** |
| movers with >=3 seasons at each of >=2 schools | **{movers['3_seasons_each_school']}** |
| connected components | {result['components']} |
| largest component | **{lcc['schools']} schools, {lcc['coaches']} coaches** |
| largest-component team-seasons | **{lcc['team_seasons']:,}** ({_pct(lcc['fraction_of_matched'])} of matched; {_pct(lcc['fraction_of_all_fbs'])} of all FBS) |
| team-seasons outside largest component, including unmatched | {result['team_seasons_outside_largest_component']:,} |
| team-seasons outside every mover-connected component, including unmatched | {result['team_seasons_outside_any_mover_component']:,} |

Unmatched examples: {examples}.

"Outside any connected component" is reported as the literal data-join quantity:
team-seasons with no coach node are outside the graph. Because every matched edge is
mathematically in some component, the table also reports the more useful identification
quantities outside the largest component and outside every component containing a
coach who moved schools.

### Seasons per coach-school spell

There are {spells['count']} consecutive spells. Mean length is
{spells['mean']:.2f} seasons; the median is {spells['quantiles']['0.5']:.0f}, p75 is
{spells['quantiles']['0.75']:.0f}, p90 is {spells['quantiles']['0.9']:.0f}, and the
maximum is {spells['quantiles']['1.0']:.0f}.

Histogram: {hist}.

### Gate verdict: {verdict}

The predeclared gate requires at least 40 two-season-at-each-school movers and about
60% of all team-seasons in the largest connected component. This graph has
{movers['2_seasons_each_school']} qualifying movers and
{_pct(lcc['fraction_of_all_fbs'])} coverage, so it is **{gate['decision']}**.

This is a stop gate, not authorization to interpret coach effects. Even with a pass,
limited-mobility bias can inflate naive coach variance and push coach-program
covariance downward. Phase 2, if approved, must report both naive and bias-corrected
decompositions and must keep existing preseason observables in the same fit.

## Reproduction

```powershell
python -m scripts.coach_effects
```

Machine-readable results are in `audit/coach_mover_graph.json`. CFBD coordinator data
does not exist; this experiment can test head coaches only.
"""
    if decomposition is not None:
        text += _phase2_markdown(decomposition)
    text += preserved_phase2 or predictive_tail
    AUDIT.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", action="store_true",
                        help="run the reviewed descriptive decomposition")
    parser.add_argument("--draws", type=int, default=750)
    parser.add_argument("--tune", type=int, default=750)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()
    result = phase0()
    JSON_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    decomposition = None
    if args.phase2:
        if not result["gate"]["pass"]:
            raise RuntimeError("Phase 0 gate failed; Phase 2 is not estimable")
        decomposition = phase2(args.draws, args.tune, args.chains)
        PHASE2_JSON.write_text(json.dumps(decomposition, indent=2), encoding="utf-8")
    write_audit(result, decomposition)
    gate = result["gate"]
    print(json.dumps(result, indent=2))
    print(f"\nPhase 0 gate: {'PASS' if gate['pass'] else 'FAIL'} — "
          f"{gate['decision']}")
    if decomposition is None:
        print(f"Stopped for review. -> {AUDIT}")
    else:
        print(f"Phase 2 complete. -> {AUDIT}")


if __name__ == "__main__":
    main()
