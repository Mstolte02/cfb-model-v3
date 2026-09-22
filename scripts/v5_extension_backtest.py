"""Score additions to the live v5 ensemble against v5 itself, fold by fold.

The full prior_decay_backtest spends most of its 40 minutes choosing each member's
knobs, score step, form half-life and stack penalty. Those choices were made inside
each outer fold's training pool, so reusing them is leak-free, and this harness does
exactly that: it refits the members' preseason models, replays the seasons, and then
changes ONE thing at a time -

* **stack additions**: extra pregame columns beside [prior_level, elo_change, dO, dD,
  hfa], with the member's own penalty C;
* **walk variants**: a different rating update inside every member (per-team gains).

Every variant is the same four-member equal average, scored on the same 2,189 games
of 2023-25 and compared with the baseline game by game, with the season-week block
bootstrap the rest of the repo uses.

Frames use the production `build_frames` WAR scaling, so the baseline is v5 as it is
served, not the backtest's slightly different scaling
(see audit/ACCURACY_RECOVERY_REVIEW.md).

    python -m scripts.v5_extension_backtest            # all registered variants
    python -m scripts.v5_extension_backtest --only base,pace_variance
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri
from scipy.stats import norm

from config import ARTIFACTS, GAME_YEARS
from scripts import prior_decay_backtest as PD
from scripts import train_live_ensemble as T
from scripts import v4_backtest as BT

OUTER = (2023, 2024, 2025)
SPECS = T.SPECS
ARM = "ewma_elo_nodecay"
BASE_COLS = PD.ARM_COLUMNS[ARM]
CACHE = ARTIFACTS / "v5_extension_cache.pkl"
OUT_JSON = ARTIFACTS / "v5_extension_backtest.json"
OUT_CSV = ARTIFACTS / "v5_extension_predictions.csv"
KEYS = ["season", "week", "home_team", "away_team"]


# ------------------------------------------------------------------ fold contexts
def fold_selections() -> dict:
    j = json.loads((ARTIFACTS / "prior_decay_backtest.json").read_text())
    out = {}
    for f in j["folds"]:
        if f["spec"] in SPECS:
            sel = f["decay_selection"][ARM]
            out[(f["season"], f["spec"])] = {
                "knobs": f["model_knobs"], "k": f["elo_k"],
                "halflife": (math.inf if sel["current_halflife"] in (None, "inf")
                             else float(sel["current_halflife"])),
                "C": float(sel["C"])}
    return out


def build_contexts(refresh=False, backtest_scaling=False):
    """Fitted preseason models for every (outer season, spec, season in play)."""
    cache = CACHE.with_name(CACHE.stem + ("_bt" if backtest_scaling else "") + ".pkl")
    if cache.exists() and not refresh:
        return pickle.loads(cache.read_bytes())
    frames, parts_by_spec, raw, _ = T.build(include_projection=False,
                                            backtest_scaling=backtest_scaling)
    sel = fold_selections()
    ctx = {}
    for test in OUTER:
        pool = [y for y in GAME_YEARS if y < test]
        for spec in SPECS:
            s = sel[(test, spec)]
            names, parts = PD.SPEC_FEATURES[spec], parts_by_spec[spec]
            fitted = {}
            for i in range(1, len(pool)):
                model, _, _ = BT.fit_predict(parts, pool[:i], pool[i], names, s["knobs"])
                fitted[pool[i]] = model
            fitted[test], _, _ = BT.fit_predict(parts, pool, test, names, s["knobs"])
            ctx[(test, spec)] = {"sel": s, "models": fitted}
    payload = {"ctx": ctx, "frames": frames, "raw": raw,
               "parts": {spec: parts_by_spec[spec] for spec in SPECS}}
    cache.write_bytes(pickle.dumps(payload))
    return payload


# ------------------------------------------------------------------ rating walks
_FORM_CACHE: dict = {}


def _forms(raw, season, weeks, halflife):
    key = (season, halflife)
    if key not in _FORM_CACHE:
        _FORM_CACHE[key] = PD.week_states(raw[season], weeks, halflife)
    return _FORM_CACHE[key]


def walk(model, frame, part, raw, season, halflife, k, variant=None, min_form=2):
    """One season of pregame rows for one member; a whole slate before its results.

    With ``variant=None`` this is PD.season_design's ewma_elo_nodecay arm exactly.
    A variant dict switches the update to a per-team extended Kalman gain:
    ``v0`` week-0 rating variance scaled by exp(beta * uncertainty_z), ``tau`` weekly
    drift, and ``gamma`` inflation after surprising results (track record).
    """
    X, y, home_flag, margins, meta = part
    initial = {t: model.team_logit_strength(frame, t) for t in frame.index}
    r = dict(initial)
    sigma = float(model.margin_sigma)
    states = _forms(raw, season, meta.week, halflife)
    order = meta.assign(row_i=np.arange(len(meta))).sort_values(["week", "row_i"])
    rows = [None] * len(meta)
    if variant:
        u = variant.get("uncertainty", {}).get(season, {})
        v = {t: variant["v0"] * math.exp(variant["beta"] * float(u.get(t, 0.0)))
             for t in frame.index}
    for week, slate in order.groupby("week", sort=True):
        od = states.get(float(week))
        change, vcut, vgrow = {}, {}, {}
        for g in slate.itertuples():
            i, h, a = int(g.row_i), g.home_team, g.away_team
            hfa = float(home_flag[i])
            have = (od is not None and h in od.index and a in od.index and
                    od.at[h, "n"] >= min_form and od.at[a, "n"] >= min_form)
            rows[i] = {"prior_level": initial[h] - initial[a],
                       "elo_change": (r[h] - initial[h]) - (r[a] - initial[a]),
                       "dO": float(od.at[h, "O"] - od.at[a, "O"]) if have else 0.0,
                       "dD": float(od.at[h, "D"] - od.at[a, "D"]) if have else 0.0,
                       "hfa": hfa, "y": float(y[i]), "season": season,
                       "week": int(g.week), "home_team": h, "away_team": a}
            gap = r[h] - r[a] + model.hfa_coef * hfa
            p = float(expit(gap))
            expected = sigma * float(ndtri(min(max(p, .01), .99)))
            if not variant:
                d = k * float(np.clip((float(margins[i]) - expected) / sigma, -2.5, 2.5))
                change[h] = change.get(h, 0.0) + d
                change[a] = change.get(a, 0.0) - d
                continue
            resid = float(np.clip(float(margins[i]) - expected, -2.5 * sigma, 2.5 * sigma))
            pc = min(max(p, .01), .99)
            H = sigma * pc * (1 - pc) / float(norm.pdf(ndtri(pc)))
            S = H * H * (v[h] + v[a]) + sigma * sigma
            change[h] = change.get(h, 0.0) + v[h] * H / S * resid
            change[a] = change.get(a, 0.0) - v[a] * H / S * resid
            z2 = resid * resid / S
            for t in (h, a):
                vcut[t] = vcut.get(t, 0.0) + v[t] * v[t] * H * H / S
                vgrow[t] = vgrow.get(t, 0.0) + variant["gamma"] * v[t] * max(z2 - 1.0, 0.0)
        for t, d in change.items():
            r[t] += d
        if variant:
            for t in v:
                v[t] = max(v[t] - vcut.get(t, 0.0), 1e-4) + vgrow.get(t, 0.0) \
                       + variant["tau"] ** 2
    return pd.DataFrame(rows)


def member_designs(payload, test, spec, variant=None, min_form=2):
    c = payload["ctx"][(test, spec)]
    s = c["sel"]
    out = {}
    for season, model in c["models"].items():
        out[season] = walk(model, payload["frames"][season],
                           payload["parts"][spec][season], payload["raw"], season,
                           s["halflife"], s["k"], variant, min_form)
    return out


# ------------------------------------------------------------------ stack + score
def fit_predict_stack(train: pd.DataFrame, test: pd.DataFrame, cols, C,
                      n_extra=0, extra_scale=1.0):
    """PD.fit_stack, with the last ``n_extra`` columns shrunk by ``extra_scale``
    after standardization (penalty x 1/extra_scale^2 on those columns only)."""
    if n_extra == 0 or extra_scale == 1.0:
        fitted = PD.fit_stack(train, cols, C)
        return PD.stack_predict(fitted, test, cols), dict(zip(cols, fitted[1].coef_[0]))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=False).fit(train[cols].to_numpy(float))
    mult = np.ones(len(cols))
    mult[-n_extra:] = extra_scale
    xt = scaler.transform(train[cols].to_numpy(float)) * mult
    model = LogisticRegression(C=float(C), fit_intercept=False, max_iter=3000)
    model.fit(xt, train.y.to_numpy(float))
    xs = scaler.transform(test[cols].to_numpy(float)) * mult
    return model.predict_proba(xs)[:, 1], dict(zip(cols, model.coef_[0] * mult))


def run_variant(payload, name, extra=None, extra_cols=(), variant=None, C_scale=1.0,
                extra_scale=1.0, min_form=2):
    """``extra_scale`` < 1 shrinks the added columns harder than the base ones: in an
    L2 logistic, scaling a standardized column by s multiplies its penalty by 1/s^2."""
    preds, coefs = [], {}
    for test in OUTER:
        member_p = []
        for spec in SPECS:
            designs = member_designs(payload, test, spec, variant, min_form)
            if extra is not None:
                designs = {y: d.merge(extra, on=KEYS, how="left") for y, d in designs.items()}
                for d in designs.values():
                    d[list(extra_cols)] = d[list(extra_cols)].fillna(0.0)

            train = pd.concat([d for y, d in designs.items() if y != test],
                              ignore_index=True)
            cols = [*BASE_COLS, *extra_cols]
            C = payload["ctx"][(test, spec)]["sel"]["C"] * C_scale
            p, coef = fit_predict_stack(train, designs[test], cols, C,
                                        len(extra_cols), extra_scale)
            member_p.append(p)
            coefs[f"{test}:{spec}"] = coef
        base = designs[test][KEYS + ["y"]].copy()
        base["p"] = np.mean(member_p, axis=0)
        preds.append(base)
    out = pd.concat(preds, ignore_index=True)
    out["variant"] = name
    return out, coefs


def summarize(results: dict[str, pd.DataFrame]) -> dict:
    base = results["base"].rename(columns={"p": "p_base"})
    report = {}
    for name, frame in results.items():
        m = base.merge(frame[KEYS + ["p"]], on=KEYS)
        m["p_var"] = m.p
        brier = float(((m.p_var - m.y) ** 2).mean())
        by = {str(s): float(((g.p_var - g.y) ** 2).mean()) for s, g in m.groupby("season")}
        entry = {"brier": brier, "by_season": by, "n": int(len(m))}
        if name != "base":
            entry["vs_base"] = PD.bootstrap(m, "p_var", "p_base")
            entry["seasons_better"] = int(sum(
                ((g.p_var - g.y) ** 2).mean() < ((g.p_base - g.y) ** 2).mean()
                for _, g in m.groupby("season")))
        report[name] = entry
    return report


# ------------------------------------------------------------------ per-team gain
def uncertainty_by_season(frames) -> dict[int, dict[str, float]]:
    """Week-0 prior doubt per team: low returning production, heavy portal churn,
    and a first-year head coach, each z-scored within season and averaged.
    All three are known before week 1."""
    coaches = json.loads((Path(__file__).resolve().parents[1] / "data" / "raw" /
                          "coaches_2014_2025.json").read_text())
    head = {}
    for c in coaches:
        for s in c.get("seasons", []):
            key = (s["school"], int(s["year"]))
            if s.get("games", 0) > head.get(key, (None, -1))[1]:
                head[key] = (c["id"], s.get("games", 0))
    out = {}
    for year, f in frames.items():
        new = pd.Series({t: float(head.get((t, year), (0,))[0] !=
                                  head.get((t, year - 1), (-1,))[0]) for t in f.index})
        parts = [-f.returning, f.portal_in + f.portal_out, new]
        z = [(x - x.mean()) / (x.std(ddof=0) or 1.0) for x in parts]
        u = sum(z) / len(z)
        u = (u - u.mean()) / (u.std(ddof=0) or 1.0)
        out[int(year)] = u.to_dict()
    return out


KALMAN_GRID = [{"v0": v0, "tau": tau, "beta": beta, "gamma": gamma}
               for v0 in (.25, .45, .7) for tau in (0.0, .08)
               for beta in (0.0, .5, 1.0) for gamma in (0.0, .5)]


def walk_brier(model, frame, part, season, params, uncertainty):
    """Brier of the member's own rating walk, expit(gap), for parameter selection."""
    X, y, home_flag, margins, meta = part
    r = {t: model.team_logit_strength(frame, t) for t in frame.index}
    sigma = float(model.margin_sigma)
    u = uncertainty.get(season, {})
    v = {t: params["v0"] * math.exp(params["beta"] * float(u.get(t, 0.0))) for t in r}
    loss = 0.0
    wk = meta.week.to_numpy()
    hs, as_ = meta.home_team.to_numpy(), meta.away_team.to_numpy()
    for week in np.unique(wk):
        idx = np.flatnonzero(wk == week)
        change, vcut, vgrow = {}, {}, {}
        for i in idx:
            h, a = hs[i], as_[i]
            gap = r[h] - r[a] + model.hfa_coef * float(home_flag[i])
            p = float(expit(gap))
            loss += (p - float(y[i])) ** 2
            pc = min(max(p, .01), .99)
            expected = sigma * float(ndtri(pc))
            resid = float(np.clip(float(margins[i]) - expected, -2.5 * sigma, 2.5 * sigma))
            H = sigma * pc * (1 - pc) / float(norm.pdf(ndtri(pc)))
            S = H * H * (v[h] + v[a]) + sigma * sigma
            change[h] = change.get(h, 0.0) + v[h] * H / S * resid
            change[a] = change.get(a, 0.0) - v[a] * H / S * resid
            z2 = resid * resid / S
            for t in (h, a):
                vcut[t] = vcut.get(t, 0.0) + v[t] * v[t] * H * H / S
                vgrow[t] = vgrow.get(t, 0.0) + params["gamma"] * v[t] * max(z2 - 1, 0)
        for t, d in change.items():
            r[t] += d
        for t in v:
            v[t] = max(v[t] - vcut.get(t, 0.0), 1e-4) + vgrow.get(t, 0.0) + params["tau"] ** 2
    return loss / len(y)


def run_kalman(payload, name, grid=None, fixed=None, extra=None, extra_cols=()):
    """Per-team gain inside every member, its four parameters chosen per member on
    that fold's earlier seasons (their out-of-sample preseason models)."""
    uncertainty = uncertainty_by_season(payload["frames"])
    grid = grid or KALMAN_GRID
    preds, chosen = [], {}
    for test in OUTER:
        member_p = []
        for spec in SPECS:
            c = payload["ctx"][(test, spec)]
            contexts = [y for y in c["models"] if y != test]
            if fixed is not None:
                best = fixed
            else:
                scores = []
                for params in grid:
                    loss = np.mean([walk_brier(c["models"][y], payload["frames"][y],
                                               payload["parts"][spec][y], y, params,
                                               uncertainty) for y in contexts])
                    scores.append((loss, json.dumps(params, sort_keys=True)))
                best = json.loads(min(scores)[1])
            chosen[f"{test}:{spec}"] = best
            variant = {**best, "uncertainty": uncertainty}
            designs = member_designs(payload, test, spec, variant)
            if extra is not None:
                designs = {y: d.merge(extra, on=KEYS, how="left").fillna(
                    {col: 0.0 for col in extra_cols}) for y, d in designs.items()}
            train = pd.concat([d for y, d in designs.items() if y != test],
                              ignore_index=True)
            p, _ = fit_predict_stack(train, designs[test], [*BASE_COLS, *extra_cols],
                                     c["sel"]["C"])
            member_p.append(p)
        base = designs[test][KEYS + ["y"]].copy()
        base["p"] = np.mean(member_p, axis=0)
        preds.append(base)
    out = pd.concat(preds, ignore_index=True)
    out["variant"] = name
    return out, chosen


# ------------------------------------------------------------------ registry
def _style():
    from src import style
    return style.build_style_features(GAME_YEARS)


def _tempo():
    from src import tempo
    return tempo.build_rolling_drive_features(GAME_YEARS)


def variants(payload):
    """name -> kwargs for run_variant. Built lazily so --only stays cheap."""
    reg = {"base": lambda: {}}

    def pace_variance():
        s = _style()
        return {"extra": s[KEYS + ["expected_pace_z"]], "extra_cols": (),
                "_interact": "expected_pace_z"}
    reg["pace_variance"] = pace_variance

    def style_family(cols):
        def build():
            s = _style()
            return {"extra": s[KEYS + list(cols)], "extra_cols": tuple(cols)}
        return build
    reg["style_pace"] = style_family(["pace_pref_diff", "pace_fit_edge",
                                      "pace_control_diff"])
    reg["style_pass"] = style_family(["pass_pref_diff", "pass_fit_edge",
                                      "pass_control_diff"])
    reg["style_edge"] = style_family(["style_edge"])
    reg["style_all"] = style_family(["pace_pref_diff", "pace_fit_edge",
                                     "pace_control_diff", "pass_pref_diff",
                                     "pass_fit_edge", "pass_control_diff"])

    def shrunk(builder, scale):
        def build():
            return {**builder(), "extra_scale": scale}
        return build
    def all_games():
        spec = SPECS[0]
        return pd.concat([part[4].assign(season=y)[KEYS]
                          for y, part in payload["parts"][spec].items()],
                         ignore_index=True)

    def pff_family(name):
        def build():
            from src import pff_form
            cols = pff_form.FAMILIES[name]
            f = pff_form.build_pff_form(all_games())
            return {"extra": f[KEYS + cols], "extra_cols": tuple(cols)}
        return build
    for fam in ("pff_outcome_form", "pff_process_form", "pff_scheme_clash",
                "pff_outcome_composite"):
        reg[fam] = pff_family(fam)

    def availability(cols):
        def build():
            from src import availability as AV
            f = AV.build_availability(all_games())
            return {"extra": f[KEYS + list(cols)], "extra_cols": tuple(cols)}
        return build
    reg["availability_war"] = availability(["missing_war_diff"])
    reg["availability_qb"] = availability(["missing_qb_war_diff"])
    reg["availability_all"] = availability(["missing_war_diff", "missing_qb_war_diff",
                                            "missing_count_diff"])
    reg["kalman_team"] = lambda: {"_kalman": {}}
    # Control for the PFF early-season gain: CFBD form after one game, not two.
    reg["cfbd_form_n1"] = lambda: {"min_form": 1}

    def leak_placebo():
        from src import pff_form
        cols = pff_form.FAMILIES["pff_outcome_composite"]
        f = pff_form.build_pff_form(all_games(), _leak_placebo=True)
        return {"extra": f[KEYS + cols], "extra_cols": tuple(cols)}
    reg["pff_LEAK_PLACEBO"] = leak_placebo

    def composite_n1():
        return {**pff_family("pff_outcome_composite")(), "min_form": 1}
    reg["pff_composite_n1"] = composite_n1

    def composite_kalman():
        built = pff_family("pff_outcome_composite")()
        return {"_kalman": built}
    reg["pff_composite_kalman"] = composite_kalman
    reg["kalman_no_prior_doubt"] = lambda: {"_kalman": {"grid": [
        g for g in KALMAN_GRID if g["beta"] == 0.0]}}
    reg["kalman_no_track_record"] = lambda: {"_kalman": {"grid": [
        g for g in KALMAN_GRID if g["gamma"] == 0.0]}}
    for scale in (.3, .1):
        reg[f"style_all_s{scale}"] = shrunk(reg["style_all"], scale)
        reg[f"style_edge_s{scale}"] = shrunk(reg["style_edge"], scale)

    def tempo_family(cols):
        def build():
            t = _tempo()
            return {"extra": t[KEYS + list(cols)], "extra_cols": tuple(cols)}
        return build
    from src import tempo as TP
    reg["tempo_identity"] = tempo_family(TP.PACE_IDENTITY)
    reg["tempo_script"] = tempo_family(TP.SCRIPT_WINDOWS)
    reg["tempo_control"] = tempo_family(TP.STATE_CONTROL)
    reg["tempo_matchup"] = tempo_family(TP.PACE_MATCHUP)
    return reg


def main(only=None, refresh=False, backtest_scaling=False):
    t0 = time.time()
    payload = build_contexts(refresh, backtest_scaling)
    print(f"contexts ready in {time.time() - t0:.0f}s", flush=True)
    reg = variants(payload)
    names = [n for n in reg if only is None or n in only]
    if "base" not in names:
        names = ["base", *names]
    results, coefs = {}, {}
    for name in names:
        t1 = time.time()
        kwargs = reg[name]()
        interact = kwargs.pop("_interact", None)
        kalman = kwargs.pop("_kalman", None)
        if kalman is not None:
            results[name], coefs[name] = run_kalman(payload, name, **kalman)
        elif interact:
            results[name], coefs[name] = run_interaction(payload, name, interact, **kwargs)
        else:
            results[name], coefs[name] = run_variant(payload, name, **kwargs)
        b = float(((results[name].p - results[name].y) ** 2).mean())
        print(f"{name:<18} Brier {b:.6f}  ({time.time() - t1:.0f}s)", flush=True)
    report = summarize(results)
    for name, e in report.items():
        if name == "base":
            print(f"\nbase {e['brier']:.6f} {e['by_season']}")
            continue
        vb = e["vs_base"]
        print(f"{name:<18} {e['brier']:.6f}  d={vb['difference']:+.6f} "
              f"[{vb['ci95'][0]:+.6f},{vb['ci95'][1]:+.6f}] "
              f"P={vb['probability_left_better']:.2f} seasons={e['seasons_better']}/3")
    previous = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    previous.update({n: {**report[n], "stack_coef": coefs[n]} for n in report})
    OUT_JSON.write_text(json.dumps(previous, indent=2, default=float))
    pd.concat(results.values()).to_csv(OUT_CSV, index=False)
    return report


def run_interaction(payload, name, col, extra, extra_cols=()):
    """Pace as variance: let the game's expected length scale the rating terms."""
    preds, coefs = [], {}
    inter = [f"prior_level_x_{col}", f"elo_change_x_{col}"]
    for test in OUTER:
        member_p = []
        for spec in SPECS:
            designs = member_designs(payload, test, spec)
            for y in designs:
                d = designs[y].merge(extra, on=KEYS, how="left")
                d[col] = d[col].fillna(0.0)
                d[inter[0]] = d.prior_level * d[col]
                d[inter[1]] = d.elo_change * d[col]
                designs[y] = d
            train = pd.concat([d for y, d in designs.items() if y != test],
                              ignore_index=True)
            cols = [*BASE_COLS, *inter]
            p, coef = fit_predict_stack(train, designs[test], cols,
                                        payload["ctx"][(test, spec)]["sel"]["C"])
            member_p.append(p)
            coefs[f"{test}:{spec}"] = coef
        base = designs[test][KEYS + ["y"]].copy()
        base["p"] = np.mean(member_p, axis=0)
        preds.append(base)
    out = pd.concat(preds, ignore_index=True)
    out["variant"] = name
    return out, coefs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated variant names")
    parser.add_argument("--refresh", action="store_true", help="rebuild fitted contexts")
    parser.add_argument("--backtest-scaling", action="store_true",
                        help="standardise WAR as prior_decay_backtest did (parity check)")
    a = parser.parse_args()
    main(set(a.only.split(",")) if a.only else None, a.refresh, a.backtest_scaling)
