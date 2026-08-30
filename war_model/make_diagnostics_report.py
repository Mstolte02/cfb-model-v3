"""One self-contained HTML report from every diagnostic the pipeline produces.

Reads collinearity.json, model_comparison.json and diagnostics_suite.json and renders
them as one page, so the answer to "did this complexity earn its place" is a file you
can open rather than six terminal transcripts.

Run after candidates.py, collinearity.py, model_lab.py and diagnostics_suite.py:
  ./rbenv/bin/python make_diagnostics_report.py
"""
import json, os
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/war_diagnostics.html"


def load(name):
    p = f"{HERE}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def table(rows, cols, headers=None, fmt=None, cls=""):
    if not rows:
        return "<p class='muted'>not available</p>"
    headers = headers or cols
    fmt = fmt or {}
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for r in rows:
        tds = []
        for c in cols:
            v = r.get(c)
            f = fmt.get(c)
            tds.append(f"<td>{esc(f(v) if f and v is not None else v)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<div class='scroll'><table class='{cls}'><thead><tr>{th}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def heatmap(corr, names, limit=26):
    """Correlation heatmap as a plain table with tinted cells - no libraries."""
    names = names[:limit]
    head = "<th></th>" + "".join(f"<th class='rot'>{esc(n)}</th>" for n in names)
    rows = []
    for a in names:
        cells = []
        for b in names:
            v = corr.get(a, {}).get(b)
            if v is None:
                cells.append("<td></td>"); continue
            if a == b:
                cells.append("<td class='diag'>1</td>"); continue
            al = min(1.0, abs(v))
            col = (f"rgba(47,96,150,{al*0.8:.2f})" if v >= 0
                   else f"rgba(168,50,38,{al*0.8:.2f})")
            strong = " strong" if al >= 0.7 else ""
            cells.append(f"<td class='cc{strong}' style='background:{col}' "
                         f"title='{esc(a)} / {esc(b)} = {v:+.2f}'>{v:+.2f}</td>")
        rows.append(f"<tr><th class='rowh'>{esc(a)}</th>{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table class='corr'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            f"<p class='muted'>First {len(names)} features by catalogue order. "
            f"Blue positive, red negative; bordered cells are |r| &ge; 0.70.</p>")


def bar(v, vmax, tint="var(--accent)"):
    w = 0 if not vmax else max(0.0, min(1.0, abs(v) / vmax)) * 100
    return (f"<div class='bar'><i style='width:{w:.1f}%;background:{tint}'></i></div>")


def ci_row(f, mean, lo, hi, span, stable):
    """A coefficient with its 95% interval, drawn on a shared scale."""
    def x(v):
        return 50 + 50 * max(-1, min(1, v / span))
    l, h, m = x(lo), x(hi), x(mean)
    col = "var(--accent)" if stable else "var(--red)"
    return (f"<tr><td class='fname'>{esc(f)}</td>"
            f"<td class='cispan'><div class='ci'><span class='zero'></span>"
            f"<i style='left:{min(l,h):.1f}%;width:{abs(h-l):.1f}%;background:{col}'></i>"
            f"<b style='left:{m:.1f}%;background:{col}'></b></div></td>"
            f"<td class='num'>{mean:+.4f}</td><td class='num'>{hi-lo:.4f}</td>"
            f"<td>{'' if stable else '<span class=flag>sign flips</span>'}</td></tr>")


def main():
    col = load("collinearity.json")
    cn = load("concepts.json")
    cv = load("concept_verdict.json")
    lc = load("learned_concepts.json")
    cmp_ = load("model_comparison.json")
    dg = load("diagnostics_suite.json")
    if not (col and cmp_ and dg):
        missing = [n for n, v in (("collinearity.json", col),
                                  ("model_comparison.json", cmp_),
                                  ("diagnostics_suite.json", dg)) if not v]
        raise SystemExit(f"missing inputs: {', '.join(missing)}")

    names = [v["feature"] for v in col["vif"]]
    cat_order = list(col["correlation"].keys())

    # ---- headline verdict --------------------------------------------------
    boot = dg["bootstrap_margin"]
    beat = [b for b in boot if b["hi95"] < 0]
    verdict = ("A learned feature set measurably beats the hand-built one."
               if beat else
               "No learned variant beats the hand-built model by a margin the data "
               "can distinguish from zero.")
    vcls = "good" if beat else "warn"

    rows = cmp_["rows"]
    base = cmp_["benchmark_rmse"]
    best = min(rows, key=lambda r: r["rmse"])

    # ---- sections ----------------------------------------------------------
    stab = dg["stability"]
    span = max(abs(s["mean"]) for s in stab) * 1.15
    top_ci = "".join(ci_row(s["feature"], s["mean"], s["lo95"], s["hi95"], span,
                            s["sign_stable"]) for s in stab[:20])
    unstable = [s for s in stab if not s["sign_stable"]]

    imp = dg["importance"]
    imax = max(i["ridge"] for i in imp) or 1
    imp_rows = "".join(
        f"<tr><td class='fname'>{esc(i['feature'])}</td>"
        f"<td>{bar(i['ridge'], imax)}</td><td class='num'>{i['ridge']:.4f}</td>"
        f"<td>{bar(i['permutation'], max(x['permutation'] for x in imp) or 1, 'var(--green)')}</td>"
        f"<td class='num'>{i['permutation']:.4f}</td>"
        f"<td class='num'>{i.get('rank_gap', 0):.0f}</td></tr>" for i in imp[:20])

    groups = col["group_recommendations"]
    grp_rows = "".join(
        f"<tr><td>{esc(', '.join(g['members']))}</td><td class='num'>{g['n']}</td>"
        f"<td class='num'>{g['pc1_explained']*100:.0f}%</td>"
        f"<td class='num'>{g['r_pc1_vs_target']:.3f}</td>"
        f"<td class='num'>{g['r_allmembers_vs_target']:.3f}</td>"
        f"<td><span class='{'good' if g['recommendation']=='combine' else 'warn'} pill'>"
        f"{esc(g['recommendation'])}</span></td>"
        f"<td class='muted small'>{esc(g['why'])}</td></tr>"
        for g in sorted(groups.values(), key=lambda x: -x["pc1_explained"]))

    pca = col["pca"]
    ev = pca["explained_variance_ratio"][:14]
    evmax = max(ev)
    pca_bars = "".join(
        f"<div class='pcacol'><span class='pv'>{v*100:.1f}</span>"
        f"<div class='pcabar' style='height:{v/evmax*100:.0f}%'></div>"
        f"<span class='px'>{i+1}</span></div>" for i, v in enumerate(ev))
    load_rows = ""
    for pc in list(pca["loadings"])[:5]:
        L = pca["loadings"][pc]
        top = sorted(L.items(), key=lambda kv: -abs(kv[1]))[:6]
        load_rows += (f"<tr><td>{esc(pc)}</td><td class='muted small'>"
                      + ", ".join(f"{esc(k)} <b>{v:+.2f}</b>" for k, v in top)
                      + "</td></tr>")

    # ---- concept blocks ----------------------------------------------------
    concept_pca_block = loco_block = fwd_block = inter_block = verdict_block = ""
    cverdict, vcls2, qbase = "not available", "warn", "-"
    if cn:
        pca_rows = "".join(
            f"<tr><td>{esc(c)}</td><td class='num'>{len(v['members'])}</td>"
            f"<td class='num'>{v['explained'][0]*100:.0f}%</td>"
            f"<td class='muted small'>"
            + ", ".join(f"{esc(k)} <b>{w:+.2f}</b>" for k, w in
                        sorted(v['pc1'].items(), key=lambda kv: -abs(kv[1]))[:4])
            + "</td></tr>"
            for c, v in sorted(cn['within_concept_pca'].items(),
                               key=lambda kv: -kv[1]['explained'][0])
            if len(v['members']) > 1)
        rep = cn['representation']
        concept_pca_block = (
            "<div class='panel'><h3>PCA within each concept</h3>"
            "<p class='muted small'>A concept whose first component explains most of "
            "its variance is one skill measured several ways. Most are not: coverage's "
            "first component carries 30% of eleven features.</p>"
            "<div class='scroll'><table><thead><tr><th>concept</th>"
            "<th class='num'>features</th><th class='num'>first component</th>"
            "<th>heaviest loadings</th></tr></thead><tbody>"
            + pca_rows + "</tbody></table></div>"
            f"<p class='muted small' style='margin-top:10px'>Compressing to concept "
            f"scores <b>loses</b> information: 86 raw features "
            f"{rep['raw_86']:.4f}, {rep['n_concept_1pc']} concept scores "
            f"{rep['concept_1pc']:.4f}, {rep['n_unit_quality']} unit qualities "
            f"{rep['unit_quality']:.4f}. The redundancy is doing work.</p></div>")

        loco_block = table(cn['leave_one_concept_out'],
                           ["concept", "features_dropped", "rmse", "delta"],
                           ["concept", "features", "RMSE", "ΔRMSE"],
                           {"rmse": lambda v: f"{v:.4f}",
                            "delta": lambda v: f"{v:+.5f}"})
        fwd = cn['forward_by_concept']
        fwd_block = ("<ol class='muted small' style='padding-left:20px'>"
                     + "".join(f"<li>{esc(c)}</li>" for c in fwd[:8])
                     + "</ol>")
        qbase = f"{cn['quality_base_rmse']:.4f}"
        inter_block = table(cn['quality_interactions'],
                            ["interaction", "rmse", "delta"],
                            ["interaction", "RMSE", "ΔRMSE"],
                            {"rmse": lambda v: f"{v:.4f}",
                             "delta": lambda v: f"{v:+.5f}"})
    if cv:
        sig = [r for r in cv['bootstrap'] if r['significant']]
        cverdict = ("A concept-level model beats the hand-built facets."
                    if sig else
                    "Six concepts reach the same accuracy as the hand-built facets "
                    "and 86 features barely improve on either. Nothing separates from "
                    "the benchmark at 95%.")
        vcls2 = "good" if sig else "warn"
        verdict_block = ("<div class='panel'><h3>Concept models vs the benchmark</h3>"
                         + table(cv['bootstrap'],
                                 ["variant", "n_features", "rmse", "mean_delta",
                                  "lo95", "hi95", "pct_better"],
                                 ["variant", "features", "RMSE", "mean ΔRMSE",
                                  "95% low", "95% high", "better in"],
                                 {"rmse": lambda v: f"{v:.4f}",
                                  "mean_delta": lambda v: f"{v:+.4f}",
                                  "lo95": lambda v: f"{v:+.4f}",
                                  "hi95": lambda v: f"{v:+.4f}",
                                  "pct_better": lambda v: f"{v*100:.0f}%"})
                         + "</div>")

    # ---- learned vs hand grouping ------------------------------------------
    learned_block = "<p class='muted'>not available</p>"
    if lc:
        sizes = sorted((len(v) for v in lc["learned_groups"].values()), reverse=True)
        biggest = max(lc["learned_groups"].items(), key=lambda kv: len(kv[1]))
        ari = lc["ari_hand_vs_learned"]
        best_ari = max(r["ari_vs_hand"] for r in lc["sweep"])
        lf = lc["learned_forward"][:4]
        hf2 = lc["hand_forward"][:6]
        f4 = lambda v: f"{v:.4f}"
        f4s = lambda v: f"{v:+.4f}"
        lf_tbl = table(lf, ["group", "rmse", "gain", "label"],
                       ["group", "RMSE", "gain", "what it is"],
                       {"rmse": f4, "gain": f4s})
        hf_tbl = table(hf2, ["group", "rmse", "gain"], ["concept", "RMSE", "gain"],
                       {"rmse": f4, "gain": f4s})
        learned_block = f"""
<p>The grouping in section 7 was hand-written, so it has to be tested like any other
assumption. Two data-driven alternatives &mdash; hierarchical clustering on
1&nbsp;&minus;&nbsp;|r|, and varimax-rotated factor loadings &mdash; were swept over
6 to 20 groups and scored against the hand version by adjusted Rand index
(1.0 identical, 0.0 chance).</p>
<div class="kpi">
  <div><b>{ari:.2f}</b><span>ARI, hand vs learned</span></div>
  <div><b>{best_ari:.2f}</b><span>best ARI in the whole sweep</span></div>
  <div><b>{sizes[0]}</b><span>features in the largest learned group</span></div>
  <div><b>{sum(1 for n in sizes if n == 1)}</b><span>singleton groups</span></div>
</div>
<div class="panel"><h3>What the data actually groups together</h3>
<p class="muted small">Cluster sizes: {', '.join(str(n) for n in sizes)}.
The largest is <code>{esc(biggest[0])}</code> &mdash;
{esc(lc['learned_labels'][biggest[0]])}.</p>
<p>That single cluster holds essentially all of offense. The data's ontology of football
is <b>offense, back seven, front</b> &mdash; three real groups, plus a tight end cluster
and eleven fragments of two features or fewer. Partialling
out a team-quality axis does not separate it; the blob grows to 44 features and starts
mixing offense with defense. Every offensive metric correlates with every other one
through plain team strength, so there is no modular structure left to recover.</p></div>
<div class="grid2">
  <div class="panel"><h3>Forward selection, learned groups</h3>
  {lf_tbl}</div>
  <div class="panel"><h3>Forward selection, hand groups</h3>
  {hf_tbl}</div>
</div>
<div class="panel verdict warn"><b>Consequence.</b> The two routes reach the same place,
so nothing in section 7's <em>accuracy</em> depends on the grouping. Its
<em>interpretation</em> does. In particular &ldquo;receiving adds nothing given
passing&rdquo; is largely an artefact: the data puts receiving and passing in one
cluster, so the honest statement is that offense is a single indivisible signal here.
Concepts that are both interpretable and data-derived would need an external football
taxonomy or player-level co-occurrence data &mdash; not this correlation matrix.</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WAR model diagnostics</title>
<style>
:root {{ --bg:#faf7f0; --panel:#fffdf8; --line:#e3ddcd; --line2:#cfc6ae;
  --text:#241f1a; --muted:#7a7266; --accent:#8a6512; --green:#3f7d3a; --red:#a83226; }}
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ background:var(--bg); color:var(--text); font:15px/1.55 Georgia,'Times New Roman',serif;
  padding:34px 22px 70px }}
.wrap {{ max-width:1120px; margin:0 auto }}
h1 {{ font-size:27px; margin-bottom:4px }}
.sub {{ color:var(--muted); margin-bottom:22px }}
h2 {{ font-size:18px; margin:30px 0 10px; padding-bottom:5px; border-bottom:1px solid var(--line) }}
h3 {{ font-size:14px; text-transform:uppercase; letter-spacing:.8px; color:var(--muted);
  margin:18px 0 8px; font-family:-apple-system,sans-serif }}
p {{ margin-bottom:11px; max-width:78ch }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:11px;
  padding:17px 19px; margin-bottom:15px }}
.verdict {{ border-left:4px solid var(--accent); font-size:16px }}
.verdict.warn {{ border-left-color:var(--red) }}
.verdict.good {{ border-left-color:var(--green) }}
table {{ border-collapse:collapse; width:100% }}
th {{ text-align:left; font:11px/1.3 -apple-system,sans-serif; text-transform:uppercase;
  letter-spacing:.6px; color:var(--muted); padding:6px 8px; border-bottom:1px solid var(--line) }}
td {{ padding:5px 8px; border-bottom:1px solid #f0ebde; font-size:13.5px }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums }}
.fname {{ font-family:-apple-system,sans-serif; font-size:12px }}
.scroll {{ overflow-x:auto }}
.muted {{ color:var(--muted) }} .small {{ font-size:12px }}
.bar {{ height:8px; background:#efe9db; border-radius:4px; width:120px; overflow:hidden }}
.bar i {{ display:block; height:100% }}
.pill {{ font:11px -apple-system,sans-serif; padding:2px 8px; border-radius:10px }}
.pill.good {{ background:rgba(63,125,58,.14); color:var(--green) }}
.pill.warn {{ background:rgba(138,101,18,.14); color:var(--accent) }}
.flag {{ font:11px -apple-system,sans-serif; color:var(--red) }}
.ci {{ position:relative; height:14px; background:#efe9db; border-radius:3px; min-width:220px }}
.ci .zero {{ position:absolute; left:50%; top:0; width:1px; height:100%; background:var(--line2) }}
.ci i {{ position:absolute; top:5px; height:4px; border-radius:2px; opacity:.55 }}
.ci b {{ position:absolute; top:2px; width:2px; height:10px }}
.cispan {{ width:46% }}
table.corr td, table.corr th {{ border:0; padding:3px 4px }}
td.cc {{ text-align:center; font-size:10.5px; min-width:44px; border-radius:3px;
  font-variant-numeric:tabular-nums }}
td.cc.strong {{ outline:1px solid var(--text); font-weight:700 }}
td.diag {{ text-align:center; color:var(--muted); font-size:10px }}
th.rot {{ font-size:9px; writing-mode:vertical-rl; transform:rotate(180deg); height:78px }}
th.rowh {{ text-align:right; font-size:10px; text-transform:none; letter-spacing:0;
  color:var(--text) }}
.pcarow {{ display:flex; align-items:flex-end; gap:5px; height:130px; margin:12px 0 4px }}
.pcacol {{ flex:1; display:flex; flex-direction:column; align-items:center;
  justify-content:flex-end; height:100% }}
.pcabar {{ width:100%; background:var(--accent); border-radius:3px 3px 0 0; min-height:2px }}
.pv {{ font-size:9.5px; color:var(--muted); font-family:-apple-system,sans-serif }}
.px {{ font-size:10px; color:var(--muted); margin-top:4px }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:15px }}
.kpi {{ display:flex; gap:22px; flex-wrap:wrap; margin-bottom:6px }}
.kpi div {{ text-align:center }}
.kpi b {{ display:block; font-size:20px; font-variant-numeric:tabular-nums }}
.kpi span {{ font:10px -apple-system,sans-serif; color:var(--muted);
  text-transform:uppercase; letter-spacing:.6px }}
code {{ background:#f2ecdd; padding:1px 5px; border-radius:3px; font-size:12.5px }}
</style></head><body><div class="wrap">

<h1>WAR model diagnostics</h1>
<p class="sub">{date.today().isoformat()} &middot;
  {col['n_features']} candidate features &middot;
  {cmp_['n_team_seasons']:,} team-seasons with a following year &middot;
  target: next season's adjusted win percentage</p>

<div class="panel verdict {vcls}">
  <b>Verdict.</b> {esc(verdict)}
  Best variant was <b>{esc(best['model'])}</b> at RMSE {best['rmse']:.4f} against the
  benchmark's {base:.4f}, but the bootstrap interval on that difference
  {'excludes' if beat else 'includes'} zero.
</div>

<h2>1 &middot; Model comparison</h2>
<div class="panel verdict warn"><b>Historical.</b> This comparison was run when
production used twenty hand-built facets. Production has since moved to the 98-feature
candidate set with non-negative weights, so &ldquo;A current&rdquo; below is the
<em>old</em> benchmark, not what ships today. The comparison is kept because it is the
evidence that motivated the move; re-running it would compare the new model against
itself.</div>
<p>Every model predicts the same thing on the same season-blocked splits: next
season's adjusted win percentage. Feature selection runs <em>inside</em> each fold,
so no model sees its test season while choosing features.</p>
{table(rows, ["model","features","rmse","mae","r2","r","calib","seconds","vs_benchmark"],
       ["model","features","RMSE","MAE","R²","r","calibration","sec","vs benchmark"],
       {"features": lambda v: f"{v:.0f}", "rmse": lambda v: f"{v:.4f}",
        "mae": lambda v: f"{v:.4f}", "r2": lambda v: f"{v:.3f}",
        "r": lambda v: f"{v:.3f}", "calib": lambda v: f"{v:.3f}",
        "vs_benchmark": lambda v: f"{v:+.4f}"})}
<h3>Is the margin real?</h3>
{table(dg["bootstrap_margin"], ["model","mean_delta_rmse","lo95","hi95","pct_better_than_A"],
       ["model","mean ΔRMSE","95% low","95% high","beats benchmark in"],
       {"mean_delta_rmse": lambda v: f"{v:+.4f}", "lo95": lambda v: f"{v:+.4f}",
        "hi95": lambda v: f"{v:+.4f}",
        "pct_better_than_A": lambda v: f"{v*100:.0f}% of resamples"})}
<p class="muted small">200 bootstrap resamples of the held-out predictions. An interval
that crosses zero means the improvement is not distinguishable from sampling noise,
however often it happens to come out ahead.</p>

<h2>2 &middot; Are these features measuring the same thing?</h2>
<div class="kpi">
  <div><b>{sum(1 for v in col['vif'] if v['vif'] > col['vif_threshold'])}</b>
    <span>features VIF &gt; {col['vif_threshold']:.0f}</span></div>
  <div><b>{len([1 for a in col['correlation'] for b in col['correlation'][a]
        if a < b and abs(col['correlation'][a][b]) >= 0.7])}</b>
    <span>pairs |r| &ge; 0.70</span></div>
  <div><b>{pca['n_components_80pct']}</b><span>components for 80% variance</span></div>
  <div><b>{pca['n_components_95pct']}</b><span>components for 95%</span></div>
</div>
<div class="panel"><h3>Correlation heatmap</h3>{heatmap(col['correlation'], cat_order)}</div>
<div class="grid2">
  <div class="panel"><h3>Variance inflation</h3>
    {table(col['vif'][:16], ["feature","vif","r2_on_rest"],
           ["feature","VIF","R² on the rest"],
           {"vif": lambda v: f"{v:.1f}", "r2_on_rest": lambda v: f"{v:.2f}"})}</div>
  <div class="panel"><h3>PCA explained variance</h3>
    <div class="pcarow">{pca_bars}</div>
    <p class="muted small">Percent of variance per component, first 14 of
      {col['n_features']}.</p>
    <table><tbody>{load_rows}</tbody></table></div>
</div>

<h2>3 &middot; Combine or keep separate?</h2>
<p>Each row is a group of features correlated at |r| &ge; 0.70. The recommendation
compares what a single component predicts against what all the members predict
together &mdash; a group is only worth collapsing if the collapse is free.</p>
<div class="panel">
{table([], [])}
<div class='scroll'><table><thead><tr><th>group</th><th class='num'>n</th>
<th class='num'>first component</th><th class='num'>r, component</th>
<th class='num'>r, all members</th><th>call</th><th>why</th></tr></thead>
<tbody>{grp_rows}</tbody></table></div></div>
<p class="muted small">Note how often the answer is <em>keep separate</em> even at
|r| above 0.95. Two metrics can be nearly identical and still have a small orthogonal
residual that carries real signal &mdash; drop rate and drop grade share 98% of their
variance, and the 2% left over is worth 0.04 of correlation with winning.</p>

<h2>4 &middot; Coefficient stability</h2>
<p>{dg['repeats']}&times;{dg['folds']}-fold, weights refitted every time.
<b>{dg['n_sign_flips']} of {col['n_features']}</b> features change sign across folds.
Bars are 95% intervals; the tick is the mean; the vertical line is zero.</p>
<div class="panel"><div class='scroll'><table><thead><tr><th>feature</th>
<th>95% interval</th><th class='num'>mean</th><th class='num'>width</th><th></th>
</tr></thead><tbody>{top_ci}</tbody></table></div></div>
<p class="muted small">The largest weights are stable &mdash; every one of the top
twenty keeps its sign in 100% of folds. The {len(unstable)} unstable features are all
small: the biggest has |mean| of
{max((abs(s['mean']) for s in unstable), default=0):.4f}. Instability is in the tail,
not in the conclusions.</p>

<h2>5 &middot; Feature importance, three ways</h2>
<p>Ridge coefficient magnitude, permutation importance, and their rank disagreement.
Disagreement is the diagnostic: a collinear feature can carry a large coefficient and
still cost nothing when permuted, because its twin covers for it.</p>
<div class="panel"><div class='scroll'><table><thead><tr><th>feature</th>
<th>ridge</th><th class='num'></th><th>permutation</th><th class='num'></th>
<th class='num'>rank gap</th></tr></thead><tbody>{imp_rows}</tbody></table></div></div>

<h2>6 &middot; Sensitivity to the assumptions nobody fitted</h2>
<div class="grid2">
  <div class="panel"><h3>Replacement level</h3>
  {table(dg['replacement'], ["replacement","mean_war","spearman_vs_15pct",
                             "mean_rank_shift","top50_overlap"],
         ["level","mean WAR","Spearman vs 15%","mean rank shift","top-50 overlap"],
         {"replacement": lambda v: f"{v*100:.1f}%", "mean_war": lambda v: f"{v:.4f}",
          "spearman_vs_15pct": lambda v: f"{v:.4f}",
          "mean_rank_shift": lambda v: f"{v:.0f}",
          "top50_overlap": lambda v: f"{v*100:.1f}%"})}
  <p class="muted small">The level is set by assumption at 15%. It moves every WAR
  figure and barely moves the order: Spearman stays above 0.96 across the whole range
  and the top fifty are 95%+ unchanged.</p></div>
  <div class="panel"><h3>Normalization baseline</h3>
  {table(dg['normalization'], ["scheme","rmse","r"], ["scheme","RMSE","r"],
         {"rmse": lambda v: f"{v:.4f}", "r": lambda v: f"{v:.4f}"})}
  <p class="muted small">Within-season, rolling three-season and global baselines land
  within 0.0001 RMSE of each other. The convention is not doing any work.</p></div>
</div>
<div class="panel"><h3>Leave one feature out</h3>
<p class="muted small">Base RMSE {dg['lofo_base_rmse']:.4f}. Coefficient drift is how
far the <em>other</em> weights move when this one is removed.</p>
{table(dg['leave_one_out'][:12], ["dropped","rmse","delta_rmse","max_coef_drift","drifted_most"],
       ["dropped","RMSE","ΔRMSE","largest coefficient drift","which moved"],
       {"rmse": lambda v: f"{v:.4f}", "delta_rmse": lambda v: f"{v:+.4f}",
        "max_coef_drift": lambda v: f"{v:.4f}"})}
<p class="muted small">No single feature is load-bearing &mdash; the worst removal
costs {max(r['delta_rmse'] for r in dg['leave_one_out']):.4f} RMSE. That is what a
redundant feature set looks like from the inside: whatever you take out, its
correlates absorb the job.</p></div>

<h2>7 &middot; Football concepts, not features</h2>
<div class="panel verdict warn"><b>These concepts were chosen by hand.</b>
The fifteen groupings below are a dict written by a person, not a structure recovered
from the data. Everything in this section is therefore <em>conditional on that
grouping</em> &mdash; it says what follows if you accept these categories, not that the
data endorses them. Section 8 tests the grouping itself.</div>
<p>Asking whether one feature helps has no power when 86 features contain six views of
every throw &mdash; the worst single removal cost 0.0006 RMSE. Removing a whole
<em>concept</em> does have power, and it changes what the model looks like.</p>
{concept_pca_block}
<div class="grid2">
  <div class="panel"><h3>Leave one concept out</h3>
  {loco_block}
  <p class="muted small">Every feature of a concept removed at once. Five concepts cost
  nothing or help slightly by their absence.</p></div>
  <div class="panel"><h3>Which concepts earn a place</h3>
  {fwd_block}
  <p class="muted small">Forward selection one concept at a time. Six carry
  effectively all of it.</p></div>
</div>
<div class="panel"><h3>Interactions on unit quality</h3>
<p class="muted small">Built on the snap-weighted mean z of each unit &mdash; quality
with volume divided out &mdash; rather than on team totals, which carry volume and
make the product meaningless. Base with qualities alone:
{qbase}.</p>
{inter_block}</div>
<div class="panel verdict {vcls2}"><b>Concept verdict.</b> {cverdict}</div>
{verdict_block}

<h2>8 &middot; Can the grouping be learned?</h2>
{learned_block}

<p class="muted small" style="margin-top:26px">Generated by
<code>make_diagnostics_report.py</code> from <code>collinearity.json</code>,
<code>model_comparison.json</code> and <code>diagnostics_suite.json</code>. Re-run
<code>candidates.py &rarr; collinearity.py &rarr; model_lab.py &rarr;
diagnostics_suite.py</code> to refresh.</p>
</div></body></html>"""

    open(OUT, "w", encoding="utf-8").write(html)
    print(f"war_diagnostics.html written ({os.path.getsize(OUT)/1024:.0f} KB)")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
