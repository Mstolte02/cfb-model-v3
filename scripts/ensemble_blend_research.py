"""Read-only 2026 chronological diagnostic; never promotes a candidate.

Fitted member models are held fixed. Each candidate learns from strictly earlier
weeks. Archived PFF/WAR pulls are retrospective: this is not prospective validation.
Run: python -m scripts.ensemble_blend_research
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scripts import ensemble_replay as ER

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'audit/output/ensemble_blend_2026.json'


def features(e, state, home, away, hfa):
    probs, raw, labels, crosses = [], [], [], []
    for m in e['members']:
        f = ER._features(m, state['ratings'][m['name']], state['form'].get(m['name']),
                         home, away, hfa, state.get('pff'), state.get('war'))
        probs.append(ER.member_probability(m, state['ratings'][m['name']],
                     state['form'].get(m['name']), home, away, hfa,
                     state.get('pff'), state.get('war'), state.get('slate')))
        cols, _, _ = ER._stack(m, state.get('pff'), state.get('war'), state.get('slate'))
        for key in cols:
            if key == 'hfa': continue
            raw.append(f[key]); labels.append(m['name']+':'+key)
        form = state['form'].get(m['name']) or {}
        a, b = form.get(home), form.get(away)
        have = a is not None and b is not None and a[2] >= 2 and b[2] >= 2
        if have:
            # Both sides use the same fitted data source. High O/D mean stronger.
            oa, ob = a[0], b[0]
            if state.get('pff') is not None:
                oa, ob = ER._pff(state['pff'], home, 0), ER._pff(state['pff'], away, 0)
                have = home in state['pff'] and away in state['pff']
            da, db = a[1], b[1]
            x, y = oa-db, ob-da
            crosses.append([x*abs(x)-y*abs(y), oa*db-ob*da] if have else [0., 0.])
        else: crosses.append([0., 0.])
    return np.array(probs), dict(zip(labels, raw)), np.mean(crosses, axis=0)


def data():
    e=json.loads((ROOT/'viz/data/model_v4.json').read_text())['ensemble']
    # Test the requested carry-forward policy throughout the diagnostic only.
    e=copy.deepcopy(e);e['pff_carry_forward_from_slate']=1
    schedule=json.loads((ROOT/'viz/data/schedule.json').read_text())
    names=set(e['members'][0]['initial'])
    finals=[{'id':g['id'],'week':g['w'],'season_type':g.get('st','regular'),
             'home':g['h'],'away':g['a'],'neutral':bool(g.get('n')),
             'home_score':g['hp'],'away_score':g['ap']}
            for g in schedule if g.get('f') and g['h'] in names and g['a'] in names]
    rows=ER.form_rows(json.loads((ROOT/'data/live/game_advanced_2026.json').read_text()),names)
    pff=json.loads((ROOT/'data/live/pff_form_2026.json').read_text())
    war=json.loads((ROOT/'data/live/inseason_war_team_2026.json').read_text())
    results=[]
    for w in sorted({g['week'] for g in finals}):
        state=ER.replay(e,finals,rows,stop_before=w,pff=pff,war=war)['state']
        for g in finals:
            if g['week'] != w: continue
            p, x, cross=features(e,state,g['home'],g['away'],float(not g['neutral']))
            results.append({**g,'p':p,'features':x,'cross':cross,
                            'y':int(g['home_score']>g['away_score'])})
    keys=sorted({k for r in results for k in r['features']})
    P=np.array([r['p'] for r in results]); Z=logit(np.clip(P,1e-8,1-1e-8))
    # Keep a fixed schema defined by the fitted stacks (not held-out data).
    keys=sorted({m['name']+':'+c for m in e['members']
                 for st in (m, m.get('stack_pff') or m, m.get('stack_war') or m)
                 for c in st['columns'] if c != 'hfa'})
    X=np.array([[r['features'].get(k,0) for k in keys]+[float(not r['neutral'])] for r in results])
    keys+=['hfa']

    return e,results,P,Z,X,keys,np.array([r['cross'] for r in results])


def logistic_fit(X,y,anchor=None,penalty=.1,offset=None):
    # No intercept/centering: reversing teams must negate the logit.
    scale=np.sqrt(np.mean(X*X,axis=0));scale=np.where(scale>1e-8,scale,1.)
    A=X/scale
    prior=np.zeros(A.shape[1]) if anchor is None else np.asarray(anchor)*scale
    off=np.zeros(len(y)) if offset is None else offset
    def loss(b):
        z=off+A@b
        value=np.mean(np.logaddexp(0,z)-y*z)+penalty*np.sum((b-prior)**2)/2
        grad=A.T@(expit(z)-y)/len(y)+penalty*(b-prior)
        return value,grad
    fit=minimize(loss,prior,jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-9})
    if not fit.success: raise RuntimeError(fit.message)
    return fit.x/scale


def convex_fit(P,y):
    n=P.shape[1];a=np.full(n,1/n)
    def loss(w):
        error=P@w-y
        return np.mean(error**2)+.01*np.sum((w-a)**2),2*P.T@error/len(y)+.02*(w-a)
    fit=minimize(loss,a,jac=True,bounds=[(0,1)]*n,
                 constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)}],
                 method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x


def metrics(y,p):
    p=np.clip(p,1e-9,1-1e-9)
    return {'n':len(y),'brier':float(np.mean((p-y)**2)),
            'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'accuracy':float(np.mean((p>=.5)==y)),
            'mean_prediction':float(np.mean(p)),'observed_rate':float(np.mean(y))}


def run():
    e,rows,P,Z,X,keys,cross=data()
    weeks=np.array([r['week'] for r in rows]);y=np.array([r['y'] for r in rows])
    names=['equal_probabilities','mean_logits','learned_convex_blend','single_logit_stack',
           'unified_raw_features','stack_plus_od_mismatch','stack_plus_od_product',
           'unified_plus_od_mismatch','unified_plus_od_product']
    names += ['member_'+m['name'] for m in e['members']]
    pred={k:np.full(len(y),np.nan) for k in names};folds=[]
    for week in sorted(set(weeks)):
        train=weeks<week;test=weeks==week
        if len(set(weeks[train]))<2:continue
        w=convex_fit(P[train],y[train])
        beta=logistic_fit(Z[train],y[train],np.full(Z.shape[1],1/Z.shape[1]))
        raw=logistic_fit(X[train],y[train])
        pred['equal_probabilities'][test]=P[test].mean(axis=1)
        pred['mean_logits'][test]=expit(Z[test].mean(axis=1))
        pred['learned_convex_blend'][test]=P[test]@w
        pred['single_logit_stack'][test]=expit(Z[test]@beta)
        pred['unified_raw_features'][test]=expit(X[test]@raw)
        offsets=Z@beta
        for i,k in enumerate(['stack_plus_od_mismatch','stack_plus_od_product']):
            # A one-feature extension of the fitted stack, same training window.
            b=logistic_fit(cross[train,i:i+1],y[train],offset=offsets[train])
            pred[k][test]=expit(offsets[test]+cross[test,i:i+1]@b)
        for i,k in enumerate(['unified_plus_od_mismatch','unified_plus_od_product']):
            joint=np.column_stack([X,cross[:,i]])
            b=logistic_fit(joint[train],y[train])
            pred[k][test]=expit(joint[test]@b)
        for i,m in enumerate(e['members']):pred['member_'+m['name']][test]=P[test,i]
        folds.append({'test_week':int(week),'train_games':int(train.sum()),'test_games':int(test.sum()),
                      'blend_weights':w.tolist(),'stack_coefficients':beta.tolist(),
                      'metrics':{k:metrics(y[test],p[test]) for k,p in pred.items()}})
    eligible=np.isfinite(pred['equal_probabilities']);base=pred['equal_probabilities'][eligible]
    yy=y[eligible]; ww=weeks[eligible]; rng=np.random.default_rng(20261011)
    clusters=sorted(set(ww));draws=rng.integers(0,len(clusters),size=(5000,len(clusters)))
    report={}
    for k,p in pred.items():
        q=p[eligible];diff=(q-yy)**2-(base-yy)**2
        sums=np.array([diff[ww==w].sum() for w in clusters]);ns=np.array([(ww==w).sum() for w in clusters])
        boot=sums[draws].sum(axis=1)/ns[draws].sum(axis=1)
        report[k]={**metrics(yy,q),'delta_brier':float(diff.mean()),
                   'week_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
                   'weeks_better':sum(f['metrics'][k]['brier']<f['metrics']['equal_probabilities']['brier'] for f in folds)}
    payload={'status':'exploratory_only_no_promotion','season':2026,
      'contract':'Train only earlier weeks; first two weeks warm up; same games for all candidates.',
      'limitations':['Only one partial season and four evaluated week clusters.',
        'This diagnostic uses only 2026; recovered multi-season inputs are tested separately by historical_ensemble_research.',
        'PFF/WAR cutoffs include later retrievals; this is retrospective, not a prospective betting backtest.',
        'The published member fits are fixed; this does not refit their historical training pipelines.',
        'Penalty constants fixed before scoring, not tuned to held-out weeks; candidates are exploratory.',
        'Week bootstrap intervals are unstable with this few clusters; no model is promoted.'],
      'penalties':{'convex_shrink_to_equal':.01,'logistic_l2':.1,'logistic_anchor':'equal logit weights for stack; zero for raw features and interactions'},
      'members':[m['name'] for m in e['members']], 'raw_feature_columns':keys,
      'games_available':len(rows),'test_weeks':list(map(int,clusters)),
      'metrics':report,'folds':folds,
      'pairwise_member_probability_correlation':np.corrcoef(P[eligible].T).tolist()}
    OUT.parent.mkdir(exist_ok=True,parents=True);OUT.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    import csv
    with (OUT.parent/'ensemble_blend_2026_predictions.csv').open('w') as f:
        out=csv.writer(f);out.writerow(['id','week','home','away','y',*names])
        for i,r in enumerate(rows):
            if eligible[i]:out.writerow([r['id'],r['week'],r['home'],r['away'],r['y'],*[float(pred[k][i]) for k in names]])
    for name,m in report.items():print(f"{name:40} n={m['n']} Brier={m['brier']:.6f} delta={m['delta_brier']:+.6f} LL={m['log_loss']:.6f}")
    return payload



def write_report(r):
    from html import escape
    labels={
      'equal_probabilities':'Current equal probability average',
      'mean_logits':'Equal logit average → one probability',
      'learned_convex_blend':'Learned probability blend',
      'single_logit_stack':'One fitted model of four member logits',
      'unified_raw_features':'One fitted model of combined live inputs',
      'stack_plus_od_mismatch':'Logit stack + nonlinear O/D gap',
      'stack_plus_od_product':'Logit stack + opposing O×D interaction',
      'unified_plus_od_mismatch':'Combined inputs + nonlinear O/D gap',
      'unified_plus_od_product':'Combined inputs + opposing O×D interaction'}
    for m in r['members']:labels['member_'+m]='Single member: '+m
    table='| Candidate | Brier ↓ | Change vs equal ↓ | Log loss ↓ | Weeks better |\n|---|---:|---:|---:|---:|\n'
    for k,m in r['metrics'].items():
        table+=f"| {labels[k]} | {m['brier']:.6f} | {m['delta_brier']:+.6f} | {m['log_loss']:.6f} | {m['weeks_better']}/4 |\n"
    text='''# PFF carry-forward, ensemble blends and opposing-unit audit

October 11, 2026. Exploratory research, not a model promotion.

## Operational fix

PFF now uses the newest nonempty cutoff strictly before the prediction slate.
Week 7 currently uses data through Week 5 until Week 6 arrives. This keeps both PFF
and in-season WAR active. It never reads same-week or future-week tables. The
published policy boundary is slate 7: previously graded slates keep their original
lookup behavior. All 329 replayed historical game predictions and frozen betting
files were verified unchanged. The explorer displays the actual PFF cutoff and
whether it was carried forward.

## What was tested

The first two weeks (100 games) train the initial candidate models. Weeks 3–6
supply 229 matched test games. Each held-out week trains on strictly earlier weeks;
all candidates receive the same games. The four pretrained member models remain
fixed. This is a chronological meta-model experiment, not a historical refit of
the complete preseason pipelines. Penalties were fixed before scoring, without
using test-week labels. No candidate is automatically promoted.

''' + table + '''
The learned blend improves in all four evaluated weeks; its weights move away from
the production-feature variants. For the final held-out week its weights are
40.6%, 32.7%, 15.0%, 11.7% (base, curvature, production, both). Those are diagnostic
weights fitted through Week 5, not new production weights.

The combined-input model improves overall but only in two of four weeks. The
strongest point estimate here is the logit stack with an opposing O×D product,
which improves in all four weeks. The member probability correlations range from
0.9929 to 0.9973, so the ensemble's members are very similar.

## What “one model” means here

The logit stack feeds four individual member logits into one regularized logistic
model and applies one final sigmoid. The combined-input model instead fits a
single logistic model to member-specific preseason differences, result-update
differences, active form inputs, PFF, WAR, and home field. Repeated inputs retain
their per-member columns; regularization is part of the stated experiment. Neither
can reconstruct the original preseason feature union from the public payload,
which stores the preseason scores as aggregates. The recovered original team
frames now support the fuller union comparison in HISTORICAL_ENSEMBLE_RESEARCH.md.

## Is production a matchup model?

Production is predominantly an additive team-strength-difference model. With
positive defense values representing better defense, its efficiency terms are

`bO * (O_A - O_B) + bD * (D_A - D_B)`.

This can be rearranged into two opposing-side advantages:

`(bO * O_A - bD * D_B) - (bO * O_B - bD * D_A)`.

Thus renaming the comparisons “offense vs defense” alone adds no new information.
What is absent in the live stack is an explicit interaction that lets an offense's
impact depend nonlinearly on the specific defense. The four-member average also
introduces a nonlinear aggregation, but it is not an explicit O×D matchup term.
The existing UI's position matchup displays do not establish that those crosses
enter the live probability.

Two antisymmetric features were tested:

- Nonlinear gap: `g(O_A - D_B) - g(O_B - D_A)`, where `g(x)=x*abs(x)`.
- Opposing product: `O_A*D_B - O_B*D_A`.

Offense uses PFF when available, otherwise CFBD form; defense uses CFBD form.
Both are standardized composites, not common-unit estimates of points. Missing
coverage or fewer than two form games zeros the interaction. The signals are
averaged over members; their four current form series happen to be identical.
Swapping teams negates each input and preserves complementary probabilities.

At the current cutoff, Notre Dame has PFF offense 1.599 and CFBD defense 2.187;
Alabama has PFF offense 1.881 and CFBD defense 1.254. These are separate standardized
scales. The linear live terms reward Alabama's offensive edge and Notre Dame's
defensive edge; the tested cross terms additionally allow that pairing itself to
matter. They are research-only.

## Limits and next validation

This is one partial season, four held-out week clusters, and multiple candidates.
The PFF/WAR cutoffs include later retrievals, so chronological outcome splitting
does not make this a prospective point-in-time backtest. The interval estimates in
the JSON resample weeks; four clusters cannot support strong significance claims.
The original run could not access the historical inputs. On October 12 the actual
local source tree was recovered; see HISTORICAL_ENSEMBLE_RESEARCH.md for the newer
multi-season comparison and the remaining upstream chronology limitations.
Older checked-in v3/v4 prediction CSVs are not a substitute for this ensemble.

Keep the existing production combination while the candidates are shadowed or
rerun on restored multi-season forward predictions. Do not declare a winner from
this sample. A full union-of-preseason-features comparison still requires those
training inputs; the current result concerns the available live inputs only.

## Reproduce

`python3 -m scripts.ensemble_blend_research`

Outputs: `audit/output/ensemble_blend_2026.json` and
`audit/output/ensemble_blend_2026_predictions.csv`. The JSON includes every fold,
weights, probability correlations, metrics, and exploratory week-bootstrap intervals.

Checks: `python3 -m unittest tests.test_pff_carry_forward tests.test_ensemble_blend_research`
and `node --test tests/test_power_xray.cjs tests/test_market_board.cjs`.
'''
    (ROOT/'audit/PFF_CARRY_FORWARD_AND_ENSEMBLE_RESEARCH.md').write_text(text)
    body=''.join(f"<tr><td>{escape(labels[k])}</td><td>{m['brier']:.6f}</td><td>{m['delta_brier']:+.6f}</td><td>{m['log_loss']:.6f}</td><td>{m['weeks_better']}/4</td></tr>" for k,m in r['metrics'].items())
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>What should combine the models? · CFB X-Ray</title><link rel="stylesheet" href="style.css"></head><body><header><a href="./">← Power Rating X-Ray</a><span>RESEARCH / OCTOBER 11, 2026</span></header><main><section class="intro"><div class="eyebrow">RESEARCH, NOT A MODEL CHANGE</div><h1>Four views.<br><em>One better prediction?</em></h1><p>Testing learned blends, one combined model, and offense-versus-opposing-defense interactions.</p></section><section><h2>The PFF fix is live. The candidates are exploratory.</h2><p>PFF now carries forward the latest eligible weekly table, keeping in-season WAR active too. Frozen betting predictions remain unchanged. The explorer shows the data cutoff.</p><p class="note">The research below uses 229 games in Weeks 3–6, training only on earlier weeks. One partial season, four test weeks, and retrospective source pulls are not enough to choose a replacement. The live ensemble still averages four probabilities equally.</p></section><section><div class="eyebrow">MATCHED GAMES / LOWER IS BETTER</div><h2>Early results</h2><div class="table-scroll"><table><thead><tr><th>Candidate</th><th>Brier</th><th>Δ vs equal</th><th>Log loss</th><th>Weeks better</th></tr></thead><tbody>'''+body+'''</tbody></table></div><p class="note">Brier measures squared probability error; log loss penalizes confident mistakes more strongly. “Weeks better” compares Brier on the same four held-out weeks. Multiple candidates were inspected; these are promising leads, not confirmed improvements.</p></section><section><h2>What the matchup audit found</h2><p>The live model uses differences in offensive and defensive strength. Those terms can be rearranged into offense-versus-opposing-defense comparisons, but there is no explicit nonlinear O×D interaction in the active stack.</p><p class="note">We tested both a nonlinear gap and an opposing-product feature. The product uses O<sub>A</sub> × D<sub>B</sub> − O<sub>B</sub> × D<sub>A</sub>. It changes sign when the teams swap. That lets a specific pairing matter beyond the individual team strengths. Standardized offense and defense are not predicted points.</p><h2>Why not switch immediately?</h2><p>The members are highly correlated (0.993–0.997). Learned blending helps in this sample, and combining live inputs into one logistic model also improves overall. But the multi-season prediction exports and original preseason frames are unavailable. This test holds the existing member fits fixed; it does not retrain the complete preseason models.</p><p class="note">Next evidence needed: multi-season forward validation and prospective shadow results. The current production blend remains unchanged.</p></section><section><h2>Inspect the work</h2><p><a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/PFF_CARRY_FORWARD_AND_ENSEMBLE_RESEARCH.md">Full methodology and matchup explanation</a> · <a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/output/ensemble_blend_2026.json">Fold metrics and weights</a> · <a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/output/ensemble_blend_2026_predictions.csv">Game-level predictions</a></p></section></main></body></html>'''
    (ROOT/'viz/power-xray/research.html').write_text(html)
    historical = ROOT/'audit/output/historical_ensemble_research.json'
    if historical.exists():
        from scripts.historical_ensemble_research import write_report as write_history
        write_history(json.loads(historical.read_text()))

if __name__=='__main__':write_report(run())
