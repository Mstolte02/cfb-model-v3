"""Recovered multi-season research. Never writes production or licensed inputs.

Run with the existing local source tree: python -m scripts.historical_ensemble_research
Member selections reuse recorded earlier-season tuning; fresh frames and fits bypass
the unversioned pickle cache. Historical WAR updates are excluded because their
leave-season-out calibration includes future years. Source vintages remain retrospective.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from config import ROOT, PFF_API_DIR, PFF_DIR, TWODEEP_2026, PFSN_MASTER
from scripts import ensemble_blend_research as B
from scripts import ensemble_replay as ER
from scripts import prior_decay_backtest as PD
from scripts import train_live_ensemble as T
from scripts import v5_extension_backtest as V
from scripts import v4_backtest as BT
from src import live_ensemble as LE
from src import v4 as M
from src.data import war

OUT = ROOT / 'audit/output/historical_ensemble_research.json'
UNION = list(dict.fromkeys(c for spec in T.SPECS for c in PD.SPEC_FEATURES[spec]))
SELECTIONS = ROOT / 'audit/output/historical_fold_selections.json'


def fingerprint(path):
    path = Path(path)
    return {'file': path.name, 'exists': path.exists(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


def calibration(y, p):
    bins = []
    for low in np.arange(0., 1., .1):
        take = (p >= low) & (p < low + .1 + (1e-12 if low > .89 else 0))
        if take.any():
            bins.append({'lower': float(low), 'n': int(take.sum()),
                         'predicted': float(p[take].mean()), 'observed': float(y[take].mean())})
    return {'bins': bins, 'ece': float(sum(x['n'] * abs(x['predicted']-x['observed'])
                                          for x in bins) / len(y))}


def chronological_masks(seasons, test):
    return seasons < test, seasons == test


def uncertainty(y, p, base, seasons, weeks):
    blocks = sorted(set(zip(seasons, weeks)))
    rng = np.random.default_rng(20261012)
    draws = rng.integers(len(blocks), size=(5000, len(blocks)))
    losses = {'brier': (p-y)**2-(base-y)**2,
              'log_loss': -(y*np.log(p)+(1-y)*np.log1p(-p))
                          +(y*np.log(base)+(1-y)*np.log1p(-base))}
    result = {'season_week_blocks': len(blocks)}
    counts = np.array([np.sum((seasons == s) & (weeks == w)) for s,w in blocks])
    for name, diff in losses.items():
        sums = np.array([diff[(seasons == s) & (weeks == w)].sum() for s,w in blocks])
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        result[name] = {'difference': float(diff.mean()),
                        'season_week_bootstrap_95': np.quantile(boot,[.025,.975]).tolist()}
    # Whole-season blocks expose how little independent season evidence exists.
    ss = sorted(set(seasons)); sd = rng.integers(len(ss), size=(5000,len(ss)))
    sums = np.array([losses['brier'][seasons==s].sum() for s in ss])
    counts = np.array([np.sum(seasons==s) for s in ss])
    result['season_blocks'] = len(ss)
    result['season_bootstrap_brier_95'] = np.quantile(sums[sd].sum(axis=1)/counts[sd].sum(axis=1),[.025,.975]).tolist()
    return result


def recover():
    # No stale cache and no substitution of the older expanded WAR tree.
    production = fingerprint(war.PRESEASON_TEAM_WAR)
    expanded = fingerprint(PFF_API_DIR/'war_all_facets_v52/preseason_team_war.csv')
    if not production['exists'] or production['sha256'] != expanded['sha256']:
        raise ValueError('Production preseason WAR does not match the recovered v52 lineage')
    frames,parts,raw,_ = T.build(include_projection=False)
    selections=json.loads(SELECTIONS.read_text())['folds']
    contexts={}
    for fold in selections:
        test,spec=fold['test_season'],fold['member']; pool=fold['train_seasons']
        if any(year>=test for year in pool): raise ValueError('Fold selection uses future seasons')
        sel=dict(fold['selection']);sel['halflife']=sel['halflife'] if sel['halflife'] is not None else np.inf
        models={}
        for i in range(1,len(pool)):
            models[pool[i]]=BT.fit_predict(parts[spec],pool[:i],pool[i],PD.SPEC_FEATURES[spec],sel['knobs'])[0]
        models[test]=BT.fit_predict(parts[spec],pool,test,PD.SPEC_FEATURES[spec],sel['knobs'])[0]
        contexts[(test,spec)]={'sel':sel,'models':models}
    payload={'frames':frames,'parts':parts,'raw':raw,'ctx':contexts}
    # Private regenerated frames support later research, never public artifacts.
    private=ROOT/'artifacts/historical_research';private.mkdir(parents=True,exist_ok=True)
    for year,frame in frames.items():frame.to_csv(private/f'frame_{year}.csv',index_label='team')
    extra = T.pff_features(payload['parts'])
    records, reconciliations = [], []
    for test in V.OUTER:
        members, expected = [], []
        for spec in T.SPECS:
            context = payload['ctx'][(test,spec)]; sel = context['sel']
            designs = V.member_designs(payload,test,spec)
            train = pd.concat([d for year,d in designs.items() if year<test],ignore_index=True)
            cols = PD.ARM_COLUMNS[T.ARM]
            scaler, model = PD.fit_stack(train,cols,sel['C'])
            joined = train.merge(extra,on=V.KEYS,how='left').fillna({c:0. for c in T.PFF_COLUMNS})
            pcols = T._lean([*cols,*T.PFF_COLUMNS])
            ps, pm = PD.fit_stack(joined,pcols,sel['C'])
            def stack(s,m,c):
                return {'columns':c,'scale':s.scale_.tolist(),'coef':m.coef_[0].tolist(),'C':sel['C']}
            members.append({'name':spec,'weight':.25,'features':PD.SPEC_FEATURES[spec],
                            'model':LE.model_payload(context['models'][test]),'score_k':sel['k'],
                            'current_halflife':sel['halflife'] if np.isfinite(sel['halflife']) else None,
                            'stack':stack(scaler,model,cols),'stack_pff':stack(ps,pm,pcols),
                            'stack_war':None})
            d = designs[test].merge(extra,on=V.KEYS,how='left').fillna({c:0. for c in T.PFF_COLUMNS})
            expected.append(PD.stack_predict((ps,pm),d,pcols))
        frame = payload['frames'][test]
        manifest = {'members':members,'schema_version':3,'model_version':'5.3-research-without-inseason-war',
                    'architecture':'equal_ensemble_expanded_war_score_innovation_ewma',
                    'temporal_contract':'retrospective sources; outcome folds use earlier seasons',
                    'pff_carry_forward_from_slate':1000,
                    'training_seasons':[y for y in payload['frames'] if y<test]}
        block = LE.ensemble_block(manifest,frame,frame)
        part = payload['parts'][T.SPECS[0]][test]
        _,y,hfa,margins,meta = part
        finals = [{'week':int(g.week),'season_type':'regular','home':g.home_team,'away':g.away_team,
                   'neutral':bool(g.neutral_site),'home_score':float(margins[i]),'away_score':0}
                  for i,g in enumerate(meta.itertuples())]
        stats = payload['raw'][test]
        form = stats[list(ER.FORM_FIELDS)].to_dict('records')
        pff = T.staged_pff_payload(test,list(frame.index))
        got = []
        for week in sorted(meta.week.unique()):
            state = ER.replay(block,finals,form,stop_before=int(week),pff=pff)['state']
            for i in np.flatnonzero(meta.week.to_numpy()==week):
                g=finals[i]; p,live,cross=B.features(block,state,g['home'],g['away'],float(hfa[i]))
                # True deduplicated preseason union; shared form appears once.
                x=dict(zip(UNION,M.matchup_vector(frame,g['home'],g['away'],UNION)))
                for name,value in live.items():
                    member,key=name.split(':')
                    if key=='prior_level': continue # replaced by the underlying feature union
                    column=member+':'+key if key=='elo_change' else key
                    if column in x and not np.isclose(x[column],value,rtol=0,atol=1e-12):
                        raise ValueError(f'{column} is not shared across members; retain separate feature columns')
                    x[column]=float(value)
                x['hfa']=float(hfa[i])
                records.append({'season':test,'week':int(week),'home':g['home'],'away':g['away'],
                                'y':int(y[i]),'p':p,'x':x,'cross':cross})
                # Reconcile only exact PFF-table weeks against pandas; missing weeks use base stack.
                if str(int(week)-1) in pff['cutoffs']:
                    got.append(float(abs(p.mean()-np.mean(expected,axis=0)[i])))
        error=max(got,default=0.)
        if error>1e-8: raise AssertionError(f'{test} runtime differs from fitted stack by {error}')
        reconciliations.append({'season':test,'games':len(meta),'pff_reconciled_games':len(got),'max_abs_error':error})
        print(f'Recovered {test}: {len(meta)} games, runtime error {error:.3g}',flush=True)
    return records, {'production_war':production,'expanded_v52_war':expanded,
                     'older_expanded_war':fingerprint(PFF_API_DIR/'war_all_facets/preseason_team_war.csv'),
                     'depth_chart':fingerprint(TWODEEP_2026),'pfsn_master':fingerprint(PFSN_MASTER),
                     'pff_export_files':len(list(PFF_DIR.glob('*.csv'))),
                     'pff_weekly_files':len(list((PFF_API_DIR/'team_stats_weekly').glob('*.csv'))),
                     'reconciliation':reconciliations,
                     'frame_rows':{str(y):len(f) for y,f in frames.items()},
                     'inherited_input_coverage':{str(y):{k:float(f.attrs[k]) for k in
                         ('pff_coverage','war_coverage','war_projected_coverage') if k in f.attrs}
                         for y,f in frames.items()},
                     'player_production':fingerprint(ROOT/'war_model/preseason_player_components.csv'),
                     'fold_selections':fingerprint(SELECTIONS)}


def evaluate(rows):
    seasons=np.array([r['season'] for r in rows]); weeks=np.array([r['week'] for r in rows]); y=np.array([r['y'] for r in rows])
    P=np.array([r['p'] for r in rows]); Z=logit(np.clip(P,1e-8,1-1e-8))
    keys=[*UNION,*[s+':elo_change' for s in T.SPECS],'dO','dD','pff_O_diff','hfa']
    X=np.array([[r['x'].get(k,0.) for k in keys] for r in rows]); crosses=np.array([r['cross'] for r in rows])
    names=['equal_probabilities','learned_convex_blend','single_logit_stack','feature_union',
           'stack_plus_od_product','union_plus_od_product','union_plus_od_gap']
    predictions={n:np.full(len(y),np.nan) for n in names}; folds=[]
    for test in sorted(set(seasons))[1:]:
        train,held=chronological_masks(seasons,test)
        weights=B.convex_fit(P[train],y[train]); beta=B.logistic_fit(Z[train],y[train],np.full(4,.25))
        raw=B.logistic_fit(X[train],y[train])
        predictions['equal_probabilities'][held]=P[held].mean(axis=1)
        predictions['learned_convex_blend'][held]=P[held]@weights
        predictions['single_logit_stack'][held]=expit(Z[held]@beta)
        predictions['feature_union'][held]=expit(X[held]@raw)
        b=B.logistic_fit(crosses[train,1:2],y[train],offset=Z[train]@beta)
        predictions['stack_plus_od_product'][held]=expit(Z[held]@beta+crosses[held,1:2]@b)
        for name,index in [('union_plus_od_product',1),('union_plus_od_gap',0)]:
            A=np.column_stack([X,crosses[:,index]]); b=B.logistic_fit(A[train],y[train])
            predictions[name][held]=expit(A[held]@b)
        folds.append({'test_season':int(test),'train_seasons':sorted(map(int,set(seasons[train]))),
                      'train_games':int(train.sum()),'test_games':int(held.sum()),'blend_weights':weights.tolist()})
    take=np.isfinite(predictions['equal_probabilities']); yy=y[take]; ss=seasons[take]; ww=weeks[take]
    base=predictions['equal_probabilities'][take]
    report={}
    for name,values in predictions.items():
        p=np.clip(values[take],1e-9,1-1e-9)
        report[name]={**B.metrics(yy,p),'calibration':calibration(yy,p),
                      'paired':uncertainty(yy,p,base,ss,ww),
                      'by_season':{str(s):B.metrics(yy[ss==s],p[ss==s]) for s in sorted(set(ss))},
                      'by_season_week':{f'{s}:{w}':B.metrics(yy[(ss==s)&(ww==w)],p[(ss==s)&(ww==w)]) for s,w in sorted(set(zip(ss,ww)))}}
    export=pd.DataFrame([{k:r[k] for k in ('season','week','home','away','y')} for r in rows])
    for name,p in predictions.items(): export[name]=p
    export.loc[take].to_csv(OUT.with_name('historical_ensemble_predictions.csv'),index=False)
    return {'metrics':report,'folds':folds,'feature_columns':keys,'member_names':T.SPECS,
            'warmup_season':int(seasons.min()),'games_recovered':len(rows),'evaluation_games':int(take.sum())}


def write_report(report):
    from html import escape
    labels={'equal_probabilities':'Equal probability average (lean baseline)',
            'learned_convex_blend':'Learned constrained probability weights',
            'single_logit_stack':'Regularized member logit model',
            'feature_union':'Deduplicated preseason feature union + live inputs',
            'stack_plus_od_product':'Member logit model + opposing O×D',
            'union_plus_od_product':'Feature union + opposing O×D',
            'union_plus_od_gap':'Feature union + nonlinear opposing gap'}
    table='| Candidate | Brier | Δ vs equal | Log loss | 2024 Brier | 2025 Brier |\n|---|---:|---:|---:|---:|---:|\n'
    htmlrows=[]
    for name,m in report['metrics'].items():
        values=[f"{m['brier']:.6f}",f"{m['paired']['brier']['difference']:+.6f}",f"{m['log_loss']:.6f}",
                f"{m['by_season']['2024']['brier']:.6f}",f"{m['by_season']['2025']['brier']:.6f}"]
        table+='| '+labels[name]+' | '+' | '.join(values)+' |\n'
        htmlrows.append('<tr><td>'+escape(labels[name])+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    text='''# Historical recovery and ensemble comparison

October 12, 2026. Research only; production continues to average four member probabilities equally.

## What was recovered

The actual local checkout under `C:/Users/marks/Projects/04 - College Football Model`
contains the previously missing external source tree. Both purchased workbooks,
2014–2025 player PFF exports, historical team-season and weekly PFF tables, CFBD
caches, expanded-WAR trees, and earlier multi-season prediction exports were found.
Frames for 2021–2025 were regenerated through the production loaders and saved
privately under `artifacts/historical_research/`; no licensed rows are published.
The committed production preseason WAR file is byte-identical to the external
`war_all_facets_v52/preseason_team_war.csv`. The older `war_all_facets` file differs
and was not substituted. The output JSON records hashes and coverage counts.

Local PFF access successfully retrieved a 2025 team directory (265 rows), 2024
blocking (5,808 rows), and historical 2025 offense/defense tables (281 rows each).
The repository `PFF_API_KEY` is now configured. A fresh Actions probe successfully recovered both CFBD and PFF (run 38340063670).
See `audit/output/source_recovery_status.json` for both status-only probes.
The local cache and fresh Actions advanced-game responses have 3,314 and 3,316
rows respectively: source retrieval vintages differ even when credentials work.

## Baseline and chronological contract

Fresh member fits reuse the recorded tuning selections made inside each earlier-season
training pool, bypassing the unversioned pickle cache. Sanitized selections are now
committed so the experiment no longer depends on an ignored JSON artifact. Preseason
projection training uses earlier years; full season roster membership remains a
retrospective approximation to the actual preseason roster.

The lean v5.3 PFF/base architecture was replayed on 2,189 games in 2023–2025.
Canonical runtime probabilities reconcile with independently fitted pandas stacks
on eligible exact-table PFF weeks to at most 6.2e-12. Missing historical tables
use the base stack; the 2026 carry-forward policy is not retroactively imposed.
The current 2026 full production architecture is separately reconciled against all
138 published power ratings. These are two different baseline checks.

The historical in-season WAR signal is excluded. Its saved calibration uses
leave-season-out rules: the 2023 rule was trained on 2022, 2024 and 2025.
That avoids its own year but uses future years, so it is unsuitable for a strictly
chronological comparison. A complete current-WAR multi-season baseline still needs
earlier-year-only calibration of the player filter, facet weights and production
conversion, followed by fresh team histories and member stack fits.

2023 supplies 729 out-of-fold member predictions for training. Test 2024 uses only
2023; test 2025 uses 2023–2024. All alternatives are evaluated on the identical 1,460
games in 2024–2025. New logistic scalers and coefficients and constrained blend
weights use training rows only. Penalties were fixed before scoring; there is no
held-out hyperparameter search or automatic candidate selection. Inherited baseline
PCA/residualization and standardization use each year's known preseason team universe;
this is not a complete refit of every upstream transform on a pooled training-only
universe. Other upstream WAR parameter provenance still needs auditing.

The feature-union candidate uses the actual deduplicated union of the four preseason
feature lists, replacing aggregate member preseason logits, plus member-specific score
updates, shared available form and home field. It is one final logistic probability.
This is substantially more complete than the earlier aggregate-live-input experiment.
It retains the existing score update mechanics rather than estimating a new state model.

## Matched results

'''+table+'''
Learned weights stay close to 25% each and are effectively tied with equal averaging.
The logit model improves slightly in 2025 but loses more in 2024. The full feature
union loses in both seasons at the fixed penalty. Opposing-unit products and nonlinear
gaps do not repair that loss. The 2026 product result did not generalize in this
recovered comparison. This does not rule out other carefully regularized feature
unions or interaction families, and is not evidence for a production replacement.

The product `O_A*D_B - O_B*D_A` and nonlinear gap
`g(O_A-D_B) - g(O_B-D_A)`, `g(x)=x*abs(x)`, negate when teams swap.
Logistic models have no intercept or centering, so neutral probabilities remain
complementary. Offense and defense are standardized signals, not predicted points.

The JSON reports log loss, calibration bins/ECE, every season-week score, paired
season-week bootstrap intervals and whole-season bootstrap intervals. The paired
intervals include zero for all alternatives. There are only two evaluated seasons;
week blocks do not capture all team dependence, and these intervals do not correct
for historical candidate selection. Retrospective source pulls do not establish
prospective betting performance. Original timestamped historical source snapshots
cannot be recovered from current API responses alone.

## Remaining dependencies and decision

Keep production unchanged. Remaining work is specific: reconstruct chronological
in-season WAR rules and audit all upstream facet/conversion fits; establish source
vintages or collect prospective shadow predictions. The repository PFF secret is now
configured and verified for future captures. The unique workbooks are available locally but are
not public or API-regenerable. API credentials and licensed raw tables stay private.

## Reproduce

`python -m scripts.historical_ensemble_research` with the external source tree and
the existing Python dependencies. It regenerates frames privately and emits aggregate
JSON, game-level predictions and this report. It never writes production model,
kickoff ledgers, frozen predictions or locked bets. Tests:
`python -m unittest tests.test_historical_ensemble_research tests.test_pff_carry_forward tests.test_ensemble_blend_research tests.test_ensemble_replay`.
Use `PYTHON` to select the interpreter when running the Node X-Ray tests on Windows.
'''
    (ROOT/'audit/HISTORICAL_ENSEMBLE_RESEARCH.md').write_text(text,encoding='utf-8')
    section='''<section id="historical-recovery"><div class="eyebrow">RECOVERED HISTORY / OCTOBER 12, 2026</div><h2>Multi-season evidence changes the picture</h2><p>The local source tree was recovered: historical PFF tables, expanded WAR, and the original feature inputs are available. Fresh lean member fits reproduce the runtime on 2,189 games from 2023–2025. Local and GitHub Actions PFF API access both work; the repository secret was configured and verified in run 38340063670.</p><p class="note">This comparison excludes historical in-season WAR because its saved calibration uses future seasons. It tests the lean PFF/base architecture, not the complete current WAR ensemble. The current 2026 production ratings were reconciled separately.</p><p>Train on 2023, test 2024; then train on 2023–2024, test 2025. All candidates share 1,460 held-out games. The feature union uses the actual preseason inputs once each, plus live inputs. New scaling and weights use training rows only; inherited preseason transforms and source vintages remain retrospective.</p><div class="table-scroll"><table><thead><tr><th>Candidate</th><th>Brier</th><th>Δ vs equal</th><th>Log loss</th><th>2024 Brier</th><th>2025 Brier</th></tr></thead><tbody>'''+''.join(htmlrows)+'''</tbody></table></div><p>Learned weights are effectively tied with equal averaging. The feature union loses in both seasons, and the interaction candidates do not improve the pooled score. The favorable 2026 interaction result did not generalize here. Production remains unchanged.</p><p class="note">Paired uncertainty intervals include zero. Only two test seasons, retrospective source pulls, and incomplete upstream WAR chronology prevent a production decision. A complete historical WAR comparison needs earlier-season-only player calibration and rebuilt team histories.</p><p><a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/HISTORICAL_ENSEMBLE_RESEARCH.md">Recovery, methodology and remaining dependencies</a> · <a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/output/historical_ensemble_research.json">Calibration, stability and paired intervals</a> · <a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/output/historical_ensemble_predictions.csv">Matched game predictions</a></p></section>'''
    page=ROOT/'viz/power-xray/research.html'; html=page.read_text(encoding='utf-8')
    start='<!-- recovered-history:start -->';end='<!-- recovered-history:end -->'
    if start in html:
        html=html[:html.index(start)]+html[html.index(end)+len(end):]
    html=html.replace('But the multi-season prediction exports and original preseason frames are unavailable.',
                      'The local historical source tree has since been recovered; the multi-season comparison above supplies newer evidence.')
    html=html.replace('<section><div class="eyebrow">MATCHED GAMES / LOWER IS BETTER',
                      start+section+end+'<section><div class="eyebrow">MATCHED GAMES / LOWER IS BETTER',1)
    page.write_text(html,encoding='utf-8')


def main():
    rows,provenance=recover(); report=evaluate(rows)
    e=json.loads((ROOT/'viz/data/model_v4.json').read_text())['ensemble']
    ratings=json.loads((ROOT/'viz/data/ratings.json').read_text())['teams']
    powers={r['team']:r['power'] for r in ER.power_table(e,e['state'],[r['team'] for r in ratings])}
    error=max(abs(powers[r['team']]-r['power']) for r in ratings)
    if error>2e-6:raise AssertionError('Published 2026 power ratings do not reconcile')
    report['current_production_reconciliation']={'teams':len(ratings),'max_power_difference':error,
          'manifest':fingerprint(ROOT/'viz/data/model_v4.json'),
          'locked_bets':fingerprint(ROOT/'viz/data/locked_results_2026.json'),
          'bet_tracking':fingerprint(ROOT/'viz/data/bet_tracking.json')}
    report.update({'status':'retrospective_reconstruction_no_promotion','provenance':provenance,
                   'penalties':{'logistic_l2':.1,'convex_shrink_to_equal':.01,'selection':'fixed before scoring; no candidate hyperparameter search'},
                   'limitations':[
                       'Historical in-season WAR omitted: its leave-season-out rules use future seasons. This is the lean PFF/base architecture, not the complete current WAR ensemble.',
                       '2023 supplies earlier-season out-of-fold member predictions; alternatives are evaluated only on 2024 and 2025.',
                       'Retrospectively retrieved PFF/CFBD data and roster membership cannot establish original pregame availability.',
                       'Inherited preprocessing and expanded-WAR projections need further provenance auditing; this does not establish every upstream parameter was fit chronologically.',
                       'Member tuning reuses recorded earlier-season selections; multiple families were historically examined. New penalties are fixed, not selected on these held-out games.',
                       'Two held-out seasons and many candidates cannot establish a production winner; block intervals do not account for model selection.',
                       'Licensed raw inputs remain local. Public outputs contain aggregate metrics and game predictions only.']})
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_report(report)
    for n,m in report['metrics'].items(): print(n,m['brier'],m['paired']['brier'],flush=True)


if __name__=='__main__': main()
