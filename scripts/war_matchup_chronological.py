"""Matched chronological ablations of rebuilt WAR and opposing-unit interactions.

Read-only production contract. Scaling, penalties and coefficients fit on earlier
data. Main-effect controls accompany every interaction. Licensed input tables stay
under source-data/artifacts; public outputs contain probabilities and attribution.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score,balanced_accuracy_score

from config import ROOT
from scripts import historical_ensemble_research as H,train_live_ensemble as T
from scripts import v5_extension_backtest as V,v4_backtest as BT,prior_decay_backtest as PD
from scripts import chronological_war_rebuild as R
from src import inseason_war as IW,pff_form as PF

OUT=ROOT/'audit/output/chronological_war_matchups.json'
C_GRID=(.01,.03,.1,.3,1.)
PAIRS={
 'passing_coverage':(('offense_passing','epa_per_pass_play',1),('defense_passing','epa_per_pass_play_allowed',-1)),
 'rushing_front':(('offense_rushing','epa_per_run_play',1),('defense_rushing','epa_per_run_play_allowed',-1)),
 'protection_pass_rush':(('offense_passing','pressure_rate_against',-1),('defense_passing','pressure_rate',1)),
 'run_blocking_front':(('offense_rushing','yards_before_contact_per_carry',1),('defense_rushing','yards_before_contact_per_carry_allowed',-1)),
 'explosive_pass_prevention':(('offense_passing','explosive_pass_rate',1),('defense_passing','explosive_pass_rate_allowed',-1)),
 'after_contact_tackling':(('offense_rushing','yards_after_contact_per_carry',1),('defense_rushing','yards_after_contact_per_carry_allowed',-1)),
}
MAIN=[f'{p}_{s}_diff' for p in PAIRS for s in ('O','D')]
PRODUCT=[f'{p}_product' for p in PAIRS]
GAP=[f'{p}_gap' for p in PAIRS]
BASE=['prior_level','elo_change','dD','hfa','pff_O_diff']
FAMILIES={
 'no_inseason_war':[],
 'chronological_war':['war_delta_diff'],
 'war_units_linear':['war_delta_diff',*MAIN],
 'war_units_product':['war_delta_diff',*MAIN,*PRODUCT],
 'war_units_gap':['war_delta_diff',*MAIN,*GAP],
 'war_units_product_gap':['war_delta_diff',*MAIN,*PRODUCT,*GAP],
 'no_war_units_product':[*MAIN,*PRODUCT],
}
# Targeted hypotheses have matched main effects and the same penalty search.
# These are exploratory families, not independent confirmatory significance tests.
for pair in PAIRS:
    linear=['war_delta_diff',f'{pair}_O_diff',f'{pair}_D_diff']
    FAMILIES[f'{pair}_linear']=linear
    FAMILIES[f'{pair}_product']=[*linear,f'{pair}_product']
    FAMILIES[f'{pair}_gap']=[*linear,f'{pair}_gap']


def build_contexts():
    frames,parts,raw,_=T.build(False)
    contexts={}
    for fold in json.loads(H.SELECTIONS.read_text())['folds']:
        test,spec=fold['test_season'],fold['member'];pool=fold['train_seasons'];sel=dict(fold['selection'])
        sel['halflife']=np.inf if sel['halflife'] is None else sel['halflife']
        models={pool[i]:BT.fit_predict(parts[spec],pool[:i],pool[i],PD.SPEC_FEATURES[spec],sel['knobs'])[0]
                for i in range(1,len(pool))}
        models[test]=BT.fit_predict(parts[spec],pool,test,PD.SPEC_FEATURES[spec],sel['knobs'])[0]
        contexts[(test,spec)]={'sel':sel,'models':models}
    return {'frames':frames,'parts':parts,'raw':raw,'ctx':contexts}


def unit_rows(games):
    cache={};rows=[]
    for g in games.itertuples(index=False):
        key=(int(g.season),int(g.week)-1)
        if key not in cache:
            tables={};ids=PF._directory(key[0])
            for category in {d[0] for pair in PAIRS.values() for d in pair}:
                path=PF.WEEKLY/f'{category}_{key[0]}_thru_w{key[1]:02d}.csv'
                if path.exists():
                    d=pd.read_csv(path,low_memory=False);d['team']=d.team_id.map(ids)
                    tables[category]=d.dropna(subset=['team']).drop_duplicates('team').set_index('team')
            cache[key]=tables
        row={k:getattr(g,k) for k in V.KEYS}
        for pair,definitions in PAIRS.items():
            for side,(category,col,sign) in zip(('O','D'),definitions):
                t=cache[key].get(category)
                for flag,team in (('h',g.home_team),('a',g.away_team)):
                    row[f'{pair}_{side}_{flag}']=float(t.at[team,col])*sign if t is not None and team in t.index and col in t else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def unit_transform(train,test):
    """Fit separate unit scales on training observations only; then cross opponents."""
    a=train.copy();b=test.copy();parameters={}
    for pair in PAIRS:
        for side in ('O','D'):
            h,f=f'{pair}_{side}_h',f'{pair}_{side}_a'
            values=train[[h,f]].to_numpy().reshape(-1);values=values[np.isfinite(values)]
            mu=float(values.mean()) if len(values) else 0.;sd=float(values.std()) if len(values) else 1.
            sd=sd if sd>1e-10 else 1.;parameters[pair+'_'+side]={'mean':mu,'scale':sd,'observations':len(values)}
            for frame in (a,b):
                for column in (h,f):frame[column]=(frame[column]-mu)/sd
                frame[pair+'_'+side+'_diff']=frame[h].fillna(0)-frame[f].fillna(0)
        for frame in (a,b):
            oh,oa,dh,da=[frame[f'{pair}_{s}_{f}'] for s,f in [('O','h'),('O','a'),('D','h'),('D','a')]]
            valid=np.isfinite(oh)&np.isfinite(oa)&np.isfinite(dh)&np.isfinite(da)
            frame[pair+'_product']=(oh*da-oa*dh).where(valid,0.)
            x,z=oh-da,oa-dh
            frame[pair+'_gap']=(x*abs(x)-z*abs(z)).where(valid,0.)
    return a,b,parameters


def select_c(designs,extras):
    """Nested chronological validation. No held-out outer labels or scales."""
    years=sorted(designs)
    train0=pd.concat([designs[y] for y in years[:-1]],ignore_index=True) if len(years)>1 else designs[years[0]][designs[years[0]].week<9]
    valid0=designs[years[-1]] if len(years)>1 else designs[years[0]][designs[years[0]].week>=9]
    if len(train0)<30 or len(valid0)<20:return .1,[]
    train,valid,_=unit_transform(train0,valid0);cols=BASE+extras
    scores=[]
    for c in C_GRID:
        fitted=PD.fit_stack(train,cols,c);p=np.clip(PD.stack_predict(fitted,valid,cols),1e-9,1-1e-9);y=valid.y.to_numpy()
        scores.append({'C':c,'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))})
    return min(scores,key=lambda s:s['log_loss'])['C'],scores


def metrics(y,p):
    p=np.clip(np.asarray(p),1e-9,1-1e-9);y=np.asarray(y)
    z=logit(p)
    def objective(b):
        t=b[0]+b[1]*z
        return float(np.mean(np.logaddexp(0,t)-y*t))
    calibration=minimize(objective,[0.,1.],method='BFGS').x
    confident=(p>=.8)|(p<=.2);wrong=(p>=.5)!=y
    return {**H.B.metrics(y,p),'auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)) if len(set(y))==2 else None,
            'calibration_intercept':float(calibration[0]),'calibration_slope':float(calibration[1]),
            **H.calibration(y,p),'confident_games':int(confident.sum()),
            'confident_error_rate':float(wrong[confident].mean()) if confident.any() else None,
            'mean_entropy':float(-np.mean(p*np.log(p)+(1-p)*np.log1p(-p)))}


def run():
    payload=build_contexts()
    games=pd.concat([part[4].assign(season=s)[V.KEYS] for s,part in payload['parts'][T.SPECS[0]].items()],ignore_index=True)
    if games.duplicated(V.KEYS).any():raise ValueError('Duplicate game keys')
    rawunits=unit_rows(games);extra=T.pff_features(payload['parts'])
    deltas=pd.read_csv(R.PRIVATE/'team_history.csv')
    extra['war_delta_diff']=IW.game_war_column(extra,deltas)
    extra=extra.merge(rawunits,on=V.KEYS,validate='one_to_one')
    records=[];folds=[];explanations=[]
    for test in V.OUTER:
        preds={n:[] for n in FAMILIES};terms={n:[] for n in FAMILIES};traces=[]
        for spec in T.SPECS:
            designs={y:d.merge(extra,on=V.KEYS,validate='one_to_one') for y,d in V.member_designs(payload,test,spec).items()}
            for d in designs.values():d[['pff_O_diff','war_delta_diff']]=d[['pff_O_diff','war_delta_diff']].fillna(0.)
            earlier={y:d for y,d in designs.items() if y<test}
            train0=pd.concat(list(earlier.values()),ignore_index=True);held0=designs[test]
            train,held,scales=unit_transform(train0,held0)
            for name,extras in FAMILIES.items():
                c,trace=select_c(earlier,extras);cols=BASE+extras
                fitted=PD.fit_stack(train,cols,c);p=PD.stack_predict(fitted,held,cols)
                coef=fitted[1].coef_[0]/fitted[0].scale_
                individual=held[cols].to_numpy(float)*coef
                if not np.allclose(expit(individual.sum(axis=1)),p,atol=1e-12):raise AssertionError('Attribution mismatch')
                if not np.allclose(expit(-individual.sum(axis=1))+p,1.,atol=1e-12):raise AssertionError('Reciprocity failed')
                preds[name].append(p);terms[name].append((cols,individual))
                traces.append({'member':spec,'candidate':name,'C':c,'inner_scores':trace,'training_seasons':sorted(earlier),
                               'scales':scales,'coefficients':dict(zip(cols,map(float,coef)))})
        meta=held0[V.KEYS+['y']].copy()
        for name,p in preds.items():meta[name]=np.mean(p,axis=0)
        records.append(meta)
        for i,g in enumerate(meta.itertuples(index=False)):
            # Each member's exact log-odds decomposition; ensemble = mean sigmoid.
            explanations.append({'season':test,'week':int(g.week),'home':g.home_team,'away':g.away_team,
                'outcome':int(g.y),'candidates':{n:{'probability':float(meta.iloc[i][n]),
                    'members':[dict(zip(cols,map(float,values[i]))) for cols,values in terms[n]]} for n in FAMILIES}})
        folds.append({'season':test,'train_seasons':sorted(earlier),'games':len(meta),'fits':traces})
        print(f'{test}: evaluated {len(meta)} matched games across {len(FAMILIES)} candidates',flush=True)
    allrows=pd.concat(records,ignore_index=True)
    # 2022 is player-rule warmup; 2023 is first OOF WAR history and therefore the
    # first season from which the team stack can learn its WAR coefficient.
    evaluated=allrows[allrows.season>=2024].copy();y=evaluated.y.to_numpy();ss=evaluated.season.to_numpy();ww=evaluated.week.to_numpy()
    report={}
    for name in FAMILIES:
        p=evaluated[name].to_numpy()
        controls=['no_inseason_war']
        if name not in ('no_inseason_war','chronological_war'):controls+=['chronological_war','war_units_linear']
        for pair in PAIRS:
            if name in (f'{pair}_product',f'{pair}_gap'):controls.append(f'{pair}_linear')
        report[name]={'overall':metrics(y,p),
          'by_season':{str(s):metrics(d.y,d[name]) for s,d in evaluated.groupby('season')},
          'by_season_week':{f'{s}:{w}':H.B.metrics(d.y.to_numpy(),d[name].to_numpy()) for (s,w),d in evaluated.groupby(['season','week'])},
          'paired':{c:H.uncertainty(y,np.clip(p,1e-9,1-1e-9),np.clip(evaluated[c].to_numpy(),1e-9,1-1e-9),ss,ww) for c in controls}}
    result={'status':'retrospective_chronological_rebuild_no_automatic_promotion','evaluation_games':len(evaluated),
      'evaluation_seasons':[2024,2025],'warmup_seasons':[2022,2023],'metrics':report,'folds':folds,
      'war_provenance':json.loads((R.PRIVATE/'provenance.json').read_text()),
      'decision_criteria':'Joint Brier/log-loss changes, calibration intercept/slope and ECE, AUC, confidence errors, season/week stability and paired uncertainty. Interaction controls include the same main effects. No automatic promotion.',
      'limitations':['PFF/CFBD sources are retrospective, not archived as-of-kickoff snapshots.',
        'This rebuild corrects the complete in-season WAR measurement/calibration chain. The four preseason member inputs and structural feature choices retain the recovered production lineage.',
        'Only two seasons have both earlier calibrated WAR and earlier team-stack training; 2023 is reported as warmup, not credited as independent WAR evidence.',
        'PFF facet-WAR omits CFBD play-value window facets; historical injuries are not reconstructed.',
        'Several interaction families are evaluated; paired intervals do not correct for this multiplicity or all cross-week team dependence.'],
      'unit_pairs':{p:{'offense':list(a),'defense':list(b)} for p,(a,b) in PAIRS.items()}}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    allrows.to_csv(OUT.with_name('chronological_war_matchup_predictions.csv'),index=False)
    from scripts.publish_war_research import publish
    publish({'metrics':report,'games':explanations,'status':result['status'],'pair_definitions':result['unit_pairs']})
    for name,d in report.items():print(name,{k:d['overall'][k] for k in ('brier','log_loss','auc','calibration_slope','ece')},flush=True)
    return result


if __name__=='__main__':run()
