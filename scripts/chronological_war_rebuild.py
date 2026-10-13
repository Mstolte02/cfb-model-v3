"""Rebuild WAR calibration in private, strictly earlier-season work directories.

Never run the production writer against the public WAR tree. Every supervised
facet/composite/reliability/wins-map fit sees years strictly before its target.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from config import ROOT

PRIVATE = ROOT/'artifacts/chronological_war'
WAR = ROOT/'war_model'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(year):
    dest=PRIVATE/str(year);dest.mkdir(parents=True,exist_ok=True)
    source={n:digest(WAR/n) for n in ('candidate_values.parquet','cfbd_facet_values.parquet','records.csv','schedule.csv')}
    stamp={'target_season':year,'sources':source,'contract':'all fitting seasons strictly before target','version':1}
    marker=dest/'weights_manifest.json'
    if marker.exists():
        cached=json.loads(marker.read_text())
        if cached.get('inputs')==stamp and cached.get('outputs') and all(
                (dest/name).exists() and digest(dest/name)==expected
                for name,expected in cached['outputs'].items()):
            print(f'{year}: verified cached chronological facet build',flush=True);return dest
    for name in ('candidate_values.parquet','cfbd_facet_values.parquet'):
        d=pd.read_parquet(WAR/name);d=d[d.season<year].copy()
        if d.empty or d.season.max()>=year:raise ValueError('Invalid facet cutoff')
        d.to_parquet(dest/name,index=False)
    for name in ('records.csv','schedule.csv'):
        d=pd.read_csv(WAR/name);d[d.season<year].to_csv(dest/name,index=False)
    for name in ('team_map.json','candidate_catalogue.json'):
        shutil.copyfile(WAR/name,dest/name)
    # Imports use the repository's production implementation, with all data/output
    # roots redirected before main runs. Reliability is recomputed for this fold.
    sys.path.insert(0,str(WAR))
    build=importlib.import_module('build_hybrid')
    tlw=importlib.import_module('two_level_weights')
    build.HERE=str(dest);tlw.HERE=str(dest)
    build.YEARS=sorted(pd.read_csv(dest/'records.csv').season.unique().tolist())
    os.environ['WAR_FEATURES']='candidates';os.environ['WAR_CONSOLIDATE']='1'
    os.environ['WAR_TARGET']='next_season';os.environ['WAR_ROLE_NORM']='pooled'
    # Do not inherit a selected reliability cache from another fit.
    reliability=dest/tlw.RELIABILITY
    if reliability.exists():reliability.unlink()
    print(f'{year}: fitting facet/composite/WAA pipeline through {max(build.YEARS)}',flush=True)
    with (dest/'build.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        build.main()
    marker.write_text(json.dumps({'inputs':stamp,'fit_seasons':build.YEARS,
        'outputs':{n:digest(dest/n) for n in ('hybrid_facet_war.parquet','hybrid_player_war.csv','hybrid_wins_map.json','consolidated_facets.json')}},indent=2))
    print(f'{year}: chronological facet build complete',flush=True)
    return dest


def measurements(files, season, cache_name):
    """Unstandardized licensed measurements, cached only in ignored artifacts."""
    from src import war_window as ww
    import candidates as cand
    cache=PRIVATE/'raw';cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{cache_name}.parquet';meta=path.with_suffix('.json')
    signature={str(p):digest(p) for p in files.values()}
    if path.exists() and meta.exists() and json.loads(meta.read_text())==signature:
        return pd.read_parquet(path)
    players=ww.load_players_from(files,season);rows=[]
    for name,metric,denom,positions,kind in cand.build_catalogue():
        if metric not in players or denom not in players:continue
        d=players.loc[players.position.isin(positions),['player_id','player','position','team_name']].copy()
        d['metric']=pd.to_numeric(players.loc[d.index,metric],errors='coerce')
        d['snaps']=pd.to_numeric(players.loc[d.index,denom],errors='coerce')
        d=d[(d.snaps>=cand.MIN_DENOM[kind]) & d.metric.notna()]
        if kind==cand.COUNT:d['metric']/=d.snaps
        if metric in cand.LOWER_IS_BETTER:d['metric']=-d.metric
        rows.append(d.assign(facet=name,season=season))
    out=pd.concat(rows,ignore_index=True);out['player_id']=out.player_id.astype(str)
    out.to_parquet(path,index=False);meta.write_text(json.dumps(signature))
    return out


def fitted_moments(raw, target):
    train=raw[raw.season<target]
    rows={}
    for facet,d in train.groupby('facet'):
        if len(d)<200:continue
        mu=float(np.average(d.metric,weights=d.snaps))
        sd=float(np.sqrt(np.average((d.metric-mu)**2,weights=d.snaps)))
        if sd>1e-12:rows[facet]=(mu,sd)
    return rows


def score(raw, moments, dest, target):
    """Apply earlier-season moments, PCA loadings, weights and team-total scales."""
    from src import war_window as ww
    fv=raw[raw.facet.isin(moments)].copy()
    fv['value']=np.clip((fv.metric-fv.facet.map({k:v[0] for k,v in moments.items()})) /
                        fv.facet.map({k:v[1] for k,v in moments.items()}),-6,6)*fv.snaps
    prod=ww._production();ws=int(prod.season.max())
    if ws>=target:raise ValueError('Facet scoring reads the target season')
    pw=prod[prod.season==ws].drop_duplicates('facet').set_index('facet')
    coefs,_=ww.recover_composites()
    columns=['season','player_id','facet','value','snaps']
    parts=[fv[fv.facet.isin(pw.index)][columns]]
    for (s,comp),cf in coefs.items():
        if s!=ws:continue
        d=fv[fv.facet.isin(cf)].copy();d['value']*=d.facet.map(cf)
        parts.append(d.groupby(['season','player_id'],as_index=False).agg(value=('value','sum'),snaps=('snaps','max')).assign(facet=comp))
    f=pd.concat(parts,ignore_index=True)
    f['fc']=f.facet.map(pw.w)*f.value/f.facet.map(pw.sigma)
    info=fv.drop_duplicates(['season','player_id']).set_index(['season','player_id'])[['player','position','team_name']]
    out=f.groupby(['season','player_id']).agg(fc=('fc','sum'),snaps=('snaps','max')).join(info).reset_index()
    out['group']=out.position.map(ww.GROUP)
    return out


def player_frames(year):
    from scripts import war_inseason_backtest as B
    from src import inseason_war as IW,war_window as ww,opponent_strength as O
    dest=weights(year)
    # Deliberate cache invalidation when switching the private fold root.
    ww.WAR=dest;ww._production.cache_clear();ww.recover_composites.cache_clear()
    raw=pd.concat([measurements(ww.season_files(s),s,f'season_{s}')
                   for s in B.SEASONS_ALL if s<year],ignore_index=True)
    moments=fitted_moments(raw,year)
    (dest/'moments.json').write_text(json.dumps(moments))
    hist=score(raw,moments,dest,year);hist.to_parquet(dest/'history.parquet',index=False)
    pr=B.priors(hist,year);gp=pr.groupby('group').agg(mu=('mu','first'),tau2=('tau2','first'))
    team_map=json.loads((WAR/'team_map.json').read_text())
    def enrich(d):
        d=d.copy();d['team']=d.team_name.map(lambda x:IW.canonical_team([x],team_map))
        d=d.merge(pr[['player_id','group','m','P']],on=['player_id','group'],how='left',validate='many_to_one')
        d['m']=d.m.fillna(d.group.map(gp.mu));d['P']=d.P.fillna(d.group.map(gp.tau2))
        return d
    rows=[];cal=[]
    for cut in (3,6,9):
        def window(a,b):
            directory=B.window_dir(year,a,b)
            files={B.PREFIX[n]:directory/f'{n}.csv' for n in (*B.LEGACY,*B.POSITION)}
            return score(measurements(files,year,f'{year}_{a}_{b}'),moments,dest,year)
        early=enrich(window(1,cut));late=window(cut+1,16)
        early=early[(early.snaps>=IW.MIN_SNAPS)&early.team.notna()&early.group.notna()].copy()
        early['obs']=early.fc/early.snaps*1000
        early['diff_w']=O.player_difficulty(early.team,early.group,O.window_difficulty(year,1,cut,through=cut))
        early['cut']=cut;rows.append(early)
        d=early.merge(late[['player_id','snaps','fc']],on='player_id',suffixes=('_w','_t'),validate='one_to_one')
        d=d[d.snaps_t>=B.MIN_TARGET].copy();d['target']=d.fc_t/d.snaps_t*1000
        d['diff_t']=O.player_difficulty(d.team,d.group,O.window_difficulty(year,cut+1,16,through=16))
        cal.append(d)
    player=pd.concat(rows,ignore_index=True);fr=pd.concat(cal,ignore_index=True)
    player.to_parquet(dest/'window_rows.parquet',index=False);fr.to_parquet(dest/'calibration_rows.parquet',index=False)
    # Conversion is refitted against this fold's rebuilt production WAA, on past years.
    w=pd.read_csv(dest/'hybrid_player_war.csv',dtype={'player_id':str})
    w=w.groupby(['season','player_id'],as_index=False).waa.sum()
    joined=hist.merge(w,on=['season','player_id'],validate='one_to_one')
    k={g:float(np.polyfit(d.fc,d.waa,1)[0]) for g,d in joined.groupby('group')}
    (dest/'conversion.json').write_text(json.dumps(k))
    print(f'{year}: rebuilt {len(player)} player-cut inputs and {len(fr)} calibration targets',flush=True)


def rebuild():
    from scripts import war_inseason_backtest as B
    from src import inseason_war as IW
    output=[];audit=[]
    for year in (2022,2023,2024,2025):
        player_frames(year)
        dest=PRIVATE/str(year);prior=[s for s in (2022,2023,2024,2025) if s<year]
        if not prior:
            audit.append({'season':year,'status':'calibration_warmup','training_seasons':[]});continue
        train=pd.concat([pd.read_parquet(PRIVATE/str(s)/'calibration_rows.parquet') for s in prior],ignore_index=True)
        if train.season.max()>=year:raise ValueError('Calibration reads future seasons')
        beta=B.fit_opponent_beta(train);adjusted=B.apply_opponent(train,beta)
        rule=IW.fit_rule(adjusted);k=json.loads((dest/'conversion.json').read_text())
        rows=pd.read_parquet(dest/'window_rows.parquet')
        rows['obs']-=rows.group.map(beta).fillna(0)*rows.diff_w
        delta=IW.team_deltas(rows,rule,k)
        output.append(delta)
        params={'training_seasons':prior,'rule':rule,'k':k,'opp_beta':beta}
        (dest/'params.json').write_text(json.dumps(params,indent=2))
        audit.append({'season':year,'status':'rebuilt','training_seasons':prior,'team_cuts':len(delta),
                      'facet_fit_max_season':year-1,'normalization_max_season':year-1,
                      'conversion_max_season':year-1,'opponent_observation_cutoff':'through window cut',
                      'rule_sha256':digest(dest/'params.json')})
        print(f'{year}: calibrated only on {prior}; {len(delta)} team-cut WAR deltas',flush=True)
    pd.concat(output,ignore_index=True).to_csv(PRIVATE/'team_history.csv',index=False)
    (PRIVATE/'provenance.json').write_text(json.dumps({'folds':audit,'definition':'chronological PFF facet-WAR update; CFBD play-value facets have no window source'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--year',type=int)
    args=ap.parse_args()
    if args.year:weights(args.year)
    else:rebuild()
