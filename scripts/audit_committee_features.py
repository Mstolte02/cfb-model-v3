"""Chronological ablation; select on 2018-23, report 2024-25 separately.
Run from repository root: python -m scripts.audit_committee_features.
The 2024-25 data have been viewed in earlier research; not a fresh holdout.
"""
import json
import numpy as np
from sklearn.linear_model import Ridge
from scripts import train_committee_timeline as t
from collections import defaultdict
base=tuple(k for k in t.FEATURES if k not in ('prior_sq','recent_margin'))
D=json.loads((t.ROOT/'data/committee_history.json').read_text())
f=t.snapshots(D);c=t.snapshots(D,True)
for name,cap,mode in [('main',35,'none'),('nonlinear',35,'none'),('combined_raw',35,'signed'),('cap14',14,'signed'),('cap21',21,'signed'),('log',35,'log'),('wins',35,'wins'),('quality_margin',35,'quality'),('close_win',7,'close')]:
 for frame in [f,c]:
  vals=[]
  for r in frame.itertuples():
   polls=D['seasons'][str(r.season)]['polls'];current=next(p for p in polls if p['week']==r.week);prev=polls[polls.index(current)-1]
   val=0
   for g in D['seasons'][str(r.season)]['games']:
    if not(prev['week']-1<g['week']<r.week) or g['title'] or r.team not in (g['home'],g['away']):continue
    m=(g['home_score']-g['away_score'])*(1 if g['home']==r.team else -1)
    if mode=='signed':v=np.clip(m,-cap,cap)/7
    elif mode=='log':v=np.sign(m)*np.log1p(abs(m))/np.log(8)
    elif mode=='wins':v=max(0,min(m,cap))/7
    elif mode=='quality':v=np.clip(m,-cap,cap)/7*(30-prev['rankings'].get(g['away'] if g['home']==r.team else g['home'],30))/30
    else:v=int(0<m<=7)
    val+=v
   vals.append(val)
  frame['margin_candidate']=vals
 features=base if name=='main' else base+('prior_sq',) if name=='nonlinear' else base+('prior_sq','margin_candidate');t.FEATURES=features;weeks=[]
 for year in [2018,2019,2021,2022,2023,2024,2025]:
  train=f[(f.season<year)&(f.season!=2020)];model=Ridge(alpha=30).fit(train[list(features)],train['rank'])
  weeks+=t.weekly_predictions(model,c[c.season==year],D,.25)
 dev=np.mean([x['mae'] for x in weeks if x['season']<=2023]);hold=np.mean([x['mae'] for x in weeks if x['season']>=2024])
 print(name,round(dev,4),round(hold,4),flush=True)
