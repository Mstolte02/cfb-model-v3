/* Published-results rankings. Independent of roster what-if state. */
(function(root) {
  'use strict';
  function movement(ratings, period = 'week') {
    const snapshots = (ratings.history || []).filter(s => Array.isArray(s.teams));
    const current = snapshots.at(-1);
    const baseline = period === 'season' ? snapshots.find(s => s.label === 'Preseason') : snapshots.at(-2);
    if (!current || !baseline || current === baseline) return {rows:[], current, baseline};
    const before = new Map(baseline.teams.map(r => [r.team,r]));
    const rows = current.teams.flatMap(r => {
      const b = before.get(r.team);
      return b && [r.rank,b.rank,r.power,b.power].every(Number.isFinite)
        ? [{...r, previousRank:b.rank, change:b.rank-r.rank, powerChange:100*(r.power-b.power)}] : [];
    });
    return {current,baseline,rows};
  }
  function z(values) {
    if (!values.length) return [];
    const mean=values.reduce((a,b)=>a+b,0)/values.length;
    const sd=Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/values.length);
    return values.map(v=>sd>1e-12?(v-mean)/sd:0);
  }
  // Pivoted Gaussian elimination solves the same ridge Massey system as the research.
  function solve(a,b) {
    a=a.map((row,i)=>[...row,b[i]]);const n=b.length;
    for(let k=0;k<n;k++) {
      let p=k;for(let i=k+1;i<n;i++)if(Math.abs(a[i][k])>Math.abs(a[p][k]))p=i;
      [a[k],a[p]]=[a[p],a[k]];
      for(let i=k+1;i<n;i++) {const f=a[i][k]/a[k][k];for(let j=k;j<=n;j++)a[i][j]-=f*a[k][j];}
    }
    const x=Array(n).fill(0);for(let i=n-1;i>=0;i--) {let v=a[i][n];for(let j=i+1;j<n;j++)v-=a[i][j]*x[j];x[i]=v/a[i][i];}return x;
  }
  function deserving(schedule, metadata, fbsNames, model, powerRatings = []) {
    const fbs=new Set(fbsNames);
    const powerRank=new Map(powerRatings.map(r=>[r.team,Number.isFinite(r.rank)?r.rank:Infinity]));
    const powerTie=(a,b)=>(powerRank.get(a.team)??Infinity)-(powerRank.get(b.team)??Infinity)||a.team.localeCompare(b.team);
    const finals=schedule.filter(g=>g.f && Number.isFinite(g.hp) && Number.isFinite(g.ap) && g.st!=='postseason');
    const rated=finals.filter(g=>fbs.has(g.h)&&fbs.has(g.a));
    // Match the audited universe: at least one completed FBS matchup.
    const names=[...new Set(rated.flatMap(g=>[g.h,g.a]))].sort();const n=names.length;
    if(!n)return [];
    const idx=new Map(names.map((t,i)=>[t,i]));
    const counts=Array(n).fill(0),total=Array(n).fill(0),wins=Array(n).fill(0),ties=Array(n).fill(0),direct=Array(n).fill(0),opps=names.map(()=>[]);
    const a=names.map((_,i)=>names.map((_,j)=>i===j?model.ridge:0));
    for(const g of rated) {
      const i=idx.get(g.h),j=idx.get(g.a),margin=Math.max(-model.margin_cap,Math.min(model.margin_cap,g.hp-g.ap))-(g.n?0:model.home_field_points);
      direct[i]+=margin;direct[j]-=margin;counts[i]++;counts[j]++;opps[i].push(j);opps[j].push(i);
      a[i][i]++;a[j][j]++;a[i][j]--;a[j][i]--;
    }
    const first=direct.map((v,i)=>v/counts[i]),second=first.map((v,i)=>v+opps[i].reduce((s,j)=>s+first[j],0)/counts[i]);
    const fixed=solve(a,direct),mean=fixed.reduce((s,v)=>s+v,0)/n;fixed.forEach((v,i)=>fixed[i]=v-mean);
    const sd=Math.sqrt(fixed.reduce((s,v)=>s+v*v,0)/n),sos=Array(n).fill(0);
    for(const g of finals)for(const [t,o,scored,allowed] of [[g.h,g.a,g.hp,g.ap],[g.a,g.h,g.ap,g.hp]]) {
      const i=idx.get(t);if(i===undefined)continue;total[i]++;wins[i]+=Number(scored>allowed);ties[i]+=Number(scored===allowed);
      sos[i]+=idx.has(o)?fixed[idx.get(o)]:-2*sd;
    }
    const fz=z(first),sz=z(second),lift=z(sz.map((v,i)=>v-fz[i])),quality=z(fixed),sosz=z(sos.map((v,i)=>v/total[i]));
    const rows=names.map((team,i)=>({team,win_pct:wins[i]/total[i],rating_z:quality[i],sos_z:sosz[i],second_pass_lift:lift[i],
      p4:Number(['SEC','Big Ten','Big 12','ACC','Pac-12','Pac-10'].includes(metadata[team]?.conference)||team==='Notre Dame'),
      wins:wins[i],losses:total[i]-wins[i]-ties[i],ties:ties[i],games:total[i],h2h:0}));
    const score=(r,w)=>Object.entries(w).reduce((s,[k,v])=>s+v*r[k],0);
    const order=rows.slice().sort((a,b)=>score(b,model.provisional_weights)-score(a,model.provisional_weights)||powerTie(a,b));
    const ranks=new Map(order.map((r,i)=>[r.team,i]));
    const pair=new Map();for(const g of rated)if(g.hp!==g.ap) {const w=g.hp>g.ap?g.h:g.a,l=g.hp>g.ap?g.a:g.h;pair.set(w+'|'+l,1);pair.set(l+'|'+w,-1);}
    for(const r of rows) {for(const o of names)if(Math.abs(ranks.get(r.team)-ranks.get(o))<=model.h2h_within)r.h2h+=pair.get(r.team+'|'+o)||0;
      r.score=score(r,model.weights);r.liftContribution=r.second_pass_lift*model.weights.second_pass_lift;}
    rows.sort((a,b)=>Math.abs(b.score-a.score)>=1e-10?b.score-a.score:powerTie(a,b));
    rows.forEach((r,i)=>r.rank=i+1);
    return rows;
  }
  root.RankingHistory={movement,deserving,z};if(typeof module!=='undefined')module.exports=root.RankingHistory;
})(typeof window!=='undefined'?window:globalThis);
