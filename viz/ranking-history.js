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
  // Pivoted Gaussian elimination solves the schedule-adjusted win/loss system.
  function solve(a,b) {
    a=a.map((row,i)=>[...row,b[i]]);const n=b.length;
    for(let k=0;k<n;k++) {
      let p=k;for(let i=k+1;i<n;i++)if(Math.abs(a[i][k])>Math.abs(a[p][k]))p=i;
      [a[k],a[p]]=[a[p],a[k]];
      for(let i=k+1;i<n;i++) {const f=a[i][k]/a[k][k];for(let j=k;j<=n;j++)a[i][j]-=f*a[k][j];}
    }
    const x=Array(n).fill(0);for(let i=n-1;i>=0;i--) {let v=a[i][n];for(let j=i+1;j<n;j++)v-=a[i][j]*x[j];x[i]=v/a[i][i];}return x;
  }
  // Colley: (games + 2) * rating - sum(opponent ratings) = 1 + (wins - losses)/2.
  // Conference metadata and predictive power ratings are deliberately unused.
  function deserving(schedule, metadata, fbsNames, model) {
    if (model?.method !== 'colley') throw new Error('Unsupported résumé method');
    const fbs=new Set(fbsNames);
    const finals=schedule.filter(g=>g.f && Number.isFinite(g.hp) && Number.isFinite(g.ap)
      && g.st!=='postseason' && fbs.has(g.h) && fbs.has(g.a));
    const names=[...new Set(finals.flatMap(g=>[g.h,g.a]))].sort(),n=names.length;
    if(!n)return [];
    const idx=new Map(names.map((team,i)=>[team,i]));
    const wins=Array(n).fill(0),losses=Array(n).fill(0),ties=Array(n).fill(0),opps=names.map(()=>[]);
    const matrix=names.map((_,i)=>names.map((_,j)=>i===j?2:0));
    for(const g of finals) {
      const i=idx.get(g.h),j=idx.get(g.a);
      opps[i].push(j);opps[j].push(i);
      matrix[i][i]++;matrix[j][j]++;matrix[i][j]--;matrix[j][i]--;
      if(g.hp>g.ap){wins[i]++;losses[j]++;}
      else if(g.hp<g.ap){wins[j]++;losses[i]++;}
      else {ties[i]++;ties[j]++;}
    }
    const rating=solve(matrix,wins.map((w,i)=>1+(w-losses[i])/2));
    const rows=names.map((team,i)=>{
      const games=opps[i].length,sos=opps[i].reduce((s,j)=>s+rating[j],0)/games;
      const recordBase=(wins[i]+ties[i]/2+1)/(games+2);
      const scheduleAdjustment=games/(games+2)*(sos-.5);
      return {team,wins:wins[i],losses:losses[i],ties:ties[i],games,
        win_pct:(wins[i]+ties[i]/2)/games,rating:rating[i],sos,
        recordBase,scheduleAdjustment,score:100*rating[i]};
    });
    // Equal résumés share a rank. Alphabetical ordering is display-only.
    rows.sort((a,b)=>b.score-a.score||a.team.localeCompare(b.team));
    let tieScore;
    rows.forEach((r,i)=>{
      if(i===0||Math.abs(r.score-tieScore)>1e-8){r.rank=i+1;tieScore=r.score;}
      else r.rank=rows[i-1].rank;
    });
    return rows;
  }
  root.RankingHistory={movement,deserving,z};if(typeof module!=='undefined')module.exports=root.RankingHistory;
})(typeof window!=='undefined'?window:globalThis);

