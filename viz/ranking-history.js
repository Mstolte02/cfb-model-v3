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
  // Schedule-adjusted Colley opponent ratings are an input, NOT the final rank.
  // All adjustments below are in loss-equivalent units, so comparing two
  // teams never requires a pairwise override (which could be non-transitive).
  const DESERVING_V1 = Object.freeze({
    sosPerRatingPoint: 4.0,  // provisional: 0.25 SOS gap cancels one loss
    marginWeight: 0.30,     // smaller than a loss; capped and saturating
    winWeight: 0.16,        // rewards additional earned wins at the same losses
    marginScale: 21
  });
  function deserving(schedule, metadata, fbsNames, model) {
    if (model?.method !== 'colley' && model?.method !== 'record-first')
      throw new Error('Unsupported résumé method');
    const fbs=new Set(fbsNames);
    const finals=schedule.filter(g=>g.f && Number.isFinite(g.hp) && Number.isFinite(g.ap)
      && g.st!=='postseason' && fbs.has(g.h) && fbs.has(g.a) && g.h!==g.a);
    const names=[...new Set(finals.flatMap(g=>[g.h,g.a]))].sort(),n=names.length;
    if(!n)return [];
    const idx=new Map(names.map((team,i)=>[team,i]));
    const wins=Array(n).fill(0),losses=Array(n).fill(0),ties=Array(n).fill(0);
    const opps=names.map(()=>[]),played=names.map(()=>[]);
    const matrix=names.map((_,i)=>names.map((_,j)=>i===j?2:0));
    for(const g of finals) {
      const i=idx.get(g.h),j=idx.get(g.a),margin=g.hp-g.ap;
      opps[i].push(j);opps[j].push(i);
      played[i].push({opponent:j,margin});
      played[j].push({opponent:i,margin:-margin});
      matrix[i][i]++;matrix[j][j]++;matrix[i][j]--;matrix[j][i]--;
      if(margin>0){wins[i]++;losses[j]++;}
      else if(margin<0){wins[j]++;losses[i]++;}
      else {ties[i]++;ties[j]++;}
    }
    const rating=solve(matrix,wins.map((w,i)=>1+(w-losses[i])/2));
    const cfg=DESERVING_V1;
    const rows=names.map((team,i)=>{
      const games=opps[i].length;
      const sos=opps[i].reduce((s,j)=>s+rating[j],0)/games;
      const cappedMargin=played[i].reduce((s,g)=>
        s+Math.tanh(Math.max(-28,Math.min(28,g.margin))/cfg.marginScale),0)/games;
      const recordBase=-losses[i];
      const scheduleAdjustment=cfg.sosPerRatingPoint*(sos-0.5);
      const marginAdjustment=cfg.marginWeight*cappedMargin;
      const winsAdjustment=cfg.winWeight*wins[i]/12;
      const qualityAdjustment=scheduleAdjustment+marginAdjustment+winsAdjustment;
      const score=recordBase+qualityAdjustment;
      return {team,wins:wins[i],losses:losses[i],ties:ties[i],games,
        win_pct:(wins[i]+ties[i]/2)/games,rating:rating[i],sos,
        recordBase,scheduleAdjustment,marginAdjustment,winsAdjustment,
        qualityAdjustment,score,
        // For a team with one fewer loss to pass this team, compare the
        // quality difference with the loss gap, not with an arbitrary rank.
        rankingMethod:'record-first-v1'};
    });
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

