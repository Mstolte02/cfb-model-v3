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

  /* ------------------------------------------------------------------------
     Most Deserving: earned results against a playoff-caliber reference team.

     The reference team has a FIXED strength for the whole snapshot: the average
     end-of-season rating of the k-th best FBS team (k = 12, the playoff cutoff)
     in earlier seasons. It never depends on any current team's record.

     q_g = chance the reference team wins game g (same opponent, same venue).
     Every completed regular-season FBS-v-FBS game g of team i adds

        loss term         -omega * (1 - W_g)              (W = 1 win, 0 loss, .5 tie)
        schedule term      1 - q_g                         (reference team's loss chance)
        performance term   alpha * (s_g - W_g)             (s = margin-implied dominance)

     so   score = -omega * L + E + P   in loss-equivalent units. E is the number of
     games the reference team would be expected to lose on the same schedule.
     With omega = 1 and alpha = 0 the score is W - sum(q): wins above the
     reference team. It is one number per team, so the order is transitive and
     deterministic. A team with one more loss passes another only when its
     (E + P) is more than omega higher.

     Résumé rarity (reported with every row, exact, no simulation):
        W* = sum of independent Bernoulli(q_g),  p = P(W* >= W),  R = -log10(p).

     q_g uses opponent ratings fitted from this season's FBS results only, with
     every game of team i removed (leave-team-out). Team i's own results never
     change how strong its opponents look to it, so a better result can never
     lower team i's score. See audit/MOST_DESERVING_V2.md.
     ------------------------------------------------------------------------ */
  const RESUME_DEFAULTS = Object.freeze({
    calibrated: Object.freeze({ridge_games:2, margin_cap:60, home_field:3, win_sigma:15.0465, dominance_scale:25.5903,
      reference_ratings:Object.freeze({4:21.2893,8:16.5856,12:14.3922,16:12.1457,25:9.1906})}),
    assumptions: Object.freeze({benchmark_rank:12, convincingness_weight:0, loss_aversion:1, performance_cap:28})
  });
  const METHODS = new Set(['earned-results']);
  function erfc(x) { // Numerical Recipes erfcc: fractional error below 1.2e-7 everywhere.
    const zz=Math.abs(x),t=1/(1+0.5*zz);
    const r=t*Math.exp(-zz*zz-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806
      +t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
    return x>=0?r:2-r;
  }
  const normalCdf=x=>0.5*erfc(-x/Math.SQRT2);
  const clip=(x,c)=>Math.max(-c,Math.min(c,x));
  // Dense Cholesky solve. (Laplacian + ridge * I) is symmetric positive definite.
  function cholSolve(A,b,n) {
    const L=new Float64Array(n*n);
    for(let j=0;j<n;j++) {
      let d=A[j*n+j];
      for(let k=0;k<j;k++)d-=L[j*n+k]*L[j*n+k];
      const ljj=Math.sqrt(d);L[j*n+j]=ljj;
      for(let i=j+1;i<n;i++) {
        let v=A[i*n+j];
        for(let k=0;k<j;k++)v-=L[i*n+k]*L[j*n+k];
        L[i*n+j]=v/ljj;
      }
    }
    const y=new Float64Array(n);
    for(let i=0;i<n;i++){let v=b[i];for(let k=0;k<i;k++)v-=L[i*n+k]*y[k];y[i]=v/L[i*n+i];}
    const x=new Float64Array(n);
    for(let i=n-1;i>=0;i--){let v=y[i];for(let k=i+1;k<n;k++)v-=L[k*n+i]*x[k];x[i]=v/L[i*n+i];}
    return x;
  }
  // Ridge least squares on capped margins: r_i - r_j + h*v = clip(margin) + noise.
  // skip = a team whose games are all removed (leave-team-out fit).
  function fitRatings(n,games,cal,skip=-1) {
    const A=new Float64Array(n*n),b=new Float64Array(n);
    for(let k=0;k<n;k++)A[k*n+k]=cal.ridge_games;
    for(const g of games) {
      if(g.i===skip||g.j===skip)continue;
      A[g.i*n+g.i]++;A[g.j*n+g.j]++;A[g.i*n+g.j]--;A[g.j*n+g.i]--;
      const t=g.y-cal.home_field*g.v;b[g.i]+=t;b[g.j]-=t;
    }
    return cholSolve(A,b,n);
  }
  // Exact distribution of the number of wins of independent Bernoulli games.
  // dist[k] = P(W* = k). O(G^2), no sampling noise.
  function poissonBinomial(qs) {
    let dist=new Float64Array([1]);
    for(const q of qs) {
      const next=new Float64Array(dist.length+1);
      for(let k=0;k<dist.length;k++){next[k]+=dist[k]*(1-q);next[k+1]+=dist[k]*q;}
      dist=next;
    }
    return dist;
  }
  function upperTail(dist,wins) { // P(W* >= wins); a tie counts half, so round up.
    let s=0;for(let k=dist.length-1;k>=Math.ceil(wins-1e-12);k--)s+=dist[k];
    return Math.min(1,s);
  }
  function eligible(schedule,fbsNames) {
    const fbs=new Set(fbsNames);
    return schedule.filter(g=>g.f && Number.isFinite(g.hp) && Number.isFinite(g.ap)
      && g.st!=='postseason' && fbs.has(g.h) && fbs.has(g.a) && g.h!==g.a)
      // Canonical form: a neutral-site game is the same game whichever side is listed first.
      .map(g=>{
        const neutral=!!g.n,swap=neutral&&g.a<g.h;
        return swap?{h:g.a,a:g.h,hp:g.ap,ap:g.hp,n:true,d:g.d||'',id:g.id}:{h:g.h,a:g.a,hp:g.hp,ap:g.ap,n:neutral,d:g.d||'',id:g.id};
      })
      // Fixed order: identical inputs in any order give bit-identical sums.
      .sort((x,y)=>String(x.d).localeCompare(String(y.d))||x.h.localeCompare(y.h)||x.a.localeCompare(y.a)||x.hp-y.hp||x.ap-y.ap);
  }
  function resolveModel(model) {
    if(!METHODS.has(model?.method))throw new Error('Unsupported résumé method');
    const calibrated={...RESUME_DEFAULTS.calibrated,...(model.calibrated||{})};
    calibrated.reference_ratings={...RESUME_DEFAULTS.calibrated.reference_ratings,...(model.calibrated?.reference_ratings||{})};
    return {calibrated,assumptions:{...RESUME_DEFAULTS.assumptions,...(model.assumptions||{})},
      sensitivity:model.sensitivity||null};
  }
  // Opponent strength for every team's games. This is the only expensive step and
  // it does not depend on the philosophical assumptions, so sensitivity reuses it.
  function prepare(schedule,fbsNames,model) {
    const {calibrated:cal}=resolveModel(model);
    const finals=eligible(schedule,fbsNames);
    const names=[...new Set(finals.flatMap(g=>[g.h,g.a]))].sort(),n=names.length;
    if(!n)return {cal,names,teams:[]};
    const idx=new Map(names.map((t,i)=>[t,i]));
    const games=finals.map(g=>({i:idx.get(g.h),j:idx.get(g.a),y:clip(g.hp-g.ap,cal.margin_cap),v:g.n?0:1,g}));
    const teams=names.map((team,t)=>{
      const own=games.filter(x=>x.i===t||x.j===t);
      const loo=fitRatings(n,games,cal,t);
      return {team,games:own.map(x=>{
        const home=x.i===t,opp=home?x.j:x.i,g=x.g;
        return {opponent:names[opp],date:g.d,id:g.id,site:g.n?'neutral':home?'home':'away',venue:g.n?0:home?1:-1,
          pf:home?g.hp:g.ap,pa:home?g.ap:g.hp,oppRating:loo[opp]};
      })};
    });
    return {cal,names,teams};
  }
  function referenceRating(cal,a) {
    if(Number.isFinite(a.reference_rating))return a.reference_rating;
    const table=cal.reference_ratings,keys=Object.keys(table).map(Number).sort((x,y)=>x-y),k=a.benchmark_rank;
    if(Number.isFinite(table[k]))return table[k];
    // Linear interpolation between tabulated ranks; flat beyond the ends.
    if(k<=keys[0])return table[keys[0]];if(k>=keys.at(-1))return table[keys.at(-1)];
    const hi=keys.find(x=>x>k),lo=keys.filter(x=>x<k).at(-1);
    return table[lo]+(table[hi]-table[lo])*(k-lo)/(hi-lo);
  }
  function score(ctx,assumptions) {
    const cal=ctx.cal,a={...RESUME_DEFAULTS.assumptions,...(assumptions||{})};
    const omega=a.loss_aversion,alpha=a.convincingness_weight,ref=referenceRating(cal,a);
    const rows=ctx.teams.map(t=>{
      let wins=0,losses=0,ties=0,lossPenalty=0,fromWins=0,fromLosses=0,performance=0;
      const ledger=t.games.map(g=>{
        const margin=g.pf-g.pa,W=margin>0?1:margin<0?0:0.5;
        const q=normalCdf((ref-g.oppRating+cal.home_field*g.venue)/cal.win_sigma);
        // Venue-adjusted, capped margin: how clearly the game showed the better team.
        const s=normalCdf((clip(margin,a.performance_cap)-cal.home_field*g.venue)/cal.dominance_scale);
        const lossTerm=-omega*(1-W),scheduleTerm=1-q,performanceTerm=alpha*(s-W);
        if(W===1){wins++;fromWins+=scheduleTerm;}
        else if(W===0){losses++;fromLosses+=scheduleTerm;}
        else {ties++;fromWins+=scheduleTerm/2;fromLosses+=scheduleTerm/2;}
        lossPenalty+=lossTerm;performance+=performanceTerm;
        return {...g,result:W===1?'W':W===0?'L':'T',refWin:q,dominance:s,lossTerm,scheduleTerm,performanceTerm,
          total:lossTerm+scheduleTerm+performanceTerm};
      });
      const schedule=fromWins+fromLosses,games=ledger.length;
      const dist=poissonBinomial(ledger.map(g=>g.refWin)),earned=wins+ties/2;
      const matchProbability=upperTail(dist,earned);
      return {team:t.team,wins,losses,ties,games,win_pct:earned/games,
        lossPenalty,schedule,scheduleFromWins:fromWins,scheduleFromLosses:fromLosses,performance,
        score:lossPenalty+schedule+performance,sos:schedule,ledger,referenceRating:ref,
        referenceWins:games-schedule,matchProbability,rarity:-Math.log10(Math.max(matchProbability,1e-300))};
    });
    // Equal résumés share a rank. Alphabetical order only arranges ties on screen.
    rows.sort((x,y)=>y.score-x.score||x.team.localeCompare(y.team));
    let tieScore;
    rows.forEach((r,i)=>{
      if(i===0||Math.abs(r.score-tieScore)>1e-9){r.rank=i+1;tieScore=r.score;}
      else r.rank=rows[i-1].rank;
    });
    const bySchedule=rows.map(r=>r.schedule),byRarity=rows.map(r=>r.rarity);
    rows.forEach(r=>{
      r.scheduleRank=1+bySchedule.filter(v=>v>r.schedule+1e-12).length;
      r.rarityRank=1+byRarity.filter(v=>v>r.rarity+1e-12).length;
    });
    return rows;
  }
  function deserving(schedule,metadata,fbsNames,model,options={}) {
    const resolved=resolveModel(model);
    const ctx=options.context||prepare(schedule,fbsNames,model);
    return score(ctx,{...resolved.assumptions,...(options.assumptions||{})});
  }
  // Why does A rank ahead of B? Uses the same components as the score, so the
  // net always equals the score difference exactly.
  function compare(a,b) {
    return {losses:a.lossPenalty-b.lossPenalty,schedule:a.schedule-b.schedule,
      scheduleFromWins:a.scheduleFromWins-b.scheduleFromWins,scheduleFromLosses:a.scheduleFromLosses-b.scheduleFromLosses,
      performance:a.performance-b.performance,net:a.score-b.score,lossGap:a.losses-b.losses,
      rarity:a.rarity-b.rarity,referenceWins:a.referenceWins-b.referenceWins};
  }
  function cartesian(grid) {
    return Object.entries(grid).reduce((acc,[k,vals])=>acc.flatMap(c=>vals.map(v=>({...c,[k]:v}))),[{}]);
  }
  // Rank range and top-12 share across every combination of the assumption grid.
  function sensitivity(ctx,model,cutoff=12) {
    const {assumptions,sensitivity:grid}=resolveModel(model);
    const configs=cartesian({...Object.fromEntries(Object.entries(assumptions).map(([k,v])=>[k,[v]])),...(grid||{})});
    const ranks=new Map(ctx.teams.map(t=>[t.team,[]]));
    for(const c of configs)for(const r of score(ctx,c))ranks.get(r.team).push(r.rank);
    const out=new Map();
    for(const [team,rs] of ranks) {
      const sorted=[...rs].sort((x,y)=>x-y);
      out.set(team,{min:sorted[0],max:sorted.at(-1),median:sorted[Math.floor((sorted.length-1)/2)],
        inTop:rs.filter(r=>r<=cutoff).length/rs.length,configs:rs.length});
    }
    return {configs,byTeam:out};
  }
  root.RankingHistory={movement,deserving,z,prepare,score,compare,sensitivity,fitRatings,normalCdf,eligible,
    poissonBinomial,upperTail,referenceRating,RESUME_DEFAULTS};
  if(typeof module!=='undefined')module.exports=root.RankingHistory;
})(typeof window!=='undefined'?window:globalThis);
