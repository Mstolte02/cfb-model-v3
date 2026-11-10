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
      reference_ratings:Object.freeze({4:21.2893,8:16.5856,12:14.3922,16:12.1457,25:9.1906}),
      reference_sd:Object.freeze({4:3,8:2.5,12:2.3,16:2,25:1.5})}),
    assumptions: Object.freeze({benchmark_rank:12, convincingness_weight:0, loss_aversion:1, performance_cap:28, reference_uncertainty:true, head_to_head_window:0.39})
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
  // priorMean (optional): the ridge pulls each rating toward this value instead of 0
  // (early-season prior-season blend; see priorWeight).
  function fitRatings(n,games,cal,skip=-1,priorMean=null) {
    const A=new Float64Array(n*n),b=new Float64Array(n);
    for(let k=0;k<n;k++){A[k*n+k]=cal.ridge_games;if(priorMean)b[k]+=cal.ridge_games*(priorMean[k]||0);}
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
        const w=Number.isFinite(g.w)?g.w:null;
        return swap?{h:g.a,a:g.h,hp:g.ap,ap:g.hp,n:true,d:g.d||'',id:g.id,w}:{h:g.h,a:g.a,hp:g.hp,ap:g.ap,n:neutral,d:g.d||'',id:g.id,w};
      })
      // Fixed order: identical inputs in any order give bit-identical sums.
      .sort((x,y)=>String(x.d).localeCompare(String(y.d))||x.h.localeCompare(y.h)||x.a.localeCompare(y.a)||x.hp-y.hp||x.ap-y.ap);
  }
  function resolveModel(model) {
    if(!METHODS.has(model?.method))throw new Error('Unsupported résumé method');
    const calibrated={...RESUME_DEFAULTS.calibrated,...(model.calibrated||{})};
    calibrated.reference_ratings={...RESUME_DEFAULTS.calibrated.reference_ratings,...(model.calibrated?.reference_ratings||{})};
    calibrated.reference_sd={...RESUME_DEFAULTS.calibrated.reference_sd,...(model.calibrated?.reference_sd||{})};
    return {calibrated,assumptions:{...RESUME_DEFAULTS.assumptions,...(model.assumptions||{})},
      sensitivity:model.sensitivity||null};
  }
  /* Early-season stability: through the first weeks the ridge pulls each opponent
     toward a share of its prior-season rating instead of toward average. The share
     decays linearly to zero:  weight(week) = max(0, 1 - (week - 1) / decay_weeks).
     Both the share (regression) and decay_weeks are fitted on chronological
     out-of-sample log loss (scripts/deserving_calibration.cjs). Prior-season ratings
     come from last season's FBS results only: no talent, recruiting or polls. */
  function priorWeight(prior,week) {
    if(!prior||!prior.ratings||!(prior.decay_weeks>0)||!Number.isFinite(week))return 0;
    return Math.max(0,Math.min(1,1-(week-1)/prior.decay_weeks));
  }
  // Opponent strength for every team's games. This is the only expensive step and
  // it does not depend on the philosophical assumptions, so sensitivity reuses it.
  function prepare(schedule,fbsNames,model,options={}) {
    const {calibrated:cal}=resolveModel(model);
    const finals=eligible(schedule,fbsNames);
    const names=[...new Set(finals.flatMap(g=>[g.h,g.a]))].sort(),n=names.length;
    const prior=options.prior===undefined?model.prior||null:options.prior;
    const weeks=finals.map(g=>g.w).filter(Number.isFinite),week=weeks.length?Math.max(...weeks):null;
    const pw=priorWeight(prior,week);
    if(!n)return {cal,names,teams:[],week,priorWeight:pw};
    const idx=new Map(names.map((t,i)=>[t,i]));
    const games=finals.map(g=>({i:idx.get(g.h),j:idx.get(g.a),y:clip(g.hp-g.ap,cal.margin_cap),v:g.n?0:1,g}));
    const priorMean=pw>0?Float64Array.from(names,t=>pw*(prior.regression??1)*(Number(prior.ratings[t])||0)):null;
    // This season's playoff-caliber band so far (ranks k-4..k+4 of a full fit), shown beside
    // the fixed reference as a drift check. It never enters any score.
    const full=[...fitRatings(n,games,cal,-1,priorMean)].sort((x,y)=>y-x);
    const band=k=>{const v=full.slice(Math.max(0,k-5),k+4),m=v.reduce((s,x)=>s+x,0)/v.length;return {mean:m,sd:Math.sqrt(v.reduce((s,x)=>s+(x-m)**2,0)/v.length),teams:v.length};};
    const teams=names.map((team,t)=>{
      const own=games.filter(x=>x.i===t||x.j===t);
      const loo=fitRatings(n,games,cal,t,priorMean);
      return {team,games:own.map(x=>{
        const home=x.i===t,opp=home?x.j:x.i,g=x.g;
        return {opponent:names[opp],date:g.d,id:g.id,site:g.n?'neutral':home?'home':'away',venue:g.n?0:home?1:-1,
          pf:home?g.hp:g.ap,pa:home?g.ap:g.hp,oppRating:loo[opp]};
      })};
    });
    return {cal,names,teams,week,priorWeight:pw,inSeasonBand:n>=20?band(12):null};
  }
  /* The playoff-caliber reference team is a distribution, not one number:
     N(mean, sd) fitted to the end-of-season strengths of the teams ranked
     k-4 .. k+4 in earlier seasons (both the width of "playoff-caliber" and its
     year-to-year drift). Every team is judged against the same distribution. */
  function lookupRank(table,k) {
    const keys=Object.keys(table||{}).map(Number).filter(Number.isFinite).sort((x,y)=>x-y);
    if(!keys.length)return NaN;
    if(Number.isFinite(table[k]))return table[k];
    if(k<=keys[0])return table[keys[0]];if(k>=keys.at(-1))return table[keys.at(-1)];
    const hi=keys.find(x=>x>k),lo=keys.filter(x=>x<k).at(-1);
    return table[lo]+(table[hi]-table[lo])*(k-lo)/(hi-lo);
  }
  function referenceDistribution(cal,a) {
    const mean=Number.isFinite(a.reference_rating)?a.reference_rating:lookupRank(cal.reference_ratings,a.benchmark_rank);
    const sd=a.reference_uncertainty===false?0:Number.isFinite(a.reference_sd)?a.reference_sd:(lookupRank(cal.reference_sd,a.benchmark_rank)||0);
    return {mean,sd};
  }
  function referenceRating(cal,a){return referenceDistribution(cal,a).mean;}
  // Standard-normal quantiles at equal-probability midpoints, for ranks and rarity
  // under reference uncertainty. Bisection on normalCdf: deterministic everywhere.
  function normalQuantile(u){let lo=-9,hi=9;for(let k=0;k<80;k++){const m=(lo+hi)/2;if(normalCdf(m)<u)lo=m;else hi=m;}return (lo+hi)/2;}
  const NODES=25,NODE_Z=Array.from({length:NODES},(_,j)=>normalQuantile((j+0.5)/NODES)),Z80=normalQuantile(0.9);

  // Résumé components for one reference strength. sigma = win-probability scale.
  function evaluate(ctx,a,ref,sigma,withLedger) {
    const cal=ctx.cal,omega=a.loss_aversion,alpha=a.convincingness_weight;
    return ctx.teams.map(t=>{
      let wins=0,losses=0,ties=0,lossPenalty=0,fromWins=0,fromLosses=0,performance=0;
      const qs=[],ledger=[];
      for(const g of t.games) {
        const margin=g.pf-g.pa,W=margin>0?1:margin<0?0:0.5;
        const q=normalCdf((ref-g.oppRating+cal.home_field*g.venue)/sigma);
        // Venue-adjusted, capped margin: how clearly the game showed the better team.
        const s=normalCdf((clip(margin,a.performance_cap)-cal.home_field*g.venue)/cal.dominance_scale);
        const lossTerm=-omega*(1-W),scheduleTerm=1-q,performanceTerm=alpha*(s-W);
        if(W===1){wins++;fromWins+=scheduleTerm;}
        else if(W===0){losses++;fromLosses+=scheduleTerm;}
        else {ties++;fromWins+=scheduleTerm/2;fromLosses+=scheduleTerm/2;}
        lossPenalty+=lossTerm;performance+=performanceTerm;qs.push(q);
        if(withLedger)ledger.push({...g,result:W===1?'W':W===0?'L':'T',refWin:q,dominance:s,lossTerm,scheduleTerm,performanceTerm,
          total:lossTerm+scheduleTerm+performanceTerm});
      }
      return {team:t.team,wins,losses,ties,games:t.games.length,lossPenalty,fromWins,fromLosses,performance,qs,ledger,
        base:lossPenalty+fromWins+fromLosses+performance};
    });
  }
  /* Head-to-head between close résumés: a capped, soft Bayesian update.

     The résumé score D is noisy: the gap between two nearby résumés has a measured
     standard deviation W (the head-to-head window, 0.39 loss units). When team w beat
     team l (on balance), the result is evidence that w's résumé is the better one.
     With base gap g = D_l - D_w (positive when the winner is behind):

       prior on the true gap   Delta = D_w - D_l + noise,  noise ~ N(0, W^2)
       evidence                P(w beat l | Delta) = Phi(Delta / W)
       posterior mean shift    m(g) = (W / sqrt2) * phi(z) / Phi(z),  z = -g / (W sqrt2)

     The update only operates inside the noise band: full strength while the winner
     trails by less than W/2, fading linearly to zero at W (no jump), and nothing
     beyond. The pair's gap closes by t = min(W, m(g) * taper(g)); the winner gains
     t/2 and the loser gives up t/2. A team's net head-to-head adjustment is capped at
     +/- W/2, so an extra win (+1) or tie (+0.5) can never be cancelled by it.

     Ranking = résumé score + head-to-head adjustment: one number per team, so the
     order is transitive, deterministic and independent of the order pairs are visited
     (no path dependence). A winner within ~0.24 of the team it beat passes it;
     further apart, the gap only narrows. */
  function headToHeadNet(ctx) { // per team: opponent -> wins minus losses against it
    return ctx.teams.map(t=>{const m=new Map();for(const g of t.games){if(g.pf===g.pa)continue;m.set(g.opponent,(m.get(g.opponent)||0)+(g.pf>g.pa?1:-1));}return m;});
  }
  const phi=x=>Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);
  function headToHeadTransfer(gap,window) {
    if(!(window>0)||!(gap<window))return 0;
    const S=window*Math.SQRT2,zz=-gap/S,m=window/Math.SQRT2*phi(zz)/Math.max(1e-300,normalCdf(zz));
    const taper=gap<=window/2?1:(window-gap)/(window/2);
    return Math.max(0,Math.min(window,m*taper));
  }
  function headToHeadAdjust(names,scores,net,window) {
    const n=names.length,raw=new Float64Array(n),detail=names.map(()=>[]);
    if(window>0){
      const idx=new Map(names.map((t,i)=>[t,i]));
      for(let w=0;w<n;w++)for(const [opp,v] of net[w]){
        const l=idx.get(opp);if(l===undefined||!(v>0))continue;
        const gap=scores[l]-scores[w],t=headToHeadTransfer(gap,window);
        if(t<=0)continue;
        raw[w]+=t/2;raw[l]-=t/2;
        detail[w].push({team:opp,won:true,gap,value:t/2});detail[l].push({team:names[w],won:false,gap:-gap,value:-t/2});
      }
    }
    const cap=window/2,adj=Float64Array.from(raw,x=>clip(x,cap));
    detail.forEach((d,i)=>{d.sort((x,y)=>Math.abs(y.value)-Math.abs(x.value)||x.team.localeCompare(y.team));
      if(Math.abs(adj[i]-raw[i])>1e-15)d.push({team:null,cap:true,value:adj[i]-raw[i]});});
    return {adj,detail,cap};
  }
  function orderTeams(names,scores) {
    const order=scores.map((_,i)=>i).sort((x,y)=>scores[y]-scores[x]||names[x].localeCompare(names[y]));
    const ranks=Array(order.length);
    order.forEach((i,k)=>{const p=order[k-1];ranks[i]=k>0&&Math.abs(scores[i]-scores[p])<=1e-9?ranks[p]:k+1;});
    return {order,ranks};
  }
  function nodeRanks(ctx,a,ref,sigma,net) {
    const ev=evaluate(ctx,a,ref,sigma,false),names=ev.map(e=>e.team),base=ev.map(e=>e.base);
    const {adj}=headToHeadAdjust(names,base,net,a.head_to_head_window),scores=base.map((x,i)=>x+adj[i]);
    return {ev,scores,ranks:orderTeams(names,scores).ranks};
  }
  function score(ctx,assumptions,options={}) {
    const cal=ctx.cal,a={...RESUME_DEFAULTS.assumptions,...(assumptions||{})},cutoff=options.cutoff||12;
    const dist=referenceDistribution(cal,a),net=headToHeadNet(ctx);
    // Expected credit over the reference distribution: E[Phi((R - r + hv)/s)] with
    // R ~ N(mean, sd^2) is exactly Phi((mean - r + hv)/sqrt(s^2 + sd^2)).
    const sigmaEff=Math.sqrt(cal.win_sigma**2+dist.sd**2);
    const ev=evaluate(ctx,a,dist.mean,sigmaEff,true);
    const names=ev.map(e=>e.team),base=ev.map(e=>e.base);
    const h2h=headToHeadAdjust(names,base,net,a.head_to_head_window);
    const rows=ev.map((e,i)=>{
      const schedule=e.fromWins+e.fromLosses,earned=e.wins+e.ties/2;
      return {team:e.team,wins:e.wins,losses:e.losses,ties:e.ties,games:e.games,win_pct:earned/e.games,
        lossPenalty:e.lossPenalty,schedule,scheduleFromWins:e.fromWins,scheduleFromLosses:e.fromLosses,performance:e.performance,
        resumeScore:e.base,headToHead:h2h.adj[i],headToHeadGames:h2h.detail[i],score:e.base+h2h.adj[i],
        sos:schedule,ledger:e.ledger,referenceRating:dist.mean,referenceSd:dist.sd,referenceWins:e.games-schedule,
        quality:qualitySummary(e.ledger,cal,a),headToHeadPassed:[],headToHeadPassedBy:[]};
    });
    const scores=rows.map(r=>r.score),{order,ranks}=orderTeams(names,scores);
    // Every pair whose order the head-to-head adjustment reversed (résumé score says
    // one thing, the ranking the other), and whether the team placed higher won the game.
    const beat=(x,y)=>(net[x].get(names[y])||0)>0;
    for(let p=0;p<order.length;p++)for(let q=p+1;q<order.length;q++){
      const x=order[p],y=order[q];if(!(base[x]<base[y]-1e-12&&scores[x]>scores[y]+1e-12))continue;
      rows[x].headToHeadPassed.push({team:names[y],gap:base[y]-base[x],won:beat(x,y)});
      rows[y].headToHeadPassedBy.push({team:names[x],gap:base[y]-base[x],won:beat(x,y)});
    }
    rows.forEach((r,i)=>{r.rank=ranks[i];});
    if(!options.light) {
      // Range and rarity under reference uncertainty. A stronger reference team
      // expects fewer losses, so the low end of the range uses the 90th-percentile reference.
      const lo=nodeRanks(ctx,a,dist.mean+Z80*dist.sd,cal.win_sigma,net),hi=nodeRanks(ctx,a,dist.mean-Z80*dist.sd,cal.win_sigma,net);
      const nodes=dist.sd>0?NODE_Z.map(zz=>nodeRanks(ctx,a,dist.mean+zz*dist.sd,cal.win_sigma,net)):[nodeRanks(ctx,a,dist.mean,cal.win_sigma,net)];
      rows.forEach((r,i)=>{
        r.scoreLow=lo.scores[i];r.scoreHigh=hi.scores[i];
        const earned=r.wins+r.ties/2,tail=nd=>upperTail(poissonBinomial(nd.ev[i].qs),earned);
        r.matchProbability=nodes.reduce((s,nd)=>s+tail(nd),0)/nodes.length;
        // A weaker reference team matches a record less often: that end is the rarer one.
        r.matchProbabilityLow=tail(hi);r.matchProbabilityHigh=tail(lo);
        r.rarity=-Math.log10(Math.max(r.matchProbability,1e-300));
        r.topProbability=nodes.filter(nd=>nd.ranks[i]<=cutoff).length/nodes.length;
        r.rankRangeReference=[Math.min(lo.ranks[i],hi.ranks[i],r.rank),Math.max(lo.ranks[i],hi.ranks[i],r.rank)];
      });
    }
    const sorted=order.map(i=>rows[i]);
    const bySchedule=sorted.map(r=>r.schedule),byRarity=sorted.map(r=>r.rarity);
    sorted.forEach(r=>{
      r.scheduleRank=1+bySchedule.filter(v=>v>r.schedule+1e-12).length;
      if(r.rarity!==undefined)r.rarityRank=1+byRarity.filter(v=>v>r.rarity+1e-12).length;
    });
    return sorted;
  }
  /* Quality wins and losses, judged with the same leave-team-out opponent strengths
     and the same venue adjustment as the score. Tiers are set by the calibrated
     reference table: "playoff-caliber" = at least as strong (venue-adjusted) as the
     reference team; "top-25 caliber" = at least the #25 reference; "bad loss" = a
     loss to a below-average FBS team. */
  function qualitySummary(ledger,cal,a) {
    const top=lookupRank(cal.reference_ratings,a.benchmark_rank),t25=lookupRank(cal.reference_ratings,25);
    const out={playoffWins:0,top25Wins:0,playoffLosses:0,top25Losses:0,badLosses:0,tiers:[]};
    for(const g of ledger){
      const s=g.oppRating-cal.home_field*g.venue; // opponent strength from this team's side of the venue
      const tier=s>=top?'playoff':s>=t25?'top25':s<0?'below-average':'average';
      g.tier=tier;g.adjustedOpp=s;
      if(g.result==='W'){if(tier==='playoff')out.playoffWins++;if(tier==='playoff'||tier==='top25')out.top25Wins++;}
      if(g.result==='L'){if(tier==='playoff')out.playoffLosses++;if(tier==='playoff'||tier==='top25')out.top25Losses++;if(tier==='below-average')out.badLosses++;}
    }
    return out;
  }
  function deserving(schedule,metadata,fbsNames,model,options={}) {
    const resolved=resolveModel(model);
    const ctx=options.context||prepare(schedule,fbsNames,model);
    return score(ctx,{...resolved.assumptions,...(options.assumptions||{})},options);
  }
  // Why does A rank ahead of B? Uses the same components as the ranking score, so the
  // net always equals the score difference exactly, head-to-head adjustment included.
  function compare(a,b) {
    const meetings=a.ledger.filter(g=>g.opponent===b.team);
    const resume=a.resumeScore-b.resumeScore,net=a.score-b.score;
    // The adjustment decided the order when it reversed the sign of the résumé gap.
    const decidedBy=resume<-1e-12&&net>1e-12?a.team:resume>1e-12&&net<-1e-12?b.team:null;
    return {losses:a.lossPenalty-b.lossPenalty,schedule:a.schedule-b.schedule,
      scheduleFromWins:a.scheduleFromWins-b.scheduleFromWins,scheduleFromLosses:a.scheduleFromLosses-b.scheduleFromLosses,
      performance:a.performance-b.performance,resume,headToHead:a.headToHead-b.headToHead,net,lossGap:a.losses-b.losses,
      meetings,headToHeadNet:meetings.reduce((s,g)=>s+(g.result==='W'?1:g.result==='L'?-1:0),0),decidedBy,
      rarity:(a.rarity??NaN)-(b.rarity??NaN),referenceWins:a.referenceWins-b.referenceWins};
  }
  function cartesian(grid) {
    return Object.entries(grid).reduce((acc,[k,vals])=>acc.flatMap(c=>vals.map(v=>({...c,[k]:v}))),[{}]);
  }
  // Rank range and top-12 share across every combination of the assumption grid.
  function sensitivity(ctx,model,cutoff=12) {
    const {assumptions,sensitivity:grid}=resolveModel(model);
    const configs=cartesian({...Object.fromEntries(Object.entries(assumptions).map(([k,v])=>[k,[v]])),...(grid||{})});
    const ranks=new Map(ctx.teams.map(t=>[t.team,[]]));
    for(const c of configs)for(const r of score(ctx,c,{light:true}))ranks.get(r.team).push(r.rank);
    const out=new Map();
    for(const [team,rs] of ranks) {
      const sorted=[...rs].sort((x,y)=>x-y);
      out.set(team,{min:sorted[0],max:sorted.at(-1),median:sorted[Math.floor((sorted.length-1)/2)],
        inTop:rs.filter(r=>r<=cutoff).length/rs.length,configs:rs.length});
    }
    return {configs,byTeam:out};
  }
  root.RankingHistory={movement,deserving,z,prepare,score,compare,sensitivity,fitRatings,normalCdf,eligible,
    poissonBinomial,upperTail,referenceRating,referenceDistribution,orderTeams,headToHeadTransfer,headToHeadAdjust,priorWeight,RESUME_DEFAULTS};
  if(typeof module!=='undefined')module.exports=root.RankingHistory;
})(typeof window!=='undefined'?window:globalThis);
