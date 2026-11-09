#!/usr/bin/env node
/* Calibrate and backtest the Most Deserving engine on 2014-2025 FBS results.

   Usage:  node scripts/deserving_calibration.cjs [rawDir]
           (rawDir defaults to $CFB_RAW_DIR, then data/raw; it needs CFBD
           games_YYYY.json and, optionally, committee_rankings.json)

   Writes  viz/data/deserving-model.json            calibrated + assumed parameters
           audit/output/deserving_calibration.json  every number in the report

   The production engine (viz/ranking-history.js) is imported, not re-implemented,
   so every backtest number comes from the code that ranks teams on the site.
   2020 is excluded: conference-only, disconnected schedules.
*/
'use strict';
const fs=require('fs'),path=require('path');
const R=require('../viz/ranking-history.js');
const ROOT=path.resolve(__dirname,'..');
const RAW=(require.main===module&&process.argv[2])||process.env.CFB_RAW_DIR||path.join(ROOT,'data','raw');
const TRAIN=[2014,2015,2016,2017,2018,2019,2021],TEST=[2022,2023,2024,2025],ALL=[...TRAIN,...TEST];
const RANKS=[4,8,12,16,25];
const clip=(x,c)=>Math.max(-c,Math.min(c,x));
const Phi=R.normalCdf;
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:NaN;
const sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(x=>(x-m)**2)));};
const round=(x,d=4)=>Number.isFinite(x)?+x.toFixed(d):x;
const roundObj=o=>Object.fromEntries(Object.entries(o).map(([k,v])=>[k,typeof v==='number'?round(v):v]));
const log=(...a)=>console.error(...a);

function load(y) {
  const G=JSON.parse(fs.readFileSync(path.join(RAW,`games_${y}.json`),'utf8'));
  const games=G.filter(x=>x.completed&&x.seasonType==='regular'&&x.homeClassification==='fbs'&&x.awayClassification==='fbs'
      &&Number.isFinite(x.homePoints)&&Number.isFinite(x.awayPoints))
    .map(x=>({h:x.homeTeam,a:x.awayTeam,hp:x.homePoints,ap:x.awayPoints,n:x.neutralSite?1:0,d:x.startDate.slice(0,10),w:x.week,id:x.id,f:1}));
  return {season:y,games,teams:[...new Set(games.flatMap(g=>[g.h,g.a]))].sort()};
}
const seasons=new Map(ALL.map(y=>[y,load(y)]));
let committee={};
try{committee=JSON.parse(fs.readFileSync(path.join(RAW,'committee_rankings.json'),'utf8'));}catch{log('No committee file; descriptive comparison skipped');}

/* ---------- 1. Opponent-strength model: chronological, out-of-sample ---------- */
function indexGames(S,cal){
  const idx=new Map(S.teams.map((t,i)=>[t,i]));
  return S.games.map(g=>({i:idx.get(g.h),j:idx.get(g.a),y:clip(g.hp-g.ap,cal.margin_cap),v:g.n?0:1,w:g.w,m:g.hp-g.ap}));
}
function predictions(S,cal,minWeek=3) {
  const n=S.teams.length,gs=indexGames(S,cal);
  const weeks=[...new Set(gs.map(g=>g.w))].sort((a,b)=>a-b),out=[];
  for(const w of weeks) {
    if(w<minWeek)continue;
    const train=gs.filter(g=>g.w<w);if(!train.length)continue;
    const r=R.fitRatings(n,train,cal);
    for(const g of gs)if(g.w===w&&g.m!==0)out.push({mu:r[g.i]-r[g.j]+cal.home_field*g.v,win:g.m>0?1:0,y:g.y,w});
  }
  return out;
}
function logLoss(P,s){return -mean(P.map(p=>{const q=Math.min(1-1e-9,Math.max(1e-9,Phi(p.mu/s)));return p.win?Math.log(q):Math.log(1-q);}));}
function brier(P,s){return mean(P.map(p=>(Phi(p.mu/s)-p.win)**2));}
function fitSigma(P){let a=4,b=45;const g=(Math.sqrt(5)-1)/2;for(let k=0;k<60;k++){const c=b-g*(b-a),d=a+g*(b-a);if(logLoss(P,c)<logLoss(P,d))b=d;else a=c;}return (a+b)/2;}
const GRID={ridge_games:[1,2,3,4,6,9],margin_cap:[14,21,28,38,60],home_field:[1.5,2,2.5,3,3.5]};
function calibrateRatings(years) {
  const results=[];
  for(const ridge_games of GRID.ridge_games)for(const margin_cap of GRID.margin_cap)for(const home_field of GRID.home_field) {
    const cal={ridge_games,margin_cap,home_field};
    const P=years.flatMap(y=>predictions(seasons.get(y),cal));
    const win_sigma=fitSigma(P);
    results.push({...cal,win_sigma,log_loss:logLoss(P,win_sigma),brier:brier(P,win_sigma),n:P.length});
  }
  return results.sort((a,b)=>a.log_loss-b.log_loss);
}
function calibrationTable(P,s) {
  const bins=Array.from({length:5},(_,k)=>({lo:.5+k/10,n:0,p:0,obs:0}));
  for(const x of P){const q=Phi(x.mu/s),fav=q>=.5?q:1-q,won=q>=.5?x.win:1-x.win;const b=bins[Math.min(4,Math.floor((fav-.5)*10))];b.n++;b.p+=fav;b.obs+=won;}
  return bins.filter(b=>b.n).map(b=>({range:`${b.lo.toFixed(1)}-${(b.lo+.1).toFixed(1)}`,games:b.n,predicted:round(b.p/b.n),observed:round(b.obs/b.n)}));
}
function byPhase(P,s){
  return [['Weeks 3-5',3,5],['Weeks 6-9',6,9],['Weeks 10+',10,99]].map(([phase,a,b])=>{
    const Q=P.filter(p=>p.w>=a&&p.w<=b);return {phase,games:Q.length,log_loss:round(logLoss(Q,s)),brier:round(brier(Q,s))};});
}
// Game noise and true-strength spread give the margin -> "better team" mapping:
// P(winner is the better team | margin m) = Phi(m / (sigma*sqrt(tau^2+sigma^2)/tau)).
function dominanceScale(years,cal) {
  const late=years.flatMap(y=>predictions(seasons.get(y),cal,9));
  const noiseVar=mean(late.map(p=>(p.y-p.mu)**2));
  const all=years.flatMap(y=>seasons.get(y).games.map(g=>clip(g.hp-g.ap,cal.margin_cap)-(g.n?0:cal.home_field)));
  const m=mean(all),totalVar=mean(all.map(x=>(x-m)**2));
  const tauVar=Math.max(1,totalVar-noiseVar),sigma=Math.sqrt(noiseVar),tau=Math.sqrt(tauVar);
  return {dominance_scale:sigma*Math.sqrt(tauVar+noiseVar)/tau,game_noise_sd:sigma,true_spread_sd:tau};
}
const pick=r=>({ridge_games:r.ridge_games,margin_cap:r.margin_cap,home_field:r.home_field,win_sigma:r.win_sigma});

log('Calibrating opponent-strength model on',TRAIN.join(','));
const trainGrid=calibrateRatings(TRAIN);
const calTrain={...pick(trainGrid[0])};Object.assign(calTrain,dominanceScale(TRAIN,calTrain));
const testP=TEST.flatMap(y=>predictions(seasons.get(y),calTrain));
log('Refitting on all seasons for production');
const allGrid=calibrateRatings(ALL);
const calProd={...pick(allGrid[0])};Object.assign(calProd,dominanceScale(ALL,calProd));
const naive={ridge_games:3,margin_cap:1000,home_field:2.5},naiveP=TEST.flatMap(y=>predictions(seasons.get(y),naive)),naiveSigma=fitSigma(naiveP);

/* ---------- 2. Playoff-caliber reference strength: a distribution, prior seasons only ---------- */
// End-of-season ratings of the 30 best FBS teams, per season.
const endSorted=new Map(ALL.map(y=>{
  const S=seasons.get(y),cal=TEST.includes(y)?calTrain:calProd;
  return [y,[...R.fitRatings(S.teams.length,indexGames(S,cal),cal)].sort((a,b)=>b-a).slice(0,30)];
}));
const endRatings=new Map(ALL.map(y=>[y,Object.fromEntries(RANKS.map(k=>[k,endSorted.get(y)[k-1]]))]));
const BAND=4; // "playoff-caliber around #k" = the teams ranked k-4 .. k+4
function bandStats(years){
  const out={ratings:{},sd:{},season_sd:{}};
  for(const k of RANKS){
    const pool=years.flatMap(s=>endSorted.get(s).slice(Math.max(0,k-1-BAND),k+BAND));
    out.ratings[k]=mean(pool);out.sd[k]=sd(pool);out.season_sd[k]=sd(years.map(s=>endRatings.get(s)[k]));
  }
  return out;
}
function referenceFor(y){
  // Chronological: seasons strictly before y. The first three seasons have too
  // little history and use 2014-2016 (flagged as in-sample).
  let prior=ALL.filter(s=>s<y);const inSample=prior.length<3;if(inSample)prior=ALL.slice(0,3);
  return {...bandStats(prior),from:prior,inSample};
}
const refProd=bandStats(ALL);
const calFor=y=>{const f=referenceFor(y);return {...(TEST.includes(y)?calTrain:calProd),reference_ratings:f.ratings,reference_sd:f.sd};};

/* ---------- 3. Snapshots ---------- */
const modelFor=cal=>({method:'earned-results',calibrated:cal});
function snapshot(y,uptoWeek=99,games=null){
  const S=seasons.get(y);return R.prepare((games||S.games).filter(g=>g.w<=uptoWeek),S.teams,modelFor(calFor(y)));
}
log('Building end-of-season snapshots');
const finals=new Map(ALL.map(y=>[y,snapshot(y)]));

/* ---------- 4. Candidate methods ---------- */
function solveDense(A,b,n){const M=Array.from({length:n},(_,i)=>[...A.slice(i*n,i*n+n),b[i]]);
  for(let k=0;k<n;k++){let p=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[p][k]))p=i;[M[k],M[p]]=[M[p],M[k]];
    for(let i=k+1;i<n;i++){const f=M[i][k]/M[k][k];for(let j=k;j<=n;j++)M[i][j]-=f*M[k][j];}}
  const x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let v=M[i][n];for(let j=i+1;j<n;j++)v-=M[i][j]*x[j];x[i]=v/M[i][i];}return x;}
function colleyRows(S){ // The former production method, and the v1 branch formula built on it.
  const names=S.teams,n=names.length,idx=new Map(names.map((t,i)=>[t,i]));
  const A=new Float64Array(n*n),b=new Float64Array(n).fill(1),W=Array(n).fill(0),L=Array(n).fill(0),opps=names.map(()=>[]),margins=names.map(()=>[]);
  for(let i=0;i<n;i++)A[i*n+i]=2;
  for(const g of S.games){const i=idx.get(g.h),j=idx.get(g.a);A[i*n+i]++;A[j*n+j]++;A[i*n+j]--;A[j*n+i]--;opps[i].push(j);opps[j].push(i);
    margins[i].push(g.hp-g.ap);margins[j].push(g.ap-g.hp);if(g.hp>g.ap){W[i]++;L[j]++;b[i]+=.5;b[j]-=.5;}else if(g.hp<g.ap){W[j]++;L[i]++;b[j]+=.5;b[i]-=.5;}}
  const r=solveDense(A,b,n);
  return names.map((team,i)=>{const sos=mean(opps[i].map(j=>r[j]));
    return {team,wins:W[i],losses:L[i],colley:r[i],v1:-L[i]+4*(sos-.5)+.3*mean(margins[i].map(m=>Math.tanh(clip(m,28)/21)))+.16*W[i]/12};});
}
function ranked(rows,key){
  const s=rows.map(r=>({...r,score:r[key]})).sort((a,b)=>b.score-a.score||a.team.localeCompare(b.team));
  s.forEach((r,i)=>r.rank=i>0&&Math.abs(r.score-s[i-1].score)<1e-9?s[i-1].rank:i+1);return s;
}
/* Head-to-head closeness window. "Close" is measured, not guessed: the standard
   deviation of the score gap between two nearby top-25 teams when opponent ratings
   move by one standard error and the reference strength is redrawn. Two résumés
   closer than one SD are not reliably different, so head-to-head may order them. */
function residualNoise(reps=40){
  const r=rng(5),sds=[];
  const base={benchmark_rank:12,convincingness_weight:0,loss_aversion:1,performance_cap:28,reference_uncertainty:false,head_to_head_window:0};
  for(const y of ALL){
    const ctx=finals.get(y),cal=ctx.cal,top=R.score(ctx,base,{light:true}).filter(x=>x.rank<=25).map(x=>x.team);
    const gamesOf=new Map(ctx.teams.map(t=>[t.team,t.games.length])),draws=new Map(top.map(t=>[t,[]]));
    for(let k=0;k<reps;k++){
      const noise=new Map(ctx.names.map(t=>[t,gauss(r)*cal.game_noise_sd/Math.sqrt((gamesOf.get(t)||0)+cal.ridge_games)]));
      const pert={...ctx,teams:ctx.teams.map(t=>({...t,games:t.games.map(g=>({...g,oppRating:g.oppRating+noise.get(g.opponent)}))}))};
      const refDraw=cal.reference_ratings[12]+gauss(r)*cal.reference_sd[12];
      const sc=new Map(R.score(pert,{...base,reference_rating:refDraw},{light:true}).map(x=>[x.team,x.score]));
      for(const t of top)draws.get(t).push(sc.get(t));
    }
    for(let i=0;i<top.length;i++)for(let j=i+1;j<Math.min(top.length,i+6);j++){
      const d=draws.get(top[i]).map((v,k)=>v-draws.get(top[j])[k]);sds.push(sd(d));
    }
  }
  return {pairwise_gap_sd:mean(sds),pairs:sds.length};
}
const noise=residualNoise();
const H2H_WINDOW=round(noise.pairwise_gap_sd,2);
log('Head-to-head window',H2H_WINDOW,'from pairwise gap SD',round(noise.pairwise_gap_sd,3));
const A0={benchmark_rank:12,convincingness_weight:0,loss_aversion:1,performance_cap:28,reference_uncertainty:true,head_to_head_window:H2H_WINDOW};
const modelA=(y,a=A0)=>R.score(finals.get(y),a);
const modelB=(y,a=A0)=>ranked(R.score(finals.get(y),a),'rarity');

/* ---------- 5. Overrides: when does a team with more losses rank higher? ---------- */
const KINDS={'1 loss over unbeaten':[1,0],'2 losses over unbeaten':[2,0],'2 losses over 1 loss':[2,1],'3 losses over 1 loss':[3,1],'3 losses over unbeaten':[3,0]};
function overrides(rowsBySeason,top=25){
  const out={};
  for(const [k,[hi,lo]] of Object.entries(KINDS)){
    let pairs=0,over=0;const examples=[];
    for(const [y,rows] of rowsBySeason){
      const t=rows.filter(r=>r.rank<=top);
      for(const a of t)for(const b of t){if(a.losses!==hi||b.losses!==lo)continue;pairs++;
        if(a.rank<b.rank){over++;examples.push({season:y,above:`${a.team} ${a.wins}-${a.losses} (#${a.rank})`,below:`${b.team} ${b.wins}-${b.losses} (#${b.rank})`,
          score_gap:round(a.score-b.score,3),schedule_gap:a.schedule!==undefined?round(a.schedule-b.schedule,3):undefined});}}
    }
    out[k]={pairs,overrides:over,rate:pairs?round(over/pairs,3):null,examples:examples.sort((x,y)=>x.season-y.season).slice(0,40)};
  }
  return out;
}
function committeeOverlap(rowsBySeason){
  const per=[];for(const [y,rows] of rowsBySeason){const c=committee[y];if(!c)continue;
    const cset=new Set(c.filter(r=>r.rank<=12).map(r=>r.team));per.push({season:y,overlap:rows.filter(r=>r.rank<=12&&cset.has(r.team)).length});}
  return {mean_top12_overlap:per.length?round(mean(per.map(p=>p.overlap)),2):null,per_season:per};
}
const bySeason=fn=>new Map(ALL.map(y=>[y,fn(y)]));

/* ---------- 6. Stability and stress tests ---------- */
function topOverlap(a,b,k=12){const s=new Set(a.filter(r=>r.rank<=k).map(r=>r.team));return b.filter(r=>r.rank<=k&&s.has(r.team)).length;}
function spearmanTop(a,b,k=25){ // over the union of both top-k lists
  const ra=new Map(a.map(r=>[r.team,r.rank])),rb=new Map(b.map(r=>[r.team,r.rank]));
  const teams=[...new Set([...a,...b].filter(r=>r.rank<=k).map(r=>r.team))];
  const x=teams.map(t=>ra.get(t)),y=teams.map(t=>rb.get(t)),mx=mean(x),my=mean(y);
  return mean(x.map((v,i)=>(v-mx)*(y[i]-my)))/(sd(x)*sd(y));
}
function benchmarkStability(method){
  const pairs=[[8,12],[12,16],[4,12],[12,25]],out={};
  for(const [p,q] of pairs){const ov=[],sp=[];
    for(const y of ALL){const a=method(y,{...A0,benchmark_rank:p}),b=method(y,{...A0,benchmark_rank:q});ov.push(topOverlap(a,b));sp.push(spearmanTop(a,b));}
    out[`#${p} vs #${q}`]={top12_overlap:round(mean(ov),2),spearman_top25:round(mean(sp),3)};}
  // One season-to-season SD of the #12 reference strength, either way.
  const ov=[];for(const y of ALL){const base=calFor(y).reference_ratings[12],s=referenceFor(y).sd[12];
    const a=method(y,{...A0,reference_rating:base-s}),b=method(y,{...A0,reference_rating:base+s});ov.push(topOverlap(a,b));}
  out['#12 reference distribution -1 SD vs +1 SD']={top12_overlap:round(mean(ov),2)};
  return out;
}
function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
function gauss(r){let u=0,v=0;while(!u)u=r();while(!v)v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
// Opponent-strength uncertainty: perturb every opponent rating by its approximate
// standard error and re-rank. Same perturbation for both methods.
function opponentUncertainty(reps=60){
  const r=rng(7),res={A:[],B:[],Amove:[],Bmove:[]};
  for(const y of ALL){
    const ctx=finals.get(y),cal=ctx.cal,baseA=R.score(ctx,A0),baseB=ranked(baseA,'rarity');
    const gamesOf=new Map(ctx.teams.map(t=>[t.team,t.games.length]));
    for(let k=0;k<reps;k++){
      const noise=new Map(ctx.names.map(t=>[t,gauss(r)*cal.game_noise_sd/Math.sqrt((gamesOf.get(t)||0)+cal.ridge_games)]));
      const pert={...ctx,teams:ctx.teams.map(t=>({...t,games:t.games.map(g=>({...g,oppRating:g.oppRating+noise.get(g.opponent)}))}))};
      const a=R.score(pert,A0),b=ranked(a,'rarity');
      res.A.push(topOverlap(baseA,a));res.B.push(topOverlap(baseB,b));
      const mv=(base,x)=>{const m=new Map(x.map(r=>[r.team,r.rank]));return mean(base.filter(r=>r.rank<=25).map(r=>Math.abs(m.get(r.team)-r.rank)));};
      res.Amove.push(mv(baseA,a));res.Bmove.push(mv(baseB,b));
    }
  }
  return {A_top12_overlap:round(mean(res.A),2),B_top12_overlap:round(mean(res.B),2),
    A_mean_abs_rank_move_top25:round(mean(res.Amove),2),B_mean_abs_rank_move_top25:round(mean(res.Bmove),2)};
}
// Can a team buy résumé credit by running up scores or adding a weak opponent?
function manipulation(a){
  const gains={margin:[],cupcake_A:[],cupcake_B:[]};
  for(const y of ALL){
    const ctx=finals.get(y),base=R.score(ctx,a);
    const oppMean=new Map();for(const t of ctx.teams)for(const g of t.games){const v=oppMean.get(g.opponent)||[];v.push(g);oppMean.set(g.opponent,v);}
    const weakest=[...oppMean.entries()].sort((p,q)=>mean(p[1].map(g=>g.oppRating))-mean(q[1].map(g=>g.oppRating)))[0];
    for(const r of base.filter(x=>x.rank<=10)){
      const t=ctx.teams.find(x=>x.team===r.team);
      const blow={...ctx,teams:ctx.teams.map(x=>x===t?{...x,games:x.games.map(g=>g.pf>g.pa?{...g,pf:g.pa+70}:g)}:x)};
      gains.margin.push(R.score(blow,a).find(x=>x.team===r.team).score-r.score);
      const extraGame={...weakest[1][0],opponent:weakest[0],site:'home',venue:1,pf:56,pa:0};
      const extra={...ctx,teams:ctx.teams.map(x=>x===t?{...x,games:[...x.games,extraGame]}:x)};
      const after=R.score(extra,a).find(x=>x.team===r.team);
      gains.cupcake_A.push(after.score-r.score);gains.cupcake_B.push(after.rarity-r.rarity);
    }
  }
  return {max_gain_all_wins_by_70:round(Math.max(...gains.margin),3),mean_gain_all_wins_by_70:round(mean(gains.margin),3),
    max_gain_extra_cupcake_A:round(Math.max(...gains.cupcake_A),3),max_gain_extra_cupcake_B_rarity:round(Math.max(...gains.cupcake_B),3)};
}
// Changing results AFTER the snapshot week must not change the snapshot.
function leakageCheck(){
  const r=rng(11);let checked=0,changed=0;
  for(const y of TEST)for(const w of [5,9]){
    const S=seasons.get(y),base=R.score(snapshot(y,w),A0);
    const mutated=S.games.map(g=>g.w>w?{...g,hp:Math.floor(r()*60),ap:Math.floor(r()*60)}:g);
    const after=R.score(snapshot(y,w,mutated),A0);checked++;
    if(JSON.stringify(base.map(x=>[x.team,x.score]))!==JSON.stringify(after.map(x=>[x.team,x.score])))changed++;
  }
  return {snapshots_checked:checked,snapshots_changed_by_future_results:changed,
    reference_strength_uses:Object.fromEntries(ALL.map(y=>{const f=referenceFor(y);return [y,f.inSample?'2014-2016 (in-sample)':`${f.from[0]}-${f.from.at(-1)}`];}))};
}
// Early season: weekly volatility and how small samples behave.
function earlySeason(){
  const out=[];
  for(const w of [3,5,7,9,11]){
    const ov=[],moveA=[],moveB=[],thin=[];
    for(const y of TEST){
      const a=R.score(snapshot(y,w),A0),b=ranked(a,'rarity'),fa=modelA(y);
      ov.push(topOverlap(a,fa));thin.push(a.filter(r=>r.rank<=12&&r.games<=Math.max(2,w-3)).length);
      const nextA=R.score(snapshot(y,w+1),A0),nextB=ranked(nextA,'rarity');
      const mv=(x,z)=>{const m=new Map(z.map(r=>[r.team,r.rank]));return mean(x.filter(r=>r.rank<=25).map(r=>Math.abs(m.get(r.team)-r.rank)));};
      moveA.push(mv(a,nextA));moveB.push(mv(b,nextB));
    }
    out.push({through_week:w,top12_overlap_with_final_A:round(mean(ov),2),weekly_abs_rank_move_top25_A:round(mean(moveA),2),
      weekly_abs_rank_move_top25_B:round(mean(moveB),2),top12_teams_with_few_games:round(mean(thin),2)});
  }
  return out;
}

/* ---------- 7. Schedule fairness (exact expectations, oracle opponent strength) ---------- */
// A schedule-fair rule gives the same expected value to equally good teams whatever
// schedule they faced. Truth = end-of-season ratings, noise = game noise SD.
const GH=[[-2.0201828705,0.0199532421],[-0.9585724646,0.3936193232],[0,0.9453087205],[0.9585724646,0.3936193232],[2.0201828705,0.0199532421]];
function expectedValues(games,quality,cal,a){
  const ref=R.referenceRating(cal,a);let A=0;const qs=[],ps=[];
  for(const g of games){
    const delta=quality-g.oppRating+cal.home_field*g.venue,p=Phi(delta/cal.game_noise_sd),q=Phi((ref-g.oppRating+cal.home_field*g.venue)/cal.win_sigma);
    let es=0;for(const [x,w] of GH){const m=delta+Math.SQRT2*cal.game_noise_sd*x;es+=w/Math.sqrt(Math.PI)*Phi((clip(m,a.performance_cap)-cal.home_field*g.venue)/cal.dominance_scale);}
    A+=-a.loss_aversion*(1-p)+(1-q)+a.convincingness_weight*(es-p);qs.push(q);ps.push(p);
  }
  const truth=R.poissonBinomial(ps),refd=R.poissonBinomial(qs);let B=0;
  truth.forEach((pk,k)=>{B+=pk*-Math.log10(Math.max(R.upperTail(refd,k),1e-300));});
  return {A,B};
}
function fairness(a){
  const out={};
  for(const k of [4,12,25]){
    const spreadA=[],spreadB=[];
    for(const y of ALL){
      const ctx=finals.get(y),cal=ctx.cal,quality=calFor(y).reference_ratings[k];
      const rows=R.score(ctx,a).filter(r=>r.rank<=40);
      const vals=rows.map(r=>{const t=ctx.teams.find(x=>x.team===r.team);const e=expectedValues(t.games,quality,cal,a);
        return {E:r.schedule/r.games,A:e.A/r.games*12,B:e.B};});
      vals.sort((p,q)=>p.E-q.E);const d=Math.max(1,Math.floor(vals.length/5));
      spreadA.push(mean(vals.slice(-d).map(v=>v.A))-mean(vals.slice(0,d).map(v=>v.A)));
      spreadB.push(mean(vals.slice(-d).map(v=>v.B))-mean(vals.slice(0,d).map(v=>v.B)));
    }
    out[`#${k}-caliber team`]={A_hardest_minus_easiest_quintile_loss_units_per_12:round(mean(spreadA),3),B_hardest_minus_easiest_quintile_rarity:round(mean(spreadB),3)};
  }
  return out;
}

/* ---------- 8. Simulation: are overrides signal or noise? ---------- */
function simulate(a,reps=150,method='A'){
  let o10=0,o10right=0,keep10=0,keep10right=0,o20=0,o20right=0,n=0;const r=rng(20261108);
  for(const y of ALL){
    const S=seasons.get(y),cal=calFor(y),gs=indexGames(S,cal),truth=R.fitRatings(S.teams.length,gs,cal),ref=R.referenceRating(cal,a);
    for(let rep=0;rep<reps;rep++){
      n++;const rows=S.teams.map(()=>({W:0,L:0,score:0,qs:[]}));
      for(const g of gs){
        const m=truth[g.i]-truth[g.j]+cal.home_field*g.v+cal.game_noise_sd*gauss(r);
        for(const [t,o,sign] of [[g.i,g.j,1],[g.j,g.i,-1]]){
          const venue=g.v*sign,mm=sign*m,W=mm>0?1:0,q=Phi((ref-truth[o]+cal.home_field*venue)/cal.win_sigma);
          const s=Phi((clip(mm,a.performance_cap)-cal.home_field*venue)/cal.dominance_scale);
          rows[t].W+=W;rows[t].L+=1-W;rows[t].qs.push(q);rows[t].score+=-a.loss_aversion*(1-W)+(1-q)+a.convincingness_weight*(s-W);
        }
      }
      if(method==='B')rows.forEach(x=>{x.score=-Math.log10(Math.max(R.upperTail(R.poissonBinomial(x.qs),x.W),1e-300));});
      const order=rows.map((x,t)=>({...x,t})).sort((x,z)=>z.score-x.score).slice(0,25);order.forEach((x,k)=>x.rank=k);
      for(const A of order)for(const B of order){
        if(A.L===1&&B.L===0){if(A.rank<B.rank){o10++;if(truth[A.t]>truth[B.t])o10right++;}else{keep10++;if(truth[B.t]>truth[A.t])keep10right++;}}
        if(A.L===2&&B.L===0&&A.rank<B.rank){o20++;if(truth[A.t]>truth[B.t])o20right++;}
      }
    }
  }
  return {simulated_seasons:n,one_loss_over_unbeaten_rate:round(o10/(o10+keep10),3),
    when_overridden_one_loss_team_truly_stronger:round(o10right/Math.max(1,o10),3),
    when_not_overridden_unbeaten_truly_stronger:round(keep10right/Math.max(1,keep10),3),
    two_loss_over_unbeaten_per_season:round(o20/n,3),when_two_loss_over_unbeaten_truly_stronger:round(o20right/Math.max(1,o20),3)};
}

/* ---------- 9. Model C: what loss unit do the rarity probabilities imply? ---------- */
// Pairwise agreement between Model A (loss weight omega) and Model B over each
// season's top 40. If B's tail probabilities imply omega ~ 1, the additive
// loss-equivalent score already carries B's information.
function impliedLossUnit(){
  const out=[];
  for(const omega of [0.6,0.8,0.9,1,1.1,1.2,1.4,1.7,2]){
    let agree=0,total=0;
    for(const y of ALL){
      const a=R.score(finals.get(y),{...A0,loss_aversion:omega}),bmap=new Map(ranked(a,'rarity').map(r=>[r.team,r.rank]));
      const top=a.filter(r=>r.rank<=40||bmap.get(r.team)<=40);
      for(let i=0;i<top.length;i++)for(let j=i+1;j<top.length;j++){total++;if((top[i].rank<top[j].rank)===(bmap.get(top[i].team)<bmap.get(top[j].team)))agree++;}
    }
    out.push({loss_aversion:omega,pairwise_agreement_with_B:round(agree/total,4)});
  }
  return out;
}

/* ---------- 10. Assumption sweep ---------- */
const SWEEP={benchmark_rank:[4,8,12,16,25],convincingness_weight:[0,0.05,0.1,0.2],loss_aversion:[1,1.25,1.5]};
function sweepTable(){
  const out=[];
  for(const benchmark_rank of SWEEP.benchmark_rank)for(const convincingness_weight of SWEEP.convincingness_weight)for(const loss_aversion of SWEEP.loss_aversion){
    const a={benchmark_rank,convincingness_weight,loss_aversion,performance_cap:28},rs=bySeason(y=>modelA(y,a)),ov=overrides(rs);
    out.push({...a,one_over_unbeaten:ov['1 loss over unbeaten'].rate,two_over_unbeaten:ov['2 losses over unbeaten'].rate,
      two_over_one:ov['2 losses over 1 loss'].rate,committee_top12_overlap:committeeOverlap(rs).mean_top12_overlap});
  }
  return out;
}
// Every order that head-to-head reversed, and close meetings it could not reverse
// counting only pairs where the winner was placed above the team it beat.
function headToHeadBacktest(a){
  const reversed=[];let swapsAll=0;
  for(const y of ALL){
    const rows=modelA(y,a),by=new Map(rows.map(r=>[r.team,r]));
    for(const r of rows)for(const p of r.headToHeadPassed.filter(x=>x.won)){
      swapsAll++;const o=by.get(p.team);
      if(Math.min(r.rank,o.rank)<=25)reversed.push({season:y,winner:r.team+' '+r.wins+'-'+r.losses+' (#'+r.rank+')',loser:o.team+' '+o.wins+'-'+o.losses+' (#'+o.rank+')',score_gap:round(r.score-o.score,3)});
    }
  }
  return {orders_reversed_all:swapsAll,orders_reversed_top25:reversed.length,reversed};
}
function seasonTables(a){
  return Object.fromEntries(ALL.map(y=>{const rows=modelA(y,a);return [y,rows.slice(0,16).map(r=>({rank:r.rank,team:r.team,record:`${r.wins}-${r.losses}`,
    score:round(r.score,3),range:[round(r.scoreLow,2),round(r.scoreHigh,2)],head_to_head_passed:r.headToHeadPassed.map(p=>p.team),top12_probability:round(r.topProbability,2),losses:round(r.lossPenalty,2),schedule:round(r.schedule,3),performance:round(r.performance,3),
    reference_wins:round(r.referenceWins,2),match_probability:round(r.matchProbability,4),rarity:round(r.rarity,3),rarity_rank:r.rarityRank}))];}));
}

module.exports={seasons,finals,calTrain,calProd,calFor,referenceFor,refProd,endRatings,modelA,modelB,ranked,colleyRows,overrides,committeeOverlap,
  benchmarkStability,opponentUncertainty,manipulation,leakageCheck,earlySeason,fairness,simulate,impliedLossUnit,sweepTable,seasonTables,bySeason,A0,ALL,TRAIN,TEST,round};

if(require.main===module){
  const DEFAULT=process.env.DESERVING_DEFAULT?JSON.parse(process.env.DESERVING_DEFAULT):A0;
  log('Default assumptions',DEFAULT);
  const candidates={
    'Colley (former production)':bySeason(y=>ranked(colleyRows(seasons.get(y)),'colley')),
    'Record-first v1 branch':bySeason(y=>ranked(colleyRows(seasons.get(y)),'v1')),
    'Model A: wins above playoff-caliber (alpha 0)':bySeason(y=>modelA(y)),
    'Model A without head-to-head':bySeason(y=>modelA(y,{...A0,head_to_head_window:0})),
    'Model A, point reference (no uncertainty)':bySeason(y=>modelA(y,{...A0,reference_uncertainty:false})),
    'Model B: résumé rarity -log10 P(W*>=W)':bySeason(y=>modelB(y)),
    'Model A + 5% convincingness (rejected)':bySeason(y=>modelA(y,{...DEFAULT,convincingness_weight:0.05})),
  };
  const candidateReport=Object.fromEntries(Object.entries(candidates).map(([k,rs])=>[k,{overrides:overrides(rs),committee:committeeOverlap(rs)}]));
  log('Stress tests');
  const report={
    generated:new Date().toISOString().slice(0,10),
    data:`CFBD regular-season FBS-v-FBS results ${ALL[0]}-${ALL.at(-1)} (2020 excluded), conference title games included`,
    rating_model:{train_seasons:TRAIN,test_seasons:TEST,grid:GRID,selected_on_train:roundObj(calTrain),train_log_loss:round(trainGrid[0].log_loss),
      test:{games:testP.length,log_loss:round(logLoss(testP,calTrain.win_sigma)),brier:round(brier(testP,calTrain.win_sigma)),
        accuracy:round(mean(testP.map(p=>(p.mu>0)===(p.win===1)?1:0))),by_phase:byPhase(testP,calTrain.win_sigma),calibration:calibrationTable(testP,calTrain.win_sigma)},
      test_reference_uncapped_ridge3:{log_loss:round(logLoss(naiveP,naiveSigma)),brier:round(brier(naiveP,naiveSigma))},
      top_train_configs:trainGrid.slice(0,6).map(roundObj),production:roundObj(calProd)},
    reference_strength:{definition:'average end-of-season rating (points vs average FBS team, neutral field) of the k-th best FBS team in earlier seasons',
      per_season_end_ratings:Object.fromEntries(ALL.map(y=>[y,roundObj(endRatings.get(y))])),production:roundObj(refProd.ratings),production_sd:roundObj(refProd.sd),production_season_sd:roundObj(refProd.season_sd),band:`ranks k-${BAND}..k+${BAND}`},
    default_assumptions:DEFAULT,
    head_to_head:{window:H2H_WINDOW,noise:{pairwise_gap_sd:round(noise.pairwise_gap_sd,4),pairs:noise.pairs},backtest:headToHeadBacktest(DEFAULT)},
    candidates:candidateReport,
    implied_loss_unit:impliedLossUnit(),
    stability:{benchmark_A:benchmarkStability(modelA),benchmark_B:benchmarkStability(modelB),opponent_uncertainty:opponentUncertainty()},
    manipulation:manipulation(DEFAULT),
    leakage:leakageCheck(),
    early_season:earlySeason(),
    fairness:{model_A_alpha0:fairness(A0),convincingness_0_05:fairness({...DEFAULT,convincingness_weight:0.05}),loss_aversion_1_5:fairness({...DEFAULT,loss_aversion:1.5}),top4_reference:fairness({...DEFAULT,benchmark_rank:4})},
    simulation:{model_A_alpha0:simulate(A0),model_B:simulate(A0,150,'B'),convincingness_0_05:simulate({...DEFAULT,convincingness_weight:0.05}),loss_aversion_1_5:simulate({...DEFAULT,loss_aversion:1.5}),top4_reference:simulate({...DEFAULT,benchmark_rank:4})},
    sweep:sweepTable(),
    season_tables:seasonTables(DEFAULT),
  };
  fs.mkdirSync(path.join(ROOT,'audit','output'),{recursive:true});
  fs.writeFileSync(path.join(ROOT,'audit','output','deserving_calibration.json'),JSON.stringify(report,null,1));
  const model={version:3,method:'earned-results',
    description:'Wins above a playoff-caliber reference team, in loss-equivalent units, with exact résumé rarity',
    games:'completed regular-season FBS-versus-FBS only (conference title games included)',
    calibrated:{...roundObj(calProd),reference_ratings:roundObj(refProd.ratings),reference_sd:roundObj(refProd.sd),reference_season_sd:roundObj(refProd.season_sd),
      reference_band:`end-of-season strengths of the teams ranked k-${BAND}..k+${BAND}, pooled over calibration seasons`,
      score_gap_noise_sd:round(noise.pairwise_gap_sd,4),
      calibration_seasons:ALL,source:'scripts/deserving_calibration.cjs'},
    assumptions:DEFAULT,
    assumption_notes:{benchmark_rank:'Reference team = the k-th best FBS team (12 = playoff cutoff). Philosophical.',
      convincingness_weight:'Share of each game decided by how clearly it was won rather than whether it was won. Philosophical.',
      loss_aversion:'Loss-equivalent penalty per loss. 1 keeps the score schedule-fair for a reference-caliber team. Philosophical.',
      performance_cap:'Margins beyond this many points earn nothing more. Philosophical.',
      reference_uncertainty:'Judge against the whole playoff-caliber distribution (true) or only its mean (false).',
      head_to_head_window:'Score gap (loss units) inside which a team that won the head-to-head game moves directly above the team it beat, passing only teams also within this gap that did not beat it: one SD of the measured score-gap noise. Scores are unchanged.'},
    sensitivity:{benchmark_rank:[8,12,16],convincingness_weight:[0,0.05,0.1],loss_aversion:[1,1.25],head_to_head_window:[0,H2H_WINDOW]},
    equal_scores:'shared rank',report:'audit/MOST_DESERVING_V2.md'};
  // A compact summary the page shows in "How it works". Every number is from the report above.
  const fin=candidateReport['Model A: wins above playoff-caliber (alpha 0)'],rb=candidateReport['Model B: résumé rarity -log10 P(W*>=W)'];
  const best=report.implied_loss_unit.reduce((a,b)=>b.pairwise_agreement_with_B>a.pairwise_agreement_with_B?b:a);
  model.validation={seasons:`${ALL[0]}-${ALL.at(-1)} (2020 excluded)`,
    opponent_model:{test_seasons:`${TEST[0]}-${TEST.at(-1)}`,test_games:report.rating_model.test.games,log_loss:report.rating_model.test.log_loss,brier:report.rating_model.test.brier,accuracy:report.rating_model.test.accuracy},
    overrides:Object.fromEntries(['1 loss over unbeaten','2 losses over unbeaten','3 losses over unbeaten'].map(k=>[k,{model_a:`${fin.overrides[k].overrides} of ${fin.overrides[k].pairs}`,model_b:`${rb.overrides[k].overrides} of ${rb.overrides[k].pairs}`}])),
    two_loss_over_unbeaten:fin.overrides['2 losses over unbeaten'].examples.map(e=>`${e.season}: ${e.above} over ${e.below}`),
    implied_loss_weight_from_rarity:best.loss_aversion,
    rarity_agreement_at_loss_weight_1:report.implied_loss_unit.find(x=>x.loss_aversion===1).pairwise_agreement_with_B,
    schedule_bias_reference_caliber:{loss_weight_1:report.fairness.model_A_alpha0['#12-caliber team'].A_hardest_minus_easiest_quintile_loss_units_per_12,
      loss_weight_1_5:report.fairness.loss_aversion_1_5['#12-caliber team'].A_hardest_minus_easiest_quintile_loss_units_per_12},
    future_results_leak:report.leakage.snapshots_changed_by_future_results,
    head_to_head:{window:H2H_WINDOW,noise_sd:round(noise.pairwise_gap_sd,3),orders_reversed:report.head_to_head.backtest.orders_reversed_top25,
      examples:report.head_to_head.backtest.reversed.slice(0,10).map(e=>`${e.season}: ${e.winner} over ${e.loser} (score gap ${e.score_gap})`)}};
  fs.writeFileSync(path.join(ROOT,'viz','data','deserving-model.json'),JSON.stringify(model,null,2)+'\n');
  log('Wrote audit/output/deserving_calibration.json and viz/data/deserving-model.json');
}
