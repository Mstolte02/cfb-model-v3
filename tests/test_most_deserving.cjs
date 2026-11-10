/* Most Deserving engine: invariants, exact résumé rarity and score decomposition.
   Run: node --test tests/test_most_deserving.cjs */
'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const R=require('../viz/ranking-history.js');
const model=require('../viz/data/deserving-model.json');
const close=(a,b,tol=1e-9,msg='')=>assert.ok(Math.abs(a-b)<=tol,`${msg} ${a} != ${b}`);
const game=(h,a,hp,ap,extra={})=>({h,a,hp,ap,f:1,n:0,d:'2026-09-01',...extra});
const rank=(games,names,assumptions)=>R.deserving(games,{},names,model,{assumptions});
function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}

// A small connected league: six teams, ten games, mixed venues and margins.
const NAMES=['A','B','C','D','E','F'];
const LEAGUE=[
  game('A','B',24,17),game('A','C',28,10,{d:'2026-09-08'}),game('D','A',7,21,{d:'2026-09-15'}),
  game('E','B',31,10),game('E','C',17,14,{d:'2026-09-08',n:1}),game('D','E',24,21,{d:'2026-09-22'}),
  game('F','A',3,14,{d:'2026-09-22'}),game('F','B',13,20,{d:'2026-09-29'}),game('C','F',35,0,{d:'2026-10-06'}),game('B','D',27,24,{d:'2026-10-06',n:1})
];
const compact=rows=>rows.map(r=>[r.team,r.rank,r.score]);

test('model file is the calibrated earned-results model',()=>{
  assert.equal(model.method,'earned-results');
  for(const k of ['ridge_games','margin_cap','home_field','win_sigma','dominance_scale'])assert.ok(Number.isFinite(model.calibrated[k]),k);
  assert.ok(Number.isFinite(model.calibrated.reference_ratings[12]));
  assert.ok(model.calibrated.reference_ratings[4]>model.calibrated.reference_ratings[12]);
  assert.throws(()=>R.deserving(LEAGUE,{},NAMES,{method:'colley'}),/Unsupported/);
  assert.deepEqual(R.deserving([],{},NAMES,model),[]);
});

test('exact Poisson-binomial matches hand calculations',()=>{
  const d=R.poissonBinomial([.5,.8]);
  close(d[0],.1);close(d[1],.5);close(d[2],.4);
  close(R.upperTail(d,2),.4);close(R.upperTail(d,1),.9);close(R.upperTail(d,0),1);
  const t=R.poissonBinomial([.9,.9,.9]);
  close(t[3],.729);close(R.upperTail(t,2),.972);
  const e=R.poissonBinomial([]);close(e[0],1);
  // A tie counts as half a win: P(W* >= 1.5) = P(W* >= 2).
  close(R.upperTail(d,1.5),.4);
});

test('exact tails agree with a large Monte Carlo run within sampling error',()=>{
  const r=rng(42),qs=Array.from({length:12},()=>.35+.6*r()),dist=R.poissonBinomial(qs),N=200000,counts=Array(13).fill(0);
  for(let s=0;s<N;s++){let w=0;for(const q of qs)if(r()<q)w++;counts[w]++;}
  let tail=0;
  for(let k=12;k>=0;k--){
    tail+=counts[k]/N;const exact=R.upperTail(dist,k),se=Math.sqrt(exact*(1-exact)/N);
    assert.ok(Math.abs(tail-exact)<=4*se+1e-12,`k=${k}: MC ${tail} vs exact ${exact}`);
  }
});

test('score decomposes exactly into its displayed components',()=>{
  const rows=rank(LEAGUE,NAMES);
  for(const r of rows){
    close(r.resumeScore,r.lossPenalty+r.schedule+r.performance,1e-12,'résumé score');
    close(r.score,r.resumeScore+r.headToHead,1e-12,'ranking score = résumé + head-to-head');
    close(r.headToHead,r.headToHeadGames.reduce((s,g)=>s+g.value,0),1e-12,'head-to-head lines add up');
    close(r.schedule,r.scheduleFromWins+r.scheduleFromLosses,1e-12,'schedule');
    close(r.resumeScore,r.ledger.reduce((s,g)=>s+g.total,0),1e-12,'ledger');
    close(r.referenceWins+r.schedule,r.games,1e-12,'reference wins');
    close(r.lossPenalty,-model.assumptions.loss_aversion*(r.losses+r.ties/2),1e-12,'loss penalty');
    close(r.rarity,-Math.log10(r.matchProbability),1e-12,'rarity');
  }
  // The head-to-head explanation reconciles with the ranking for every pair.
  for(const a of rows)for(const b of rows){
    const c=R.compare(a,b);
    close(c.net,a.score-b.score,1e-12);close(c.losses+c.schedule+c.performance,c.resume,1e-12);close(c.resume+c.headToHead,c.net,1e-12);
    // Conditional override, stated in loss units: A ranks above B with more losses
    // only when its schedule plus performance edge exceeds the extra losses.
    if(a.losses>b.losses&&a.rank<b.rank&&!c.decidedBy)assert.ok(c.schedule+c.performance>-c.losses);
  }
});

test('ranking is transitive, deterministic and independent of game order',()=>{
  const rows=rank(LEAGUE,NAMES);
  rows.forEach((r,i)=>{if(i)assert.ok(rows[i-1].score>=r.score-1e-12,'sorted by one number');});
  rows.forEach((r,i)=>{if(i&&rows[i-1].score-r.score>1e-9)assert.ok(r.rank>rows[i-1].rank);});
  assert.deepEqual(compact(rank([...LEAGUE].reverse(),NAMES)),compact(rows));
  const r=rng(3),shuffled=[...LEAGUE].sort(()=>r()-.5);
  assert.deepEqual(compact(rank(shuffled,NAMES)),compact(rows));
  assert.deepEqual(compact(rank(LEAGUE,NAMES)),compact(rows));
});

test('a neutral-site game is the same game whichever team is listed first',()=>{
  const swapped=LEAGUE.map(g=>g.n?{...g,h:g.a,a:g.h,hp:g.ap,ap:g.hp}:g);
  assert.deepEqual(compact(rank(swapped,NAMES)),compact(rank(LEAGUE,NAMES)));
});

test('a better result never lowers a team\'s résumé score or rarity, and a win always outweighs head-to-head (live schedule)',()=>{
  const schedule=require('../viz/data/schedule.json'),names=require('../viz/data/ratings.json').teams.map(t=>t.team);
  const base=new Map(R.deserving(schedule,{},names,model).map(r=>[r.team,r]));
  const finals=schedule.map((g,i)=>[g,i]).filter(([g])=>g.f&&Number.isFinite(g.hp)&&names.includes(g.h)&&names.includes(g.a));
  const r=rng(9);
  for(let k=0;k<12;k++){
    const [g,i]=finals[Math.floor(r()*finals.length)],home=r()<.5,team=home?g.h:g.a;
    const before=base.get(team),margin=home?g.hp-g.ap:g.ap-g.hp;
    // Flip a loss into a win by the same margin, or add a touchdown to a win.
    const newMargin=margin<0?-margin:margin+7;
    const changed=schedule.map((x,j)=>j!==i?x:home?{...x,hp:x.ap+newMargin}:{...x,ap:x.hp+newMargin});
    const after=R.deserving(changed,{},names,model).find(x=>x.team===team);
    assert.ok(after.resumeScore>=before.resumeScore-1e-12,`${team} résumé score fell`);
    assert.ok(after.rarity>=before.rarity-1e-12,`${team} rarity fell`);
    // The ranking score adds a head-to-head term capped at +/- W/2, so a flipped loss
    // (+1) always nets at least 1 - W; a margin change moves it by less than W.
    if(margin<0)assert.ok(after.score-before.score>=1-model.assumptions.head_to_head_window-1e-9,'a loss turned into a win must be worth a lot');
    else assert.ok(Math.abs(after.score-before.score)<=model.assumptions.head_to_head_window+1e-9);
  }
});

test('a team\'s own results never change how strong its opponents look to it',()=>{
  const base=R.prepare(LEAGUE,NAMES,model),blown=R.prepare(LEAGUE.map(g=>g.h==='A'?{...g,hp:g.hp+40}:g.a==='A'?{...g,ap:g.ap+40}:g),NAMES,model);
  const ta=base.teams.find(t=>t.team==='A'),tb=blown.teams.find(t=>t.team==='A');
  ta.games.forEach((g,i)=>close(g.oppRating,tb.games[i].oppRating,1e-12));
});

test('margin restraint: margins past the cap earn nothing more',()=>{
  const cap=model.assumptions.performance_cap;
  const base=[...LEAGUE,game('C','F',cap,0,{d:'2026-10-13'})],run=[...LEAGUE,game('C','F',cap+70,0,{d:'2026-10-13'})];
  const a=rank(base,NAMES).find(r=>r.team==='C'),b=rank(run,NAMES).find(r=>r.team==='C');
  close(a.resumeScore,b.resumeScore,1e-12);
  // The margin reshapes how OTHER teams' opponents look, which can shift C's head-to-head
  // gaps slightly; that second-order effect stays tiny even in a six-team league.
  assert.ok(Math.abs(a.score-b.score)<0.01);
  // A whole season of maximal margins is worth at most alpha * games * (1 - s(cap)) less than nothing.
  assert.ok(Math.abs(a.performance)<=model.assumptions.convincingness_weight*a.games+1e-12);
});

test('loss accountability and schedule sensitivity',()=>{
  // Identical schedules; Y loses the game X wins. X must rank higher.
  const names=['X','Y','O1','O2','O3'];
  const g=[game('X','O1',21,14),game('X','O2',21,14),game('Y','O1',21,14),game('Y','O2',14,21),game('O1','O3',10,7),game('O2','O3',10,7),game('O3','X',3,21),game('O3','Y',3,21)];
  const rows=rank(g,names),x=rows.find(r=>r.team==='X'),y=rows.find(r=>r.team==='Y');
  // (Schedules differ slightly: each team's fit keeps the other team's results.)
  assert.ok(x.score>y.score+0.5);assert.equal(x.losses+1,y.losses);
  // Two unbeaten teams: S beats the strong side of the league, W beats the weak side.
  const n2=['S','W','P1','P2','Q1','Q2'];
  const g2=[game('P1','Q1',35,7),game('P2','Q2',35,7),game('P1','Q2',35,7),game('P2','Q1',35,7),
    game('S','P1',24,21),game('S','P2',24,21),game('W','Q1',24,21),game('W','Q2',24,21)];
  const r2=rank(g2,n2),s=r2.find(r=>r.team==='S'),w=r2.find(r=>r.team==='W');
  assert.ok(s.schedule>w.schedule&&s.score>w.score&&s.rarity>w.rarity);
});

test('a road win is worth more than the same home win',()=>{
  const names=['T','U','V'];
  const home=rank([game('T','U',20,17),game('V','U',20,17)],names),road=rank([game('U','T',17,20),game('V','U',20,17)],names);
  assert.ok(road.find(r=>r.team==='T').score>home.find(r=>r.team==='T').score);
});

test('only completed regular-season FBS-v-FBS games count',()=>{
  const ignored=[game('A','FCS',30,0),game('FCS','B',30,0),{...game('A','B',1,0),f:0},{...game('A','B',1,0),hp:null},{...game('A','B',9,0),st:'postseason'}];
  assert.deepEqual(compact(rank([...LEAGUE,...ignored],NAMES)),compact(rank(LEAGUE,NAMES)));
});

test('small samples stay finite and disconnected schedules still solve',()=>{
  const rows=rank([game('A','B',7,3),game('C','D',10,3)],NAMES);
  assert.equal(rows.length,4);assert.ok(rows.every(r=>Number.isFinite(r.score)&&Number.isFinite(r.rarity)));
  // Equal résumés share a rank.
  assert.equal(rows[0].rank,rows[1].rank);
});

test('sensitivity summarises every assumption combination',()=>{
  const ctx=R.prepare(LEAGUE,NAMES,model),s=R.sensitivity(ctx,model);
  const expected=Object.values(model.sensitivity).reduce((p,v)=>p*v.length,1);
  assert.equal(s.configs.length,expected);
  for(const v of s.byTeam.values()){assert.ok(v.min<=v.median&&v.median<=v.max);assert.ok(v.inTop>=0&&v.inTop<=1);assert.equal(v.configs,expected);}
  // A stronger reference team expects fewer losses on every schedule; the reference is the same for every team.
  const lo=R.score(ctx,{...model.assumptions,benchmark_rank:25,head_to_head_window:0}),hi=R.score(ctx,{...model.assumptions,benchmark_rank:4,head_to_head_window:0});
  assert.ok(new Set(hi.map(r=>r.referenceRating)).size===1);
  for(const r of hi)assert.ok(r.schedule<lo.find(x=>x.team===r.team).schedule+1e-12);
});

test('head-to-head explanation reconciles and states the override in loss units',()=>{
  const Lab=require('../viz/deserving-lab.js');
  const schedule=require('../viz/data/schedule.json'),names=require('../viz/data/ratings.json').teams.map(t=>t.team);
  const rows=R.deserving(schedule,{},names,model),by=new Map(rows.map(r=>[r.team,r]));
  for(const [i,j] of [[0,1],[2,3],[0,10],[5,20]]){
    const a=rows[i],b=rows[j],e=Lab.explain(a,b,model.assumptions);
    const total=e.rows.find(r=>r.total).value,parts=e.rows.filter(r=>!r.total&&!r.subtotal).reduce((s,r)=>s+r.value,0);
    close(total,a.score-b.score,1e-12);close(parts,total,1e-12);
    close(e.rows.find(r=>r.subtotal).value,a.resumeScore-b.resumeScore,1e-12);
    assert.match(e.rarity,/playoff-caliber reference team/);
  }
  const [ta,tb]=Lab.defaultPair(rows),a=by.get(ta),b=by.get(tb);
  if(a.losses>b.losses&&!R.compare(a,b).decidedBy){assert.ok(a.rank<b.rank);assert.match(Lab.explain(a,b,model.assumptions).conclusion,/despite \d+ more loss/);}
  assert.equal(Lab.band(1).key,'secure');assert.equal(Lab.band(.5).key,'bubble');assert.equal(Lab.band(0).key,'out');
  assert.equal(Lab.signed(0.5),'+0.50');assert.equal(Lab.signed(-1),'−1.00');assert.equal(Lab.signed(0),'0.00');
});

test('reference uncertainty: exact expected credit, honest range and averaged rarity',()=>{
  const ctx=R.prepare(LEAGUE,NAMES,model),a=model.assumptions,dist=R.referenceDistribution(ctx.cal,a);
  assert.ok(dist.sd>0,'the reference team is a distribution');
  const rows=R.score(ctx,a),point=R.score(ctx,{...a,reference_uncertainty:false});
  // Closed form E[Phi((R-r)/s)] = Phi((mean-r)/sqrt(s^2+sd^2)) agrees with brute-force averaging.
  const rr=rng(17),N=4000,draws=Array.from({length:N},()=>{let u=0,v=0;while(!u)u=rr();while(!v)v=rr();return dist.mean+dist.sd*Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);});
  for(const r of rows){
    const avg=draws.reduce((s,ref)=>s+R.score(ctx,{...a,reference_rating:ref,reference_uncertainty:false,head_to_head_window:0},{light:true}).find(x=>x.team===r.team).schedule,0)/N;
    close(r.schedule,avg,0.02,r.team+' expected schedule credit');
    assert.ok(r.scoreLow<=r.score+0.05&&r.score<=r.scoreHigh+0.05,r.team+' range brackets the score');
    assert.ok(r.matchProbabilityLow<=r.matchProbability+1e-12&&r.matchProbability<=r.matchProbabilityHigh+1e-12,r.team+' rarity range');
    assert.ok(r.topProbability>=0&&r.topProbability<=1);
  }
  assert.ok(point.every(r=>Math.abs(r.scoreHigh-r.scoreLow)<1e-12),'no uncertainty, no range');
});

test('head-to-head is a capped adjustment inside the noise band and leaves the résumé score alone (live)',()=>{
  const schedule=require('../viz/data/schedule.json'),names=require('../viz/data/ratings.json').teams.map(t=>t.team);
  const a=model.assumptions,W=a.head_to_head_window,ctx=R.prepare(schedule,names,model);
  assert.ok(W>0,'head-to-head is on in production');
  const on=R.score(ctx,a,{light:true}),off=R.score(ctx,{...a,head_to_head_window:0},{light:true});
  const offBy=new Map(off.map(r=>[r.team,r]));let pairSum=0;
  for(const r of on){
    close(r.resumeScore,offBy.get(r.team).score,1e-12,'résumé score unchanged');
    assert.ok(Math.abs(r.headToHead)<=W/2+1e-12,'no team moves more than W/2');
    for(const g of r.headToHeadGames){if(g.cap)continue;pairSum+=g.value;
      assert.ok(g.won?g.value>0:g.value<0,'winners gain, losers give up');
      assert.ok(Math.abs(g.value)<=W/2+1e-12,'no pair closes by more than W');
      assert.ok(g.won?g.gap<W:-g.gap<W,'only inside the band');}
  }
  close(pairSum,0,1e-9,'every pair is zero-sum before the team cap');
  on.forEach((r,i)=>{if(i)assert.ok(r.rank>=on[i-1].rank&&on[i-1].score>=r.score-1e-12,'ranks follow the ranking score');});
});
test('the transfer: Bayesian shift, tapering to zero at the band edge, no jumps',()=>{
  const W=0.39,t=g=>R.headToHeadTransfer(g,W);
  assert.equal(t(W),0);assert.equal(t(W+0.5),0);assert.equal(R.headToHeadTransfer(0,0),0);
  for(let g=-2;g<W+0.1;g+=0.001){assert.ok(t(g)>=0&&t(g)<=W);assert.ok(Math.abs(t(g+0.001)-t(g))<0.002,`continuous at ${g}`);}
  // Confirming results (winner already ahead) move little and fade out.
  assert.ok(t(-1)<0.03&&t(-0.5)<t(-0.2)&&t(-0.2)<t(0));
  // A winner trailing by less than ~0.24 moves ahead; further back, the gap only narrows.
  assert.ok(t(0.2)>0.2);assert.ok(t(0.25)<0.25&&t(0.25)>0);
});
test('head-to-head reverses a near-tie, only narrows a wider gap, and ignores one outside the band',()=>{
  // Opponent strengths are set directly so the résumé gaps are exact.
  const cal=R.prepare(LEAGUE,NAMES,model).cal,a={...model.assumptions,reference_uncertainty:false},W=a.head_to_head_window;
  const ref=R.referenceRating(cal,a),q=r=>R.normalCdf((ref-r)/cal.win_sigma);
  const g=(opponent,pf,pa,oppRating)=>({opponent,pf,pa,oppRating,venue:0,site:'neutral',date:'2026-09-01'});
  const ratingFor=target=>{let lo=-60,hi=60;for(let k=0;k<200;k++){const m=(lo+hi)/2;if(q(m)<target)hi=m;else lo=m;}return (lo+hi)/2;};
  // X beat Y. X: win over Y, loss to Z. Y: loss to X, win over V. Résumé gap X - Y = -d.
  const make=(d,extra=[])=>({cal,names:['X','Y'],teams:[{team:'X',games:[g('Y',17,16,10),g('Z',10,20,ratingFor(q(25)+d))]},{team:'Y',games:[g('X',16,17,10),g('V',20,10,25)]},...extra]});
  const run=d=>R.score(make(d),a,{light:true});
  const near=run(0.05);
  assert.deepEqual(near.map(r=>r.team),['X','Y'],'inside the band: the winner goes first');
  close(near[1].resumeScore-near[0].resumeScore,0.05,1e-6,'résumé scores keep their real gap');
  close(near[0].headToHead,R.headToHeadTransfer(0.05,W)/2,1e-6);close(near[1].headToHead,-R.headToHeadTransfer(0.05,W)/2,1e-6);
  assert.deepEqual(near[0].headToHeadPassed.map(p=>p.team),['Y']);
  const wide=run(0.3),wx=wide.find(r=>r.team==='X'),wy=wide.find(r=>r.team==='Y');
  assert.deepEqual(wide.map(r=>r.team),['Y','X'],'a wider gap only narrows');
  assert.ok(wy.score-wx.score<wy.resumeScore-wx.resumeScore-1e-6);
  const out=run(W+0.01);assert.deepEqual(out.map(r=>r.team),['Y','X']);
  for(const r of out)close(r.headToHead,0,1e-12,'outside the band: nothing');
  // A third team that played neither can be passed (the cost of a soft adjustment), and the explanation says so.
  const C={team:'C',games:[g('W',20,10,ratingFor(q(25)+0.02)),g('U',10,20,10)]};
  const withC=R.score(make(0.05,[C]),a,{light:true});
  assert.equal(withC[0].team,'X');
  const Lab=require('../viz/deserving-lab.js'),x=withC[0],c=withC.find(r=>r.team==='C');
  if(R.compare(x,c).decidedBy)assert.match(Lab.explain(x,c,a).conclusion,/through head-to-head results against other teams/);
});

test('head-to-head cycles cancel: no path dependence',()=>{
  const names=['A','B','C'],cal=R.prepare(LEAGUE,NAMES,model).cal,a={...model.assumptions,reference_uncertainty:false};
  const g=(opponent,pf,pa)=>({opponent,pf,pa,oppRating:10,venue:0,site:'neutral',date:'2026-09-01'});
  // A beat B, B beat C, C beat A: identical résumés, a perfect cycle.
  const ctx={cal,names,teams:[{team:'A',games:[g('B',2,1),g('C',1,2)]},{team:'B',games:[g('C',2,1),g('A',1,2)]},{team:'C',games:[g('A',2,1),g('B',1,2)]}]};
  const rows=R.score(ctx,a,{light:true});
  assert.deepEqual(rows.map(r=>r.rank),[1,1,1],'a perfect cycle is a tie');
  for(const r of rows)close(r.headToHead,0,1e-12);
  const rev={...ctx,teams:[...ctx.teams].reverse()};
  assert.deepEqual(R.score(rev,a,{light:true}).map(r=>[r.team,r.rank]),rows.map(r=>[r.team,r.rank]),'order of teams does not matter');
});

test('early-season prior: decays to zero, and changes nothing once it has',()=>{
  const prior={ratings:{A:10,B:-5},regression:1,decay_weeks:5};
  assert.equal(R.priorWeight(prior,1),1);close(R.priorWeight(prior,3),0.6,1e-12);assert.equal(R.priorWeight(prior,6),0);assert.equal(R.priorWeight(prior,12),0);
  assert.equal(R.priorWeight(null,1),0);assert.equal(R.priorWeight(prior,NaN),0);
  const wk=(games,w)=>games.map(x=>({...x,w}));
  const early=wk(LEAGUE,2),late=wk(LEAGUE,8);
  const withP={...model,prior},noP={...model,prior:null};
  const e1=R.prepare(early,NAMES,withP),e0=R.prepare(early,NAMES,noP),l1=R.prepare(late,NAMES,withP),l0=R.prepare(late,NAMES,noP);
  assert.equal(e1.week,2);close(e1.priorWeight,0.8,1e-12);
  const opp=(c,t,o)=>c.teams.find(x=>x.team===t).games.find(x=>x.opponent===o).oppRating;
  assert.ok(opp(e1,'B','A')>opp(e0,'B','A'),'a strong prior-season team looks stronger early');
  l1.teams.forEach((t,i)=>t.games.forEach((x,j)=>close(x.oppRating,l0.teams[i].games[j].oppRating,1e-12)));
  // Production carries last season's ratings and a fitted decay.
  assert.ok(model.prior&&model.prior.decay_weeks>0&&model.prior.decay_weeks<=5&&Object.keys(model.prior.ratings).length>100);
});

test('quality wins and losses use the score\'s own opponent strengths',()=>{
  const rows=rank(LEAGUE,NAMES),cal=R.prepare(LEAGUE,NAMES,model).cal,top=R.referenceRating(cal,model.assumptions);
  for(const r of rows){
    const wins=r.ledger.filter(g=>g.result==='W'&&g.oppRating-cal.home_field*g.venue>=top).length;
    assert.equal(r.quality.playoffWins,wins);assert.ok(r.quality.top25Wins>=r.quality.playoffWins);
    assert.equal(r.quality.badLosses,r.ledger.filter(g=>g.result==='L'&&g.tier==='below-average').length);
  }
});

test('the head-to-head explanation names the game and the closeness window',()=>{
  const Lab=require('../viz/deserving-lab.js');
  const cal=R.prepare(LEAGUE,NAMES,model).cal,a={...model.assumptions,reference_uncertainty:false};
  const ref=R.referenceRating(cal,a),q=r=>R.normalCdf((ref-r)/cal.win_sigma);
  const ratingFor=target=>{let lo=-60,hi=60;for(let k=0;k<200;k++){const m=(lo+hi)/2;if(q(m)<target)hi=m;else lo=m;}return (lo+hi)/2;};
  const g=(opponent,pf,pa,oppRating)=>({opponent,pf,pa,oppRating,venue:0,site:'neutral',date:'2026-10-03'});
  const ctx={cal,names:['X','Y'],teams:[{team:'X',games:[g('Y',17,16,10),g('Z',10,20,ratingFor(q(25)+0.05))]},{team:'Y',games:[g('X',16,17,10),g('V',20,10,25)]}]};
  const rows=R.score(ctx,a),x=rows.find(r=>r.team==='X'),y=rows.find(r=>r.team==='Y');
  const e=Lab.explain(x,y,a);
  assert.match(e.conclusion,/X ranks ahead on head-to-head/);assert.match(e.headToHead,/X won 17–16/);assert.match(e.headToHead,/inside the/);assert.match(e.headToHead,/closer/);
  close(e.rows.find(r=>r.total).value,x.score-y.score,1e-12,'table still reconciles to the score gap');
});
