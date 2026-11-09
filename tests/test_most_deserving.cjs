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
    close(r.score,r.lossPenalty+r.schedule+r.performance,1e-12,'score');
    close(r.schedule,r.scheduleFromWins+r.scheduleFromLosses,1e-12,'schedule');
    close(r.score,r.ledger.reduce((s,g)=>s+g.total,0),1e-12,'ledger');
    close(r.referenceWins+r.schedule,r.games,1e-12,'reference wins');
    close(r.lossPenalty,-model.assumptions.loss_aversion*(r.losses+r.ties/2),1e-12,'loss penalty');
    close(r.rarity,-Math.log10(r.matchProbability),1e-12,'rarity');
  }
  // The head-to-head explanation reconciles with the ranking for every pair.
  for(const a of rows)for(const b of rows){
    const c=R.compare(a,b);
    close(c.net,a.score-b.score,1e-12);close(c.losses+c.schedule+c.performance,c.net,1e-12);
    // Conditional override, stated in loss units: A ranks above B with more losses
    // only when its schedule plus performance edge exceeds the extra losses.
    if(a.losses>b.losses&&a.rank<b.rank)assert.ok(c.schedule+c.performance>-c.losses);
  }
});

test('ranking is transitive, deterministic and independent of game order',()=>{
  const rows=rank(LEAGUE,NAMES);
  rows.forEach((r,i)=>{if(i)assert.ok(rows[i-1].score>=r.score);});
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

test('a better result never lowers a team\'s score or rarity (live schedule)',()=>{
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
    assert.ok(after.score>=before.score-1e-12,`${team} score fell`);
    assert.ok(after.rarity>=before.rarity-1e-12,`${team} rarity fell`);
    if(margin<0)assert.ok(after.score-before.score>=0.5,'a loss turned into a win must be worth a lot');
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
  close(a.score,b.score,1e-12);
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
  const lo=R.score(ctx,{...model.assumptions,benchmark_rank:25}),hi=R.score(ctx,{...model.assumptions,benchmark_rank:4});
  assert.ok(new Set(hi.map(r=>r.referenceRating)).size===1);
  for(const r of hi)assert.ok(r.schedule<lo.find(x=>x.team===r.team).schedule+1e-12);
});

test('head-to-head explanation reconciles and states the override in loss units',()=>{
  const Lab=require('../viz/deserving-lab.js');
  const schedule=require('../viz/data/schedule.json'),names=require('../viz/data/ratings.json').teams.map(t=>t.team);
  const rows=R.deserving(schedule,{},names,model),by=new Map(rows.map(r=>[r.team,r]));
  for(const [i,j] of [[0,1],[2,3],[0,10],[5,20]]){
    const a=rows[i],b=rows[j],e=Lab.explain(a,b,model.assumptions);
    const total=e.rows.find(r=>r.total).value,parts=e.rows.filter(r=>!r.total).reduce((s,r)=>s+r.value,0);
    close(total,a.score-b.score,1e-12);close(parts,total,1e-12);
    assert.match(e.rarity,/playoff-caliber reference team/);
  }
  const [ta,tb]=Lab.defaultPair(rows),a=by.get(ta),b=by.get(tb);
  if(a.losses>b.losses){assert.ok(a.rank<b.rank);assert.match(Lab.explain(a,b,model.assumptions).conclusion,/despite \d+ more loss/);}
  assert.equal(Lab.band(1).key,'secure');assert.equal(Lab.band(.5).key,'bubble');assert.equal(Lab.band(0).key,'out');
  assert.equal(Lab.signed(0.5),'+0.50');assert.equal(Lab.signed(-1),'−1.00');assert.equal(Lab.signed(0),'0.00');
});
