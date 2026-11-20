/* Committee Watch: the browser port of the committee proxy must match the R simulation,
   and the projection must start from the committee's own order. */
'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const C=require('../viz/committee-watch.js');
const timeline=require('../viz/data/committee_timeline_model.json');
const backtest=require('../viz/data/committee_backtest.json');
const playoff=require('../viz/data/playoff_current.json'),schedule=require('../viz/data/schedule.json'),poll=require('../viz/data/cfp_rankings.json');
const list=Array.isArray(playoff.teams)?playoff.teams:Object.values(playoff.teams);
const teams=list.map(t=>({team:t.team,conference:t.conference,rating:t.rating}));

test('matches the legacy R committee proxy and anchor when the saved run uses it',()=>{
  if(playoff.committee_anchor?.method?.startsWith('Weekly'))return;
  // The stored R run can predate a corrected poll cutoff; compare its own snapshot.
  const snapshotPoll={...poll,games_through:playoff.committee_anchor.games_through};
  const res=C.project({schedule,teams,poll:snapshotPoll,options:{weights:playoff.selection_weights}}),by=new Map(res.rows.map(r=>[r.team,r]));
  for(const t of list){if(!t.resume_score_parts)continue;const r=by.get(t.team);
    // Ratings are exported to 3 decimals, so allow a little rounding.
    assert.ok(Math.abs(r.static-t.resume_score_parts.staticScore)<0.005,`${t.team} static`);
    assert.ok(Math.abs(r.anchor-t.resume_score_parts.baseline)<0.005,`${t.team} anchor`);}
  if(playoff.committee_anchor?.rank_spacing)assert.ok(Math.abs(res.spacing-playoff.committee_anchor.rank_spacing)<1e-3);
});

test('with no new results the projection is the committee order',()=>{
  const cut=String(poll.games_through).slice(0,10),old=schedule.map(g=>String(g.d).slice(0,10)>cut?{...g,f:0,hp:null,ap:null}:g);
  const res=C.project({schedule:old,teams,poll,model:timeline});
  assert.deepEqual(res.rows.slice(0,25).map(r=>r.team),poll.rankings.map(r=>r.team));
  assert.equal(res.newResults,false);
});

test('a new loss drops a team and a new win never drops it',()=>{
  const cut=String(poll.games_through).slice(0,10),base=schedule.filter(g=>String(g.d).slice(0,10)<=cut);
  const team=poll.rankings[0].team,d=C.weeklyWindow(new Date(cut+'T12:00:00Z').getTime()+7*86400000).today;
  const projectionDate=d+'T12:00:00Z';
  const lose=[...base,{h:team,a:'Nobody FCS',hp:0,ap:7,f:1,d}],win=[...base,{h:team,a:'Nobody FCS',hp:7,ap:0,f:1,d}];
  const o={weights:playoff.committee_anchor?.method?.startsWith('Weekly')?undefined:playoff.selection_weights};
  const L=C.project({schedule:lose,teams,poll,options:o,now:projectionDate}).rows.find(r=>r.team===team),W=C.project({schedule:win,teams,poll,options:o,now:projectionDate}).rows.find(r=>r.team===team);
  assert.ok(L.projected>1);assert.equal(W.projected,1);assert.equal(L.results[0].result,'L');
});

test('no poll, no projection',()=>{assert.ok(C.project({schedule,teams,poll:null}).error);});

// Fixed regression season: live data refreshes must not move these calendar fixtures.
const fixtureTeams=[['Notre Dame',10],['Miami',9],['Boston College',0],['Georgia',20]].map(([team,rating])=>({team,rating,conference:'ACC'}));
const fixturePoll={released:'2026-11-10',games_through:'2026-11-09',rankings:fixtureTeams.map((t,i)=>({team:t.team,rank:i+1}))};
const fixtureSchedule=[
  {h:'Notre Dame',a:'Boston College',d:'2026-11-01',hp:40,ap:10,f:1},
  {h:'Notre Dame',a:'Miami',d:'2026-11-08',hp:31,ap:24,f:1},
  {h:'Notre Dame',a:'Boston College',d:'2026-11-14',hp:41,ap:7,f:1}
];
const runAt=(now,extra={})=>C.project({schedule:fixtureSchedule,teams:fixtureTeams,poll:fixturePoll,now,...extra});
test('Sunday and Monday include only Notre Dame’s Boston College result',()=>{
  for(const now of ['2026-11-15T12:00:00Z','2026-11-16T12:00:00Z']){
    const res=runAt(now),nd=res.rows.find(r=>r.team==='Notre Dame');
    assert.equal(res.status,'projection');
    assert.deepEqual(nd.results.map(g=>g.opponent),['Boston College']);
    assert.equal(nd.recordThen,'2–0');assert.equal(nd.recordNow,'3–0');
  }
});
test('Sunday projects through the fitted weekly transition model',()=>{
  const res=runAt('2026-11-15T12:00:00Z',{model:timeline});
  assert.equal(res.status,'projection');
  assert.equal(new Set(res.rows.map(r=>r.score)).size,fixtureTeams.length);
  assert.ok(res.rows.find(r=>r.team==='Notre Dame').score>res.rows.find(r=>r.team==='Boston College').score);
  assert.equal(res.rows.find(r=>r.team==='Notre Dame').recordNow,'3–0');
});
test('historical accuracy includes held-out rankings and cutoff misses',()=>{
  assert.equal(backtest.model_version,timeline.version);
  assert.ok(backtest.weeks.length>=30);
  for(const w of backtest.weeks){
    assert.ok(w.mae>=0&&w.top12_overlap>=0&&w.top12_overlap<=12);
    assert.equal(w.rows.filter(r=>r.actual<=25).length,25);
    assert.equal(new Set(w.rows.map(r=>r.team)).size,w.rows.length);
  }
  const final=backtest.weeks.find(w=>w.season===2025&&w.week===16);
  assert.equal(final.rows.find(r=>r.actual===1).team,'Indiana');
});
test('conference title weekend receives the separate title-game feature',()=>{
  assert.equal(C.isTitle({d:'2026-12-05',v:'Neutral Site'}),true);
  assert.equal(C.isTitle({d:'2026-11-28',v:'Campus'}),false);
  assert.equal(C.isTitle({d:'2026-12-12',v:'Army-Navy'}),false);
});
test('Tuesday through Saturday hold the official order, regardless of new finals',()=>{
  for(let d=17;d<=21;d++){
    const res=runAt(`2026-11-${d}T12:00:00Z`);
    assert.equal(res.status,'static');assert.equal(res.newResults,false);
    assert.deepEqual(res.rows.map(r=>r.team),fixturePoll.rankings.map(r=>r.team));
    assert.ok(res.rows.every(r=>r.results.length===0&&r.recordNow===r.recordThen));
  }
});
test('weekly rollover uses Eastern midnight, including daylight saving time',()=>{
  assert.equal(C.weeklyWindow('2026-11-15T04:59:59Z').active,false);
  assert.equal(C.weeklyWindow('2026-11-15T05:00:00Z').active,true);
  assert.equal(C.weeklyWindow('2026-11-17T04:59:59Z').active,true);
  assert.equal(C.weeklyWindow('2026-11-17T05:00:00Z').active,false);
  assert.equal(C.weeklyWindow('2026-10-25T03:59:59Z').active,false);
  assert.equal(C.weeklyWindow('2026-10-25T04:00:00Z').active,true);
});
test('stale polls never accumulate a second week of results',()=>{
  const res=runAt('2026-11-23T12:00:00Z');
  assert.equal(res.status,'stale');assert.equal(res.newResults,false);
  assert.ok(res.rows.every(r=>!r.results.length));
});
test('Sunday waits for unfinished games, then ignores future finals',()=>{
  const last=fixtureSchedule.find(g=>g.d==='2026-11-14'&&g.f);
  const pending=fixtureSchedule.map(g=>g===last?{...g,f:0}:g);
  assert.equal(runAt('2026-11-15T12:00:00Z',{schedule:pending}).status,'pending');
  const future={h:'Notre Dame',a:'Nobody FCS',d:'2026-11-22',hp:0,ap:7,f:1};
  const res=runAt('2026-11-15T12:00:00Z',{schedule:[...fixtureSchedule,future]});
  assert.deepEqual(res.rows.find(r=>r.team==='Notre Dame').results.map(g=>g.opponent),['Boston College']);
});

test('title margin changes ranking, caps at 35 and stays finite without titles',()=>{
  const p={released:'2026-12-01',games_through:'2026-11-30',rankings:poll.rankings};
  const ts=p.rankings.map(r=>({team:r.team,conference:'Big Ten',rating:1}));
  const base=ts.map(t=>({h:t.team,a:'FCS '+t.team,d:'2026-11-28',hp:21,ap:7,f:1}));
  const winner=p.rankings[7].team,loser=p.rankings[3].team;
  const m={version:1,intercept:0,pod_head_to_head_bonus:0,weights:{prior_rank:1,title_margin:-1,title_win_opponent:0,title_loss_opponent:0}};
  const run=margin=>C.project({schedule:[...base,{h:winner,a:loser,d:'2026-12-05',hp:margin+7,ap:7,f:1,game_type:'CONF_CHAMP'}],teams:ts,poll:p,model:m,now:'2026-12-06T18:00:00Z'});
  const close=run(1),big=run(35),capped=run(70);
  assert.ok(close.rows.every(r=>Number.isFinite(r.score)));
  assert.ok(close.rows.find(r=>r.team===winner).projected>close.rows.find(r=>r.team===loser).projected);
  assert.ok(big.rows.find(r=>r.team===winner).projected<big.rows.find(r=>r.team===loser).projected);
  assert.deepEqual(big.rows.map(r=>r.team),capped.rows.map(r=>r.team));
});
