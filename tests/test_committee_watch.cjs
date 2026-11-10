/* Committee Watch: the browser port of the committee proxy must match the R simulation,
   and the projection must start from the committee's own order. */
'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const C=require('../viz/committee-watch.js');
const playoff=require('../viz/data/playoff_current.json'),schedule=require('../viz/data/schedule.json'),poll=require('../viz/data/cfp_rankings.json');
const list=Array.isArray(playoff.teams)?playoff.teams:Object.values(playoff.teams);
const teams=list.map(t=>({team:t.team,conference:t.conference,rating:t.rating}));

test('matches the R committee proxy and anchor (scripts/simulate_playoff.R)',()=>{
  const res=C.project({schedule,teams,poll,options:{weights:playoff.selection_weights}}),by=new Map(res.rows.map(r=>[r.team,r]));
  for(const t of list){if(!t.resume_score_parts)continue;const r=by.get(t.team);
    // Ratings are exported to 3 decimals, so allow a little rounding.
    assert.ok(Math.abs(r.static-t.resume_score_parts.staticScore)<0.005,`${t.team} static`);
    assert.ok(Math.abs(r.anchor-t.resume_score_parts.baseline)<0.005,`${t.team} anchor`);}
  if(playoff.committee_anchor?.rank_spacing)assert.ok(Math.abs(res.spacing-playoff.committee_anchor.rank_spacing)<1e-3);
});

test('with no new results the projection is the committee order',()=>{
  const cut=String(poll.games_through).slice(0,10),old=schedule.map(g=>String(g.d).slice(0,10)>cut?{...g,f:0,hp:null,ap:null}:g);
  const res=C.project({schedule:old,teams,poll,options:{weights:playoff.selection_weights}});
  assert.deepEqual(res.rows.slice(0,25).map(r=>r.team),poll.rankings.map(r=>r.team));
  assert.equal(res.newResults,false);
});

test('a new loss drops a team and a new win never drops it',()=>{
  const cut=String(poll.games_through).slice(0,10),base=schedule.map(g=>String(g.d).slice(0,10)>cut?{...g,f:0,hp:null,ap:null}:g);
  const team=poll.rankings[0].team,d='2099-01-01';
  const lose=[...base,{h:team,a:'Nobody FCS',hp:0,ap:7,f:1,d}],win=[...base,{h:team,a:'Nobody FCS',hp:7,ap:0,f:1,d}];
  const o={weights:playoff.selection_weights};
  const L=C.project({schedule:lose,teams,poll,options:o}).rows.find(r=>r.team===team),W=C.project({schedule:win,teams,poll,options:o}).rows.find(r=>r.team===team);
  assert.ok(L.projected>1);assert.equal(W.projected,1);assert.equal(L.results[0].result,'L');
});

test('no poll, no projection',()=>{assert.ok(C.project({schedule,teams,poll:null}).error);});
