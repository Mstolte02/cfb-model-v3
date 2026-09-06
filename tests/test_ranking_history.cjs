const assert=require('node:assert/strict'),{movement,deserving}=require('../viz/ranking-history.js');
const model=require('../viz/data/deserving-model.json');
const snapshot=(label,week,teams)=>({label,week,teams:teams.map(([team,rank,power])=>({team,rank,power}))});
const ratings={history:[snapshot('Preseason',0,[['A',3,.3],['B',1,.7]]),snapshot('Week 1',1,[['A',2,.4],['B',1,.7]]),snapshot('Week 2 to date',2,[['A',1,.8],['B',3,.5]])]};
assert.equal(movement(ratings).rows[0].change,1);assert.equal(movement(ratings,'season').rows[0].change,2);
assert.deepEqual(movement({history:[ratings.history[0]]}).rows,[]);
assert.equal(movement({history:ratings.history.slice(1)},'season').rows.length,0);
const game=(h,a,hp,ap)=>({h,a,hp,ap,n:true,f:true}),names=['A','B','C'],meta={};
const cycle=[game('A','B',21,14),game('B','C',21,14),game('C','A',21,14)];
const power=[{team:'C',rank:1},{team:'A',rank:2},{team:'B',rank:3}];
const rows=deserving(cycle,meta,names,model,power);assert(rows.every(r=>Math.abs(r.rating_z)<1e-9));
assert.deepEqual(rows.map(r=>[r.team,r.rank]),[['C',1],['A',2],['B',3]]);
const games=[game('A','B',28,7),game('A','C',10,17),game('B','C',3,35)];
const normal=deserving(games,meta,names,model),reverse=deserving(games.map(g=>({...g,hp:g.ap,ap:g.hp})),meta,names,model);
for(const r of normal){const rev=reverse.find(v=>v.team===r.team);assert(Math.abs(r.second_pass_lift+rev.second_pass_lift)<1e-9);assert(Math.abs(r.rating_z+rev.rating_z)<1e-9);}
assert.deepEqual(deserving([...games,{...game('A','B',99,0),f:false},{...game('A','B',99,0),st:'postseason'}],meta,names,model),normal);
assert.deepEqual(deserving([],meta,names,model),[]);
const rated=deserving([game('A','B',14,7),game('A','FCS',30,0)],meta,names,model);assert.equal(rated.find(r=>r.team==='A').wins,2);assert(!rated.some(r=>r.team==='C'));
console.log('Ranking history: movement, missing baselines, power tiebreakers, symmetry, incomplete games, postseason and FCS invariants passed');
