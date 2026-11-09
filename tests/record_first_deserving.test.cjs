/* Run with: node tests/record_first_deserving.test.cjs */
'use strict';
const assert=require('node:assert/strict');
const {deserving}=require('../viz/ranking-history.js');
const game=(h,a,hp,ap)=>({h,a,hp,ap,f:true,st:'regular'});
const names=['A','B','C','D','E','F'];
const schedule=[
  game('A','B',24,17),game('A','C',28,10),game('A','D',21,7),
  game('E','B',31,10),game('E','C',17,14),game('D','E',24,21),
  game('F','A',3,14),game('F','B',13,20)
];
function rank(games){
  return deserving(games,{},names,{method:'record-first'});
}
const rows=rank(schedule),by=new Map(rows.map(r=>[r.team,r]));
assert.equal(rows.length,6);
assert.ok(rows.every(r=>Number.isFinite(r.score)&&Number.isFinite(r.sos)));
assert.ok(rows.every((r,i)=>i===0||rows[i-1].score>=r.score));
assert.ok(rows.every(r=>Math.abs(r.score-(-r.losses+r.qualityAdjustment))<1e-10));
assert.ok(rows.every(r=>Math.abs(r.qualityAdjustment-r.scheduleAdjustment-r.marginAdjustment-r.winsAdjustment)<1e-10));
// Order of input games must never determine the result.
const compact=xs=>xs.map(r=>({team:r.team,rank:r.rank,score:+r.score.toFixed(10)}));
assert.deepEqual(compact(rank([...schedule].reverse())),compact(rows));
// Swapping home/away while preserving each team's points changes nothing.
assert.deepEqual(compact(rank(schedule.map(g=>({...g,h:g.a,a:g.h,hp:g.ap,ap:g.hp})))),compact(rows));
// A one-loss disadvantage is overridden exactly when the résumé-quality gap clears one unit.
for(const a of rows)for(const b of rows) {
  const delta=a.score-b.score;
  const expected=-(a.losses-b.losses)+(a.qualityAdjustment-b.qualityAdjustment);
  assert.ok(Math.abs(delta-expected)<1e-10);
}
// A 100-point blowout is not worth more than a 28-point win.
const capA=rank([...schedule,game('C','F',28,0)]);
const capB=rank([...schedule,game('C','F',100,0)]);
assert.deepEqual(compact(capA),compact(capB));
assert.deepEqual(deserving([],{},names,{method:'record-first'}),[]);
assert.throws(()=>deserving(schedule,{},names,{method:'bad'}),/Unsupported/);
console.log('Record-first résumé ranking invariants passed');
