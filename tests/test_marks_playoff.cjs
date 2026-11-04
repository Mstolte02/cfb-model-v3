const assert=require('node:assert/strict');
const {selectField,comparison}=require('../viz/marks-playoff.js');
const {deserving}=require('../viz/ranking-history.js');
const rows=Array.from({length:30},(_,i)=>({team:`T${i+1}`,rank:i+1,score:100-i}));
const meta=Object.fromEntries(rows.map(r=>[r.team,{conference:'SEC'}]));
meta.T1.conference='ACC';meta.T2.conference='Big Ten';meta.T20.conference='Big 12';meta.T25.conference='Mountain West';
let f=selectField(rows,meta);
assert.equal(f.field.length,12);assert.equal(new Set(f.field.map(r=>r.team)).size,12);
assert.equal(f.field[10].team,'T20');assert.equal(f.field[11].team,'T25');
assert.deepEqual(f.field.slice(0,4).map(r=>r.team),['T1','T2','T3','T4']);
assert.equal(f.out[0].team,'T11');
// An actual conference champion overrides the résumé leader, and can displace a Top 12 team.
f=selectField(rows,meta,{SEC:'T29'});assert(f.field.some(r=>r.team==='T29'&&r.bid==='SEC champion'));assert.equal(f.field.length,12);
const nd=rows.map(r=>({...r,team:r.team==='T10'?'Notre Dame':r.team}));
assert(selectField(nd,{...meta,'Notre Dame':{conference:'FBS Independents'}}).field.some(r=>r.team==='Notre Dame'));
assert(selectField(rows,{...meta,T20:{conference:'SEC'}}).error);
assert.deepEqual(comparison(rows,{rankings:[{team:'T1',rank:3},{team:'T2',rank:1}]}).slice(0,3).map(r=>r.delta),[-2,1,null]);
// Run the actual résumé model and field against the committed live snapshot.
const schedule=require('../viz/data/schedule.json'),teams=require('../viz/data/teams.json');
const ratings=require('../viz/data/ratings.json'),model=require('../viz/data/deserving-model.json');
const live=deserving(schedule,teams,ratings.teams.map(r=>r.team),model);
const liveField=selectField(live,teams);assert.equal(liveField.field.length,12);assert.equal(liveField.error,null);
console.log('Mark playoff selection passed: P4/G6 inclusion, straight byes, champion override, Notre Dame, missing data, comparison signs and live field');
