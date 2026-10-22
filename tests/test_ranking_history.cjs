const assert=require('node:assert/strict'),{movement,deserving}=require('../viz/ranking-history.js');
const model=require('../viz/data/deserving-model.json');
const snapshot=(label,week,teams)=>({label,week,teams:teams.map(([team,rank,power])=>({team,rank,power}))});
const ratings={history:[snapshot('Preseason',0,[['A',3,.3],['B',1,.7]]),snapshot('Week 1',1,[['A',2,.4],['B',1,.7]]),snapshot('Week 2 to date',2,[['A',1,.8],['B',3,.5]])]};
assert.equal(movement(ratings).rows[0].change,1);assert.equal(movement(ratings,'season').rows[0].change,2);
assert.deepEqual(movement({history:[ratings.history[0]]}).rows,[]);
assert.equal(movement({history:ratings.history.slice(1)},'season').rows.length,0);
const game=(h,a,hp=21,ap=14)=>({h,a,hp,ap,n:true,f:true});
const rank=(games,names=['A','B','C'],meta={},power=[])=>deserving(games,meta,names,model,power);
const close=(a,b)=>assert(Math.abs(a-b)<1e-9,`${a} != ${b}`);
// Analytical two-team solution, including small-sample shrinkage.
const single=rank([game('A','B')]);close(single[0].score,62.5);close(single[1].score,37.5);
// Symmetric resumes share rank; predictive ratings and conference labels have no effect.
const cycle=[game('A','B'),game('B','C'),game('C','A')];
const equal=rank(cycle);assert(equal.every(r=>r.rank===1));assert(equal.every(r=>Math.abs(r.score-50)<1e-9));
assert.deepEqual(rank(cycle,['A','B','C'],{A:{conference:'SEC'},B:{conference:'Sun Belt'}},[{team:'C',rank:1}]),equal);
const games=[game('A','B',28,7),game('A','C',10,17),game('B','C',3,35)];
const normal=rank(games),reverse=rank(games.map(g=>({...g,hp:g.ap,ap:g.hp})));
for(const r of normal){const rev=reverse.find(v=>v.team===r.team);close(r.score+rev.score,100);close(r.score,100*(r.recordBase+r.scheduleAdjustment));}
// Neither FCS results (wins OR losses), postseason, missing scores, nor unfinished games count.
const ignored=[game('A','FCS',30,0),game('FCS','B',30,0),{...game('A','B'),f:false},{...game('A','B'),hp:null},{...game('A','B'),st:'postseason'}];
assert.deepEqual(rank([...games,...ignored]),normal);assert.deepEqual(rank([]),[]);
assert.deepEqual(rank([game('A','FCS')]),[]);
// Margins and venues cannot change the result-based score.
assert.deepEqual(rank(games.map(g=>({...g,hp:g.hp>g.ap?100:0,ap:g.hp>g.ap?0:1,n:false}))),normal);
assert.deepEqual(rank([...games].reverse()),normal);
const tied=rank([game('A','B',7,7)]);assert(tied.every(r=>r.ties===1&&r.win_pct===.5&&r.score===50));
// Identical 1-0 records: beating the stronger opponent earns a higher score.
const strength=rank([game('A','X'),game('B','Y'),game('X','C'),game('X','D'),game('C','Y'),game('D','Y')],['A','B','X','Y','C','D']);
const a=strength.find(r=>r.team==='A'),b=strength.find(r=>r.team==='B');
assert.equal(a.win_pct,b.win_pct);assert(a.sos>b.sos);assert(a.score>b.score);
// A repeated opponent contributes once per actual game.
const repeated=rank([game('A','B'),game('A','B')]);assert.equal(repeated[0].games,2);close(repeated[0].score,100*2/3);
assert.throws(()=>deserving(games,{},['A','B','C'],{weights:{}}),/Unsupported/);
console.log('Ranking history passed: movement, analytical solutions, SOS credit, shared ranks, conference/power/margin independence, FCS exclusion, ties and repeat opponents');
