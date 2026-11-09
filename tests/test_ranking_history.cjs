const assert=require('node:assert/strict'),{movement}=require('../viz/ranking-history.js');
const snapshot=(label,week,teams)=>({label,week,teams:teams.map(([team,rank,power])=>({team,rank,power}))});
const ratings={history:[snapshot('Preseason',0,[['A',3,.3],['B',1,.7]]),snapshot('Week 1',1,[['A',2,.4],['B',1,.7]]),snapshot('Week 2 to date',2,[['A',1,.8],['B',3,.5]])]};
assert.equal(movement(ratings).rows[0].change,1);assert.equal(movement(ratings,'season').rows[0].change,2);
assert.deepEqual(movement({history:[ratings.history[0]]}).rows,[]);
assert.equal(movement({history:ratings.history.slice(1)},'season').rows.length,0);
// Most Deserving has its own suite: tests/test_most_deserving.cjs
console.log('Ranking history passed: weekly and season movement, missing baselines');
