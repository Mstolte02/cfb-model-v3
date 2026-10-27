const assert = require('node:assert/strict');
const {percent, modalRecords, lossColumns, rankGames, rootingGuide} = require('../viz/playoff-paths.js');
// Equal-loss columns are absolute loss totals, even for teams on different records.
assert.deepEqual(lossColumns([[{losses:1},{losses:2}], [{losses:3},{losses:4}]]), [0,1,2,3,4]);
assert.deepEqual(lossColumns([]), []);
// A mode is not the rounded expected record; tied modes are all retained.
const cells = [{wins:12,losses:0,samples:30},{wins:11,losses:1,samples:30},{wins:10,losses:2,samples:1}];
assert.deepEqual(modalRecords(cells).map(c=>c.wins), [12,11]);
assert.deepEqual(modalRecords([{wins:12,losses:0,samples:0}]), []);
assert.equal(percent(0), '0%'); assert.equal(percent(null), '—');
assert.equal(percent(.00001), '<0.1%'); assert.equal(percent(.99999), '>99.9%');
const game=(id, week, home, away, swing, se=.005, supported=true)=>({id,week,home,away,supported,combined_swing:Math.abs(swing),effects:{A:{swing,standard_error:se,home_win:.5+swing/2,away_win:.5-swing/2}}});
const games=[game('negative',9,'B','C',-.2),game('positive',10,'D','E',.1),game('own',9,'A','F',.8),game('negligible',9,'G','H',.001),game('noise',9,'I','J',.06,.08),game('thin',9,'K','L',.5,.001,false),game('repeated-comparison-noise',9,'M','N',.031,.012)];
const impacts={rooting_teams:['A'],rooting_min_swing:.02,rooting_z:3.8,games};
const all=rootingGuide(impacts,'A','all');
assert.deepEqual(all.map(r=>r.winner),['C','D']);
assert.equal(all[0].preferred,.6);assert.equal(all[0].opposite,.4);assert.equal(all[0].swing,.2);
assert.deepEqual(rootingGuide(impacts,'A','9').map(r=>r.winner),['C']);
assert.deepEqual(rootingGuide(impacts,'BelowThreshold','all'),[]);
assert.deepEqual(rootingGuide({...impacts,games:[{...games[0],effects:{A:{swing:null,home_win:null,away_win:1}}}]},'A','all'),[]);
// Small-sample games never outrank well-supported games on a noisy extreme.
const ranked=rankGames([games[5],games[1],games[0],{...games[5],id:'unknown',combined_swing:null}]);
assert.deepEqual(ranked.map(g=>g.id),['negative','positive','thin','unknown']);
assert.equal(games[0].id,'negative'); // selectors do not mutate inputs
console.log('Playoff paths passed: equal-loss alignment, true modal records and ties, null odds, rooting direction, eligibility, weeks, sample uncertainty and leverage ordering');
