const {test} = require('node:test');
const assert = require('node:assert/strict');
const O = require('../viz/team-overview.js');

test('historical percentile is an empirical midrank', () => {
  const v = [.1, .2, .3, .4, .5];
  assert.equal(O.percentile(v, .3), 50);
  assert.equal(O.percentile(v, .05), 0);
  assert.equal(O.percentile(v, .9), 100);
  assert.equal(O.percentile([.2, .2, .2, .2], .2), 50);
  assert.equal(O.percentile([], .2), null);
});

test('histogram bin assignment uses fixed half-open bins', () => {
  const bins = [{lo: 0, hi: .25}, {lo: .25, hi: .5}, {lo: .5, hi: .75}];
  assert.equal(O.binIndex(bins, 0), 0);
  assert.equal(O.binIndex(bins, .25), 1);
  assert.equal(O.binIndex(bins, .74), 2);
  assert.equal(O.binIndex(bins, .75), 2);
  assert.equal(O.binIndex(bins, -1), 0);
});

test('injured players are kept out of trending down and listed in the injury report', () => {
  const ps = [
    {n: 'Hurt', g: 'QB', dwin: -.4, win: .2, base: .4, inj: 'questionable'},
    {n: 'Out', g: 'WR', dwin: -.3, win: 0, base: .3, out: true},
    {n: 'Slump', g: 'CB', dwin: -.05, win: .1},
    {n: 'Riser', g: 'LB', dwin: .08, win: .3},
  ];
  const m = O.movers(ps);
  assert.deepEqual(m.down.map(p => p.n), ['Slump']);
  assert.deepEqual(m.up.map(p => p.n), ['Riser']);
  const r = O.injuryReport(ps);
  assert.deepEqual(r.map(p => p.n), ['Out', 'Hurt']);
  assert.ok(Math.abs(r[1].impact + .2) < 1e-12);
});

test('empty injury state', () => {
  assert.deepEqual(O.injuryReport([{n: 'A', dwin: .1}]), []);
  assert.deepEqual(O.injuryReport(undefined), []);
});

test('every unit the roster has appears in unit rankings, ordered by rank', () => {
  const byGroup = {QB: 1, RB: .5, WR: .9, OT: .3, IOL: .2, TE: .1,
                   DT: .4, EDGE: .6, LB: .3, CB: .7, SAF: .5};
  const league = {};
  for (const g of Object.keys(byGroup)) league[g] = {n: 138, rank: {X: 1 + Object.keys(league).length * 7}};
  const u = O.unitRankings(byGroup, league, 'X');
  assert.equal(u.length, 11);
  assert.deepEqual(u.map(x => x.group).sort(), Object.keys(byGroup).sort());
  for (let i = 1; i < u.length; i++) assert.ok(u[i - 1].rank <= u[i].rank);
  assert.equal(u[0].tier, 'elite');
});

test('rating journey is chronological with pre and post ratings', () => {
  const card = {
    points: [{week: 0, power: .5}, {week: 1, power: .55}, {week: 3, power: .6}],
    games: [{week: 3, result: 'W'}, {week: 1, result: 'L'}],
  };
  const j = O.journey(card);
  assert.deepEqual(j.map(g => g.week), [1, 3]);
  assert.equal(j[0].pre, .5); assert.equal(j[0].post, .55);
  assert.equal(j[1].pre, .55); assert.equal(j[1].post, .6);
});

test('comparables use each team-season once, at its final rating', () => {
  const obs = [[2024, 3, 'A', .72], [2024, 15, 'A', .60], [2023, 15, 'B', .70], [2026, 5, 'X', .72]];
  const c = O.comparables(obs, .72, 5, {season: 2026, team: 'X'});
  assert.deepEqual(c.map(x => x.team), ['B', 'A']);
  assert.equal(c[1].week, 15); assert.equal(c[1].power, .60);
});
