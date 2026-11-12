const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../viz/app.js'), 'utf8');
const bet = source.slice(source.indexOf('  function betToPlace('),
  source.indexOf('  let weeklyMarket'));
function board(now, lockedResults = {bets: []}) {
  const context = vm.createContext({Date: class extends Date {
    static now() { return Date.parse(now); }
  }, lockedResults, BET_RULES: {spread: {minGap: 3}},
  picksOtherSide: () => false, esc: x => x, abbr: x => x});
  vm.runInContext(bet, context);
  return context;
}
const game = {id: 1, week: 6, start: '2026-10-11T00:00Z', home: 'A', away: 'B',
  gap: 4, marketValue: -3};
test('locked bets remain visible throughout the week and after kickoff without quote checks', () => {
  for (const day of ['06', '10', '12']) {
    assert.equal(board(`2026-10-${day}T22:45Z`).betToPlace(game, 'spread'), 'A -3.0');
  }
});
test('lock visibility preserves exclusions and the original betting gate', () => {
  const ctx = board('2026-10-06T22:54:02Z');
  assert.equal(ctx.betToPlace({...game, bettingExcluded: true}, 'spread'), null);
  assert.equal(ctx.betToPlace({...game, gap: 1}, 'spread'), null);
  assert.equal(ctx.betToPlace({...game, marketValue: null}, 'spread'), null);
});
test('historical board cannot fall back to current ratings', () => {
  const fn = source.slice(source.indexOf('  function boardPredict('), source.indexOf('  function marketRows('));
  const ctx = vm.createContext({predict: () => ({pA: .99, margin: 40, total: 50})});
  vm.runInContext(fn,ctx);
  const g = {home:'A',away:'B',week:7,start:'2026-01-01T00:00:00Z'};
  assert.equal(ctx.boardPredict(g),null);
  const frozen = ctx.boardPredict({...g,modelSnapshot:{homeWinProbability:.4,homeMargin:-3}});
  assert.equal(frozen.pA,.4);
  assert.equal(frozen.margin,-3);
});

test('restored weeks have genuine pregame model and DraftKings observations', () => {
  const read = name => JSON.parse(fs.readFileSync(require.resolve('../viz/data/'+name+'.json'),'utf8'));
  const teams = new Set(read('ratings').teams.map(t=>t.team));
  const games = read('odds').weekly.filter(g=>[7,8].includes(g.week) && teams.has(g.home) && teams.has(g.away));
  assert(games.length > 100);
  for (const g of games) {
    assert(Date.parse(g.modelSnapshot.recordedAt) < Date.parse(g.start), `late model ${g.id}`);
    assert(Date.parse(g.books.DraftKings.retrieved_at) < Date.parse(g.start), `late line ${g.id}`);
  }
});

for (const week of [1,2,7,8]) {
  for (const flip of [true,false]) {
    test(`gap exception applies in week ${week}, opposite winner ${flip}`, () => {
      const ctx = vm.createContext({BET_RULES:{spread:{minGap:8},moneyline:{minGap:.2,minModelP:.5}},
        picksOtherSide:g=>g.flip,esc:x=>x,abbr:x=>x,americanOdds:String});
      vm.runInContext(bet,ctx);
      const g={...game,week,home:'A',away:'B',flip};
      for (const gap of [4,-4,7.999,-7.999]) assert.equal(Boolean(ctx.betToPlace({...g,gap},'spread')),flip);
      for (const gap of [8,-8]) assert(ctx.betToPlace({...g,gap},'spread'));
      for (const gap of [.1,-.1,.1999,-.1999]) assert.equal(Boolean(ctx.betToPlace({...g,gap,modelValue:.6},'moneyline')),flip);
      for (const gap of [.2,-.2]) assert(ctx.betToPlace({...g,gap,modelValue:.6},'moneyline'));
      assert.equal(ctx.betToPlace({...g,gap:.2,modelValue:.5},'moneyline'),null);
    });
  }
}

const disagreement = source.slice(source.indexOf('  function picksOtherSide('),
  source.indexOf('  let weeklyMarket'));
function actualRules() {
  const ctx = vm.createContext({BET_RULES:{spread:{minGap:8},moneyline:{minGap:.2,minModelP:.5}},
    esc:x=>x,abbr:x=>x,americanOdds:String});
  vm.runInContext(disagreement,ctx);
  return ctx;
}
test('wrong team favored qualifies either side for spread and moneyline below the gap', () => {
  const ctx = actualRules();
  for (const home of [true,false]) {
    const sign = home ? 1 : -1;
    const side = home ? 'A' : 'B';
    const spread = {...game,gap:sign*3,marketValue:sign*2,
      modelSpread:-sign,consensus:sign*2};
    assert.equal(ctx.betToPlace(spread,'spread'),`${side} +2.0`);
    const ml = {...game,gap:sign*.07,marketValue:120,modelValue:.52,
      r:{pA:home ? .52 : .48},marketHomeP:home ? .45 : .55};
    assert.equal(ctx.betToPlace(ml,'moneyline'),`${side} 120`);
    for (const g of [spread,ml]) {
      const market = g.r ? 'moneyline' : 'spread';
      assert.equal(ctx.betToPlace({...g,bettingExcluded:true},market),null);
      assert.equal(ctx.betToPlace({...g,marketValue:null},market),null);
      assert.equal(ctx.betToPlace({...g,gap:null},market),null);
    }
    for (const p of [.5,.49])
      assert.equal(ctx.betToPlace({...ml,modelValue:p},'moneyline'),null);
  }
});
test('a pickem or exactly 50% is not a disagreement and cannot waive the gap', () => {
  const ctx = actualRules();
  for (const [modelSpread,consensus] of [[0,2],[-2,0],[-2,-1],[2,1]])
    assert.equal(ctx.betToPlace({...game,gap:1,modelSpread,consensus},'spread'),null);
  for (const [pA,marketHomeP] of [[.5,.45],[.55,.5],[.55,.52],[.45,.42]])
    assert.equal(ctx.betToPlace({...game,gap:.03,modelValue:.55,r:{pA},marketHomeP},'moneyline'),null);
});
