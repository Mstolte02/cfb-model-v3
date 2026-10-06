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
test('week-one history still comes exclusively from its authoritative ledger', () => {
  const historical = {...game, week: 1};
  assert.equal(board('2026-10-06T22:54Z').betToPlace(historical, 'spread'), null);
  const ctx = board('2026-10-06T22:54Z', {bets: [
    {game_id: 1, market: 'spread', bet: 'Original +7.0'}]});
  assert.equal(ctx.betToPlace(historical, 'spread'), 'Original +7.0');
});
