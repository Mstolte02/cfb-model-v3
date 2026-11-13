const assert=require('node:assert/strict');
global.PlayoffPaths=require('../viz/playoff-paths.js');
const {likelyPath}=require('../viz/simulation-studio.js');
const render=path=>likelyPath('A & B',{game_impacts:{paths:{'A & B':path}}},{});
assert(render(null).includes('after the next season simulation'));
assert(render({status:'no_qualifying_draws'}).includes('does not prove elimination'));
assert(render({status:'no_material_games'}).includes('No remaining game clears'));
const html=render({status:'available',qualifying_samples:20,min_samples:30,baseline:.2,playoff:.8,pattern_probability:.1,
  share_of_qualifying:.4,tied_paths:2,steps:[{week:12,home:'A & B',away:'<Opponent>',
    winner:'A & B',own_game:true,result_probability:.6,swing:.2,playoff:.8,samples:12,supported:false}]});
assert(html.includes('A &amp; B beats &lt;Opponent&gt;'));
assert(html.includes('20.0% → 80.0%'));
assert(html.includes('12 simulations · small sample'));
assert(html.includes('2 paths tie'));
assert(html.includes('small qualifying sample'));
assert(html.includes('Unlisted games and conference titles remain simulated'));
console.log('Likely path rendering: conditional odds, escaping, ties, uncertainty and empty states passed');
