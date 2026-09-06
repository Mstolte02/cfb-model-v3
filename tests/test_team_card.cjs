const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {data, mascot} = require('../viz/team-card.js');
test('away results complement pregame probabilities and reverse rating changes', () => {
  const ratings={teams:[{team:'B',rank:2,power:.6}],history:[
    {label:'Preseason',week:0,teams:[{team:'B',rank:5,power:.4}]},
    {label:'Week 1',week:1,teams:[{team:'B',rank:2,power:.6}]}],
    game_history:[{home:'A',away:'B',week:1,p_home:.8,home_rating_delta:-.25}]};
  const c=data('B',ratings,[{h:'A',a:'B',w:1,f:1,hp:14,ap:21},{h:'B',a:'C',w:2,hp:7,ap:0}]);
  assert.equal(c.rankChange,3);assert.equal(c.wins,1);assert.equal(c.games.length,1);
  assert.equal(c.games[0].scored,21);assert.equal(c.games[0].delta,.25);
  assert.ok(Math.abs(c.games[0].probability-.2)<1e-10);
});
test('unrated opponents and absent baseline do not invent forecasts or rank movement',()=>{
  const c=data('A',{teams:[{team:'A',rank:3,power:.7}]},[{h:'A',a:'FCS',w:1,f:1,hp:0,ap:3}]);
  assert.equal(c.rankChange,null);assert.equal(c.games[0].probability,null);
  assert.equal(c.games[0].delta,null);assert.equal(c.losses,1);
});
test('all published teams resolve to valid art and finite snapshots',()=>{
  const teams=require('../viz/data/teams.json'),ratings=require('../viz/data/ratings.json');
  const schedule=require('../viz/data/schedule.json');
  for(const r of ratings.teams){
    const m=mascot(r.team,teams[r.team]||{});assert.ok(m.sheet>=0&&m.sheet<3&&m.cell>=0&&m.cell<16,r.team);
    const c=data(r.team,ratings,schedule);assert.ok(Number.isFinite(c.last.power),r.team);
  }
  assert.equal(mascot('Michigan',teams.Michigan).family,'wolverine');
  assert.equal(mascot('Oregon',teams.Oregon).family,'duck');
});
test('ranking crests keep team-colour tiles and use sourced white assets',()=>{
  const teams=require('../viz/data/teams.json');
  let white=0;
  for(const [team,m] of Object.entries(teams)){
    assert.equal(m.crest.plate,'color',team);
    assert.ok(fs.existsSync(path.join(__dirname,'..','viz',m.crest.mark)),team);
    assert.equal(m.crest.white,false,team);
    if(m.crest.mark.startsWith('logos-white/')) white++;
  }
  assert.equal(white,47);
});
