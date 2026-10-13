const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const X=require('../viz/power-xray/war-research.js');
const folder=path.join(__dirname,'../viz/power-xray');
test('every historical contribution reconstructs its four-member prediction',()=>{
  const summary=JSON.parse(fs.readFileSync(path.join(folder,'war-research.json')));
  const chunks=new Map();let games=0;
  for(const meta of summary.games){
    if(!chunks.has(meta.chunk))chunks.set(meta.chunk,JSON.parse(fs.readFileSync(path.join(folder,'war-research',meta.chunk+'.json'))));
    const game=chunks.get(meta.chunk)[meta.index];
    assert.equal(game.home,meta.home);assert.equal(game.away,meta.away);assert.equal(game.season,meta.season);assert.equal(game.week,meta.week);
    for(const [name,candidate] of Object.entries(game.candidates)){
      assert.equal(candidate.members.length,4);
      const result=X.explain(candidate,summary.columns[name],0);
      assert.ok(Math.abs(result.probability-candidate.probability)<1e-10);
      assert.ok(result.terms.every(t=>Number.isFinite(t.value)));
      const swapped={members:candidate.members.map(m=>m.map(v=>-v))};
      assert.ok(Math.abs(X.explain(swapped,summary.columns[name],0).probability+result.probability-1)<1e-10);
    }
    games++;
  }
  assert.equal(games,2189);
  assert.equal(summary.games.filter(g=>g.season>=2024).length,1460);
});
