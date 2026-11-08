const assert=require('node:assert/strict');
const {rows,html}=require('../viz/resume-scores.js');
const snapshots=require('../viz/data/resume_scores.json');
const ratings=require('../viz/data/ratings.json'),meta=require('../viz/data/teams.json'),schedule=require('../viz/data/schedule.json'),committee=require('../viz/data/cfp_rankings.json');
for(const version of ['current','preseason']){
  const playoff=require(`../viz/data/playoff_${version}.json`);
  const ctx={playoff,snapshots,ratings:version==='current'?ratings.teams:ratings.history.find(h=>h.week===0).teams,meta,schedule,committee};
  const ledger=rows(ctx);assert.equal(ledger.length,138);
  assert(ledger.every(t=>Number.isFinite(t.score)));
  for(const t of ledger){
    assert(Math.abs(t.score-(version==='current'?t.baseline+t.recordChange:t.record+t.staticScore))<2e-8);

  }
  if(version==='current'){
    const ranked=ledger.filter(t=>t.current_cfp_rank).sort((a,b)=>a.current_cfp_rank-b.current_cfp_rank);
    assert.equal(ranked.length,25);
    for(let i=1;i<25;i++)assert(Math.abs(ranked[i-1].baseline-ranked[i].baseline-playoff.committee_anchor.rank_spacing)<2e-8);
    assert(ledger.filter(t=>!t.current_cfp_rank).every(t=>t.baseline<ranked[24].baseline));
  }
  const page=html({...ctx,esc:s=>String(s),crestOf:()=>({mark:'logo.png'})});
  assert(page.includes('Average ranking points'));assert(page.includes('not probabilities'));assert(page.includes('Avg. simulated rank'));assert(!page.includes('NaN'));
}
// Missing simulation counts are never replaced with selected-only score_parts or rounded avg_wins.
const p=require('../viz/data/playoff_current.json');
const missing={...p,record_odds:null,teams:p.teams.map(({predicted_resume_score,resume_score_parts,...t})=>t)};
assert(rows({playoff:missing,ratings:ratings.teams,meta,schedule,committee}).every(t=>t.score===null));
const fs=require('fs'),index=fs.readFileSync('viz/index.html','utf8');
assert(index.includes('data-view="resume-scores"'));assert(index.includes('id="view-resume-scores"'));assert(!index.includes('How the field is chosen'));
console.log('Resume scores: both snapshots, exported score decomposition, CFP baseline ordering, incomplete data and tab wiring passed');

// A new simulation must never borrow the previous simulation's ranking scores.
assert(rows({playoff:{...missing,record_odds:{generated_at:'new-run'}},snapshots}).every(t=>t.score===null));
assert(rows({playoff:p,snapshots}).every(t=>Number.isFinite(t.score)));
const direct={...p,teams:[{...p.teams[0],predicted_resume_score:42,resume_score_parts:{baseline:40,recordChange:2}}]};
assert.equal(rows({playoff:direct,snapshots})[0].score,42);
