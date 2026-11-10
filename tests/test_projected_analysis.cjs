const assert=require('node:assert/strict');
const {races,table,outside}=require('../viz/projected-analysis.js');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const teams=Array.from({length:30},(_,i)=>({team:`Team ${i+1}`,conference:i<6?'SEC':'Mountain West',
  predicted_resume_score:30-i,projected_rank:i+1,conf_champ:i===0?.8:0,g6_bid:i===6?.8:0,
  playoff:1-i/30,avg_wins:10,avg_losses:2}));
const playoff={teams,bracket:{seeds:teams.slice(0,12).map(t=>t.team),auto_bids:['Team 1','Team 7']}};
const ctx={playoff,esc,logoURL:()=>'/logo.png',color:()=>'#abc',crestOf:()=>({mark:'/logo.png'})};
let html=races(ctx);
assert(html.includes('width:80%'));assert(html.includes('aria-valuenow="80"'));
assert.equal((html.match(/data-team=/g)||[]).length,54); // 30 conference rows + 24 G6 rows
assert(html.includes('Show more (2)'));assert(html.includes('Show more (16)'));
assert(html.includes('Team 30'));
html=table(ctx);assert.equal((html.match(/<tr class=/g)||[]).length,30);
assert(html.includes('Automatic qualifier · SEC champion'));assert(html.includes('highest-ranked G6 team'));
assert(html.includes('Outside projected field'));assert(html.includes('All season draws'));
teams[0].seed_outlooks=[{score:42,wins:12,losses:1,samples:100}];
html=table(ctx);assert(html.includes('42.00'));assert(html.includes('12.0–1.0'));assert(html.includes('Seed #1 draws (100)'));
const out=outside(teams,playoff.bracket.seeds);assert.equal(out.length,13);assert.equal(out[0].team,'Team 13');assert.equal(out[12].team,'Team 25');
const {projectedHTML}=require('../viz/marks-playoff.js');
html=projectedHTML({games:[],seeds:playoff.bracket.seeds,teams,season:2026,esc,crestOf:ctx.crestOf,meta:{}});
assert(html.includes('OUTSIDE LOOKING IN'));assert(!html.includes('PROJECTED SEEDS'));assert(html.includes('>13</b>'));assert(html.includes('>25</b>'));
console.log('Analysis: absolute percentage bars, expanded full fields, qualification table, seed-conditioned scores, and outside 13–25 passed');
