/* Presentation of the same conference and committee draws used by the bracket. */
(function(root){
  'use strict';
  const P4=['ACC','Big 12','Big Ten','SEC'];
  const G6=['American Athletic','Conference USA','Mid-American','Mountain West','Pac-12','Sun Belt'];
  const chance=value=>Math.max(0,Math.min(1,Number(value)||0));
  function ordered(teams){
    return [...teams].sort((a,b)=>(b.predicted_resume_score??-Infinity)-(a.predicted_resume_score??-Infinity)
      ||(a.projected_rank??Infinity)-(b.projected_rank??Infinity)||a.team.localeCompare(b.team));
  }
  function outside(teams,seeds){return ordered(teams.filter(t=>!seeds.includes(t.team))).slice(0,13);}
  function races({playoff,esc,logoURL,color}){
    const row=(t,key)=>`<div class="ccg-row"><img src="${logoURL(t.team)}" alt="" loading="lazy"><button class="team-link" data-team="${esc(t.team)}">${esc(t.team)}</button><div class="bar" role="meter" aria-label="${esc(t.team)} ${key==='g6_bid'?'G6 bid':'conference champion'} probability" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${100*chance(t[key])}"><i style="width:${100*chance(t[key])}%;background:${color(t.team)}"></i></div><span class="pct">${(100*chance(t[key])).toFixed(1)}%</span></div>`;
    const pool=(teams,key,limit)=>{
      const sorted=[...teams].sort((a,b)=>chance(b[key])-chance(a[key])||a.team.localeCompare(b.team));
      return sorted.slice(0,limit).map(t=>row(t,key)).join('')+(sorted.length>limit?`<details class="ccg-more"><summary><span class="show-more">Show more (${sorted.length-limit})</span><span class="show-less">Show less</span></summary>${sorted.slice(limit).map(t=>row(t,key)).join('')}</details>`:'');
    };
    const group=(title,confs,auto)=>`<h4 class="ccg-h">${title}</h4><div class="ccg-grid">${confs.map(c=>{
      const teams=playoff.teams.filter(t=>t.conference===c);
      return teams.length?`<article class="ccg-card"><div class="ccg-name">${esc(c)}<span class="ccg-tag">${auto?'champion is in':'title probability'}</span></div>${pool(teams,'conf_champ',4)}</article>`:'';
    }).join('')}</div>`;
    return `<p class="board-note">Every bar uses a fixed 0–100% scale. Conference title odds follow simulated conference standings, tiebreakers and championship outcomes. The G6 bid goes to the highest committee-ranked G6 team; a conference title is not required.</p>`
      +group('Power 4 — automatic qualifiers',playoff.p4_confs||P4,true)
      +group('Group of 6 — conference title races',playoff.g6_confs||G6,false)
      +`<h4 class="ccg-h">Who takes the Group of 6 bid</h4><div class="ccg-card wide">${pool(playoff.teams.filter(t=>(playoff.g6_confs||G6).includes(t.conference)),'g6_bid',8)}</div>`
      +`<details class="conference-method"><summary>Conference rules and tiebreaker assumptions</summary><p class="board-note">${esc(playoff.conference_method||'cfbseedR applies conference-specific standings and tiebreaker cascades. Model power ratings stand in for unavailable external analytics composites; future CFP ranking tiebreakers are unavailable. The Pac-12 uses the engine’s generic cascade.')} <a href="https://cfbseedr.sportsdataverse.org/reference/cfb_standings.html" target="_blank" rel="noopener">Standings methodology ↗</a></p></details>`;
  }
  function seedMatrix({playoff,esc,crestOf}){
    const data=[...playoff.teams].sort((a,b)=>chance(b.playoff)-chance(a.playoff)
      ||(a.projected_rank??Infinity)-(b.projected_rank??Infinity)||a.team.localeCompare(b.team));
    const bracket=playoff.bracket?.seeds||[];
    const pct=x=>(100*chance(x)).toFixed(1)+'%';
    const numeric=(x,digits=1)=>x!=null&&Number.isFinite(Number(x))?Number(x).toFixed(digits):'—';
    const row=t=>{
      const probabilities=Array.from({length:12},(_,i)=>chance(t.seeds?.[i]));
      const detail=probabilities.map((prob,i)=>{
        if(!prob)return '';
        const o=t.seed_outlooks?.[i];
        return `<div class="seed-route"><strong>Seed ${i+1} · ${pct(prob)}</strong><span>${o?`${numeric(o.wins)}–${numeric(o.losses)} average record · ${numeric(o.score,2)} committee points · ${Number(o.samples)||0} draws${o.conf_champ==null?'':` · ${pct(o.conf_champ)} won conference`}`:'Seed-conditioned record and score unavailable for this snapshot'}</span></div>`;
      }).filter(Boolean).join('');
      const current=bracket.indexOf(t.team)+1;
      const outside=Math.max(0,1-probabilities.reduce((sum,p)=>sum+p,0));
      return `<details class="seed-team"><summary class="seed-grid"><span class="seed-identity"><img src="${esc(crestOf(t.team).mark)}" alt="" loading="lazy"><span><b>${esc(t.team)}</b><small>${esc(t.conference)}${current?' · bracket #'+current:''}</small></span></span><span class="seed-cell seed-make">${pct(t.playoff)}</span>${probabilities.map((p,i)=>`<span class="seed-cell ${p?'has-chance':''}" aria-label="Seed ${i+1}: ${pct(p)}" style="--seed-strength:${p}">${pct(p)}</span>`).join('')}<span class="seed-cell seed-miss">${pct(outside)}</span></summary><div class="seed-team-detail"><div class="seed-overview"><span>Current CFP rank <b>${numeric(t.current_cfp_rank,0)}</b></span><span>Power rank <b>${numeric(t.power_rank,0)}</b></span><span>Average final rank <b>${numeric(t.projected_rank,2)}</b></span><span>All-draw record <b>${numeric(t.avg_wins)}–${numeric(t.avg_losses)}</b></span><span>Playoff <b>${pct(t.playoff)}</b></span><span>Bye <b>${pct(t.bye)}</b></span><span>Conference title <b>${pct(t.conf_champ)}</b></span>${G6.includes(t.conference)?`<span>G6 bid <b>${pct(t.g6_bid)}</b></span>`:''}</div><p>Each route below uses only simulations where ${esc(t.team)} earned that seed. The record and committee points are averages within those draws; title odds are the share of those draws where the team won its conference.</p><div class="seed-routes">${detail||'<p>No seeded draws in this simulation.</p>'}</div><p class="seed-model-note">Power ratings drive game outcomes. The committee score then updates week by week from prior rank, record, opponent record, quality wins, bad losses, recent results, conference context, and title-game results. Nearby head-to-head results affect ranking pools. Scores are model ranking points, not official CFP scores.</p></div></details>`;
    };
    return `<p class="board-note">Seed percentages use ${Number(playoff.n_sims||0).toLocaleString()} simulated seasons. Each row shows the chance of every seed and missing the field; values may differ by 0.1 percentage point from rounding. Sorted by playoff chance. Open a team to see the records and scores in the draws that produced each seed.</p><div class="seed-matrix-scroll"><div class="seed-matrix" role="region" aria-label="Playoff seed probabilities by team"><div class="seed-grid seed-grid-head"><span>Team</span><span>In</span>${Array.from({length:12},(_,i)=>`<span>#${i+1}</span>`).join('')}<span>Out</span></div>${data.map(row).join('')}</div></div>`;
  }
  root.ProjectedAnalysis={ordered,outside,races,seedMatrix};
  if(typeof module!=='undefined')module.exports=root.ProjectedAnalysis;
})(typeof window!=='undefined'?window:globalThis);
