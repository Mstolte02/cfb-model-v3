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
  function table({playoff,esc,crestOf}){
    const seeds=playoff.bracket?.seeds||[],auto=new Set(playoff.bracket?.auto_bids||[]),data=ordered(playoff.teams);
    const num=x=>Number.isFinite(x)?x.toFixed(2):'—';
    return `<p class="board-note">All ${data.length} teams, ordered by average projected committee score. Scores are model ranking points, not official CFP scores. Seed rows show the average score and record in simulations where the team earned that seed, when available; other rows show averages across all simulations. The bracket combines seed frequencies, so average score order can differ from its seeds.</p><div class="resume-score-scroll"><table class="resume-score-table projected-selection-table"><caption>How the projected field is selected</caption><thead><tr><th scope="col">Score order</th><th scope="col">Team</th><th scope="col">Bracket seed</th><th scope="col">Committee score</th><th scope="col">Projected record</th><th scope="col">Record / score basis</th><th scope="col">Qualification</th><th scope="col">Make CFP</th><th scope="col">Conference title</th><th scope="col">G6 bid</th></tr></thead><tbody>${data.map((t,i)=>{
      const seed=seeds.indexOf(t.team)+1,conditioned=seed?t.seed_outlooks?.[seed-1]:null;
      const score=conditioned?.score??t.predicted_resume_score,wins=conditioned?.wins??t.avg_wins,losses=conditioned?.losses??t.avg_losses;
      const bid=auto.has(t.team)?P4.includes(t.conference)?`Automatic qualifier · ${t.conference} champion`:G6.includes(t.conference)?'Automatic qualifier · highest-ranked G6 team':'Notre Dame Top 12 protection':seed?'At-large':'Outside projected field';
      return `<tr class="${seed?'in-field':''}"><td>${i+1}</td><th scope="row"><button class="team-link" data-team="${esc(t.team)}"><img src="${esc(crestOf(t.team).mark)}" alt="" loading="lazy">${esc(t.team)}</button><small>${esc(t.conference)}</small></th><td>${seed?'#'+seed:'—'}</td><td><b>${num(score)}</b></td><td>${Number.isFinite(wins)&&Number.isFinite(losses)?wins.toFixed(1)+'–'+losses.toFixed(1):'—'}</td><td>${conditioned?'Seed #'+seed+' draws ('+conditioned.samples+')':'All season draws'}</td><td class="selection-route">${esc(bid)}</td><td>${(100*chance(t.playoff)).toFixed(1)}%</td><td>${(100*chance(t.conf_champ)).toFixed(1)}%</td><td>${G6.includes(t.conference)?(100*chance(t.g6_bid)).toFixed(1)+'%':'—'}</td></tr>`;
    }).join('')}</tbody></table></div>`;
  }
  root.ProjectedAnalysis={ordered,outside,races,table};
  if(typeof module!=='undefined')module.exports=root.ProjectedAnalysis;
})(typeof window!=='undefined'?window:globalThis);
