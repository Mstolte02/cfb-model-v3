/* The selection model's ranking points, averaged over the published season draws. */
(function(root){
  'use strict';
  function rows({playoff,snapshots}){
    const saved=snapshots?.[playoff.version];
    const snapshot=saved && saved.generated_at===playoff.record_odds?.generated_at && saved.season===playoff.season && saved.n_sims===playoff.n_sims?saved:null;
    return playoff.teams.map(t=>{
      const score=Number.isFinite(t.predicted_resume_score)?t:snapshot?.teams?.[t.team];
      return {...t,
      score:score?.predicted_resume_score??null,
      ...score?.resume_score_parts
    };}).sort((a,b)=>(b.score??-Infinity)-(a.score??-Infinity)||a.team.localeCompare(b.team));
  }
  function html(ctx){
    const {playoff,esc,crestOf}=ctx,anchored=Boolean(playoff.committee_anchor),timeline=playoff.committee_anchor?.method?.startsWith('Weekly'),data=rows(ctx),seeds=playoff.bracket?.seeds||[];
    const num=v=>Number.isFinite(v)?v.toFixed(2):'—';
    return `<p class="board-note">${playoff.n_sims.toLocaleString()} simulated seasons. ${timeline?'Each draw advances the latest official poll week by week, accounting for opponent quality, result timing, conference context and head-to-head comparisons. Score = 31 minus projected rank in each draw.':anchored?`The CFP release of ${esc(playoff.committee_anchor.released)} supplies the baseline. Predicted score = baseline + simulated record change.`:'The preseason score combines simulated winning percentage, team quality, schedule strength and the power-conference term.'} Higher scores rank higher within each simulation. These are ranking points, not probabilities or official CFP scores.</p>
      <div class="resume-score-scroll"><table class="resume-score-table"><caption>Average ranking points across every simulated season</caption><thead><tr><th scope="col">Score order</th><th scope="col">Team</th><th scope="col">Predicted score</th><th scope="col">${timeline?'Prior CFP rank':anchored?'CFP baseline':'Record points'}</th><th scope="col">${timeline?'Model adjustment':anchored?'Record change':'Other points'}</th><th scope="col">Avg. simulated rank</th><th scope="col">Current CFP</th><th scope="col">Projected record</th><th scope="col">Bracket seed</th><th scope="col">Make CFP</th></tr></thead><tbody>${data.map((t,i)=>`<tr class="${seeds.includes(t.team)?'in-field':''}"><td>${t.score===null?'—':i+1}</td><th scope="row"><button class="team-link" data-team="${esc(t.team)}"><img src="${esc(crestOf(t.team).mark)}" alt="" loading="lazy">${esc(t.team)}</button><small>${esc(t.conference)}</small></th><td><b>${num(t.score)}</b></td><td>${timeline?(t.priorRank?'#'+t.priorRank:'NR'):num(anchored?t.baseline:t.record)}</td><td>${timeline?num(t.priorRank?t.score-(31-t.priorRank):null):num(anchored?t.recordChange:t.staticScore)}</td><td>${num(t.projected_rank)}</td><td>${t.current_cfp_rank?'#'+t.current_cfp_rank:'NR'}</td><td>${t.avg_wins.toFixed(1)}–${t.avg_losses.toFixed(1)}</td><td>${seeds.includes(t.team)?'#'+(seeds.indexOf(t.team)+1):'—'}</td><td>${(100*t.playoff).toFixed(1)}%</td></tr>`).join('')}</tbody></table></div>
      <p class="board-note">The current bracket uses projected automatic qualifiers and the average score order. An individual draw can have a different field and seed order. Select a team to inspect its season outlook.</p>${data.some(t=>t.score===null)?'<p class="board-note">— indicates that this snapshot does not include a complete score distribution.</p>':''}`;
  }
  root.ResumeScores={rows,html};if(typeof module!=='undefined')module.exports={rows,html};
})(typeof window!=='undefined'?window:globalThis);
