/* Most Deserving résumé lab: ledger, head-to-head, cutoff sensitivity, method.
   Every number shown comes from RankingHistory.score, so the explanations always
   reconcile with the ranking. Predictive ratings and committee projections are not used. */
(function(root) {
  'use strict';
  const signed=(v,d=2)=>`${v>0.0000005?'+':v<-0.0000005?'−':''}${Math.abs(v).toFixed(d)}`;
  const pctText=(p,d=0)=>{const v=100*p;return v>0&&v<1?'<1%':v>99&&v<100?'>99%':`${v.toFixed(d)}%`;};
  const record=r=>`${r.wins}–${r.losses}${r.ties?'–'+r.ties:''}`;
  const lossWord=n=>Math.abs(n)===1?'loss':'losses';
  const gapText=g=>Math.abs(g)<0.01?Math.abs(g).toFixed(3):Math.abs(g).toFixed(2);

  // Most informative default comparison: the highest pair where the team with more
  // losses ranks ahead (a conditional override); otherwise #1 against #2.
  function defaultPair(rows) {
    const top=rows.filter(r=>r.rank<=25);
    for(const a of top)for(const b of top)if(a.rank<b.rank&&a.losses>b.losses)return [a.team,b.team];
    return rows.length>1?[rows[0].team,rows[1].team]:[];
  }
  function band(share) {
    if(share>=0.999)return {key:'secure',label:'Secure'};
    if(share>=2/3)return {key:'likely',label:'Likely in'};
    if(share>1/3)return {key:'bubble',label:'On the bubble'};
    if(share>0)return {key:'outside',label:'Needs help'};
    return {key:'out',label:'Out in every case'};
  }
  const h2hText=list=>list.filter(x=>!x.cap&&Math.abs(x.value)>=0.0005).map(x=>`${x.won?'beat':'lost to'} ${x.team} ${signed(x.value,3)}`).join('; ');
  // Quality wins and losses, in one line.
  function qualityText(r) {
    const q=r.quality||{};
    return `${q.playoffWins||0} win${q.playoffWins===1?'':'s'} over playoff-caliber teams, ${q.top25Wins||0} over top-25 caliber; ${q.top25Losses||0} loss${q.top25Losses===1?'':'es'} to top-25 caliber, ${q.badLosses||0} to below-average FBS teams`;
  }
  // "Why is A ahead of B?" Components are the ranking score's own components.
  function explain(a,b,assumptions) {
    const R=root.RankingHistory,c=R.compare(a,b),omega=assumptions.loss_aversion;
    const rows=[{label:c.lossGap===0?'Losses (equal)':`${Math.abs(c.lossGap)} ${c.lossGap>0?'additional':'fewer'} ${lossWord(c.lossGap)}`,value:c.losses,
        note:`Each FBS loss costs ${omega.toFixed(2)}`},
      {label:'Schedule quality',value:c.schedule,note:'Extra games a playoff-caliber team would expect to lose',children:[
        {label:'from games won (quality wins)',value:c.scheduleFromWins},{label:'from games lost (softens those losses)',value:c.scheduleFromLosses}]}];
    if(assumptions.convincingness_weight>0||Math.abs(c.performance)>1e-12)rows.push({label:'Performance',value:c.performance,note:'How clearly games were won or lost'});
    rows.push({label:'Résumé score difference',value:c.resume,subtotal:true});
    rows.push({label:'Head-to-head adjustment',value:c.headToHead,note:'Capped update from results between close résumés',children:[
      {label:`${a.team}: ${h2hText(a.headToHeadGames)||'none'}`,value:a.headToHead},{label:`${b.team}: ${h2hText(b.headToHeadGames)||'none'}`,value:-b.headToHead}]});
    rows.push({label:'Net ranking advantage',value:c.net,total:true});
    const ahead=a.rank<b.rank?a:b.rank<a.rank?b:null,behind=ahead===a?b:a,window=assumptions.head_to_head_window||0;
    const sideOf=t=>t===a?1:-1;
    const games=c.meetings.map(g=>`${g.result==='W'?a.team:b.team} won ${Math.max(g.pf,g.pa)}–${Math.min(g.pf,g.pa)}${g.date?` (${String(g.date).slice(5).replace('-','/')})`:''}`);
    let headToHead=null;
    if(c.meetings.length){
      const winner=c.headToHeadNet>0?a:c.headToHeadNet<0?b:null,loser=winner===a?b:a;
      const pair=winner&&winner.headToHeadGames.find(x=>x.team===loser.team&&x.won);
      const gap=winner?loser.resumeScore-winner.resumeScore:0;
      const why=!winner?'the series is split, so it adjusts nothing'
        :pair?`${gap>0?`${winner.team} trailed by ${gapText(gap)} on résumé score, inside the ${window.toFixed(2)} band where two résumés are not reliably different`:`${winner.team} was already ${gapText(-gap)} ahead on résumé score`}, so the result moved the pair ${(2*pair.value).toFixed(3)} closer (${winner.team} ${signed(pair.value,3)}, ${loser.team} ${signed(-pair.value,3)})`
        :`${winner.team} trailed by ${gap.toFixed(2)}, outside the ${window.toFixed(2)} band, so the game counts only through its own result`;
      headToHead=`Head-to-head: ${games.join('; ')}. ${why[0].toUpperCase()+why.slice(1)}.`;
    }
    let conclusion;
    if(!ahead)conclusion=`${a.team} and ${b.team} have equal ranking scores and share a rank.`;
    else if(c.decidedBy){
      const direct=c.meetings.length&&((c.headToHeadNet>0&&ahead===a)||(c.headToHeadNet<0&&ahead===b));
      const edge=Math.abs(c.resume),swing=Math.abs(c.headToHead);
      conclusion=direct
        ?`${ahead.team} ranks ahead on head-to-head. ${behind.team}'s résumé score is ${gapText(edge)} higher, inside the ${window.toFixed(2)} band, and ${ahead.team} won the game between them; the head-to-head adjustments swing ${swing.toFixed(3)} its way.`
        :`${ahead.team} ranks ahead through head-to-head results against other teams. ${behind.team}'s résumé score is ${gapText(edge)} higher, but the adjustments (${ahead.team}: ${h2hText(ahead.headToHeadGames)||'none'}; ${behind.team}: ${h2hText(behind.headToHeadGames)||'none'}) swing ${swing.toFixed(3)}. The two did not decide it between themselves.`;
    }
    else{
      const lossGap=ahead.losses-behind.losses,edge=Math.abs(c.net),quality=Math.abs(c.schedule+c.performance),s=sideOf(ahead);
      if(lossGap>0)conclusion=`${ahead.team} earned the higher ranking despite ${lossGap} more ${lossWord(lossGap)}: `+
        `its schedule and results are worth ${quality.toFixed(2)} loss equivalents more, which clears the ${(omega*lossGap).toFixed(2)} cost of the extra ${lossWord(lossGap)}.`;
      else if(lossGap<0)conclusion=`${ahead.team} ranks ahead with ${-lossGap} fewer ${lossWord(-lossGap)}. `+
        `${behind.team}'s schedule edge (${Math.max(0,-s*(c.schedule+c.performance)).toFixed(2)}) is not enough to overcome the ${(omega*-lossGap).toFixed(2)} loss gap.`;
      else conclusion=`Same number of losses, so schedule quality decides it: ${ahead.team} leads by ${Math.abs(c.resume).toFixed(2)} loss equivalents on résumé score.`;
      if(Math.abs(c.headToHead)>=0.0005)conclusion+=` Head-to-head adjustments move the gap ${signed(s*c.headToHead,3)}, leaving ${ahead.team} ${edge.toFixed(2)} ahead.`;
    }
    const rarity=`A playoff-caliber reference team would match or beat ${a.team}'s ${record(a)} on ${a.team}'s schedule ${pctText(a.matchProbability,1)} of the time, and ${b.team}'s ${record(b)} on ${b.team}'s schedule ${pctText(b.matchProbability,1)} of the time.`;
    return {rows,conclusion,rarity,headToHead,compare:c};
  }

  function mount(o) {
    const {host,ctx,model,esc,crestOf,wireTeamLinks,games_through}=o;
    const R=root.RankingHistory,official=model.assumptions;
    const rows=R.score(ctx,official),byTeam=new Map(rows.map(r=>[r.team,r]));
    const sens=R.sensitivity(ctx,model),cal=ctx.cal,v=model.validation||{};
    const refDist=R.referenceDistribution(cal,official),refPts=refDist.mean,win=official.head_to_head_window||0;
    const range=r=>r.scoreLow===undefined?'':`<small>80%: ${signed(r.scoreLow)} to ${signed(r.scoreHigh)}</small>`;
    const h2hTag=r=>r.headToHeadPassed?.length?` <span class="md-h2h-tag" title="${esc('Head-to-head adjustment '+signed(r.headToHead,3)+' ('+h2hText(r.headToHeadGames)+'); now above '+r.headToHeadPassed.map(p=>p.team).join(', ')+', whose résumé scores are higher')}">H2H</span>`:'';
    const band12=ctx.inSeasonBand,prior=model.prior,pw=ctx.priorWeight||0;
    const refSeasons=cal.reference_seasons||[],refFrom=refSeasons.length?`${refSeasons[0]}–${refSeasons.at(-1)}`:'earlier seasons';
    // Drift check (shown, never used): this season's #8-#16 band so far against the same
    // week's reading in past seasons. The in-season fit is shrunk, so comparing it with the
    // end-of-season reference directly would always look like a weak year.
    const byWeek=v.reference_band?.in_season_by_week||{},wk=Object.keys(byWeek).map(Number).filter(w=>w<=ctx.week).sort((x,y)=>y-x)[0];
    const past=wk!==undefined?byWeek[wk]:null,drift=band12&&past&&ctx.week>=6?band12.mean-past.mean:null;
    const crest=t=>{const c=crestOf(t);return `<span class="md-crest" style="--plate:${c.plate}"><img src="${c.mark}" alt="" loading="lazy"></span>`;};
    const teamBtn=(t,small='')=>`<button type="button" class="md-team team-link" data-team="${esc(t)}">${crest(t)}<span>${esc(t)}${small?`<small>${small}</small>`:''}</span></button>`;
    const showPerf=official.convincingness_weight>0;
    const grid=model.sensitivity||{};
    const fact=(label,value,sub)=>`<div><dt>${label}</dt><dd>${value}</dd>${sub?`<small>${sub}</small>`:''}</div>`;
    host.innerHTML=`<article class="md-lab">
      <header class="md-head"><span class="eyebrow">The résumé lab</span><h3>Earned on the field. Explained to the decimal.</h3>
        <p>How many more games did each team win than a playoff-caliber team would have, on the same schedule at the same venues? An extra loss costs one point; a schedule can pay it back only when it is worth a whole extra expected loss.</p>
        <dl class="md-facts">
          ${fact('Reference team',`+${refPts.toFixed(1)} <i>± ${refDist.sd.toFixed(1)}</i>`,`pts vs average FBS · #${Math.max(1,official.benchmark_rank-4)}–#${official.benchmark_rank+4}, ${esc(refFrom)}`)}
          ${drift!==null?fact('Playoff band this year',Math.abs(drift)>past.sd?'Unusual':'Normal',`${drift>=0?'+':'−'}${Math.abs(drift).toFixed(1)} pts vs past seasons at week ${wk}`):''}
          ${fact('Head-to-head band',win.toFixed(2),'capped adjustment between close résumés')}
          ${pw>0?fact('Early-season anchor',`${Math.round(100*pw)}%`,`opponents tied to ${prior.season} ratings`):''}
          ${fact('Results through',ctx.week?`Week ${ctx.week}`:'—','completed FBS-vs-FBS games')}
        </dl></header>
      <div class="md-tabs" role="tablist" aria-label="Résumé lab views">
        <button role="tab" aria-selected="true" data-md-tab="h2h">Why is A ahead of B?</button><button role="tab" aria-selected="false" data-md-tab="ledger">Résumé ledger</button><button role="tab" aria-selected="false" data-md-tab="sens">Cutoff sensitivity</button><button role="tab" aria-selected="false" data-md-tab="how">How it works</button></div>
      <section class="md-panel" data-md-panel="h2h"></section>
      <section class="md-panel" data-md-panel="ledger" hidden></section>
      <section class="md-panel" data-md-panel="sens" hidden></section>
      <section class="md-panel" data-md-panel="how" hidden></section>
    </article>`;
    const panel=k=>host.querySelector(`[data-md-panel="${k}"]`);

    /* ---- Shared pieces ---- */
    const tierInfo={playoff:['Playoff','Playoff-caliber'],top25:['Top 25','Top-25 caliber'],average:['Avg','Average FBS'],'below-average':['Weak','Below-average FBS']};
    const tierTag=g=>`<span class="md-tier md-tier-${g.tier}" title="${esc(tierInfo[g.tier]?.[1]||'')} opponent: ${signed(g.adjustedOpp,1)} pts vs an average FBS team, venue-adjusted">${tierInfo[g.tier]?.[0]||''}</span>`;
    const tierLegend=`<p class="md-legend">Opponent tier, venue-adjusted: ${Object.entries(tierInfo).map(([k,[short,long]])=>`<span class="md-tier md-tier-${k}">${short}</span> ${long}`).join(' ')}</p>`;
    const where=g=>`${g.site==='away'?'at ':g.site==='neutral'?'vs ':''}${esc(g.opponent)}`;
    const qualityShort=r=>{const q=r.quality||{};return `${q.playoffWins||0} · ${q.top25Wins||0}`;};
    const lossFlags=r=>{const q=r.quality||{},out=[];if(q.badLosses)out.push(`<span class="md-flag">${q.badLosses} bad loss${q.badLosses===1?'':'es'}</span>`);return out.join(' ');};
    const gameWhy=g=>g.result==='W'?`Win: a playoff-caliber team loses this game ${pctText(1-g.refWin)} of the time, so it is worth ${signed(g.total)}`
      :g.result==='L'?`Loss: −${official.loss_aversion.toFixed(2)}, with ${signed(g.scheduleTerm)} back for difficulty`:'Tie: half a loss';

    /* ---- Ledger ---- */
    let showAll=false;
    const detailRow=r=>`<tr class="md-games" hidden data-md-games-for="${esc(r.team)}"><td colspan="${showPerf?9:8}">
        <dl class="md-strip-stats">
          <div><dt>Résumé score</dt><dd>${signed(r.resumeScore)}</dd><small>before head-to-head</small></div>
          <div><dt>Schedule credit</dt><dd>${signed(r.schedule)}</dd><small>wins ${signed(r.scheduleFromWins)} · losses ${signed(r.scheduleFromLosses)}</small></div>
          <div><dt>Reference team</dt><dd>${r.referenceWins.toFixed(1)}–${r.schedule.toFixed(1)}</dd><small>its expected record here</small></div>
          <div><dt>Match or beat</dt><dd>${pctText(r.matchProbability,1)}</dd><small>range ${pctText(r.matchProbabilityLow,0)}–${pctText(r.matchProbabilityHigh,0)}</small></div>
          <div><dt>Rarity</dt><dd>${r.rarity.toFixed(2)}</dd><small>#${r.rarityRank} in FBS</small></div>
          <div><dt>Quality</dt><dd>${qualityShort(r)}</dd><small>${r.quality?.top25Losses||0} top-25 L · ${r.quality?.badLosses||0} bad L</small></div>
        </dl>
        <table><thead><tr><th>Date</th><th>Opponent</th><th>Result</th><th class="num" title="Chance the playoff-caliber reference team wins this game">Ref. wins</th><th class="num" title="1 − that chance. Summed, this is the schedule credit.">Ref. exp. losses</th><th class="num">Value</th></tr></thead><tbody>
        ${r.ledger.map(g=>`<tr title="${esc(gameWhy(g))}"><td>${esc(String(g.date||'').slice(5).replace('-','/'))}</td><td>${where(g)} ${tierTag(g)}</td><td class="md-res md-${g.result}">${g.result} ${g.pf}–${g.pa}</td><td class="num">${pctText(g.refWin)}</td><td class="num">${(1-g.refWin).toFixed(2)}</td><td class="num ${g.total>=0?'md-pos':'md-neg'}">${signed(g.total)}</td></tr>`).join('')}
        <tr class="md-sum"><td></td><td>Résumé score</td><td>${record(r)}</td><td class="num">${r.referenceWins.toFixed(1)} W</td><td class="num">${r.schedule.toFixed(2)}</td><td class="num">${signed(r.resumeScore)}</td></tr>
        ${Math.abs(r.headToHead)>=0.0005?`<tr class="md-sum md-h2h-line"><td></td><td colspan="4">Head-to-head <small>${esc(h2hText(r.headToHeadGames))}</small></td><td class="num">${signed(r.headToHead)}</td></tr>
        <tr class="md-sum"><td></td><td colspan="4">Ranking score</td><td class="num">${signed(r.score)}</td></tr>`:''}
        </tbody></table></td></tr>`;
    function drawLedger(query='') {
      const q=query.toLowerCase().trim(),matches=rows.filter(r=>r.team.toLowerCase().includes(q));
      const list=q||showAll?matches:matches.filter(r=>r.rank<=25);
      panel('ledger').innerHTML=`<div class="md-toolbar"><p><b>Ranking score</b> = losses + schedule credit${showPerf?' + performance':''} + head-to-head. Open <b>Details</b> for every game.</p><input type="search" placeholder="Find a team…" aria-label="Find a team in the résumé ledger" value="${esc(query)}"></div>
      <div class="md-table-wrap"><table class="md-table md-ledger"><thead><tr><th>#</th><th>Team</th><th class="num" title="−${official.loss_aversion} per FBS loss">Losses</th><th class="num" title="Games a playoff-caliber team would expect to lose on this schedule">Schedule</th>${showPerf?'<th class="num">Perf.</th>':''}<th class="num" title="Capped head-to-head adjustment between close résumés">H2H</th><th class="num">Ranking score</th><th class="num" title="Wins over playoff-caliber · top-25-caliber opponents">Quality wins</th><th title="Share of the playoff-caliber reference range under which this team ranks in the top 12">Top-12 chance</th><th></th></tr></thead>
      <tbody>${list.map(r=>`<tr class="${r.rank<=12?'md-in':''}${r.rank===12?' md-cutline':''}"><td class="md-rank">${r.rank}${h2hTag(r)}</td><td>${teamBtn(r.team,`${record(r)} FBS`)}</td>
        <td class="num md-neg">${signed(r.lossPenalty)}</td><td class="num md-pos">${signed(r.schedule)}</td>${showPerf?`<td class="num">${signed(r.performance)}</td>`:''}
        <td class="num ${r.headToHead>0.0005?'md-pos':r.headToHead<-0.0005?'md-neg':'md-quiet'}">${Math.abs(r.headToHead)<0.0005?'—':signed(r.headToHead)}</td>
        <td class="num md-score">${signed(r.score)}${range(r)}</td><td class="num">${qualityShort(r)}${lossFlags(r)?`<small>${lossFlags(r)}</small>`:''}</td>
        <td><span class="md-share md-${band(r.topProbability).key}">${pctText(r.topProbability)}</span></td><td><button type="button" class="md-expand" aria-expanded="false" data-md-games="${esc(r.team)}">Details</button></td></tr>${detailRow(r)}`).join('')||`<tr><td colspan="9">No matching teams.</td></tr>`}</tbody></table></div>
      ${!q&&matches.length>25?`<button type="button" class="md-more">${showAll?'Show the top 25 only':`Show all ${matches.length} teams`}</button>`:''}
      ${tierLegend}
      <details class="md-glossary"><summary>What the columns mean</summary><dl>
        <dt>Schedule</dt><dd>Games a playoff-caliber team would expect to lose against the same opponents and venues. It pays back losses.</dd>
        <dt>H2H</dt><dd>The capped head-to-head adjustment (at most ±${(win/2).toFixed(2)}). An <span class="md-h2h-tag">H2H</span> badge marks a team it lifted above a team with a higher résumé score.</dd>
        <dt>Ranking score</dt><dd>Losses + schedule${showPerf?' + performance':''} + H2H. The small 80% figures show the score against a stronger and a weaker playoff-caliber team.</dd>
        <dt>Quality wins</dt><dd>Wins over playoff-caliber · top-25-caliber opponents, judged with the same opponent strengths and venue adjustment as the score.</dd>
        <dt>Top-12 chance</dt><dd>Share of the playoff-caliber reference range under which the team ranks in the top 12. Not its playoff chance.</dd>
        <dt>Match or beat, rarity</dt><dd>In Details: the reference team's chance to finish with at least this record on this schedule, and −log₁₀ of it. Lower chance = rarer résumé.</dd>
        <dt>Records</dt><dd>FBS games only; FCS games do not count.</dd></dl></details>`;
      const input=panel('ledger').querySelector('input');
      input.addEventListener('input',()=>{const pos=input.selectionStart;drawLedger(input.value);const n=panel('ledger').querySelector('input');n.focus();n.setSelectionRange(pos,pos);});
      panel('ledger').querySelector('.md-more')?.addEventListener('click',()=>{showAll=!showAll;drawLedger(query);});
      panel('ledger').querySelectorAll('[data-md-games]').forEach(b=>b.addEventListener('click',()=>{
        const tr=panel('ledger').querySelector(`[data-md-games-for="${CSS.escape(b.dataset.mdGames)}"]`),open=tr.hidden;tr.hidden=!open;b.setAttribute('aria-expanded',String(open));b.textContent=open?'Hide':'Details';}));
      wireTeamLinks();
    }

    /* ---- Why is A ahead of B? ---- */
    let pair=defaultPair(rows);
    function drawH2H() {
      const [ta,tb]=pair,a=byTeam.get(ta),b=byTeam.get(tb);
      const options=sel=>rows.map(r=>`<option value="${esc(r.team)}"${r.team===sel?' selected':''}>#${r.rank} ${esc(r.team)} (${record(r)})</option>`).join('');
      if(!a||!b){panel('h2h').innerHTML='<p class="md-note">Two ranked teams are needed for a comparison.</p>';return;}
      const e=explain(a,b,official),max=Math.max(.5,...e.rows.map(x=>Math.abs(x.value)));
      const bar=x=>`<span class="md-bar-track"><i class="md-zero"></i><span class="md-bar ${x>=0?'pos':'neg'}" style="width:${Math.abs(x)/max*50}%;${x>=0?'left:50%':'right:50%'}"></span></span>`;
      const card=(r,side)=>`<div class="md-card md-${side}" style="--plate:${crestOf(r.team).plate}">${crest(r.team)}<div class="md-card-body"><span class="md-card-rank">#${r.rank} résumé</span><button type="button" class="team-link md-card-name" data-team="${esc(r.team)}">${esc(r.team)}</button>
        <dl><div><dt>FBS record</dt><dd>${record(r)}</dd></div><div><dt>Ranking score</dt><dd>${signed(r.score)}</dd>${r.scoreLow!==undefined?`<small>80%: ${signed(r.scoreLow)} to ${signed(r.scoreHigh)}</small>`:''}</div>
        <div><dt>Reference team</dt><dd>${r.referenceWins.toFixed(1)}–${r.schedule.toFixed(1)}</dd><small>expected record here</small></div><div><dt>Quality wins</dt><dd>${qualityShort(r)}</dd><small>playoff · top 25${r.quality?.badLosses?` · ${r.quality.badLosses} bad L`:''}</small></div></dl></div></div>`;
      // A per-game receipt: every value adds to the résumé score, and the head-to-head lines to the ranking score.
      const receipt=r=>{const gs=[...r.ledger].sort((x,y)=>Math.abs(y.total)-Math.abs(x.total)),hh=r.headToHeadGames.filter(x=>Math.abs(x.value)>=0.0005||x.cap);
        return `<div class="md-receipt" style="--plate:${crestOf(r.team).plate}"><h5>${crest(r.team)}${esc(r.team)}<span>${signed(r.score)}</span></h5><table><thead><tr><th>Game</th><th class="num" title="Losses a playoff-caliber team would expect in this game">Ref. exp. L</th><th class="num">Value</th></tr></thead><tbody>
        ${gs.map(g=>`<tr title="${esc(gameWhy(g))}"><td><span class="md-res md-${g.result}">${g.result}</span> ${where(g)} <span class="md-score-sm">${g.pf}–${g.pa}</span> ${tierTag(g)}</td><td class="num">${(1-g.refWin).toFixed(2)}</td><td class="num ${g.total>=0?'md-pos':'md-neg'}">${signed(g.total)}</td></tr>`).join('')}
        <tr class="md-sum"><td>Résumé score</td><td class="num">${r.schedule.toFixed(2)}</td><td class="num">${signed(r.resumeScore)}</td></tr>
        ${hh.map(x=>`<tr class="md-h2h-line"><td>${x.cap?`Head-to-head cap (±${(win/2).toFixed(2)})`:`H2H: ${x.won?'beat':'lost to'} ${esc(x.team)}`}</td><td></td><td class="num ${x.value>=0?'md-pos':'md-neg'}">${signed(x.value)}</td></tr>`).join('')}
        <tr class="md-sum md-total"><td>Ranking score</td><td></td><td class="num">${signed(r.score)}</td></tr></tbody></table></div>`;};
      panel('h2h').innerHTML=`<div class="md-pickers"><label>Team A<select data-md-pick="0">${options(ta)}</select></label><button type="button" class="md-swap" aria-label="Swap teams">⇄</button><label>Team B<select data-md-pick="1">${options(tb)}</select></label></div>
        <div class="md-versus">${card(a,'a')}<span class="md-vs">VS</span>${card(b,'b')}</div>
        <p class="md-verdict">${esc(e.conclusion)}</p>
        ${e.headToHead?`<p class="md-h2h-note">${esc(e.headToHead)}</p>`:''}
        <h4 class="md-section">Where the gap comes from <small>in loss units · positive favours ${esc(a.team)}</small></h4>
        <table class="md-compare"><thead><tr><th>Component</th><th class="num">${esc(a.team)} edge</th><th class="md-barcol"><span>← ${esc(b.team)}</span><span>${esc(a.team)} →</span></th></tr></thead><tbody>
        ${e.rows.map(x=>`<tr class="${x.total?'md-total':x.subtotal?'md-subtotal':''}"><td>${esc(x.label)}${x.note?`<small>${esc(x.note)}</small>`:''}</td><td class="num ${x.value>=0?'md-pos':'md-neg'}">${signed(x.value)}</td><td class="md-barcol">${bar(x.value)}</td></tr>${(x.children||[]).map(ch=>{const m=/^([^:]+): (.*)$/.exec(ch.label);return `<tr class="md-child"><td>${m?`${esc(m[1])}<small>${esc(m[2])}</small>`:esc(ch.label)}</td><td class="num">${signed(ch.value)}</td><td class="md-barcol"></td></tr>`;}).join('')}`).join('')}</tbody></table>
        <h4 class="md-section">Game by game <small>largest first · hover a game for why</small></h4>
        <div class="md-receipts">${receipt(a)}${receipt(b)}</div>
        ${tierLegend}
        <p class="md-note">${esc(e.rarity)} That measures how hard each record is to reproduce, not either team's playoff chance. Values are rounded; the totals are exact.</p>`;
      panel('h2h').querySelectorAll('[data-md-pick]').forEach(s=>s.addEventListener('change',()=>{pair[+s.dataset.mdPick]=s.value;drawH2H();}));
      panel('h2h').querySelector('.md-swap').addEventListener('click',()=>{pair=[pair[1],pair[0]];drawH2H();});
      wireTeamLinks();
    }

    /* ---- Sensitivity ---- */
    const choice={...official};
    const labels={benchmark_rank:['Reference team',k=>`#${k}`],convincingness_weight:['Convincingness weight',x=>`${Math.round(100*x)}%`],loss_aversion:['Cost per loss',x=>x.toFixed(2)],head_to_head_window:['Head-to-head',x=>x>0?`On (${x.toFixed(2)})`:'Off']};
    const explorerValues={benchmark_rank:[4,8,12,16,25],convincingness_weight:[0,0.05,0.1,0.2],loss_aversion:[1,1.25,1.5],head_to_head_window:[0,win]};
    function drawSens() {
      const strip=rows.filter(r=>r.rank<=20||sens.byTeam.get(r.team).min<=12).slice(0,22);
      const track=(s,r)=>{const pos=k=>`${(Math.min(k,26)-1)/25*100}%`;return `<span class="md-track"><i class="md-cut"></i><span class="md-range" style="left:${pos(s.min)};width:calc(${pos(s.max)} - ${pos(s.min)})"></span><b class="md-dot" style="left:${pos(r.rank)}"></b></span>`;};
      const alt=R.score(ctx,choice),altTop=alt.filter(r=>r.rank<=16);
      const configs=sens.configs.length,dims=Object.entries(grid).map(([k,vals])=>`${labels[k]?.[0]||k} ${vals.map(labels[k]?.[1]||String).join(' / ')}`).join(' · ');
      panel('sens').innerHTML=`<div class="md-sens-grid"><div class="md-subpanel"><span class="eyebrow">Who is safe at the cutoff?</span><h4>${configs} reasonable assumption sets</h4>
        <p>Each team is re-ranked under every combination of: ${esc(dims)}. The bar spans its best and worst rank; the dot is its official rank; the line marks the twelfth place. The badge is the share of those assumption sets with a top-12 rank. The small text under each team is a separate test: how often it ranks in the top 12 across the playoff-caliber reference range.</p>
        <div class="md-strip">${strip.map(r=>{const s=sens.byTeam.get(r.team),bd=band(s.inTop);return `<div class="md-strip-row"><span class="md-rank">${r.rank}</span>${teamBtn(r.team,`${record(r)} · ranks ${s.min}–${s.max} · top 12 in ${pctText(r.topProbability)} of the reference range`)}${track(s,r)}<span class="md-share md-${bd.key}">${pctText(s.inTop)}<small>${bd.label}</small></span></div>`;}).join('')}</div>
        <div class="md-axis"><span>#1</span><span>#12 cutoff</span><span>#25+</span></div></div>
        <div class="md-subpanel"><span class="eyebrow">Try other assumptions</span><h4>Re-rank the field</h4><p>The official ranking uses the bold choices. These settings are philosophical, not fitted: see “How it works”.</p>
        ${Object.entries(explorerValues).map(([k,vals])=>`<div class="md-choice"><span>${labels[k][0]}</span><div class="seg">${vals.map(x=>`<button type="button" class="seg-btn${choice[k]===x?' active':''}${official[k]===x?' md-official':''}" data-md-set="${k}" data-v="${x}" aria-pressed="${choice[k]===x}">${labels[k][1](x)}</button>`).join('')}</div></div>`).join('')}
        <ol class="md-alt">${altTop.map(r=>{const o=byTeam.get(r.team).rank,d=o-r.rank;return `<li><span class="md-rank">${r.rank}</span>${teamBtn(r.team,record(r))}<span class="md-alt-score">${signed(r.score)}</span><span class="md-move ${d>0?'up':d<0?'down':''}">${d>0?'▲'+d:d<0?'▼'+(-d):'—'}</span></li>`;}).join('')}</ol>
        <button type="button" class="md-reset">Restore official assumptions</button></div></div>`;
      panel('sens').querySelectorAll('[data-md-set]').forEach(b=>b.addEventListener('click',()=>{choice[b.dataset.mdSet]=+b.dataset.v;drawSens();}));
      panel('sens').querySelector('.md-reset').addEventListener('click',()=>{Object.assign(choice,official);drawSens();});
      wireTeamLinks();
    }

    /* ---- How it works ---- */
    function drawHow() {
      const ov=v.overrides||{},om=v.opponent_model||{},hh=v.head_to_head,es=v.early_season_prior,rb=v.robustness,bt=v.committee_backtest;
      const card=(eyebrow,title,body,more='',wide=false)=>`<section class="md-formula${wide?' md-wide':''}"><span class="eyebrow">${eyebrow}</span><h4>${title}</h4>${body}${more?`<details><summary>More detail</summary>${more}</details>`:''}</section>`;
      const term=(sign,name,text)=>`<div class="md-term"><b>${sign}</b><span>${name}</span><p>${text}</p></div>`;
      const nameParts=k=>{const m=/^(.*?)\s*\((.*)\)$/.exec(k);return m?`${esc(m[1])}<small>${esc(m[2])}</small>`:esc(k);};
      const windowErrors=Object.entries(v.reference_band?.window_errors||{});
      panel('how').innerHTML=`<div class="md-how">
        <section class="md-formula md-wide md-overview"><span class="eyebrow">The ranking in one line</span>
          <p class="md-eq">Ranking score = −Losses + Schedule credit + Head-to-head</p>
          <div class="md-terms">
            ${term('−','Losses',`Each FBS loss costs ${official.loss_aversion.toFixed(2)} point. Losses come first.`)}
            ${term('+','Schedule credit',`Σ(1 − q): the games a playoff-caliber team would be expected to lose on this schedule. A road win over an elite team adds a lot; a win over a weak team, almost nothing.`)}
            ${term('±','Head-to-head',`A small, capped nudge (at most ±${(win/2).toFixed(2)}) when two close résumés met on the field.`)}
          </div>
          <p class="md-fine">Every unit is one loss. A team with more losses ranks higher only when its schedule is worth more than one point per extra loss. Margin does not count${showPerf?' beyond a small capped convincingness term':''}, so running up the score earns nothing.</p></section>

        ${card('Who it is measured against','The playoff-caliber reference team',
          `<p>The teams ranked #${Math.max(1,official.benchmark_rank-4)}–#${official.benchmark_rank+4} at the end of ${esc(refFrom)}: <b>${refPts.toFixed(1)} ± ${refDist.sd.toFixed(1)} points</b> better than an average FBS team on a neutral field. Every team is judged against the same range; it never depends on anyone's current record.</p>
           ${drift!==null?`<p>This season's #8–#16 teams are ${drift>=0?'+':'−'}${Math.abs(drift).toFixed(1)} pts against past seasons at week ${wk}: ${Math.abs(drift)>past.sd?'unusual':'normal'}. A drift check only; it never enters a score.</p>`:''}`,
          `<p>q is a normal curve, so averaging over the range is exact: it widens the win-probability scale from ${cal.win_sigma.toFixed(1)} to ${Math.sqrt(cal.win_sigma**2+refDist.sd**2).toFixed(1)} points. The 80% score range uses a stronger and a weaker playoff-caliber team.</p>
           ${windowErrors.length?`<p>How many seasons to pool is picked by how well each choice anticipated the band a season actually ended with (mean miss, points):</p><table class="md-mini"><tbody>${windowErrors.map(([k,x])=>`<tr><td>${k==='all'?'Every earlier season':`Last ${k} seasons`}</td><td class="num">${Number(x).toFixed(2)}</td></tr>`).join('')}</tbody></table><p>The band bounces from year to year rather than trending, so pooling more seasons wins.</p>`:''}`)}

        ${card('What a win is worth','Opponent strength',
          `<p>From this season's completed FBS results, fitted with each team's own games removed. Beating an opponent cannot make it look weaker to you, and a better result can never lower your résumé score.</p>
           ${prior?`<p><b>Early season:</b> through week ${prior.decay_weeks}, opponents are anchored to their ${prior.season} ratings, fading to zero from week ${prior.decay_weeks+1}.${pw>0?` This week: ${Math.round(100*pw)}%.`:''}</p>`:''}`,
          es?`<table class="md-mini"><thead><tr><th>Held-out log loss</th><th class="num">No anchor</th><th class="num">Anchor</th></tr></thead><tbody>${es.test.slice(0,4).map(t=>`<tr><td>Weeks ${esc(t.weeks.replace('end','end'))}</td><td class="num">${t.log_loss_no_prior}</td><td class="num">${t.log_loss_prior}</td></tr>`).join('')}</tbody></table>
           <p>Accuracy alone would keep last season in all year. It is cut off on purpose, so from mid-season a résumé is built on this season alone. Calibrated on ${esc(om.test_seasons||'')} held out: log loss ${om.log_loss??'—'}, Brier ${om.brier??'—'}, ${om.test_games??'—'} games.</p>`:'')}

        ${card('Close résumés','Head-to-head',
          `<p>Inside ${win.toFixed(2)} (one SD of the noise in the score gap between nearby teams) two résumés are not reliably different, so the game between them is treated as evidence. A winner trailing by about 0.24 or less moves ahead; further back the gap only narrows; beyond ${win.toFixed(2)}, nothing.</p>
           ${hh?`<ul class="md-kpis"><li><b>${hh.orders_reversed}</b> winners placed above the team they beat</li><li><b>${hh.narrowed}</b> gaps narrowed, not reversed</li><li><b>${hh.third_party_top25}</b> top-25 pairs that never met reordered</li></ul><p class="md-fine">Final top 25s, ${esc(v.seasons||'')}.</p>`:''}`,
          `<p class="md-eq">gap closes by min(W, (W/√2)·φ(z)/Φ(z) × taper),&nbsp; z = −gap/(W√2)</p>
           <p>A capped Bayesian update: full strength while the winner trails by under ${(win/2).toFixed(2)}, fading to zero at ${win.toFixed(2)}. The winner gains half, the loser gives up half, and no team's total exceeds ±${(win/2).toFixed(3)}, so an extra win or tie always counts for more. One number per team, so the order is transitive and never depends on which pair is looked at first. The cost is the third-party reorders counted above. Average top-25 adjustment: ${hh?.mean_abs_adjustment_top25??'—'}.</p>
           ${hh?`<ul>${(hh.examples||[]).slice(0,6).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}`)}

        ${card('Shown beside the score','Résumé rarity',
          `<p class="md-eq">p = P(W* ≥ W),&nbsp; R = −log<sub>10</sub> p</p><p>How often the reference team would match or beat this record on this schedule, calculated exactly. A significance measure, not the ranking: it counts how many games you lost, not which ones.</p>`,
          `<p>W* is the reference team's win total on your schedule, an exact Poisson-binomial with no simulation noise, averaged at 25 points across the playoff-caliber range. ${v.rarity_agreement_at_loss_weight_1?`Rarity and the score order ${pctText(v.rarity_agreement_at_loss_weight_1)} of top-40 pairs the same way; the loss cost that agrees best with rarity is about ${v.implied_loss_weight_from_rarity}, which supports the one-point loss.`:''}</p>`)}

        ${bt?card(`Does it work? ${esc(v.seasons||'')}`,'Backtest against the committee',
          `<div class="md-table-wrap"><table class="md-table md-backtest"><thead><tr><th>Method</th><th class="num">CFP teams picked</th><th class="num">Committee top 4</th><th class="num">Committee top 12</th><th class="num">CFP games: higher-ranked won</th></tr></thead><tbody>
           ${Object.entries(bt).map(([k,x],i)=>`<tr class="${i===0?'md-highlight':''}"><td>${nameParts(k)}</td><td class="num">${esc(x.field)}</td><td class="num">${Number(x.top4).toFixed(2)}</td><td class="num">${Number(x.top12).toFixed(2)}</td><td class="num">${esc(x.cfp_games)}</td></tr>`).join('')}</tbody></table></div>
           <p class="md-fine">Final regular-season rankings under the format of the time (top 4; from 2024 five conference champions plus seven at-large). Most Deserving picks fields; it is not built to pick playoff winners, and a power rating predicts the games better. ESPN and SP+ values are post-bowl, so their CFP-game column has already seen the games.</p>`,
          `<table class="md-mini"><thead><tr><th>Method</th><th class="num">Rank correlation, committee top 25</th></tr></thead><tbody>${Object.entries(bt).map(([k,x])=>`<tr><td>${esc(k)}</td><td class="num">${x.spearman}</td></tr>`).join('')}</tbody></table>`,true):''}

        ${rb?card('Robustness','Does the answer depend on the details?',
          `<table class="md-mini md-checks"><thead><tr><th>Change</th><th class="num">Same top 12</th></tr></thead><tbody>
            ${Object.entries(rb.window).map(([k,x])=>`<tr><td>${k.startsWith('window ')?`Head-to-head band ${esc(k.slice(7))} (vs ${win.toFixed(2)})`:'Head-to-head turned off'}</td><td class="num">${x.top12_overlap} / 12</td></tr>`).join('')}
            <tr><td>Count FCS games (one pooled FCS opponent)</td><td class="num">${rb.fcs.top12_overlap} / 12</td></tr>
            <tr><td>Drop opponents that left FBS the next season</td><td class="num">${rb.left_fbs.top12_overlap} / 12</td></tr>
            <tr><td>Treat first- and second-year FBS teams as non-FBS</td><td class="num">${rb.transition.top12_overlap} / 12</td></tr>
            <tr><td>Vacate a top-12 team's best win</td><td class="num">that team −${Number(rb.vacated.mean_rank_drop).toFixed(1)} places</td></tr></tbody></table>
           <p class="md-fine">Average over ${esc(v.seasons||'')}. The one that matters is FCS: today an FCS loss costs nothing; counted, it would cost about a full loss. When a win is vacated, other teams barely move (${Number(rb.vacated.others_moving).toFixed(2)} top-25 teams per case move more than one place).</p>`):''}

        ${card('Calibrated vs assumed','What is fitted, and what is a choice',
          `<table class="md-mini md-params"><thead><tr><th>Setting</th><th>Value</th><th>Source</th></tr></thead><tbody>
            <tr><td>Opponent ratings, home field, win scale</td><td>home ${cal.home_field} pts · σ ${cal.win_sigma.toFixed(1)}</td><td><span class="md-kind">Fitted</span></td></tr>
            <tr><td>Playoff-caliber range</td><td>${refPts.toFixed(1)} ± ${refDist.sd.toFixed(1)}</td><td><span class="md-kind">Fitted</span></td></tr>
            <tr><td>Head-to-head band</td><td>${win.toFixed(2)}</td><td><span class="md-kind">Fitted</span> (noise SD)</td></tr>
            ${prior?`<tr><td>Early-season anchor</td><td>${prior.regression===1?'full':Math.round(100*prior.regression)+'%'} · ${prior.decay_weeks} weeks</td><td><span class="md-kind">Fitted</span> within a 5-week cap</td></tr>`:''}
            <tr><td>Centre of the playoff-caliber range</td><td>#${official.benchmark_rank}</td><td><span class="md-kind md-choice-k">Choice</span></td></tr>
            <tr><td>Cost per loss</td><td>${official.loss_aversion.toFixed(2)}</td><td><span class="md-kind md-choice-k">Choice</span></td></tr>
            <tr><td>Convincingness (margin)</td><td>${Math.round(100*official.convincingness_weight)}%</td><td><span class="md-kind md-choice-k">Choice</span></td></tr></tbody></table>
           <p class="md-fine">Every choice can be changed in Cutoff sensitivity. A cost of 1.00 keeps the score fair across schedules for a playoff-caliber team; at 1.50 the same team would lose ${Math.abs(v.schedule_bias_reference_caliber?.loss_weight_1_5??1).toFixed(2)} points just for a hard schedule. A convincingness term was tested and improved nothing.</p>`,
          `<p><b>History, ${esc(v.seasons||'')}, final top 25s:</b> one-loss over unbeaten in ${esc(ov['1 loss over unbeaten']?.model_a||'—')} pairs, two-loss over unbeaten in ${esc(ov['2 losses over unbeaten']?.model_a||'—')}, three-loss over unbeaten in ${esc(ov['3 losses over unbeaten']?.model_a||'—')}. Every two-loss case was against an unbeaten team with a very weak schedule:</p>
           <ul>${(v.two_loss_over_unbeaten||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`)}

        <p class="md-note">Separate systems: Power Rankings predict who wins; the projected CFP pages simulate the committee. Neither is used here. Conference, polls, recruiting and talent have no effect; last season's results enter only as the early-season anchor. <a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/MOST_DESERVING_V3.md" target="_blank" rel="noopener">Full technical report ↗</a></p></div>`;
    }

    const draws={ledger:drawLedger,h2h:drawH2H,sens:drawSens,how:drawHow},drawn=new Set();
    function show(k){
      host.querySelectorAll('[data-md-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.mdTab===k)));
      host.querySelectorAll('[data-md-panel]').forEach(p=>{p.hidden=p.dataset.mdPanel!==k;});
      if(!drawn.has(k)){draws[k]();drawn.add(k);}
    }
    host.querySelectorAll('[data-md-tab]').forEach(b=>b.addEventListener('click',()=>show(b.dataset.mdTab)));
    show('h2h');
    return {rows,sens};
  }
  root.DeservingLab={mount,explain,defaultPair,band,signed};
  if(typeof module!=='undefined')module.exports=root.DeservingLab;
})(typeof window!=='undefined'?window:globalThis);
