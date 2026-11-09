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
  // The head-to-head explanation. Components are the score's own components.
  function explain(a,b,assumptions) {
    const R=root.RankingHistory,c=R.compare(a,b),omega=assumptions.loss_aversion;
    const rows=[{label:c.lossGap===0?'Losses (equal)':`${Math.abs(c.lossGap)} ${c.lossGap>0?'additional':'fewer'} ${lossWord(c.lossGap)}`,value:c.losses,
        note:`Each FBS loss costs ${omega.toFixed(2)}`},
      {label:'Schedule quality',value:c.schedule,note:'Extra games a playoff-caliber team would expect to lose',children:[
        {label:'from games won (quality wins)',value:c.scheduleFromWins},{label:'from games lost (softens those losses)',value:c.scheduleFromLosses}]}];
    if(assumptions.convincingness_weight>0||Math.abs(c.performance)>1e-12)rows.push({label:'Performance',value:c.performance,note:'How clearly games were won or lost'});
    rows.push({label:'Net résumé advantage',value:c.net,total:true});
    let conclusion;
    const ahead=a.rank<b.rank?a:b.rank<a.rank?b:null,behind=ahead===a?b:a,window=assumptions.head_to_head_window||0;
    const games=c.meetings.map(g=>`${g.result==='W'?a.team:b.team} won ${Math.max(g.pf,g.pa)}–${Math.min(g.pf,g.pa)}${g.date?` (${String(g.date).slice(5).replace('-','/')})`:''}`);
    let headToHead=null;
    if(c.meetings.length){
      const gap=Math.abs(c.net),winner=c.headToHeadNet>0?a:c.headToHeadNet<0?b:null;
      const why=!winner?'the series is split, so it decides nothing':c.decidedBy?`it decides the order because the scores are ${gapText(gap)} apart, inside the ${window.toFixed(2)} closeness window`
        :winner===ahead?`${winner.team} is already ahead on résumé score`:gap>=window?`the scores are ${gap.toFixed(2)} apart, more than the ${window.toFixed(2)} closeness window`
        :`other teams sit between them in the order, and head-to-head only swaps neighbours`;
      headToHead=`Head-to-head: ${games.join('; ')}. ${why[0].toUpperCase()+why.slice(1)}.`;
    }
    if(!ahead)conclusion=`${a.team} and ${b.team} have equal résumé scores and share a rank.`;
    else if(c.decidedBy){
      conclusion=`${ahead.team} ranks ahead on head-to-head. ${behind.team}'s résumé score is ${gapText(c.net)} higher, but ${ahead.team} beat ${behind.team}, and the two scores are within the ${window.toFixed(2)} closeness window, where the résumés are not reliably different.`;
    }
    else{
      const lossGap=ahead.losses-behind.losses,edge=Math.abs(c.net),quality=Math.abs(c.schedule+c.performance);
      if(lossGap>0)conclusion=`${ahead.team} earned the higher ranking despite ${lossGap} more ${lossWord(lossGap)}: `+
        `its schedule and results are worth ${quality.toFixed(2)} loss equivalents more, which clears the ${(omega*lossGap).toFixed(2)} cost of the extra ${lossWord(lossGap)} by ${edge.toFixed(2)}.`;
      else if(lossGap<0)conclusion=`${ahead.team} ranks ahead with ${-lossGap} fewer ${lossWord(-lossGap)}. `+
        `${behind.team}'s schedule edge (${Math.max(0,(behind===a?1:-1)*(c.schedule+c.performance)).toFixed(2)}) is not enough to overcome the ${(omega*-lossGap).toFixed(2)} loss gap.`;
      else conclusion=`Same number of losses, so schedule quality decides it: ${ahead.team} leads by ${edge.toFixed(2)} loss equivalents.`;
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
    const h2hTag=r=>r.headToHeadPassed?.length?` <span class="md-h2h-tag" title="${esc('Ahead of '+r.headToHeadPassed.map(p=>p.team).join(', ')+' on head-to-head (scores within '+win.toFixed(2)+')')}">H2H</span>`:'';
    const crest=t=>{const c=crestOf(t);return `<span class="md-crest" style="--plate:${c.plate}"><img src="${c.mark}" alt="" loading="lazy"></span>`;};
    const teamBtn=(t,small='')=>`<button type="button" class="md-team team-link" data-team="${esc(t)}">${crest(t)}<span>${esc(t)}${small?`<small>${small}</small>`:''}</span></button>`;
    const showPerf=official.convincingness_weight>0;
    const grid=model.sensitivity||{};
    host.innerHTML=`<article class="md-lab">
      <header class="md-head"><span class="eyebrow">The résumé lab</span><h3>Earned on the field. Explained to the decimal.</h3>
        <p>Every score answers one question: how many more games did this team win than a playoff-caliber team would be expected to win against the same schedule, at the same venues? An extra loss costs one full point. Schedule difficulty can pay that point back, but only when the schedule is worth more than a whole extra expected loss.</p>
        <p class="md-stamp">Reference team: a playoff-caliber team, like the #${Math.max(1,official.benchmark_rank-4)}–#${official.benchmark_rank+4} teams of earlier seasons: ${refPts.toFixed(1)} ± ${refDist.sd.toFixed(1)} points better than an average FBS team on a neutral field · Head-to-head orders neighbours within ${win.toFixed(2)} · ${esc(games_through||'Completed games to date')}</p></header>
      <div class="md-tabs" role="tablist" aria-label="Résumé lab views">
        <button role="tab" aria-selected="true" data-md-tab="ledger">Résumé ledger</button><button role="tab" aria-selected="false" data-md-tab="h2h">Why is A ahead?</button><button role="tab" aria-selected="false" data-md-tab="sens">Cutoff sensitivity</button><button role="tab" aria-selected="false" data-md-tab="how">How it works</button></div>
      <section class="md-panel" data-md-panel="ledger"></section>
      <section class="md-panel" data-md-panel="h2h" hidden></section>
      <section class="md-panel" data-md-panel="sens" hidden></section>
      <section class="md-panel" data-md-panel="how" hidden></section>
    </article>`;
    const panel=k=>host.querySelector(`[data-md-panel="${k}"]`);

    /* ---- Ledger ---- */
    const gameRows=r=>r.ledger.map(g=>`<tr><td>${esc(String(g.date||'').slice(5).replace('-','/'))}</td><td>${g.site==='away'?'at ':g.site==='neutral'?'vs ':''}${esc(g.opponent)}${g.site==='neutral'?' <i>(neutral)</i>':''}</td>
      <td class="md-res md-${g.result}">${g.result} ${g.pf}–${g.pa}</td><td>${pctText(g.refWin)}</td>
      <td class="${g.total>=0?'md-pos':'md-neg'}">${signed(g.total)}</td><td class="md-why">${g.result==='W'?`Credit: a playoff-caliber team loses this game ${pctText(1-g.refWin)} of the time`:g.result==='L'?`Cost: −${official.loss_aversion.toFixed(2)} loss, ${signed(g.scheduleTerm)} back for difficulty`:'Tie: half a loss'}</td></tr>`).join('');
    function drawLedger(query='') {
      const list=rows.filter(r=>r.team.toLowerCase().includes(query.toLowerCase().trim()));
      panel('ledger').innerHTML=`<div class="md-toolbar"><p>Résumé score = losses + schedule credit${showPerf?' + performance':''}. Select <b>Games</b> to see every game's contribution.</p><input type="search" placeholder="Find a team…" aria-label="Find a team in the résumé ledger" value="${esc(query)}"></div>
      <div class="md-table-wrap"><table class="md-table"><thead><tr><th>Rank</th><th>Team</th><th>FBS record</th><th title="−${official.loss_aversion} per FBS loss">Losses</th><th title="Games a playoff-caliber team would expect to lose on this schedule">Schedule credit</th>${showPerf?'<th>Performance</th>':''}<th>Résumé score</th><th title="Expected record of the playoff-caliber reference team on this schedule">Reference team</th><th title="Chance the reference team matches or beats this record. Not this team's playoff chance.">Match or beat</th><th title="−log10 of the match-or-beat chance">Rarity</th><th title="Share of the playoff-caliber reference range under which this team ranks in the top 12">Top-12 chance</th><th></th></tr></thead>
      <tbody>${list.map(r=>{return `<tr class="${r.rank<=12?'md-in':''}"><td class="md-rank">${r.rank}${h2hTag(r)}</td><td>${teamBtn(r.team)}</td><td>${record(r)}</td>
        <td class="md-neg">${signed(r.lossPenalty)}</td><td class="md-pos">${signed(r.schedule)}<small>wins ${signed(r.scheduleFromWins)} · losses ${signed(r.scheduleFromLosses)}</small></td>${showPerf?`<td>${signed(r.performance)}</td>`:''}
        <td class="md-score">${signed(r.score)}${range(r)}</td><td>${r.referenceWins.toFixed(1)}–${r.schedule.toFixed(1)}</td><td>${pctText(r.matchProbability,1)}<small>${pctText(r.matchProbabilityLow,0)}–${pctText(r.matchProbabilityHigh,0)}</small></td><td>${r.rarity.toFixed(2)}<small>#${r.rarityRank}</small></td>
        <td><span class="md-share md-${band(r.topProbability).key}">${pctText(r.topProbability)}</span></td><td><button type="button" class="md-expand" aria-expanded="false" data-md-games="${esc(r.team)}">Games</button></td></tr>
        <tr class="md-games" hidden data-md-games-for="${esc(r.team)}"><td colspan="${showPerf?12:11}"><table><thead><tr><th>Date</th><th>Opponent</th><th>Result</th><th>Reference team wins</th><th>Game value</th><th></th></tr></thead><tbody>${gameRows(r)}</tbody></table></td></tr>`;}).join('')||`<tr><td colspan="12">No matching teams.</td></tr>`}</tbody></table></div>
      <p class="md-note">“Playoff-caliber” is a range, not one team: every score is averaged over that range, and the small 80% figures show the score against a stronger and a weaker playoff-caliber team. “Reference team” is the expected record of a playoff-caliber team against the same opponents and venues. “Match or beat” is that team's chance to finish with at least this many wins (its range below it): the lower it is, the rarer the résumé. “Top-12 chance” is the share of the playoff-caliber range under which the team ranks in the top 12. None of these is the team's chance of making the playoff. <b>H2H</b> marks a team placed directly above a neighbour it beat, with scores within ${win.toFixed(2)}. Records exclude FCS games.</p>`;
      const input=panel('ledger').querySelector('input');
      input.addEventListener('input',()=>{const pos=input.selectionStart;drawLedger(input.value);const n=panel('ledger').querySelector('input');n.focus();n.setSelectionRange(pos,pos);});
      panel('ledger').querySelectorAll('[data-md-games]').forEach(b=>b.addEventListener('click',()=>{
        const tr=panel('ledger').querySelector(`[data-md-games-for="${CSS.escape(b.dataset.mdGames)}"]`),open=tr.hidden;tr.hidden=!open;b.setAttribute('aria-expanded',String(open));b.textContent=open?'Hide':'Games';}));
      wireTeamLinks();
    }

    /* ---- Head to head ---- */
    let pair=defaultPair(rows);
    function drawH2H() {
      const [ta,tb]=pair,a=byTeam.get(ta),b=byTeam.get(tb);
      const options=sel=>rows.map(r=>`<option value="${esc(r.team)}"${r.team===sel?' selected':''}>#${r.rank} ${esc(r.team)} (${record(r)})</option>`).join('');
      if(!a||!b){panel('h2h').innerHTML='<p class="md-note">Two ranked teams are needed for a comparison.</p>';return;}
      const e=explain(a,b,official),max=Math.max(.5,...e.rows.map(x=>Math.abs(x.value)));
      const bar=x=>`<span class="md-bar-track"><i class="md-zero"></i><span class="md-bar ${x>=0?'pos':'neg'}" style="width:${Math.abs(x)/max*50}%;${x>=0?'left:50%':'right:50%'}"></span></span>`;
      const card=(r,side)=>`<div class="md-card md-${side}" style="--plate:${crestOf(r.team).plate}">${crest(r.team)}<div><span class="md-card-rank">#${r.rank} résumé</span><button type="button" class="team-link md-card-name" data-team="${esc(r.team)}">${esc(r.team)}</button>
        <dl><div><dt>FBS record</dt><dd>${record(r)}</dd></div><div><dt>Résumé score</dt><dd>${signed(r.score)}${r.scoreLow!==undefined?`<small class="md-card-range">${signed(r.scoreLow)} to ${signed(r.scoreHigh)}</small>`:''}</dd></div><div><dt>Reference team</dt><dd>${r.referenceWins.toFixed(1)}–${r.schedule.toFixed(1)}</dd></div><div><dt>Match or beat</dt><dd>${pctText(r.matchProbability,1)}</dd></div></dl></div></div>`;
      const best=r=>{const w=r.ledger.filter(g=>g.result==='W').sort((x,y)=>y.total-x.total).slice(0,2),l=r.ledger.filter(g=>g.result==='L').sort((x,y)=>x.total-y.total).slice(0,2);
        return `<li><b>${esc(r.team)}</b> — best wins: ${w.map(g=>`${esc(g.opponent)} ${signed(g.total)}`).join(', ')||'none'}; costliest losses: ${l.map(g=>`${esc(g.opponent)} ${signed(g.total)}`).join(', ')||'none'}</li>`;};
      panel('h2h').innerHTML=`<div class="md-pickers"><label>Team A<select data-md-pick="0">${options(ta)}</select></label><button type="button" class="md-swap" aria-label="Swap teams">⇄</button><label>Team B<select data-md-pick="1">${options(tb)}</select></label></div>
        <div class="md-versus">${card(a,'a')}<span class="md-vs">VS</span>${card(b,'b')}</div>
        <table class="md-compare"><thead><tr><th>Component</th><th>${esc(a.team)} advantage</th><th class="md-barcol"><span>favours ${esc(b.team)}</span><span>favours ${esc(a.team)}</span></th></tr></thead><tbody>
        ${e.rows.map(x=>`<tr class="${x.total?'md-total':''}"><td>${esc(x.label)}${x.note?`<small>${esc(x.note)}</small>`:''}</td><td class="${x.value>=0?'md-pos':'md-neg'}">${signed(x.value)}</td><td class="md-barcol">${bar(x.value)}</td></tr>${(x.children||[]).map(ch=>`<tr class="md-child"><td>${esc(ch.label)}</td><td>${signed(ch.value)}</td><td class="md-barcol"></td></tr>`).join('')}`).join('')}</tbody></table>
        ${e.headToHead?`<p class="md-h2h-note">${esc(e.headToHead)}</p>`:''}<p class="md-verdict">${esc(e.conclusion)}</p><p class="md-note">${esc(e.rarity)} This is a measure of how hard each record is to reproduce, not either team's chance of making the playoff. The net advantage equals the difference in résumé scores exactly; head-to-head never changes a score, only the order of close neighbours.</p>
        <ul class="md-drivers">${best(a)}${best(b)}</ul>`;
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
      const ov=v.overrides||{},om=v.opponent_model||{};
      panel('how').innerHTML=`<div class="md-how">
        <div class="md-formula"><span class="eyebrow">The score</span><p class="md-eq">Résumé score = −(FBS losses) + Σ (1 − q<sub>g</sub>)</p>
          <p>q<sub>g</sub> is the chance that a playoff-caliber team wins game g: the same opponent at the same venue. Σ(1 − q) is the number of games that team would be expected to lose on your schedule. The score is wins above a playoff-caliber team: W − Σq. Every unit is one loss.</p>
          <p><b>Conditional override.</b> Team A, with more losses, ranks above team B only when A's schedule credit is more than one point higher for each extra loss. There are no pairwise rules, so the ranking is always transitive.</p>
          <p><b>Each game, in loss units.</b> A win adds 1 − q: a road win over an elite team adds a lot, and a win over a weak team adds almost nothing. A loss costs q: a loss to a weak team costs nearly a full point, and a narrow road loss to a champion costs much less. Margin does not count${showPerf?' beyond a small capped convincingness term':''}, so running up the score earns nothing.</p></div>
        <div class="md-formula"><span class="eyebrow">Résumé rarity</span><p class="md-eq">p = P(W* ≥ W),&nbsp; R = −log<sub>10</sub> p</p>
          <p>W* is the number of games the reference team wins on your schedule. Its distribution is calculated exactly (Poisson-binomial), with no simulation noise. R is shown next to every score as a significance measure. It is a one-sided tail probability under a stated reference model. It is not proof that a team is deserving, and no 5% threshold is used.</p>
          <p>Why it is not the ranking: rarity counts only how many games you lost, not which ones. A loss to the worst FBS team would count the same as a loss to the best. ${v.rarity_agreement_at_loss_weight_1?`Historically, rarity and the score order ${pctText(v.rarity_agreement_at_loss_weight_1)} of top-40 pairs the same way, and the loss cost that agrees best with rarity is about ${v.implied_loss_weight_from_rarity}, so rarity supports the one-point loss cost rather than a heavier one.`:''}</p></div>
        <div class="md-formula"><span class="eyebrow">The reference team</span>
          <p>Playoff-caliber is a range, not one number: the end-of-season strengths of the teams ranked #${Math.max(1,official.benchmark_rank-4)}–#${official.benchmark_rank+4} in earlier seasons (${esc(v.seasons||'')}), pooled. That is ${refPts.toFixed(1)} ± ${refDist.sd.toFixed(1)} points better than an average FBS team on a neutral field. The spread covers both how different playoff-caliber teams are and how the bar moves from year to year. Every team is judged against the same range, and it never depends on anyone's current record.</p><p>The score is the average over that range. Each game's win chance is a normal curve, so the average is exact: the range widens the win-probability scale from ${cal.win_sigma.toFixed(1)} to ${Math.sqrt(cal.win_sigma**2+refDist.sd**2).toFixed(1)} points. Rarity is averaged the same way (exact Poisson-binomial at 25 points across the range). The 80% score range uses a stronger and a weaker playoff-caliber team.</p>
          <p>Opponent strength comes only from this season's completed FBS results, fitted with each team's own games removed. Your result against an opponent therefore cannot make that opponent look weaker or stronger to you, and a better result can never lower your score.</p></div>
        <div class="md-formula"><span class="eyebrow">Head-to-head</span><p>When two neighbours in the order played each other, and their scores differ by less than ${win.toFixed(2)}, the team that won the game (on balance) is placed first. Scores do not change. ${win.toFixed(2)} is one standard deviation of the measured noise in the score gap between nearby teams: closer than that, two résumés are not reliably different, so the result on the field decides.</p><p>Only neighbours swap, and each pair swaps once, so a three-way cycle cannot loop and no third team moves because of a game it did not play.${v.head_to_head?` Across ${esc(v.seasons||'')}, it reversed ${v.head_to_head.orders_reversed} final top-25 orders, for example:`:''}</p>${v.head_to_head?`<ul>${(v.head_to_head.examples||[]).slice(0,6).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}</div>
        <div class="md-formula"><span class="eyebrow">Calibrated vs assumed</span>
          <p><b>Calibrated on history</b> (chronological, ${esc(om.test_seasons||'')} held out): opponent ratings, home field (${cal.home_field} pts), the win-probability scale, the playoff-caliber range and the head-to-head window. Held-out log loss ${om.log_loss??'—'}, Brier ${om.brier??'—'}, ${om.test_games??'—'} games.</p>
          <p><b>Assumed, and shown in Cutoff sensitivity:</b> the centre of the playoff-caliber range (#${official.benchmark_rank}), head-to-head on, cost per loss ${official.loss_aversion.toFixed(2)}, convincingness ${Math.round(100*official.convincingness_weight)}%. A cost of 1.00 keeps the score fair across schedules for a playoff-caliber team. At 1.50, the same team would lose ${Math.abs(v.schedule_bias_reference_caliber?.loss_weight_1_5??1).toFixed(2)} points just for playing the hardest schedules. A convincingness term was tested; it did not improve any measure, so it is off.</p>
          <p><b>History, ${esc(v.seasons||'')} final regular seasons, top 25:</b> a one-loss team ranked above an unbeaten team in ${esc(ov['1 loss over unbeaten']?.model_a||'—')} pairs, two-loss over unbeaten in ${esc(ov['2 losses over unbeaten']?.model_a||'—')}, three-loss over unbeaten in ${esc(ov['3 losses over unbeaten']?.model_a||'—')}. Every two-loss case was against an unbeaten team with a very weak schedule:</p>
          <ul>${(v.two_loss_over_unbeaten||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div>
        <p class="md-note">Separate systems: Power Rankings predict who wins; the projected CFP pages simulate the committee. Neither is used here, and this page does not change them. Conference, polls, recruiting, talent and preseason ratings have no effect. <a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/MOST_DESERVING_V2.md" target="_blank" rel="noopener">Full technical report ↗</a></p></div>`;
    }

    const draws={ledger:drawLedger,h2h:drawH2H,sens:drawSens,how:drawHow},drawn=new Set();
    function show(k){
      host.querySelectorAll('[data-md-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.mdTab===k)));
      host.querySelectorAll('[data-md-panel]').forEach(p=>{p.hidden=p.dataset.mdPanel!==k;});
      if(!drawn.has(k)){draws[k]();drawn.add(k);}
    }
    host.querySelectorAll('[data-md-tab]').forEach(b=>b.addEventListener('click',()=>show(b.dataset.mdTab)));
    show('ledger');
    return {rows,sens};
  }
  root.DeservingLab={mount,explain,defaultPair,band,signed};
  if(typeof module!=='undefined')module.exports=root.DeservingLab;
})(typeof window!=='undefined'?window:globalThis);
