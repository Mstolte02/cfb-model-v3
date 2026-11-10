/* Most Deserving résumé lab: ledger, head-to-head, cutoff sensitivity, method.
   Every number shown comes from RankingHistory.score, so the explanations always
   reconcile with the ranking. Predictive ratings and committee projections are not used. */
(function(root) {
  'use strict';
  const signed=(v,d=2)=>{const r=+Number(v).toFixed(d);return `${r>0?'+':r<0?'−':''}${Math.abs(r).toFixed(d)}`;};
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

  // Plain-English verdict for the page (explain() keeps the exact, numeric version).
  function plainVerdict(a,b) {
    const R=root.RankingHistory,c=R.compare(a,b),ahead=a.rank<b.rank?a:b.rank<a.rank?b:null;
    if(!ahead)return `${a.team} and ${b.team} are dead even and share a rank.`;
    const behind=ahead===a?b:a,sgn=ahead===a?1:-1,n=ahead.losses-behind.losses,words=k=>`${k===1?'one':k===2?'two':k===3?'three':k} more loss${k===1?'':'es'}`;
    if(c.decidedBy){
      const direct=c.meetings.length&&sgn*c.headToHeadNet>0;
      return direct?`${ahead.team} ranks ahead because it beat ${behind.team}. Their résumés are otherwise nearly even, so the game between them decides it.`
        :`${ahead.team} and ${behind.team} are nearly even. Head-to-head results against other teams tip it to ${ahead.team}.`;
    }
    const schedEdge=sgn*(c.schedule+c.performance);
    if(n>0)return `${ahead.team} ranks ahead despite ${words(n)}: it played a schedule tough enough to make up for ${n===1?'it':'them'}.`;
    if(n<0)return `${ahead.team} ranks ahead with ${words(-n).replace('more','fewer')}${schedEdge<0?`. ${behind.team}'s tougher schedule is not enough to make up the difference.`:', and a schedule at least as tough.'}`;
    return `Same number of losses, so the schedule decides it: ${ahead.team}'s results came against tougher opponents.`;
  }

  function mount(o) {
    const {host,ctx,model,esc,crestOf,wireTeamLinks}=o;
    const R=root.RankingHistory,official=model.assumptions;
    const rows=R.score(ctx,official),byTeam=new Map(rows.map(r=>[r.team,r]));
    const sens=R.sensitivity(ctx,model),configs=sens.configs.length;
    const crest=t=>{const c=crestOf(t);return `<span class="md-crest" style="--plate:${c.plate}"><img src="${c.mark}" alt="" loading="lazy"></span>`;};
    const teamBtn=(t,small='')=>`<button type="button" class="md-team team-link" data-team="${esc(t)}">${crest(t)}<span>${esc(t)}${small?`<small>${small}</small>`:''}</span></button>`;
    host.innerHTML=`<article class="md-lab">
      <header class="md-head"><span class="eyebrow">The résumé lab</span><h3>Who earned it, and why.</h3>
        <p>Compare any two résumés, see how sure each ranking is, and check who is safe at the cutoff.</p></header>
      <div class="md-tabs" role="tablist" aria-label="Résumé lab views">
        <button role="tab" aria-selected="true" data-md-tab="h2h">Why is A ahead of B?</button><button role="tab" aria-selected="false" data-md-tab="ledger">Résumé scores</button><button role="tab" aria-selected="false" data-md-tab="sens">Cutoff sensitivity</button><button role="tab" aria-selected="false" data-md-tab="how">How it works</button></div>
      <section class="md-panel" data-md-panel="h2h"></section>
      <section class="md-panel" data-md-panel="ledger" hidden></section>
      <section class="md-panel" data-md-panel="sens" hidden></section>
      <section class="md-panel" data-md-panel="how" hidden></section>
    </article>`;
    const panel=k=>host.querySelector(`[data-md-panel="${k}"]`);
    const tierInfo={playoff:['Playoff','Playoff-caliber'],top25:['Top 25','Top-25 caliber'],average:['Avg','Average FBS'],'below-average':['Weak','Below-average FBS']};
    const tierTag=g=>`<span class="md-tier md-tier-${g.tier}" title="${esc(tierInfo[g.tier]?.[1]||'')} opponent">${tierInfo[g.tier]?.[0]||''}</span>`;
    const tierLegend=`<p class="md-legend">Opponent: ${Object.entries(tierInfo).map(([k,[short,long]])=>`<span class="md-tier md-tier-${k}">${short}</span> ${long}`).join(' ')}</p>`;
    const where=g=>`${g.site==='away'?'at ':g.site==='neutral'?'vs ':''}${esc(g.opponent)}`;

    /* ---- Résumé scores: score plus the range it could reasonably take ---- */
    let showAll=false;
    function drawLedger(query='') {
      const q=query.toLowerCase().trim(),matches=rows.filter(r=>r.team.toLowerCase().includes(q));
      const list=q||showAll?matches:matches.filter(r=>r.rank<=25);
      const lo=Math.min(...list.map(r=>r.scoreLow)),hi=Math.max(...list.map(r=>r.scoreHigh));
      const span=Math.max(0.5,hi-lo),min=lo-span*0.04,max=hi+span*0.04,x=v=>`${((v-min)/(max-min)*100).toFixed(2)}%`;
      const step=span>4?1:0.5,ticks=[];for(let t=Math.ceil(min/step)*step;t<=max;t+=step)ticks.push(+t.toFixed(2));
      const axis=`<div class="md-axis-row">${ticks.map(t=>`<span style="left:${x(t)}">${t>0?'+':''}${t===0?'0':t.toFixed(step<1?1:0)}</span>`).join('')}</div>`;
      const bar=r=>`<div class="md-rng" title="${esc(`${r.team}: résumé score ${signed(r.score)} · likely range ${signed(r.scoreLow)} to ${signed(r.scoreHigh)}`)}">
        ${ticks.map(t=>`<i class="md-grid${t===0?' md-grid0':''}" style="left:${x(t)}"></i>`).join('')}
        <span class="md-rng-bar" style="left:${x(r.scoreLow)};width:calc(${x(r.scoreHigh)} - ${x(r.scoreLow)})"><em class="lo">${signed(r.scoreLow,1)}</em><em class="hi">${signed(r.scoreHigh,1)}</em></span>
        <b class="md-rng-dot" style="left:${x(r.score)}"></b></div>`;
      panel('ledger').innerHTML=`<div class="md-toolbar"><p>The bar is each team's likely range: how its score would change against a slightly stronger or weaker playoff-caliber benchmark. The dot is the actual score.</p><input type="search" placeholder="Find a team…" aria-label="Find a team" value="${esc(query)}"></div>
      <div class="md-table-wrap"><table class="md-table md-scores"><thead><tr><th>#</th><th>Team</th><th>Record</th><th class="num">Résumé score</th><th class="md-rng-head">Likely range${axis}</th></tr></thead>
      <tbody>${list.map(r=>`<tr class="${r.rank<=12?'md-in':''}${r.rank===12?' md-cutline':''}"><td class="md-rank">${r.rank}</td><td>${teamBtn(r.team)}</td><td>${record(r)}</td><td class="num md-score">${signed(r.score)}</td><td class="md-rng-cell">${bar(r)}</td></tr>`).join('')||`<tr><td colspan="5">No matching teams.</td></tr>`}</tbody></table></div>
      ${!q&&matches.length>25?`<button type="button" class="md-more">${showAll?'Show the top 25 only':`Show all ${matches.length} teams`}</button>`:''}`;
      const input=panel('ledger').querySelector('input');
      input.addEventListener('input',()=>{const pos=input.selectionStart;drawLedger(input.value);const n=panel('ledger').querySelector('input');n.focus();n.setSelectionRange(pos,pos);});
      panel('ledger').querySelector('.md-more')?.addEventListener('click',()=>{showAll=!showAll;drawLedger(query);});
      wireTeamLinks();
    }

    /* ---- Why is A ahead of B? ---- */
    let pair=defaultPair(rows);
    // Step chart: start at B's score, add each reason, land on A's score.
    function gapChart(a,b) {
      const c=R.compare(a,b),lossWord2=n=>`${Math.abs(n)} ${n>0?'more':'fewer'} loss${Math.abs(n)===1?'':'es'}`;
      const steps=[
        {label:'Losses',note:c.lossGap===0?'Same number of losses':`${a.team} has ${lossWord2(c.lossGap)}`,v:c.losses},
        {label:'Who they beat',note:'Credit for the quality of wins',v:c.scheduleFromWins},
        {label:'Who they lost to',note:'Losses to good teams cost less',v:c.scheduleFromLosses}];
      if(Math.abs(c.performance)>1e-9)steps.push({label:'Margin',note:'How convincingly games were won',v:c.performance});
      steps.push({label:'Head-to-head',note:'Nudge between close résumés',v:c.headToHead});
      let cum=b.score;const pts=[0,b.score,a.score];
      for(const st of steps){st.from=cum;cum+=st.v;st.to=cum;pts.push(cum);}
      const lo=Math.min(...pts),hi=Math.max(...pts),pad=(hi-lo||1)*0.08,min=lo-pad,max=hi+pad,x=v=>(v-min)/(max-min)*100;
      const plate=t=>crestOf(t).plate;
      const anchorRow=(r,label)=>`<div class="md-wf-row md-wf-anchor"><div class="md-wf-label">${crest(r.team)}<span><b>${esc(r.team)}</b><small>${label}</small></span></div>
        <div class="md-wf-track"><i class="md-wf-zero" style="left:${x(0)}%"></i><span class="md-wf-bar" style="left:${Math.min(x(0),x(r.score))}%;width:${Math.abs(x(r.score)-x(0))}%;background:${plate(r.team)}" title="${esc(r.team)} résumé score ${signed(r.score)}"></span><em style="left:${x(Math.max(0,r.score))}%">${signed(r.score)}</em></div></div>`;
      const stepRow=st=>{const fav=st.v>0.0005?a:st.v<-0.0005?b:null,l=Math.min(x(st.from),x(st.to)),w=Math.max(0.4,Math.abs(x(st.to)-x(st.from)));
        return `<div class="md-wf-row"><div class="md-wf-label"><span><b>${st.label}</b><small>${esc(st.note)}</small></span></div>
        <div class="md-wf-track"><i class="md-wf-zero" style="left:${x(0)}%"></i><i class="md-wf-link" style="left:${x(st.from)}%"></i>
        <span class="md-wf-bar ${fav===a?'up':fav===b?'down':'flat'}" style="left:${l}%;width:${w}%" title="${esc(`${st.label}: ${signed(st.v)}${fav?` in ${fav.team}'s favour`:''}`)}"></span>
        <em style="left:${l+w}%">${Math.abs(st.v)<0.005?'even':`${signed(st.v)} <small>${esc(fav.team)}</small>`}</em></div></div>`;};
      return `<div class="md-wf">${anchorRow(b,'starting point')}${steps.map(stepRow).join('')}${anchorRow(a,'where it ends up')}</div>`;
    }
    function drawH2H() {
      const [ta,tb]=pair,a=byTeam.get(ta),b=byTeam.get(tb);
      const options=sel=>rows.map(r=>`<option value="${esc(r.team)}"${r.team===sel?' selected':''}>#${r.rank} ${esc(r.team)} (${record(r)})</option>`).join('');
      if(!a||!b){panel('h2h').innerHTML='<p class="md-note">Two ranked teams are needed for a comparison.</p>';return;}
      const best=r=>r.ledger.filter(g=>g.result==='W').sort((x,y)=>y.total-x.total)[0],worst=r=>r.ledger.filter(g=>g.result==='L').sort((x,y)=>x.total-y.total)[0];
      const gameLine=g=>g?`${where(g)} <small>${g.pf}–${g.pa}</small>`:'—';
      const card=(r,side)=>`<div class="md-card md-${side}" style="--plate:${crestOf(r.team).plate}">${crest(r.team)}<div class="md-card-body"><span class="md-card-rank">#${r.rank} résumé</span><button type="button" class="team-link md-card-name" data-team="${esc(r.team)}">${esc(r.team)}</button>
        <dl><div><dt>Record</dt><dd>${record(r)}</dd></div><div><dt>Résumé score</dt><dd>${signed(r.score)}</dd></div>
        <div><dt>Best win</dt><dd class="md-dd-text">${gameLine(best(r))}</dd></div><div><dt>Worst loss</dt><dd class="md-dd-text">${gameLine(worst(r))}</dd></div></dl></div></div>`;
      const meet=a.ledger.filter(g=>g.opponent===b.team);
      const h2hLine=meet.length?`<p class="md-h2h-note">They played: ${meet.map(g=>`${g.result==='W'?a.team:b.team} won ${Math.max(g.pf,g.pa)}–${Math.min(g.pf,g.pa)}`).join('; ')}.</p>`:'';
      const receipt=r=>{const gs=[...r.ledger].sort((x,y)=>Math.abs(y.total)-Math.abs(x.total)),hh=r.headToHeadGames.filter(x=>Math.abs(x.value)>=0.005&&!x.cap);
        return `<div class="md-receipt" style="--plate:${crestOf(r.team).plate}"><h5>${crest(r.team)}${esc(r.team)}<span>${signed(r.score)}</span></h5><table><tbody>
        ${gs.map(g=>`<tr><td><span class="md-res md-${g.result}">${g.result}</span> ${where(g)} <span class="md-score-sm">${g.pf}–${g.pa}</span> ${tierTag(g)}</td><td class="num ${g.total>=0?'md-pos':'md-neg'}">${signed(g.total)}</td></tr>`).join('')}
        ${hh.map(x=>`<tr class="md-h2h-line"><td>Head-to-head nudge: ${x.won?'beat':'lost to'} ${esc(x.team)}</td><td class="num">${signed(x.value)}</td></tr>`).join('')}
        <tr class="md-sum md-total"><td>Résumé score</td><td class="num">${signed(r.score)}</td></tr></tbody></table></div>`;};
      panel('h2h').innerHTML=`<div class="md-pickers"><label>Team A<select data-md-pick="0">${options(ta)}</select></label><button type="button" class="md-swap" aria-label="Swap teams">⇄</button><label>Team B<select data-md-pick="1">${options(tb)}</select></label></div>
        <div class="md-versus">${card(a,'a')}<span class="md-vs">VS</span>${card(b,'b')}</div>
        <p class="md-verdict">${esc(plainVerdict(a,b))}</p>${h2hLine}
        <h4 class="md-section">Where the gap comes from</h4>${gapChart(a,b)}
        <h4 class="md-section">Game by game</h4><div class="md-receipts">${receipt(a)}${receipt(b)}</div>${tierLegend}`;
      panel('h2h').querySelectorAll('[data-md-pick]').forEach(s=>s.addEventListener('change',()=>{pair[+s.dataset.mdPick]=s.value;drawH2H();}));
      panel('h2h').querySelector('.md-swap').addEventListener('click',()=>{pair=[pair[1],pair[0]];drawH2H();});
      wireTeamLinks();
    }

    /* ---- Cutoff sensitivity ---- */
    function drawSens() {
      const strip=rows.filter(r=>r.rank<=20||sens.byTeam.get(r.team).min<=12).slice(0,22);
      const pos=k=>`${(Math.min(k,26)-1)/25*100}%`;
      const track=(s,r)=>`<span class="md-track"><i class="md-cut" title="12th place: the last playoff spot"></i>
        <span class="md-range" style="left:${pos(s.min)};width:calc(${pos(s.max)} - ${pos(s.min)})" title="${esc(`${r.team}: best #${s.min}, worst #${s.max} across ${configs} combinations`)}">${s.min===s.max?'':`<em class="lo">#${s.min}</em>`}<em class="hi">#${s.max>25?'25+':s.max}</em></span>
        <b class="md-dot" style="left:${pos(r.rank)}" title="${esc(`${r.team}: official rank #${r.rank}`)}"></b></span>`;
      panel('sens').innerHTML=`<div class="md-subpanel md-sens"><span class="eyebrow">Who is safe at the cutoff?</span><h4>Every version of the ranking</h4>
        <p>Each team's highest and lowest ranking among every combination of our résumé algorithm. The dot is the official rank, the line is 12th place, and the badge shows how many combinations result in a top-12 ranking.</p>
        <div class="md-strip">${strip.map(r=>{const s=sens.byTeam.get(r.team),bd=band(s.inTop),k=Math.round(s.inTop*configs);
          return `<div class="md-strip-row"><span class="md-rank">${r.rank}</span>${teamBtn(r.team)}${track(s,r)}<span class="md-share md-${bd.key}" title="${esc(`Top 12 in ${k} of ${configs} combinations`)}">${pctText(s.inTop)}<small>in Top 12</small></span></div>`;}).join('')}</div>
        <div class="md-axis2"><span></span><span></span><span class="md-axis-track"><b style="left:0">#1</b><b style="left:46%">Cutoff</b><b style="left:100%">#25+</b></span><span></span></div></div>`;
      wireTeamLinks();
    }

    /* ---- How it works: plain language only; the math lives in the technical report ---- */
    function drawHow() {
      const bt=model.validation?.committee_backtest,prod=bt&&Object.values(bt)[0];
      const item=(t,p)=>`<div class="md-how-item"><h5>${t}</h5><p>${p}</p></div>`;
      panel('how').innerHTML=`<div class="md-how-plain">
        ${item('What the score means','How many more games a team won than a playoff-caliber team would have won against the same schedule. Every loss costs a point. Beating good teams, especially on the road, earns points back.')}
        ${item('What counts','Only results on the field. Winning margin, polls, recruiting rankings and conference do not matter.')}
        ${item('Head-to-head','When two résumés are nearly even, the team that won the game between them gets a small nudge.')}
        ${item('Early in the season','For the first few weeks, opponents are judged partly on last season, so early rankings jump around less.')}
        ${prod?item('Does it work?',`Run on every season since 2014, it would have picked ${esc(prod.field)} of the actual playoff teams.`):''}
        <p class="md-note">Separate from the Power Ratings (who would win) and the projected bracket (what the committee will do). <a href="https://github.com/Mstolte02/cfb-model-v3/blob/main/audit/MOST_DESERVING_V3.md" target="_blank" rel="noopener">Technical report ↗</a></p></div>`;
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
  root.DeservingLab={mount,explain,plainVerdict,defaultPair,band,signed};
  if(typeof module!=='undefined')module.exports=root.DeservingLab;
})(typeof window!=='undefined'?window:globalThis);
