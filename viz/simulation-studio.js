/* Shared presentation for the simulation hub. Published probabilities stay separate
   from the explicit winner-only scenario state. */
(function(root){
  'use strict';
  const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=p=>root.PlayoffPaths.percent(p);
  const num=p=>Number.isFinite(p)?Math.round(p*100):null;
  const record=r=>`${r.wins}–${r.losses}${r.ties?'–'+r.ties:''}`;
  function brand(team,ctx){const c=ctx.crestOf(team);return `--team-color:${c.plate};--team-ink:${ctx.meta[team]?.onColor||'#fff'}`;}
  function badge(team,ctx){return `<span class="sim-logo" style="${brand(team,ctx)}"><img src="${ctx.crestOf(team).mark}" alt=""></span>`;}
  function masthead(kicker,title,description){return `<div class="sim-masthead"><span class="eyebrow">${kicker}</span><h2>${title}</h2><p>${description}</p></div>`;}
  function effect(game,team){const e=game.effects?.[team];if(!e)return {win:null,loss:null};return team===game.home?{win:e.home_win,loss:e.away_win}:{win:e.away_win,loss:e.home_win};}
  function range(win,loss){
    if(!Number.isFinite(win)||!Number.isFinite(loss))return '<span class="sim-muted">Not sampled for both results</span>';
    const lo=Math.min(win,loss)*100,delta=Math.abs(win-loss)*100;
    return `<div class="sim-range" aria-hidden="true"><i style="width:${lo}%"></i><b style="left:${lo}%;width:${delta}%"></b></div>`;
  }
  function paths(host,ctx){
    const pl=ctx.playoff,data=pl.record_odds,impacts=pl.game_impacts;
    if(!data){host.innerHTML='<div class="sim-empty">Playoff paths are waiting for the season simulation.</div>';return;}
    let showAllSwings=false;
    let mode='swings',team=null,query='',conference='',week=String([...new Set((impacts?.games||[]).map(g=>g.week))].sort((a,b)=>a-b)[0]||'all'),checkpoint='final',loss='';
    const byTeam=new Map(pl.teams.map(r=>[r.team,r]));
    const names=Object.keys(data.teams).sort((a,b)=>(byTeam.get(b)?.playoff||0)-(byTeam.get(a)?.playoff||0)||a.localeCompare(b));
    const weeks=[...new Set((impacts?.games||[]).map(g=>g.week))].sort((a,b)=>a-b);
    host.innerHTML=`<div class="sim-surface">${masthead('THE ROAD TO THE FIELD','Playoff paths','One season. Every route to the playoff. Explore the games that matter, then open a team to follow its possible finishes.')}<div class="sim-toolbar"><div class="sim-tabs" role="group" aria-label="Playoff paths view"><button data-path-mode="swings" aria-pressed="true">Weekly swings</button><button data-path-mode="teams" aria-pressed="false">Team explorer</button><button data-path-mode="losses" aria-pressed="false">Same losses</button></div><span class="sim-stamp">${pl.n_sims.toLocaleString()} simulated seasons · completed results locked</span></div><div class="sim-filters"><label>Find a team<input id="paths-search" type="search" placeholder="School name…"></label><label>Conference<select id="paths-conference"><option value="">All conferences</option>${[...new Set(names.map(t=>ctx.meta[t]?.conference))].sort().map(c=>`<option>${esc(c)}</option>`).join('')}</select></label><label id="paths-week-label">Week<select id="paths-week">${weeks.map(w=>`<option value="${w}">Week ${w}</option>`).join('')}<option value="all">All remaining weeks</option></select></label><label id="paths-loss-label" hidden>Total losses<select id="paths-loss"><option value="">Any loss total</option>${Array.from({length:13},(_,i)=>`<option value="${i}">${i} losses</option>`).join('')}</select></label></div><div id="paths-content" aria-live="polite"></div><p class="sim-footnote">Regular-season records include FCS opponents. Conditional odds use only the simulations with that record or game result. Small samples are labeled; 0% and 100% are simulation results, not guarantees. Swings are percentage points and cannot be added across games.</p></div>`;
    host.querySelector('#paths-week').value=week;
    const matches=t=>(!query||t.toLowerCase().includes(query))&&(!conference||ctx.meta[t]?.conference===conference);
    function teamButton(t,cls=''){return `<button class="sim-team-button ${cls}" data-path-team="${esc(t)}">${badge(t,ctx)}<span>${esc(t)}</span></button>`;}
    function swingTeam(g,t,feature=false){const e=effect(g,t),r=data.teams[t],base=byTeam.get(t)?.playoff;
      return `<article class="sim-swing-side" style="${brand(t,ctx)}">${teamButton(t)}<small>${r?`${r.current_wins}–${r.current_losses} · `:''}${pct(base)} now</small><div class="sim-swing-values"><span>${pct(e.loss)}</span><i>→</i><strong>${pct(e.win)}</strong></div>${range(e.win,e.loss)}<div class="sim-swing-caption"><span>with a loss</span><span>with a win</span></div>${!g.supported?'<small class="sim-sample">Small sample for one result</small>':''}</article>`;}
    function swingBoard(){
      const all=root.PlayoffPaths.rankGames((impacts?.games||[]).filter(g=>(week==='all'||g.week===Number(week))&&(matches(g.home)||matches(g.away))));
      if(!all.length)return '<div class="sim-empty">No remaining games match these filters.</div>';
      const [top,...rest]=showAllSwings?all:all.slice(0,9);
      return `<div class="sim-section-heading"><span class="eyebrow">PLAYOFF MACHINE · ${week==='all'?'REMAINING SCHEDULE':'WEEK '+week}</span><h3>Biggest playoff swings.</h3><p>Playoff chances with a loss, then a win.</p></div><p class="sim-overline">MOST ON THE LINE · ${esc(top.away)} AT ${esc(top.home)}</p><div class="sim-feature-game">${swingTeam(top,top.away,true)}${swingTeam(top,top.home,true)}</div><div class="sim-game-meta">${esc(top.date||'Date TBD')} · ${Number.isFinite(top.combined_swing)?(100*top.combined_swing).toFixed(1)+' pp combined swing':'Comparison not sampled'} · ${top.samples_home.toLocaleString()} home wins / ${top.samples_away.toLocaleString()} away wins</div><h4 class="sim-overline">ALSO SWINGING ${week==='all'?'THIS SEASON':'THIS WEEK'}</h4><div class="sim-swing-list">${rest.map(g=>`<article class="sim-swing-game"><div class="sim-game-heading"><span>Week ${g.week} · ${esc(g.date||'Date TBD')}</span><b>${esc(g.away)} at ${esc(g.home)}</b><small>${Number.isFinite(g.combined_swing)?(100*g.combined_swing).toFixed(1)+' pp combined':'Not sampled'}${!g.supported?' · small sample':''}</small></div><div class="sim-game-sides">${[g.away,g.home].map(t=>{const e=effect(g,t);return `<div class="sim-swing-row">${teamButton(t)}<div>${range(e.win,e.loss)}<small>loss → win</small></div><div class="sim-inline-values">${pct(e.loss)} <i>→</i> <strong>${pct(e.win)}</strong></div></div>`;}).join('')}</div></article>`).join('')}</div>${all.length>9?`<button class="sim-secondary sim-more" data-swings-more>${showAllSwings?'Show biggest swings only':'Show all '+all.length+' games'}</button>`:''}`;
    }
    function teamDirectory(){
      const filtered=names.filter(matches);
      return `<div class="sim-section-heading"><span class="eyebrow">${mode==='losses'?'A LEVEL COMPARISON':'FOLLOW A TEAM'}</span><h3>${mode==='losses'?'Same losses. Different paths.':'Every finish has a story.'}</h3><p>${mode==='losses'?'Choose a loss total to compare conditional playoff chances, then open a team for its full distribution.':'Select a team for its record histogram, most likely finish, playoff outcomes and remaining games.'}</p></div><div class="sim-team-grid">${filtered.map(t=>{
        const r=byTeam.get(t)||{},cells=data.teams[t].checkpoints.final||[],modes=root.PlayoffPaths.modalRecords(cells),cell=loss===''?null:cells.find(c=>c.losses===Number(loss));
        if(mode==='losses'&&loss!==''&&!cell)return '';
        return `<button class="sim-team-card" data-path-team="${esc(t)}" style="${brand(t,ctx)}">${badge(t,ctx)}<span class="sim-team-card-title">${esc(t)}<small>${esc(ctx.meta[t]?.conference)} · ${data.teams[t].current_wins}–${data.teams[t].current_losses} now</small></span><strong>${pct(mode==='losses'&&loss!==''?cell?.playoff:r.playoff)}</strong><span class="sim-card-caption">${mode==='losses'&&loss!==''?`${record(cell)} finish · ${cell.samples} simulations`:`Most likely: ${modes.map(record).join(' / ')||'Not sampled'}`}</span><span class="sim-card-footer">${mode==='losses'&&loss!==''?'CFP at this record':'Overall playoff chance'} <b>Explore →</b></span></button>`;
      }).join('')||'<div class="sim-empty">No reachable records match these filters.</div>'}</div>`;
    }
    function teamDetail(){
      const r=byTeam.get(team)||{},tr=data.teams[team],cells=tr.checkpoints.final||[],modes=root.PlayoffPaths.modalRecords(cells);
      const max=Math.max(...cells.map(c=>c.record_probability),.01);
      const histogram=cells.slice().sort((a,b)=>a.wins-b.wins).map(c=>`<button class="sim-hist-bar" data-record-wins="${c.wins}" title="${record(c)}: ${pct(c.record_probability)} of simulations, ${pct(c.playoff)} CFP"><span>${pct(c.record_probability)}</span><i style="height:${Math.max(2,c.record_probability/max*140)}px"></i><b>${record(c)}</b></button>`).join('');
      const scenarioCells=tr.checkpoints[checkpoint]||[];
      const ownGames=root.PlayoffPaths.rankGames((impacts?.games||[]).filter(g=>g.home===team||g.away===team));
      return `<button class="sim-back" data-path-back>← Back to ${mode==='swings'?'weekly swings':'all teams'}</button><div class="sim-team-hero" style="${brand(team,ctx)}">${badge(team,ctx)}<div><span class="eyebrow">${esc(ctx.meta[team]?.conference)} · ${tr.current_wins}–${tr.current_losses} NOW</span><h3>${esc(team)}</h3><p>${tr.remaining} regular-season games remain</p></div><div class="sim-hero-number"><strong>${pct(r.playoff)}</strong><span>to make the playoff</span></div></div><div class="sim-metrics">${[['Most likely finish',modes.map(record).join(' / ')||'Not sampled'],['First-round bye',pct(r.bye)],['Semifinal',pct(r.sf)],['Championship game',pct(r.final)],['National title',pct(r.champ)]].map(([k,v])=>`<div><small>${k}</small><b>${v}</b></div>`).join('')}</div><div class="sim-detail-grid"><section class="sim-panel"><span class="eyebrow">RECORD DISTRIBUTION</span><h4>How the regular season ends.</h4><p>Frequency of each win–loss record across ${pl.n_sims.toLocaleString()} simulations. Select a bar to highlight its conditional odds.</p><div class="sim-histogram">${histogram}</div></section><section class="sim-panel"><span class="eyebrow">YOUR ROUTE TO THE FIELD</span><h4>What each record means.</h4><label>Checkpoint<select id="paths-checkpoint">${data.checkpoints.map(c=>`<option value="${esc(c.id)}" ${c.id===checkpoint?'selected':''}>${esc(c.label)}</option>`).join('')}</select></label><div class="sim-record-list">${scenarioCells.map(c=>`<div class="sim-record-card" data-win-record="${c.wins}"><b>${record(c)}</b><div>${range(c.playoff,0)}</div><strong>${c.samples?pct(c.playoff):'Not sampled'}</strong><small>${c.samples} simulations${c.samples<data.min_samples?' · small sample':''}</small></div>`).join('')}</div></section></div><section class="sim-panel"><span class="eyebrow">THE GAMES AHEAD</span><h4>Where the season turns.</h4><div class="sim-swing-list">${ownGames.map(g=>{const e=effect(g,team),other=g.home===team?g.away:g.home;return `<div class="sim-swing-row"><span><small>Week ${g.week} · ${esc(g.date||'TBD')}</small><b>${g.home===team?'vs':'at'} ${esc(other)}</b></span>${teamButton(other)}<div>${range(e.win,e.loss)}<small>CFP with a loss → with a win</small></div><div class="sim-inline-values">${pct(e.loss)} → <strong>${pct(e.win)}</strong></div>${!g.supported?'<small class="sim-sample">Small sample</small>':''}</div>`;}).join('')||'<p class="sim-muted">No remaining games in the current simulation.</p>'}</div></section>`;
    }
    function draw(){
      host.querySelector('#paths-week-label').hidden=mode!=='swings'||Boolean(team);
      host.querySelector('#paths-loss-label').hidden=mode!=='losses'||Boolean(team);
      host.querySelectorAll('[data-path-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.pathMode===mode)));
      const content=host.querySelector('#paths-content');content.innerHTML=team?teamDetail():mode==='swings'?swingBoard():teamDirectory();
      content.querySelectorAll('[data-path-team]').forEach(b=>b.addEventListener('click',()=>{team=b.dataset.pathTeam;checkpoint='final';draw();content.scrollIntoView({block:'start',behavior:'smooth'});}));
      content.querySelector('[data-swings-more]')?.addEventListener('click',()=>{showAllSwings=!showAllSwings;draw();});
      content.querySelector('[data-path-back]')?.addEventListener('click',()=>{team=null;draw();});
      content.querySelector('#paths-checkpoint')?.addEventListener('change',e=>{checkpoint=e.target.value;draw();});
      content.querySelectorAll('[data-record-wins]').forEach(b=>b.addEventListener('click',()=>{checkpoint='final';draw();content.querySelector(`[data-win-record="${b.dataset.recordWins}"]`)?.classList.add('selected');}));
    }
    host.querySelectorAll('[data-path-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.pathMode;team=null;draw();}));
    host.querySelector('#paths-search').addEventListener('input',e=>{query=e.target.value.toLowerCase().trim();team=null;draw();});
    host.querySelector('#paths-conference').addEventListener('change',e=>{conference=e.target.value;team=null;draw();});
    host.querySelector('#paths-week').addEventListener('change',e=>{week=e.target.value;showAllSwings=false;draw();});
    host.querySelector('#paths-loss').addEventListener('change',e=>{loss=e.target.value;draw();});
    draw();
  }
  function scenario(host,ctx){
    const L=root.ScenarioLab,storageKey='cfb-scenario-v2-explicit';
    let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch{}
    const state={picks:saved.picks||{},titles:saved.titles||{},bracket:saved.bracket||{}};
    const valid=new Map(ctx.schedule.filter(g=>!L.final(g)).map(g=>[L.key(g),g]));
    for(const k of Object.keys(state.picks))if(!valid.has(k)||![valid.get(k).h,valid.get(k).a].includes(state.picks[k]))delete state.picks[k];
    let stage='schedule',query='',conference='',showCompleted=false,week=String([...new Set(ctx.schedule.filter(g=>!L.final(g)).map(g=>g.w))].sort((a,b)=>a-b)[0]||'all');
    const conf=t=>ctx.meta[t]?.conference;
    const save=()=>{try{localStorage.setItem(storageKey,JSON.stringify(state));}catch{}};
    const resetLater=()=>{state.titles={};state.bracket={};};
    const probability=(h,a,neutral)=>{const p=ctx.predict(h,a,neutral?'N':'A');return p&&Number.isFinite(p.pA)?p.pA:.5;};
    const gameProbability=g=>ctx.probability(g);
    const weeks=[...new Set(ctx.schedule.map(g=>g.w))].sort((a,b)=>a-b);
    host.innerHTML=`<div class="sim-surface">${masthead('YOUR PICKS. YOUR SEASON.','Scenario builder','Start with the real results. Choose every remaining winner, settle the conference titles, then build your own playoff. Nothing is picked for you.')}<div class="sim-toolbar"><div class="sim-tabs" role="group" aria-label="Scenario stage"><button data-scenario-stage="schedule" aria-pressed="true">1 · Remaining schedule</button><button data-scenario-stage="titles" aria-pressed="false">2 · Conference titles</button><button data-scenario-stage="bracket" aria-pressed="false">3 · Your playoff</button></div><button id="scenario-reset" class="sim-secondary">Clear my picks</button></div><div class="sim-progress" id="scenario-progress" aria-live="polite"></div><div class="sim-autofill"><div><b>Auto-select winners</b><small>Only when you choose. Existing picks are kept.</small></div><label class="sr-only" for="scenario-method">Auto-selection method</label><select id="scenario-method"><option value="favorite">Favorites</option><option value="underdog">Underdogs</option><option value="coin">Random · 50/50</option><option value="weighted">Random · weighted by win probability</option></select><label class="sr-only" for="scenario-scope">Auto-selection scope</label><select id="scenario-scope"><option value="visible">Unpicked games in this view</option><option value="all">All unpicked games in this stage</option></select><button id="scenario-fill" class="sim-primary">Apply selections</button></div><div class="sim-filters" id="scenario-filters"><label>Find a team<input type="search" id="scenario-search" placeholder="School name…"></label><label>Conference<select id="scenario-conference"><option value="">All conferences</option>${[...new Set(ctx.ratings.map(r=>r.conference))].sort().map(c=>`<option>${esc(c)}</option>`).join('')}</select></label><label>Week<select id="scenario-week"><option value="all">All weeks</option>${weeks.map(w=>`<option value="${w}">Week ${w}</option>`).join('')}</select></label><label class="sim-checkbox"><input type="checkbox" id="scenario-completed">Show locked results</label></div><div id="scenario-content"></div><p class="sim-footnote">Completed results are locked. No scores or winners are forecast in this builder. Autofill uses the football model only when explicitly selected. Changing a schedule pick resets the title and playoff selections. Conference-title ties use head-to-head, then current power ranking as a fallback; this simplified tie-break does not reproduce every conference’s full policy. The field follows the current committee baseline and CFP bid rules.</p></div>`;
    host.querySelector('#scenario-week').value=week;
    const evaluate=()=>L.evaluate({schedule:ctx.schedule,meta:ctx.meta,ratings:ctx.ratings,committee:ctx.committee,playoff:ctx.playoff,...state});
    function filtered(S){return S.rows.filter(r=>(showCompleted||!r.locked)&&(week==='all'||String(r.g.w)===week)&&(!conference||conf(r.g.h)===conference||conf(r.g.a)===conference)&&(!query||`${r.g.h} ${r.g.a}`.toLowerCase().includes(query)));}
    function choice(t,key,selected,locked=false,type='game'){return `<button class="sim-choice${selected?' selected':''}" style="${brand(t,ctx)}" data-pick-type="${type}" data-pick-key="${esc(key)}" data-pick-team="${esc(t)}" aria-pressed="${selected}" ${locked?'disabled':''}>${badge(t,ctx)}<b>${esc(t)}</b><span>${selected?(locked?'Final winner':'Your winner'):locked?(type==='bracket'?'Waiting for opponent':'Final result'):'Select winner'}</span></button>`;}
    function gameCards(S){const rows=filtered(S);return `<div class="sim-section-heading"><span class="eyebrow">THE REMAINING SCHEDULE</span><h3>Make the season yours.</h3><p>${S.remaining} unpicked games remain across the schedule. Pick either team, or use an auto-selection option above.</p></div><div class="sim-schedule-grid">${rows.map(({g,key,locked,winner})=>`<article class="sim-pick-game${locked?' is-locked':''}"><div class="sim-game-heading"><span>Week ${g.w} · ${esc(g.d||'Date TBD')}</span><small>${locked?'🔒 Completed · locked':winner?'Selected':'Awaiting your pick'}</small></div><div class="sim-choices">${choice(g.a,key,winner===g.a,locked)}<span class="sim-at">${g.n?'VS':'AT'}</span>${choice(g.h,key,winner===g.h,locked)}</div>${winner&&!locked?`<button class="sim-clear-one" data-clear-game="${esc(key)}">Clear selection</button>`:''}</article>`).join('')||'<div class="sim-empty">No games in this view. Try another week or clear your filters.</div>'}</div>`;}
    function titleCards(S){return `<div class="sim-section-heading"><span class="eyebrow">SETTLE THE CONFERENCES</span><h3>The titles are up to you.</h3><p>These matchups come from your selected regular-season results. Choose every title-game winner to unlock your playoff.</p></div><div class="sim-schedule-grid">${S.titleGames.map(t=>`<article class="sim-pick-game"><div class="sim-game-heading"><b>${esc(t.conf)} championship</b><small>${t.winner?'Selected':'Awaiting your pick'}</small></div><div class="sim-choices">${choice(t.h,t.conf,t.winner===t.h,false,'title')}<span class="sim-at">VS</span>${choice(t.a,t.conf,t.winner===t.a,false,'title')}</div></article>`).join('')}</div>`;}
    function bracketCards(S){
      if(!S.complete)return '<div class="sim-empty">Finish the schedule and conference titles to reveal your field. No playoff participants are assumed.</div>';
      const br=ctx.bracketFromSeeds(S.field.map(r=>r.team)),G=L.resolvePicks(br.games,br.feeds,state.bracket);
      const teamSlot=(t,g,i)=>t?choice(t,String(i),g.winner===t,!(g.top&&g.bottom),'bracket'):'<div class="sim-placeholder">Awaiting your earlier winner</div>';
      return `<div class="sim-section-heading"><span class="eyebrow">YOUR FIELD · YOUR WINNERS</span><h3>Write the final chapter.</h3><p>The field follows your results. Choose both sides of every round yourself; there are no default advances.</p></div><div class="sim-user-field">${S.field.map(r=>`<span style="${brand(r.team,ctx)}"><b>${r.seed}</b>${badge(r.team,ctx)}${esc(r.team)}<small>${esc(r.bid)} · ${r.wins}–${r.losses}</small></span>`).join('')}</div><div class="sim-user-bracket" tabindex="0" role="region" aria-label="Your winner-only playoff bracket">${[['First round',[1,0,2,3]],['Quarterfinals',[5,4,6,7]],['Semifinals',[8,9]],['Championship',[10]]].map(([label,indices])=>`<div class="sim-user-round"><h4>${label}</h4>${indices.map(i=>{const g=G[i];return `<article class="sim-pick-game"><small>${g.site?'Hosted by '+esc(g.site):'Neutral site'}</small>${teamSlot(g.top,g,i)}${teamSlot(g.bottom,g,i)}</article>`;}).join('')}</div>`).join('')}</div>${G[10].winner?`<div class="sim-your-champion"><span class="eyebrow">YOUR SCENARIO CHAMPION</span>${badge(G[10].winner,ctx)}<h3>${esc(G[10].winner)}</h3></div>`:'<p class="sim-empty">Your champion is waiting for your final selection.</p>'}<p class="sim-muted">First two out: ${S.out.map(esc).join(' · ')}</p>`;
    }
    function draw(){
      const S=evaluate();
      host.querySelector('#scenario-progress').innerHTML=`<div><b>${S.rows.filter(r=>r.locked).length}</b><span>results locked</span></div><div><b>${S.rows.filter(r=>!r.locked&&r.winner).length}</b><span>schedule picks made</span></div><div><b>${S.remaining}</b><span>schedule picks left</span></div><div><b>${S.titleGames.filter(t=>t.winner).length} / ${S.titleGames.length||'—'}</b><span>title winners chosen</span></div>`;
      host.querySelectorAll('[data-scenario-stage]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.scenarioStage===stage));b.disabled=b.dataset.scenarioStage==='titles'?Boolean(S.remaining):b.dataset.scenarioStage==='bracket'?!S.complete:false;});
      host.querySelector('#scenario-filters').hidden=stage!=='schedule';host.querySelector('#scenario-scope').hidden=stage!=='schedule';
      host.querySelector('#scenario-fill').disabled=stage==='schedule'?!S.remaining:stage==='titles'?!S.titleRemaining:!S.complete;
      const content=host.querySelector('#scenario-content');content.innerHTML=stage==='schedule'?gameCards(S):stage==='titles'?titleCards(S):bracketCards(S);
      content.querySelectorAll('[data-pick-team]').forEach(b=>b.addEventListener('click',()=>{
        const {pickTeam:t,pickKey:k,pickType:type}=b.dataset;
        if(type==='game'){state.picks[k]=state.picks[k]===t?null:t;resetLater();}
        else if(type==='title'){state.titles[k]=state.titles[k]===t?null:t;state.bracket={};}
        else{const previous=state.bracket[k];for(const ix of Object.keys(state.bracket))if(Number(ix)>=Number(k))delete state.bracket[ix];if(previous!==t)state.bracket[k]=t;}
        save();draw();
      }));
      content.querySelectorAll('[data-clear-game]').forEach(b=>b.addEventListener('click',()=>{delete state.picks[b.dataset.clearGame];resetLater();save();draw();}));
      if(stage==='schedule'&&!S.remaining){content.insertAdjacentHTML('beforeend','<button class="sim-primary sim-next" data-next-stage="titles">Continue to conference titles →</button>');}
      if(stage==='titles'&&S.complete)content.insertAdjacentHTML('beforeend','<button class="sim-primary sim-next" data-next-stage="bracket">Build my playoff →</button>');
      content.querySelectorAll('[data-next-stage]').forEach(b=>b.addEventListener('click',()=>{stage=b.dataset.nextStage;draw();host.scrollIntoView({behavior:'smooth'});}));
    }
    host.querySelectorAll('[data-scenario-stage]').forEach(b=>b.addEventListener('click',()=>{stage=b.dataset.scenarioStage;draw();}));
    host.querySelector('#scenario-fill').addEventListener('click',()=>{
      const S=evaluate(),method=host.querySelector('#scenario-method').value;
      if(stage==='schedule'){const rows=host.querySelector('#scenario-scope').value==='visible'?filtered(S):S.rows;for(const {g,key,locked,winner} of rows)if(!locked&&!winner)state.picks[key]=L.choose(g.h,g.a,gameProbability(g),method);resetLater();}
      else if(stage==='titles'){for(const t of S.titleGames)if(!t.winner)state.titles[t.conf]=L.choose(t.h,t.a,probability(t.h,t.a,true),method);state.bracket={};}
      else {const br=ctx.bracketFromSeeds(S.field.map(r=>r.team));for(let i=0;i<br.games.length;i++){const g=L.resolvePicks(br.games,br.feeds,state.bracket)[i];if(!g.winner&&g.top&&g.bottom)state.bracket[i]=L.choose(g.top,g.bottom,probability(g.top,g.bottom,!g.site),method);}}
      save();draw();
    });
    host.querySelector('#scenario-reset').addEventListener('click',()=>{state.picks={};resetLater();stage='schedule';save();draw();});
    host.querySelector('#scenario-search').addEventListener('input',e=>{query=e.target.value.toLowerCase().trim();draw();});
    host.querySelector('#scenario-conference').addEventListener('change',e=>{conference=e.target.value;draw();});
    host.querySelector('#scenario-week').addEventListener('change',e=>{week=e.target.value;draw();});
    host.querySelector('#scenario-completed').addEventListener('change',e=>{showCompleted=e.target.checked;draw();});
    draw();
  }
  root.SimulationStudio={paths,scenario,effect,range};
  if(typeof module!=='undefined')module.exports=root.SimulationStudio;
})(typeof window!=='undefined'?window:globalThis);
