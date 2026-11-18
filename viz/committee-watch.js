/* Committee Watch: weekly predictions from the same historical transition model
   used by the playoff simulation. The published Top 25 remains exact until new
   results are complete. Legacy proxy helpers remain for older snapshots. */
(function(root) {
  'use strict';
  const DEFAULTS={weights:{win_pct:10.3397,rating:0.2262,sos:0.7362,power_conf:1.0864},fcsOppRating:-2,
    p4:['ACC','Big 12','Big Ten','SEC'],powerIndependents:['Notre Dame']};
  const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
  const sampleSd=a=>{const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/Math.max(1,a.length-1));};
  const day=g=>String(g.d||'').slice(0,10);
  const final=g=>g.f&&Number.isFinite(g.hp)&&Number.isFinite(g.ap)&&g.st!=='postseason';

  const shift=(date,n)=>{const d=new Date(date+'T12:00:00Z');d.setUTCDate(d.getUTCDate()+n);return d.toISOString().slice(0,10);};
  // Calendar boundaries always use Eastern Time, independent of the viewer's zone.
  function weeklyWindow(now=new Date()) {
    const parts=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(new Date(now));
    const part=t=>parts.find(p=>p.type===t).value;
    const today=`${part('year')}-${part('month')}-${part('day')}`;
    const weekday=new Date(today+'T12:00:00Z').getUTCDay();
    const active=weekday===0||weekday===1;
    // Sunday/Monday cover Tuesday through today. Other days hold the official board.
    const baseline=shift(today,weekday===0?-6:weekday===1?-7:1-weekday);
    return {today,active,baseline,through:active?today:baseline};
  }

  // The schedule-only terms, as static_committee() in simulate_playoff.R.
  function staticScores(teams,schedule,o) {
    const names=teams.map(t=>t.team),ratings=teams.map(t=>t.rating),m=mean(ratings),s=sampleSd(ratings);
    const z=new Map(names.map((t,i)=>[t,(ratings[i]-m)/s]));
    const opp=t=>z.has(t)?z.get(t):o.fcsOppRating;
    const sos=names.map(t=>{const v=[];for(const g of schedule){if(g.h===t)v.push(opp(g.a));else if(g.a===t)v.push(opp(g.h));}return v.length?mean(v):0;});
    const sm=mean(sos),ss=sampleSd(sos);
    return new Map(teams.map((t,i)=>{
      const p4=o.p4.includes(t.conference)||o.powerIndependents.includes(t.team)?1:0;
      return [t.team,{ratingZ:z.get(t.team),sosZ:(sos[i]-sm)/ss,p4,static:o.weights.rating*z.get(t.team)+o.weights.sos*(sos[i]-sm)/ss+o.weights.power_conf*p4}];
    }));
  }
  // Win share through a date (null = every completed game), FCS games included, ties half.
  function winPct(names,schedule,through) {
    const w=new Map(names.map(t=>[t,0])),n=new Map(names.map(t=>[t,0]));
    for(const g of schedule){
      if(!final(g)||(through&&day(g)>through))continue;
      for(const t of [g.h,g.a])if(n.has(t))n.set(t,n.get(t)+1);
      if(g.hp===g.ap){for(const t of [g.h,g.a])if(w.has(t))w.set(t,w.get(t)+.5);}
      else{const winner=g.hp>g.ap?g.h:g.a;if(w.has(winner))w.set(winner,w.get(winner)+1);}
    }
    return new Map(names.map(t=>[t,n.get(t)?w.get(t)/n.get(t):.5]));
  }
  function record(schedule,team,through,after=false) {
    let w=0,l=0,t=0;
    for(const g of schedule){if(!final(g)||(g.h!==team&&g.a!==team))continue;const d=day(g);if(through&&(after?d<=through:d>through))continue;
      const pf=g.h===team?g.hp:g.ap,pa=g.h===team?g.ap:g.hp;if(pf>pa)w++;else if(pf<pa)l++;else t++;}
    return `${w}–${l}${t?'–'+t:''}`;
  }
  function timelineOrder(scores,wins,bonus) {
    const remaining=new Set(scores.keys()),placed=[];let carry=[];
    while(remaining.size){
      const available=[...remaining].filter(t=>!carry.includes(t)).sort((a,b)=>scores.get(a)-scores.get(b)||a.localeCompare(b));
      const pool=[...carry,...available.slice(0,Math.max(0,6-carry.length))];
      const adjusted=new Map(pool.map(t=>[t,scores.get(t)-bonus*pool.filter(o=>o!==t&&wins.has(t+'\r'+o)).length]));
      const ordered=pool.sort((a,b)=>adjusted.get(a)-adjusted.get(b)||scores.get(a)-scores.get(b)||a.localeCompare(b));
      const take=placed.length<16?4:3;
      placed.push(...ordered.slice(0,take));carry=ordered.slice(take);
      for(const t of ordered.slice(0,take))remaining.delete(t);
    }
    return placed;
  }
  function timelineScores(schedule,teams,poll,through,model) {
    const names=teams.map(t=>t.team),cutoff=String(poll.games_through).slice(0,10);
    const games=schedule.filter(g=>final(g)&&day(g)<=through&&day(g)<='2026-12-06');
    const played=new Map([...new Set([...names,...games.flatMap(g=>[g.h,g.a])])].map(t=>[t,[]]));
    for(const g of games){played.get(g.h).push({g,opp:g.a,win:g.hp>g.ap});
      played.get(g.a).push({g,opp:g.h,win:g.ap>g.hp});}
    const pct=new Map();
    for(const [t,gs] of played)pct.set(t,gs.length?gs.filter(x=>x.win).length/gs.length:0);
    const prior=new Map(poll.rankings.map(r=>[r.team,r.rank]));
    const score=new Map();
    for(const t of teams){
      const gs=played.get(t.team),recent=gs.filter(x=>day(x.g)>cutoff),q=o=>pct.get(o)??.5;
      const wins=gs.filter(x=>x.win),losses=gs.filter(x=>!x.win);
      const values={prior_rank:prior.get(t.team)??30,win_pct:gs.length?wins.length/gs.length:0,
        sos:gs.length?mean(gs.map(x=>q(x.opp))):0,
        quality_wins:wins.reduce((s,x)=>s+q(x.opp),0),bad_losses:losses.reduce((s,x)=>s+1-q(x.opp),0),
        recent_win:recent.filter(x=>x.win).length,recent_loss:recent.filter(x=>!x.win).length,
        recent_win_quality:recent.filter(x=>x.win).reduce((s,x)=>s+q(x.opp),0),
        recent_loss_badness:recent.filter(x=>!x.win).reduce((s,x)=>s+1-q(x.opp),0),
        title_win:0,title_loss:0,power:Number(DEFAULTS.p4.includes(t.conference)||t.team==='Notre Dame'),games:gs.length};
      score.set(t.team,model.intercept+Object.entries(model.weights).reduce((s,[k,w])=>s+w*values[k],0));
    }
    const h2h=new Set(games.map(g=>(g.hp>g.ap?g.h:g.a)+'\r'+(g.hp>g.ap?g.a:g.h)));
    return new Map(timelineOrder(score,h2h,model.pod_head_to_head_bonus).map((t,i)=>[t,31-(i+1)]));
  }
  function project({schedule,teams,poll,options={},model=null,now=new Date()}) {
    const o={...DEFAULTS,...options,weights:{...DEFAULTS.weights,...(options.weights||{})}};
    if(!poll?.rankings?.length||!poll.games_through)return {error:'No committee rankings have been released yet.'};
    const names=teams.map(t=>t.team),known=new Set(names);
    const ranked=poll.rankings.filter(r=>known.has(r.team));
    if(ranked.length<poll.rankings.length)return {error:'The committee rankings name teams the model does not know.'};
    const cutoff=String(poll.games_through).slice(0,10);
    const window=weeklyWindow(now);
    const stale=window.active&&cutoff!==window.baseline;
    const pending=window.active&&!stale&&schedule.some(g=>g.st!=='postseason'&&day(g)>cutoff&&day(g)<=window.through&&!final(g));
    const status=!window.active?'static':stale?'stale':pending?'pending':'projection';
    const through=status==='projection'?window.through:cutoff;
    const st=staticScores(teams,schedule,o),then=winPct(names,schedule,cutoff),currentPct=winPct(names,schedule,through);
    const proxy=new Map(names.map(t=>[t,o.weights.win_pct*then.get(t)+st.get(t).static]));
    // committee_anchor.R: spacing from the least-squares slope of score on committee rank.
    const xs=ranked.map(r=>r.rank),ys=ranked.map(r=>proxy.get(r.team)),mx=mean(xs),my=mean(ys);
    const slope=xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0)/xs.reduce((s,x)=>s+(x-mx)**2,0);
    const spacing=Math.max(Math.abs(slope),Math.sqrt(Number.EPSILON));
    const intercept=mean(ranked.map(r=>proxy.get(r.team)+spacing*r.rank));
    const cfp=new Map(ranked.map(r=>[r.team,r.rank]));
    const anchor=new Map(names.map(t=>[t,cfp.has(t)?intercept-spacing*cfp.get(t):Math.min(proxy.get(t),intercept-spacing*26)]));
    const timeline=model?.version===1&&status==='projection'&&schedule.some(g=>final(g)&&day(g)>cutoff&&day(g)<=through)
      ?timelineScores(schedule,teams,poll,through,model):null;
    const rows=names.map(t=>({team:t,current:cfp.get(t)??null,anchor:timeline?31-(cfp.get(t)??30):anchor.get(t),
      score:timeline?timeline.get(t):anchor.get(t)+o.weights.win_pct*(currentPct.get(t)-then.get(t)),pctThen:then.get(t),pctNow:currentPct.get(t),static:st.get(t).static}));
    rows.sort((a,b)=>b.score-a.score||a.team.localeCompare(b.team));
    rows.forEach((r,i)=>{r.projected=i+1;r.change=r.current?r.current-r.projected:null;
      r.recordThen=record(schedule,r.team,cutoff);r.recordNow=record(schedule,r.team,through);
      r.results=schedule.filter(g=>final(g)&&(g.h===r.team||g.a===r.team)&&day(g)>cutoff&&day(g)<=through).map(g=>{
        const home=g.h===r.team,pf=home?g.hp:g.ap,pa=home?g.ap:g.hp;
        return {opponent:home?g.a:g.h,site:g.n?'vs':home?'':'at',result:pf>pa?'W':pf<pa?'L':'T',pf,pa,date:day(g)};});});
    const newResults=schedule.some(g=>final(g)&&day(g)>cutoff&&day(g)<=through);
    return {rows,spacing,cutoff,through,status,released:poll.released,
      newResults,dropped:ranked.filter(r=>rows.find(x=>x.team===r.team).projected>25).map(r=>r.team)};
  }

  function historicalHTML(history,esc) {
    const weeks=history?.weeks||[];
    if(!weeks.length)return '';
    const years=[...new Set(weeks.map(w=>w.season))].sort((a,b)=>b-a);
    const recent=weeks.filter(w=>w.season>=2024&&w.season<=2025),current=weeks.filter(w=>w.season===2026);
    const avg=(rows,key)=>mean(rows.map(w=>w[key]));
    return `<section class="cw-history"><header><span class="eyebrow">Historical accuracy</span><h3>How the model would have ranked them</h3><p>Each week uses only earlier seasons for training and results available at that poll. The 2024–25 releases were held out of the head-to-head adjustment choice; 2026 was excluded from all training.</p></header>
      <div class="cw-history-stats"><div><b>${current.length?avg(current,'mae').toFixed(2):'—'}</b><span>Avg. rank error · 2026 replay</span></div><div><b>${avg(recent,'mae').toFixed(2)}</b><span>Avg. rank error · 2024–25</span></div><div><b>${avg(recent,'top12_overlap').toFixed(1)} / 12</b><span>Top-12 agreement · 2024–25</span></div></div>
      <div class="cw-history-controls"><label>Season <select data-cw-season>${years.map(y=>`<option value="${y}">${y}</option>`).join('')}</select></label><label>Release <select data-cw-week></select></label></div>
      <div data-cw-history-table></div><p class="board-note">Retrospective out-of-season replay, not a log of forecasts published at the time. A predicted place beyond No. 25 is shown as NR. The comparison uses the published Top 25 and the model's entire eligible team pool.</p></section>`;
  }
  function wireHistory(host,history,esc) {
    const season=host.querySelector('[data-cw-season]'),week=host.querySelector('[data-cw-week]'),table=host.querySelector('[data-cw-history-table]');
    if(!season||!week||!table)return;
    const render=()=>{
      const selected=history.weeks.find(w=>w.season===Number(season.value)&&w.week===Number(week.value));
      if(!selected)return;
      table.innerHTML=`<p class="cw-history-summary">Week ${selected.week}: <b>${selected.mae.toFixed(2)}</b> average rank error · <b>${selected.top12_overlap}/12</b> top-12 agreement · <b>${selected.top25_overlap}/25</b> Top 25 agreement</p><div class="cw-history-scroll"><table><thead><tr><th>Actual</th><th>Team</th><th>Predicted</th><th>Previous</th><th>Miss</th></tr></thead><tbody>${selected.rows.map(r=>{
        const show=n=>n>25?'NR':'#'+n,miss=r.actual>25||r.predicted>25?'—':Math.abs(r.actual-r.predicted);
        return `<tr${r.actual>25||r.predicted>25?' class="cw-history-miss"':''}><td>${show(r.actual)}</td><th>${esc(r.team)}</th><td>${show(r.predicted)}</td><td>${show(r.previous)}</td><td>${miss}</td></tr>`;
      }).join('')}</tbody></table></div>`;
    };
    const changeSeason=()=>{const releases=history.weeks.filter(w=>w.season===Number(season.value)).sort((a,b)=>b.week-a.week);
      week.innerHTML=releases.map(w=>`<option value="${w.week}">Week ${w.week}</option>`).join('');render();};
    season.addEventListener('change',changeSeason);week.addEventListener('change',render);changeSeason();
  }

  function mount({host,schedule,teams,poll,model,history,weights,esc,crestOf,wireTeamLinks}) {
    const res=project({schedule,teams,poll,model,options:weights?{weights}:{}});
    if(res.error){host.innerHTML=`<p class="stock-empty">${esc(res.error)}</p>`;return;}
    const fmt=d=>{const [y,m,dd]=String(d).split('-').map(Number);return new Date(Date.UTC(y,m-1,dd)).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'});};
    const crest=t=>{const c=crestOf(t);return `<span class="cw-crest" style="--plate:${c.plate}"><img src="${c.mark}" alt="" loading="lazy"></span>`;};
    const by=new Map(res.rows.map(r=>[r.team,r]));
    const current=poll.rankings.map(r=>by.get(r.team));
    const projected=res.rows.slice(0,25);
    const move=r=>r.current==null?'<span class="cw-move cw-new">NEW</span>':r.change>0?`<span class="cw-move up">▲ ${r.change}</span>`:r.change<0?`<span class="cw-move down">▼ ${-r.change}</span>`:'<span class="cw-move">—</span>';
    const statusText=res.status==='static'?'Static · projections update Sunday after games finish (Eastern Time)':res.status==='stale'?'Waiting for an updated official poll before projecting this week':res.status==='pending'?'Waiting for this week’s games to finish before updating':res.newResults?`After results through ${fmt(res.through)} · updates Sunday and Monday`:'No new results yet: matches the official order';
    const games=r=>r.results.map(g=>`<span class="cw-game cw-${g.result}">${g.result} ${g.pf}–${g.pa} ${g.site?g.site+' ':''}${esc(g.opponent)}</span>`).join('')||`<span class="cw-game cw-idle">${res.status==='projection'?'No game':'—'}</span>`;
    const row=(r,rank,side)=>`<li class="cw-row${side==='proj'&&r.change<0?' cw-fell':''}${side==='proj'&&(r.change>0||r.current==null)?' cw-rose':''}">
      <b class="cw-rank">${rank}</b>${crest(r.team)}<button type="button" class="team-link cw-team" data-team="${esc(r.team)}">${esc(r.team)}<small>${side==='proj'?r.recordNow:r.recordThen}</small></button>
      ${side==='proj'?`<span class="cw-last">${games(r)}</span>${move(r)}`:''}</li>`;
    host.innerHTML=`<div class="cw">
      <div class="cw-cols">
        <section class="cw-col"><header><span class="eyebrow">Official</span><h3>Current committee rankings</h3><p>Released ${esc(fmt(res.released))} · games through ${esc(fmt(res.cutoff))}</p></header>
          <ol class="cw-list">${current.map((r,i)=>row(r,poll.rankings[i].rank,'cur')).join('')}</ol></section>
        <section class="cw-col cw-proj"><header><span class="eyebrow">Our committee model</span><h3>Projected next rankings</h3><p>${esc(statusText)}</p></header>
          <ol class="cw-list">${projected.map(r=>row(r,r.projected,'proj')).join('')}</ol>
          ${res.dropped.length?`<p class="cw-out">Projected to drop out: ${res.dropped.map(t=>`<b>${esc(t)}</b> <small>(${by.get(t).recordNow}, ${games(by.get(t)).replace(/<[^>]+>/g,' ').trim()})</small>`).join(', ')}</p>`:''}</section>
      </div>
      <p class="board-note">The projection starts from the published order and uses a historical weekly transition model. It considers the timing and opponent quality of results, schedule, conference, and head-to-head among teams in the same committee comparison pool. The projected order is an estimate, not an official committee vote.</p>${historicalHTML(history,esc)}</div>`;
    wireHistory(host,history,esc);
    wireTeamLinks();
  }
  root.CommitteeWatch={project,mount,staticScores,winPct,weeklyWindow};
  if(typeof module!=='undefined')module.exports=root.CommitteeWatch;
})(typeof window!=='undefined'?window:globalThis);
