/* Committee Watch: the committee's last published Top 25 next to where our committee
   model expects it to move after the results since then.

   The model is the same committee proxy the projected bracket uses
   (scripts/simulate_playoff.R + scripts/committee_anchor.R, fitted by
   scripts/fit_committee.py on every published CFP ranking 2014-2025):

     score = w_win * win% + w_rating * strength_z + w_sos * schedule_z + w_p4 * power league

   Anchored to the committee: at the published ranking, every team is placed exactly
   where the committee put it (evenly spaced, spacing taken from today's score-vs-rank
   fit; unranked teams sit below #25). Each team then moves by what its new results
   change in the score. Strength and schedule terms use the season's ratings and full
   schedule, so between releases only the record moves a team, as in the simulation. */
(function(root) {
  'use strict';
  const DEFAULTS={weights:{win_pct:10.3397,rating:0.2262,sos:0.7362,power_conf:1.0864},fcsOppRating:-2,
    p4:['ACC','Big 12','Big Ten','SEC'],powerIndependents:['Notre Dame']};
  const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
  const sampleSd=a=>{const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/Math.max(1,a.length-1));};
  const day=g=>String(g.d||'').slice(0,10);
  const final=g=>g.f&&Number.isFinite(g.hp)&&Number.isFinite(g.ap)&&g.st!=='postseason';

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
  function project({schedule,teams,poll,options={}}) {
    const o={...DEFAULTS,...options,weights:{...DEFAULTS.weights,...(options.weights||{})}};
    if(!poll?.rankings?.length||!poll.games_through)return {error:'No committee rankings have been released yet.'};
    const names=teams.map(t=>t.team),known=new Set(names);
    const ranked=poll.rankings.filter(r=>known.has(r.team));
    if(ranked.length<poll.rankings.length)return {error:'The committee rankings name teams the model does not know.'};
    const cutoff=String(poll.games_through).slice(0,10);
    const st=staticScores(teams,schedule,o),then=winPct(names,schedule,cutoff),now=winPct(names,schedule,null);
    const proxy=new Map(names.map(t=>[t,o.weights.win_pct*then.get(t)+st.get(t).static]));
    // committee_anchor.R: spacing from the least-squares slope of score on committee rank.
    const xs=ranked.map(r=>r.rank),ys=ranked.map(r=>proxy.get(r.team)),mx=mean(xs),my=mean(ys);
    const slope=xs.reduce((s,x,i)=>s+(x-mx)*(ys[i]-my),0)/xs.reduce((s,x)=>s+(x-mx)**2,0);
    const spacing=Math.max(Math.abs(slope),Math.sqrt(Number.EPSILON));
    const intercept=mean(ranked.map(r=>proxy.get(r.team)+spacing*r.rank));
    const cfp=new Map(ranked.map(r=>[r.team,r.rank]));
    const anchor=new Map(names.map(t=>[t,cfp.has(t)?intercept-spacing*cfp.get(t):Math.min(proxy.get(t),intercept-spacing*26)]));
    const rows=names.map(t=>({team:t,current:cfp.get(t)??null,anchor:anchor.get(t),
      score:anchor.get(t)+o.weights.win_pct*(now.get(t)-then.get(t)),pctThen:then.get(t),pctNow:now.get(t),static:st.get(t).static}));
    rows.sort((a,b)=>b.score-a.score||a.team.localeCompare(b.team));
    rows.forEach((r,i)=>{r.projected=i+1;r.change=r.current?r.current-r.projected:null;
      r.recordThen=record(schedule,r.team,cutoff);r.recordNow=record(schedule,r.team,null);
      r.results=schedule.filter(g=>final(g)&&(g.h===r.team||g.a===r.team)&&day(g)>cutoff).map(g=>{
        const home=g.h===r.team,pf=home?g.hp:g.ap,pa=home?g.ap:g.hp;
        return {opponent:home?g.a:g.h,site:g.n?'vs':home?'':'at',result:pf>pa?'W':pf<pa?'L':'T',pf,pa,date:day(g)};});});
    const through=schedule.filter(final).map(day).sort().at(-1)||cutoff;
    return {rows,spacing,cutoff,through,released:poll.released,
      newResults:through>cutoff,dropped:ranked.filter(r=>rows.find(x=>x.team===r.team).projected>25).map(r=>r.team)};
  }

  function mount({host,schedule,teams,poll,weights,esc,crestOf,wireTeamLinks}) {
    const res=project({schedule,teams,poll,options:weights?{weights}:{}});
    if(res.error){host.innerHTML=`<p class="stock-empty">${esc(res.error)}</p>`;return;}
    const fmt=d=>{const [y,m,dd]=String(d).split('-').map(Number);return new Date(Date.UTC(y,m-1,dd)).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'});};
    const crest=t=>{const c=crestOf(t);return `<span class="cw-crest" style="--plate:${c.plate}"><img src="${c.mark}" alt="" loading="lazy"></span>`;};
    const by=new Map(res.rows.map(r=>[r.team,r]));
    const current=poll.rankings.map(r=>by.get(r.team));
    const projected=res.rows.slice(0,25);
    const move=r=>r.current==null?'<span class="cw-move cw-new">NEW</span>':r.change>0?`<span class="cw-move up">▲ ${r.change}</span>`:r.change<0?`<span class="cw-move down">▼ ${-r.change}</span>`:'<span class="cw-move">—</span>';
    const games=r=>r.results.map(g=>`<span class="cw-game cw-${g.result}">${g.result} ${g.pf}–${g.pa} ${g.site?g.site+' ':''}${esc(g.opponent)}</span>`).join('')||'<span class="cw-game cw-idle">No game</span>';
    const row=(r,rank,side)=>`<li class="cw-row${side==='proj'&&r.change<0?' cw-fell':''}${side==='proj'&&(r.change>0||r.current==null)?' cw-rose':''}">
      <b class="cw-rank">${rank}</b>${crest(r.team)}<button type="button" class="team-link cw-team" data-team="${esc(r.team)}">${esc(r.team)}<small>${side==='proj'?r.recordNow:r.recordThen}</small></button>
      ${side==='proj'?`<span class="cw-last">${games(r)}</span>${move(r)}`:''}</li>`;
    host.innerHTML=`<div class="cw">
      <div class="cw-cols">
        <section class="cw-col"><header><span class="eyebrow">Official</span><h3>Current committee rankings</h3><p>Released ${esc(fmt(res.released))} · games through ${esc(fmt(res.cutoff))}</p></header>
          <ol class="cw-list">${current.map((r,i)=>row(r,poll.rankings[i].rank,'cur')).join('')}</ol></section>
        <section class="cw-col cw-proj"><header><span class="eyebrow">Our committee model</span><h3>Projected next rankings</h3><p>${res.newResults?`After results through ${esc(fmt(res.through))}`:'No new results yet: matches the official order'}</p></header>
          <ol class="cw-list">${projected.map(r=>row(r,r.projected,'proj')).join('')}</ol>
          ${res.dropped.length?`<p class="cw-out">Projected to drop out: ${res.dropped.map(t=>`<b>${esc(t)}</b> <small>(${by.get(t).recordNow}, ${games(by.get(t)).replace(/<[^>]+>/g,' ').trim()})</small>`).join(', ')}</p>`:''}</section>
      </div>
      <p class="board-note">The projection starts from the committee's own order and moves each team by what its new results change in our committee model: record first, then team strength, schedule strength and power-conference membership, weighted the way the committee has weighted them since 2014. It does not know who a team lost to, so a loss to a top-10 team and a loss to an unranked team move a team the same amount.</p></div>`;
    wireTeamLinks();
  }
  root.CommitteeWatch={project,mount,staticScores,winPct};
  if(typeof module!=='undefined')module.exports=root.CommitteeWatch;
})(typeof window!=='undefined'?window:globalThis);
