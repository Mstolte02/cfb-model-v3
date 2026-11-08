(function(root){
  'use strict';
  const pct=n=>(100*n).toFixed(1)+'%';
  function html({current,baseline,esc,crestOf,mode}){
    if(!baseline||baseline.season!==current.season)return '<p class="board-note">Weekly tracking starts with the next published simulation. Select the locked preseason baseline to explore changes now.</p>';
    const old=new Map(baseline.teams.map(t=>[t.team,t]));
    const date=p=>p.record_odds?.generated_at||p.generated_at||p.captured_at;
    const stamp=p=>date(p)?esc(new Date(date(p)).toLocaleString()):'timestamp unavailable';
    const panels=[['conf_champ','Win conference title'],['playoff','Make the CFP']];
    return `<p class="board-note">${mode==='preseason'?'Locked preseason projection':`Weekly baseline: ${stamp(baseline)}`} → current projection (${stamp(current)}). Changes are percentage points. Model and committee updates, other teams’ results, and simulation sampling can all move these odds.</p><div class="movers-grid">`+panels.map(([key,title])=>{
      const rows=current.teams.filter(t=>old.has(t.team)&&Number.isFinite(t[key])&&Number.isFinite(old.get(t.team)[key])).map(t=>({...t,before:old.get(t.team)[key],delta:100*(t[key]-old.get(t.team)[key])}));
      return `<article><h3>${title}</h3>`+[[1,'Biggest gains'],[-1,'Biggest drops']].map(([sign,label])=>{
        const ranked=rows.filter(t=>sign*t.delta>.05).sort((a,b)=>sign*(b.delta-a.delta)||a.team.localeCompare(b.team)).slice(0,10);
        const max=Math.max(1,...ranked.map(t=>Math.abs(t.delta)));
        return `<h4>${label}</h4><p class="hint">Baseline → Now · Change (points)</p>`+(ranked.length?ranked.map(t=>`<div class="mover-row"><div><button class="team-link" data-team="${esc(t.team)}"><img src="${esc(crestOf(t.team).mark)}" alt="" loading="lazy">${esc(t.team)}</button><small>${esc(t.conference)}</small></div><span>${pct(t.before)} → <b>${pct(t[key])}</b></span><div class="mover-change ${sign>0?'gain':'drop'}"><span style="width:${Math.abs(t.delta)/max*100}%"></span><b>${t.delta>0?'+':''}${t.delta.toFixed(1)}</b></div></div>`).join(''):'<p class="board-note">No changes in this direction.</p>');
      }).join('')+'</article>';
    }).join('')+'</div>';
  }
  root.OddsMovers={html};
})(globalThis);
