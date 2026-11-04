/* Résumé-only selection. Official committee data is used only for comparison. */
(function(root) {
  'use strict';
  const P4=['ACC','Big Ten','Big 12','SEC'];
  const G6=['American Athletic','Conference USA','Mid-American','Mountain West','Pac-12','Sun Belt'];
  function selectField(rows,meta,champions={}) {
    const ranked=rows.slice().sort((a,b)=>b.score-a.score||a.team.localeCompare(b.team));
    const autos=new Map();
    for(const conf of P4) {
      const actual=ranked.find(r=>r.team===champions[conf]&&meta[r.team]?.conference===conf);
      const r=actual||ranked.find(r=>meta[r.team]?.conference===conf);
      if(!r)return {field:[],out:[],error:`Waiting for résumé results in ${conf}.`};
      autos.set(r.team,actual?`${conf} champion`:`${conf} placeholder`);
    }
    const g6=ranked.find(r=>G6.includes(meta[r.team]?.conference));
    if(!g6)return {field:[],out:[],error:'Waiting for Group of 6 résumé results.'};
    autos.set(g6.team,'Group of 6 auto bid');
    const nd=ranked.find(r=>r.team==='Notre Dame'&&r.rank<=12);
    if(nd)autos.set(nd.team,'Notre Dame Top 12');
    const chosen=new Set(autos.keys());
    for(const r of ranked) {if(chosen.size>=12)break;chosen.add(r.team);}
    const field=ranked.filter(r=>chosen.has(r.team)).map((r,i)=>({...r,seed:i+1,bid:autos.get(r.team)||'At-large',auto:autos.has(r.team)}));
    return {field,out:ranked.filter(r=>!chosen.has(r.team)).slice(0,2),error:field.length===12?null:'Waiting for twelve eligible teams.'};
  }
  function comparison(rows,poll) {
    const official=new Map((poll?.rankings||[]).map(r=>[r.team,r.rank]));
    return rows.filter(r=>r.rank<=25||official.has(r.team)).map(r=>({...r,committee:official.get(r.team)??null,
      // Positive = committee places this team too high relative to Mark.
      delta:official.has(r.team)?r.rank-official.get(r.team):null}));
  }
  function mount({rows,meta,schedule,poll,season,host,esc,crestOf,wireTeamLinks}) {
    if(!rows.length){host.innerHTML='<p class="stock-empty">The playoff board opens after completed FBS games.</p>';return;}
    const champions={};
    for(const g of schedule) {
      // Championship fixtures must be explicitly identified; never treat an ordinary final as a title game.
      if(!g.f||!Number.isFinite(g.hp)||!Number.isFinite(g.ap)||g.hp===g.ap||!(g.conference_championship===true||g.conferenceChampionship===true))continue;
      const conf=meta[g.h]?.conference;
      if(conf===meta[g.a]?.conference)champions[conf]=g.hp>g.ap?g.h:g.a;
    }
    const result=selectField(rows,meta,champions);
    if(result.error){host.innerHTML=`<p class="stock-empty">${esc(result.error)}</p>`;return;}
    const {field,out}=result,bySeed=new Map(field.map(r=>[r.seed,r]));
    const card=(r,compact=false)=>{
      const c=crestOf(r.team);
      return `<button type="button" class="mp-team team-link${r.auto?' mp-auto':''}${compact?' mp-compact':''}" data-team="${esc(r.team)}" style="--plate:${c.plate};--ink:${meta[r.team]?.onColor||'#fff'}"><b class="mp-seed">${r.seed||'OUT'}</b><img src="${c.mark}" alt=""><span><strong>${esc(r.team)}</strong><small>#${r.rank} résumé · ${r.score.toFixed(2)}${compact?'':` · ${r.wins}–${r.losses} FBS`}</small></span>${r.auto?'<i title="Automatic qualifier">AQ</i>':''}</button>`;
    };
    const placeholder=label=>`<div class="mp-placeholder">${label}</div>`;
    const bracket=`<div class="mp-broadcast"><div class="mp-masthead"><span class="mp-football" aria-hidden="true">◈</span><div><small>${season} / THE EARNED FIELD</small><h3>MARK’S PLAYOFF</h3><p>Based on current résumé scores</p></div><span class="mp-edition">12 TEAMS<br>ONE CHAMPION</span></div><div class="mp-scroll" tabindex="0" role="region" aria-label="Twelve-team playoff bracket; scroll horizontally on small screens"><div class="mp-stage"><aside class="mp-field"><h4>THE FIELD</h4>${field.map(r=>card(r,true)).join('')}<h4>FIRST TWO OUT</h4>${out.map(r=>card(r,true)).join('')}</aside><div class="mp-round mp-first"><h4>FIRST ROUND <small>Higher seed hosts</small></h4>${[[12,5],[9,8],[11,6],[10,7]].map((pair,i)=>`<div class="mp-match mp-match-${i}">${pair.map(seed=>card(bySeed.get(seed))).join('')}</div>`).join('')}</div><div class="mp-round mp-quarter"><h4>QUARTERFINALS <small>Top four receive byes</small></h4>${[4,1,3,2].map((seed,i)=>`<div class="mp-match mp-match-${i}">${placeholder(`Winner ${[5,8,6,7][i]} / ${[12,9,11,10][i]}`)}${card(bySeed.get(seed))}</div>`).join('')}</div><div class="mp-round mp-semi"><h4>SEMIFINALS <small>Fixed bracket · no reseeding</small></h4><div class="mp-match mp-sf-0">${placeholder('Quarterfinal winner · 4 / 5 / 12')}${placeholder('Quarterfinal winner · 1 / 8 / 9')}</div><div class="mp-match mp-sf-1">${placeholder('Quarterfinal winner · 3 / 6 / 11')}${placeholder('Quarterfinal winner · 2 / 7 / 10')}</div></div><div class="mp-round mp-final"><h4>CHAMPIONSHIP</h4><div class="mp-trophy" aria-hidden="true"><span>◈</span><i></i><b>NATIONAL<br>CHAMPION</b></div><div class="mp-match">${placeholder('Semifinal winner')}${placeholder('Semifinal winner')}</div></div></div></div><footer><span><b>GOLD EDGE</b> Automatic qualifier</span><span><b>BOX</b> Playoff seed</span><span><b>#</b> Résumé rank</span><span>Unplayed rounds remain open</span></footer></div>`;
    host.innerHTML=`<div class="seg mp-toggle" role="group" aria-label="Mark's Playoff display"><button class="seg-btn active" aria-pressed="true" data-mp-mode="bracket">Playoff bracket</button><button class="seg-btn" aria-pressed="false" data-mp-mode="compare">Mark vs. committee</button></div><div id="mp-bracket">${bracket}<p class="board-note">A snapshot of results earned, not an end-of-season forecast. P4 champion placeholders are each conference’s highest résumé-ranked team until a recorded title-game winner is available. The highest-ranked G6 team receives its bid regardless of a conference title. Notre Dame is protected in the résumé Top 12. Remaining places go by résumé score; seeds 1–4 receive byes. Equal scores share a résumé rank; alphabetical order breaks a seeding tie only. Records exclude FCS games.</p><p class="board-note"><a href="https://collegefootballplayoff.com/sports/2024/5/29/12-team-format" target="_blank" rel="noopener">2026 CFP selection and seeding rules ↗</a></p></div><div id="mp-compare" hidden></div>`;
    const compareHost=host.querySelector('#mp-compare');
    const drawCompare=()=>{
      if(!poll||poll.season!==season||!poll.rankings?.length){compareHost.innerHTML='<p class="stock-empty">No verified committee rankings are available for this season yet. The résumé bracket is ready.</p>';return;}
      const cutoff=poll.games_through;
      const aligned=cutoff?schedule.filter(g=>g.d&&g.d.slice(0,10)<=cutoff):schedule;
      const alignedRows=root.RankingHistory.deserving(aligned,meta,rows.map(r=>r.team),{method:'colley'});
      const all=comparison(alignedRows,poll);
      const draw=()=>{
        const query=compareHost.querySelector('input').value.toLowerCase().trim();
        const order=compareHost.querySelector('select').value;
        const list=all.filter(r=>r.team.toLowerCase().includes(query)).sort((a,b)=>order==='gap'?(Math.abs(b.delta??0)-Math.abs(a.delta??0)||a.rank-b.rank):order==='committee'?((a.committee??26)-(b.committee??26)||a.rank-b.rank):a.rank-b.rank);
        const max=Math.max(1,...all.map(r=>Math.abs(r.delta??0)));
        compareHost.querySelector('.mp-bars').innerHTML=list.map(r=>`<div class="mp-bar-row"><button class="mp-bar-team team-link" data-team="${esc(r.team)}" style="--team-plate:${crestOf(r.team).plate}"><img src="${crestOf(r.team).mark}" alt=""><span>${esc(r.team)}<small>Mark #${r.rank} · CFP ${r.committee?'#'+r.committee:'NR'}</small></span></button><div class="mp-bar-track"><i class="mp-zero"></i>${r.delta!==null?`<span class="mp-bar ${r.delta>0?'over':'under'}" style="width:${Math.abs(r.delta)/max*48}%;${r.delta<0?`right:50%`:'left:50%'}"></span><b class="mp-gap ${r.delta>0?'over':'under'}" style="${r.delta<0?'right':'left'}:calc(50% + ${Math.abs(r.delta)/max*48}% + 6px)">${r.delta>0?'+':r.delta<0?'−':''}${Math.abs(r.delta)}</b>`:'<span class="mp-nr">Unranked by CFP · no exact gap</span>'}</div></div>`).join('')||'<p class="stock-empty">No matching teams.</p>';
        wireTeamLinks();
      };
      compareHost.innerHTML=`<article class="mp-differences"><div class="mp-compare-head"><span class="eyebrow">THE PERCEPTION GAP</span><h3>Same results. Different verdict.</h3><p>Mark’s résumé rankings against the actual CFP Top 25.</p><p class="mp-snapshot">Committee release: ${esc(poll.released)} · Both rankings use games through ${esc(cutoff||'the latest final')} · <a href="${esc(poll.source_url)}" target="_blank" rel="noopener">Committee source ↗</a></p></div><div class="mp-compare-controls"><input type="search" placeholder="Find a team…" aria-label="Find a team in committee comparison"><select aria-label="Comparison order"><option value="gap">Biggest disagreement</option><option value="mark">Mark’s rank</option><option value="committee">Committee rank</option></select></div><div class="mp-axis"><span>← UNDERRATED BY COMMITTEE (−)</span><span>OVERRATED BY COMMITTEE (+) →</span></div><div class="mp-bars"></div><p class="board-note">Gap = Mark’s rank − CFP rank. Positive means the committee ranks the team higher than Mark does; negative means lower. Teams outside the CFP Top 25 are shown as NR, without inventing a #26 rank. Résumé scores are recomputed at the committee’s games cutoff for a fair comparison.</p></article>`;
      compareHost.querySelector('input').addEventListener('input',draw);
      compareHost.querySelector('select').addEventListener('change',draw);draw();
    };
    host.querySelectorAll('[data-mp-mode]').forEach(b=>b.addEventListener('click',()=>{
      const compare=b.dataset.mpMode==='compare';
      host.querySelector('#mp-bracket').hidden=compare;compareHost.hidden=!compare;
      host.querySelectorAll('[data-mp-mode]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});
      if(compare)drawCompare();
    }));wireTeamLinks();
  }
  function projectedHTML({games,seeds,teams,autoBids=[],season,anchor,preseason,esc,crestOf,meta}) {
    const byTeam=new Map(teams.map(r=>[r.team,r]));
    const seedOf=new Map(seeds.map((t,i)=>[t,i+1]));
    const auto=new Set(autoBids);
    const card=(team,prob=null,compact=false,winner=false)=>{
      if(!team)return '<div class="mp-placeholder">Projected participant pending</div>';
      const r=byTeam.get(team)||{},c=crestOf(team),seed=seedOf.get(team);
      const rank=r.current_cfp_rank?`CFP #${r.current_cfp_rank}`:preseason?'Preseason baseline':'CFP NR';
      const detail=compact?`${rank} · ${Math.round(100*(r.playoff||0))}% in`:`${rank} · proj. ${Number(r.avg_wins||0).toFixed(1)}–${Number(r.avg_losses||0).toFixed(1)}`;
      return `<button type="button" class="mp-team team-link${auto.has(team)?' mp-auto':''}${compact?' mp-compact':''}${winner?' mp-advances':''}" data-team="${esc(team)}" style="--plate:${c.plate};--ink:${meta[team]?.onColor||'#fff'}"><b class="mp-seed">${seed||'OUT'}</b><img src="${c.mark}" alt=""><span><strong>${esc(team)}</strong><small>${detail}</small></span>${prob!==null?`<b class="mp-prob">${prob}%</b>`:auto.has(team)?'<i>AQ</i>':''}</button>`;
    };
    const game=(index,reverse=false)=>{
      const g=games[index];if(!g)return '<div class="mp-placeholder">Projected matchup pending</div>';
      const p=Number.isFinite(g.p)?Math.round(g.p*100):null;
      const sides=[[g.top,p],[g.bottom,p===null?null:100-p]];if(reverse)sides.reverse();
      return sides.map(([t,prob])=>card(t,prob,false,t===g.winner)).join('');
    };
    const out=teams.filter(r=>!seedOf.has(r.team)).sort((a,b)=>b.playoff-a.playoff||a.team.localeCompare(b.team)).slice(0,2);
    const champ=games[10]?.winner;
    return `<div class="mp-broadcast mp-projected"><div class="mp-masthead"><span class="mp-football" aria-hidden="true">◈</span><div><small>${season} / MODEL FORECAST · ${preseason?'PRESEASON SNAPSHOT':'COMMITTEE STARTING POINT'}</small><h3>PROJECTED PLAYOFF</h3><p>${preseason?'Preseason forecast':`Starting from CFP rankings · ${esc(anchor?.released||'awaiting first release')}`}</p></div><span class="mp-edition">PROJECTED FIELD<br>PROJECTED WINNERS</span></div><div class="mp-scroll" tabindex="0" role="region" aria-label="Projected twelve-team playoff bracket; scroll horizontally on small screens"><div class="mp-stage"><aside class="mp-field"><h4>PROJECTED SEEDS</h4>${seeds.map(t=>card(t,null,true)).join('')}<h4>NEXT CONTENDERS</h4>${out.map(r=>card(r.team,null,true)).join('')}</aside><div class="mp-round mp-first"><h4>FIRST ROUND <small>Higher seed hosts · win probability</small></h4>${[1,0,2,3].map((k,i)=>`<div class="mp-match mp-match-${i}">${game(k)}</div>`).join('')}</div><div class="mp-round mp-quarter"><h4>QUARTERFINALS <small>Projected advancement</small></h4>${[5,4,6,7].map((k,i)=>`<div class="mp-match mp-match-${i}">${game(k,true)}</div>`).join('')}</div><div class="mp-round mp-semi"><h4>SEMIFINALS <small>Projected advancement</small></h4><div class="mp-match mp-sf-0">${game(8)}</div><div class="mp-match mp-sf-1">${game(9,true)}</div></div><div class="mp-round mp-final"><h4>CHAMPIONSHIP <small>Projected matchup</small></h4><div class="mp-trophy"><span aria-hidden="true">◈</span><i></i><b>PROJECTED CHAMPION</b>${champ?card(champ,null,true):''}<small>${champ?`${Math.round(100*(byTeam.get(champ)?.champ||0))}% overall title probability`:''}</small></div><div class="mp-match">${game(10)}</div></div></div></div><footer><span><b>ALL MATCHUPS ARE PROJECTED</b></span><span><b>%</b> Matchup win probability, not a score</span><span><b>GOLD EDGE</b> Projected automatic bid</span><span><b>BOX</b> Projected seed · CFP # is the current official rank</span></footer></div><p class="board-note">A representative projected field from simulated seed frequencies. Each matchup advances the model’s more likely winner; the displayed path is not a guarantee. Overall title odds include every simulated field and path.</p>`;
  }

  root.MarksPlayoff={selectField,comparison,mount,projectedHTML};
  if(typeof module!=='undefined')module.exports=root.MarksPlayoff;
})(typeof window!=='undefined'?window:globalThis);
