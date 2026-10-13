(async function(){
  'use strict';
  const $=id=>document.getElementById(id), X=PowerXray;
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=n=>(n*100).toFixed(2)+'%', pp=n=>(n>=0?'+':'−')+Math.abs(n*100).toFixed(2)+' pp';
  const labels={full_new_war:'WAR foundation',full_new_war_curvature:'WAR + curvature',full_new_war_production:'WAR + production',full_new_war_curvature_production:'WAR + curvature + production'};
  const load=async path=>{const r=await fetch(path,{cache:'no-store'});if(!r.ok)throw new Error('Could not load '+path+' ('+r.status+')');return r.json();};
  try {
    const [model,ratings]=await Promise.all([load('../data/model_v4.json'),load('../data/ratings.json')]);
    const E=model.ensemble;
    if(!E?.state || !E.members?.length)throw new Error('Published ensemble state is unavailable.');
    const names=ratings.teams.map(r=>r.team), sorted=names.slice().sort(), rows=new Map(ratings.teams.map(r=>[r.team,r]));
    for(const m of E.members){
      const st=X.stack(m,E.state);
      if(st.columns.length!==st.scale.length || st.columns.length!==st.coef.length || st.columns.some((k,i)=>!['hfa',...X.features.map(f=>f.key)].includes(k)||!Number.isFinite(st.coef[i])||!Number.isFinite(st.scale[i])||st.scale[i]<=0))throw new Error('The published stack schema changed; this explorer needs an update.');
      for(const t of names)if(!Number.isFinite(m.initial[t]) || !Number.isFinite(E.state.ratings[m.name]?.[t]))throw new Error('The model and ratings team lists do not match. Please reload.');
    }
    const cache=new Map();const get=t=>{if(!cache.has(t))cache.set(t,X.explain(E,t,names));return cache.get(t);};
    const options=sorted.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join('');
    $('team').innerHTML=options;$('compare').insertAdjacentHTML('beforeend',options);
    const params=new URLSearchParams(location.search);$('team').value=names.includes(params.get('team'))?params.get('team'):names[0];
    if(names.includes(params.get('compare')))$('compare').value=params.get('compare');
    $('version').textContent='MODEL '+E.model_version;
    $('source-status').textContent=E.state.pff==null?'PFF unavailable: base stack active. In-season WAR requires the PFF + WAR stack.':`PFF active · data through week ${E.state.pff_cutoff ?? 'unknown'}${E.state.pff_carried_forward?' · carried forward until the next eligible update':''}. In-season WAR ${E.state.war==null?'unavailable':'active'}. Tables only use cutoffs before the prediction week.`;
    $('snapshot').textContent=`${ratings.season} · through slate ${E.state.updated_through_slate ?? 'unknown'} · ${E.state.completed_games ?? '—'} completed games in snapshot`;
    let current,mask=127,selectedOpponent;
    function scenario(){
      $('scenario').textContent=pct(current.values[mask]);
      $('scenario-change').textContent=mask===127?'All active signals on. Matches the reconstructed rating.':`${pp(current.values[mask]-current.power)} versus the full calculation. Opponent field held fixed.`;
    }
    function match(other){
      selectedOpponent=other;
      document.querySelectorAll('.opponent').forEach(b=>{const selected=b.dataset.team===other;b.classList.toggle('selected',selected);b.setAttribute('aria-pressed',String(selected));});
      $('match-math').innerHTML=`<p>The live model uses weighted differences in team strengths. Its efficiency terms can be rearranged into offense-versus-opposing-defense advantages, but it has no explicit nonlinear O×D interaction. See the research page for tests of those interactions.</p><h3>${esc(current.team)} vs ${esc(other)} · ${pct(X.matchup(E,current.team,other))}</h3><p>Neutral field · full model · these are weighted logit terms, not percentage points.</p><div class="table-scroll"><table><thead><tr><th>Signal</th>${E.members.map(m=>`<th>${esc(labels[m.name]||m.name)}</th>`).join('')}</tr></thead><tbody>${X.features.map((f,i)=>`<tr><td>${esc(f.label)}</td>${E.members.map(m=>`<td>${X.terms(E,m,current.team,other)[i].toFixed(4)}</td>`).join('')}</tr>`).join('')}<tr><td>Sum of weighted terms</td>${E.members.map(m=>`<td>${X.terms(E,m,current.team,other).reduce((a,b)=>a+b,0).toFixed(4)}</td>`).join('')}</tr><tr><td>Win probability after logistic conversion</td>${E.members.map(m=>`<td>${pct(X.sigmoid(X.terms(E,m,current.team,other).reduce((a,b)=>a+b,0)))}</td>`).join('')}</tr></tbody></table></div>`;
    }
    function opponents(){
      const query=$('search').value.toLowerCase();
      const shown=current.opponents.filter(o=>o.team.toLowerCase().includes(query));
      $('opponent-grid').innerHTML=shown.map(o=>`<button class="opponent${selectedOpponent===o.team?' selected':''}" aria-pressed="${selectedOpponent===o.team}" data-team="${esc(o.team)}"><span>${esc(o.team)}</span><b>${pct(o.power)}</b></button>`).join('')||'<p>No matching opponents.</p>';
      $('opponent-grid').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{match(b.dataset.team);$('match-detail').open=true;}));
    }
    function comparison(){
      const t=$('compare').value;
      if(!t||t===current.team){$('comparison').innerHTML='';return;}
      const other=get(t);
      $('comparison').innerHTML=`<h3>Why ${esc(current.team)} and ${esc(t)} differ</h3><p class="note">${pct(current.power)} vs ${pct(other.power)} · ${pp(current.power-other.power)} gap. Contributions use each team’s own opponent field, excluding itself.</p><div class="table-scroll"><table><thead><tr><th>Signal</th><th>${esc(current.team)}</th><th>${esc(t)}</th><th>Difference</th></tr></thead><tbody>${current.contributions.map((c,i)=>`<tr><td>${esc(c.label)}</td><td>${pp(c.value)}</td><td>${pp(other.contributions[i].value)}</td><td>${pp(c.value-other.contributions[i].value)}</td></tr>`).join('')}<tr><td>Total power</td><td>${pct(current.power)}</td><td>${pct(other.power)}</td><td>${pp(current.power-other.power)}</td></tr></tbody></table></div>`;
    }
    function url(){const q=new URLSearchParams({team:$('team').value});if($('compare').value)q.set('compare',$('compare').value);history.replaceState(null,'','?'+q);}
    function render(){
      current=get($('team').value);mask=127;const row=rows.get(current.team);
      $('team-name').textContent=current.team;$('team-meta').textContent=row.conference||'Rated team';$('power').textContent=pct(current.power);
      const delta=Math.abs(current.power-row.power), synced=delta<0.000002;
      $('rank').textContent=synced?`#${row.rank} of ${names.length} · neutral-field win rate`:'Reconstructed from model snapshot';
      const ordered=current.contributions.filter(c=>c.active).slice().sort((a,b)=>b.value-a.value),top=ordered[0],bottom=ordered[ordered.length-1];
      $('headline').textContent=`Expected to beat a uniformly selected opponent from the other ${names.length-1} rated teams ${pct(current.power)} of the time.${top.value>0?' Biggest lift: '+top.label.toLowerCase()+'.':''}${bottom.value<0?' Biggest drag: '+bottom.label.toLowerCase()+'.':''}`;
      $('reconcile').className='reconcile'+(synced?'':' warn');
      $('reconcile').textContent=synced?`✓ Reconciled with published Power Rating (${pct(row.power)}). Maximum allowed rounding difference: 0.0002 percentage points.`:`Snapshot mismatch: published rating ${pct(row.power)}, reconstructed ${pct(current.power)}. Files may be updating; reload before interpreting the difference.`;
      const max=Math.max(...current.contributions.map(c=>Math.abs(c.value)),.01);
      $('waterfall').innerHTML=`<div class="base"><span>No differences between teams</span><b>50.00%</b></div>${current.contributions.map(c=>`<div class="contribution"><details><summary>${esc(c.label)}${!c.active?'<span class="inactive">Excluded by fitted stack</span>':''}</summary><p>${esc(c.description)}</p></details><div class="track" aria-hidden="true"><span class="bar ${c.value>=0?'pos':'neg'}" style="left:${c.value>=0?50:50-Math.abs(c.value)/max*49}%;width:${Math.abs(c.value)/max*49}%"></span></div><div class="amount ${c.value>=0?'pos':'neg'}">${pp(c.value)}</div></div>`).join('')}<div class="total"><span>Reconstructed Power Rating</span><b>${pct(current.power)}</b></div>`;
      $('toggles').innerHTML=current.contributions.map((c,i)=>`<label class="toggle" for="signal-${i}"><span>${esc(c.label)}${!c.active?' · not used':''}</span><input id="signal-${i}" type="checkbox" data-bit="${i}" checked ${!c.active?'disabled':''}></label>`).join('');
      $('toggles').querySelectorAll('input').forEach(el=>el.addEventListener('change',()=>{const bit=1<<Number(el.dataset.bit);mask=el.checked?mask|bit:mask&~bit;scenario();}));
      $('members').innerHTML=current.members.map(m=>`<div class="member"><small>${esc(labels[m.name]||m.name)}</small><strong>${pct(m.power)}</strong><small>${(100/E.members.length).toFixed(0)}% ensemble weight</small><div class="mini"><span style="width:${m.power*100}%"></span></div></div>`).join('');
      $('field-note').textContent=`Average all ${current.opponents.length} probabilities below and you get ${pct(current.power)}. Ordered from most to least favorable. Select an opponent to inspect the exact weighted inputs. These use the full current model, independent of the sandbox switches.`;
      $('coverage').textContent=`PFF offensive/defensive composite: ${E.state.pff?.[current.team]?'available':'missing (zero team input used)'}. In-season WAR: ${E.state.war?.[current.team]!=null?'available':'missing (zero input used)'}. Form games by member: ${E.members.map(m=>(labels[m.name]||m.name)+' '+(E.state.form[m.name]?.[current.team]?.[2]??0)).join('; ')}. A zero contribution can mean excluded, missing, or offsetting matchup differences; it does not automatically mean average performance.`;
      $('coefficients').innerHTML=`<table><thead><tr><th>Signal</th>${E.members.map(m=>`<th>${esc(labels[m.name]||m.name)}<br>coefficient / scale</th>`).join('')}</tr></thead><tbody>${X.features.map(f=>`<tr><td>${esc(f.label)}</td>${E.members.map(m=>{const st=X.stack(m,E.state),i=st.columns.indexOf(f.key);return `<td>${i<0?'Not used':st.coef[i].toFixed(5)+' / '+st.scale[i].toFixed(5)}</td>`;}).join('')}</tr>`).join('')}</tbody></table><p>All home-field terms are zero. Active stack: ${E.state.pff==null?'base':E.state.war==null?'PFF':'PFF + WAR'}; each member’s actual columns are shown above.</p>`;
      selectedOpponent=current.opponents.find(o=>o.team===$('compare').value)?.team||current.opponents[Math.floor(current.opponents.length/2)].team;
      $('search').value='';opponents();match(selectedOpponent);comparison();scenario();url();
    }
    $('team').addEventListener('change',render);$('compare').addEventListener('change',()=>{comparison();if($('compare').value && $('compare').value!==current.team)match($('compare').value);url();});
    $('search').addEventListener('input',opponents);$('reset').addEventListener('click',()=>{mask=127;$('toggles').querySelectorAll('input').forEach(el=>el.checked=true);scenario();});
    render();$('status').hidden=true;$('app').hidden=false;
  }catch(error){$('app').hidden=true;$('status').textContent='The explorer could not load safely. '+error.message+' Reload to try again.';console.error(error);}
})();
