'use strict';
const WarResearch = {
  sigmoid: x => 1 / (1 + Math.exp(-x)),
  explain(candidate, columns, member) {
    const members = candidate.members.map(values => ({total: values.reduce((a,b)=>a+b,0), values}));
    const probability = members.reduce((s,m)=>s+WarResearch.sigmoid(m.total),0)/members.length;
    return {probability,total:members[member].total,memberProbability:WarResearch.sigmoid(members[member].total),
      terms:columns.map((name,i)=>({name,value:members[member].values[i]}))};
  },
  label: name => name.replaceAll('_',' ').replace(/\bwar\b/g,'WAR').replace(/\bO\b/g,'offense').replace(/\bD\b/g,'defense')
};
if(typeof module!=='undefined') module.exports=WarResearch;
if(typeof document!=='undefined') (async()=>{
  const $=id=>document.getElementById(id),pct=x=>(100*x).toFixed(2)+'%',f=x=>Number(x).toFixed(4);
  const option=(value,label)=>{const n=document.createElement('option');n.value=value;n.textContent=label;return n;};
  const cell=(row,value)=>{const n=document.createElement('td');n.textContent=value;row.append(n);};
  const get=async path=>{const r=await fetch(path);if(!r.ok)throw Error(`Could not load ${path}`);return r.json();};
  try {
    const data=await get('war-research.json');
    $('decision').textContent=data.decision || 'Keep the current production model while the new candidates remain in research. See the report for the evidence and remaining limitations.';
    for(const [name,report] of Object.entries(data.metrics)) {
      const m=report.overall,tr=document.createElement('tr');
      [WarResearch.label(name),f(m.brier),f(m.log_loss),f(m.auc),pct(m.accuracy),f(m.calibration_slope),pct(m.ece)].forEach(v=>cell(tr,v));$('scores').append(tr);
      $('candidate').append(option(name,WarResearch.label(name)));
    }
    $('candidate').value='chronological_war';
    [...new Set(data.games.map(g=>g.season))].reverse().forEach(s=>$('season').append(option(s,s)));
    const cache=new Map();let sequence=0;
    async function draw(){
      const seq=++sequence,meta=data.games[Number($('game').value)];if(!meta)return;
      $('status').textContent='Loading matchup…';
      try {
        if(!cache.has(meta.chunk))cache.set(meta.chunk,get(`war-research/${meta.chunk}.json`));
        const chunk=await cache.get(meta.chunk);if(seq!==sequence)return;
        const game=chunk[meta.index],name=$('candidate').value,candidate=game.candidates[name];
        const x=WarResearch.explain(candidate,data.columns[name],Number($('member').value));
        if(Math.abs(x.probability-candidate.probability)>1e-9)throw Error('Attribution verification failed');
        $('matchup').textContent=`${game.away} at ${game.home}`;
        $('outcome').textContent=`${game.season} · Week ${game.week} · ${game.outcome?'Home':'Away'} win${game.season===2023?' · Warmup; excluded from headline metrics':''}`;
        $('probability').textContent=pct(x.probability);
        $('comparison').textContent=`No in-season WAR: ${pct(game.candidates.no_inseason_war.probability)}. Rebuilt WAR: ${pct(game.candidates.chronological_war.probability)}. Selected candidate: ${pct(x.probability)}.`;
        $('terms').replaceChildren();
        x.terms.sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).forEach(t=>{const row=document.createElement('tr');cell(row,WarResearch.label(t.name));cell(row,(t.value>=0?'+':'')+t.value.toFixed(5));$('terms').append(row);});
        $('total').textContent=`${x.total.toFixed(5)} → ${pct(x.memberProbability)}`;
        $('status').textContent='Exact four-member attribution verified.';
      } catch(error){if(seq===sequence){$('status').textContent=error.message;$('probability').textContent='—';$('terms').replaceChildren();}}
    }
    function games(){
      $('game').replaceChildren();data.games.forEach((g,i)=>{if(g.season===Number($('season').value)&&g.week===Number($('week').value))$('game').append(option(i,`${g.away} at ${g.home}`));});draw();
    }
    function weeks(){
      $('week').replaceChildren();[...new Set(data.games.filter(g=>g.season===Number($('season').value)).map(g=>g.week))].sort((a,b)=>a-b).forEach(w=>$('week').append(option(w,w)));games();
    }
    $('season').addEventListener('change',weeks);$('week').addEventListener('change',games);
    ['game','candidate','member'].forEach(id=>$(id).addEventListener('change',draw));weeks();
  }catch(error){$('status').textContent=error.message;$('decision').textContent='Research results could not be loaded. Open the methodology report above.';}
})();
