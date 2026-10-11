/* Read-only mirror of scripts/ensemble_replay.py. No model state is mutated. */
(function (root) {
  'use strict';
  const features = [
    ['prior_level', 'Preseason foundation', 'The fitted preseason strength, including projected roster WAR. This is an aggregate input; the live payload does not expose its internal feature attribution.'],
    ['elo_change', 'Results vs expectation', 'Cumulative rating updates from scoring margins relative to expected margins. A close win can lower this signal; a competitive loss can raise it.'],
    ['dO', 'Offensive efficiency', 'Opponent-adjusted offensive game form. Both teams need at least two form games. Only contributes when included in the active stack.'],
    ['dD', 'Defensive efficiency', 'Opponent-adjusted defensive game form. Both teams need at least two form games. Signs follow the fitted coefficient, not a raw grade ranking.'],
    ['pff_O_diff', 'PFF offensive form', 'The difference in season-to-date PFF offensive composites, scaled and weighted by each model.'],
    ['pff_D_diff', 'PFF defensive form', 'The difference in PFF defensive composites. May be excluded by the fitted stack to avoid overlapping signals.'],
    ['war_delta_diff', 'In-season player WAR', 'The in-season team WAR adjustment relative to opponents. This is distinct from projected roster WAR already embedded in preseason strength.']
  ].map(([key,label,description]) => ({key,label,description}));
  const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40,z))));
  const mean = a => a.reduce((s,v) => s+v,0) / a.length;
  function stack(m,s) {
    let chosen;
    if (s.pff != null && s.war != null) {
      chosen=m.stack_war;
      if (m.stack_war_legacy && m.stack_war_from_slate != null && s.slate != null && s.slate < m.stack_war_from_slate) chosen=m.stack_war_legacy;
    }
    return chosen || (s.pff != null && m.stack_pff) || m;
  }
  function terms(e,m,a,b) {
    const s=e.state, f=s.form[m.name] || {}, fa=f[a], fb=f[b], r=s.ratings[m.name];
    const have=fa && fb && fa[2]>=2 && fb[2]>=2;
    const x={prior_level:m.initial[a]-m.initial[b],elo_change:(r[a]-m.initial[a])-(r[b]-m.initial[b]),dO:have?fa[0]-fb[0]:0,dD:have?fa[1]-fb[1]:0,hfa:0,pff_O_diff:(s.pff?.[a]?.[0]??0)-(s.pff?.[b]?.[0]??0),pff_D_diff:(s.pff?.[a]?.[1]??0)-(s.pff?.[b]?.[1]??0),war_delta_diff:(s.war?.[a]??0)-(s.war?.[b]??0)};
    const st=stack(m,s);
    return features.map(({key})=>{const i=st.columns.indexOf(key);return i<0?0:x[key]*st.coef[i]/st.scale[i];});
  }
  function matchup(e,a,b,mask=127) {
    return mean(e.members.map(m=>sigmoid(terms(e,m,a,b).reduce((s,t,i)=>s+((mask & (1<<i))?t:0),0))));
  }
  function explain(e,team,names) {
    if (!names.includes(team) || names.length<2) throw new Error('A rated team and at least two teams are required.');
    const opponents=names.filter(t=>t!==team), values=new Array(128).fill(0), members=[];
    for (const m of e.members) {
      let total=0;
      for (const other of opponents) {
        const t=terms(e,m,team,other), sums=new Float64Array(128);
        values[0]+=.5;
        for(let mask=1;mask<128;mask++) {
          const bit=mask & -mask, i=31-Math.clz32(bit);
          sums[mask]=sums[mask^bit]+t[i];
          const p=sigmoid(sums[mask]); values[mask]+=p;
          if(mask===127) total+=p;
        }
      }
      members.push({name:m.name,power:total/opponents.length,stack:stack(m,e.state)});
    }
    const denom=e.members.length*opponents.length;
    for(let i=0;i<128;i++)values[i]/=denom;
    // Exact Shapley values: average a signal's marginal effect over all orders.
    const fact=[1,1,2,6,24,120,720,5040];
    const contributions=features.map((f,i)=>{
      let value=0;
      for(let mask=0;mask<128;mask++)if(!(mask & (1<<i))) {
        let n=0;for(let bits=mask;bits;bits&=bits-1)n++;
        value+=fact[n]*fact[6-n]/fact[7]*(values[mask | (1<<i)]-values[mask]);
      }
      return {...f,value,active:e.members.some(m=>stack(m,e.state).columns.includes(f.key))};
    });
    return {team,power:values[127],values,members,contributions,opponents:opponents.map(other=>({team:other,power:matchup(e,team,other)})).sort((a,b)=>b.power-a.power)};
  }
  const api={features,sigmoid,stack,terms,matchup,explain};
  if(typeof module==='object' && module.exports)module.exports=api;else root.PowerXray=api;
})(typeof globalThis!=='undefined'?globalThis:this);
