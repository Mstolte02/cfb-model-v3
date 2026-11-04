/* Explicit winner-only scenarios. No unanswered game receives a default winner. */
(function(root){
  'use strict';
  const P4=['ACC','Big Ten','Big 12','SEC'];
  const G6=['American Athletic','Conference USA','Mid-American','Mountain West','Pac-12','Sun Belt'];
  const key=g=>`${g.h}|${g.a}|${g.d||g.w}`;
  const final=g=>Boolean(g.f)&&Number.isFinite(g.hp)&&Number.isFinite(g.ap);
  const winner=g=>final(g)&&g.hp!==g.ap?(g.hp>g.ap?g.h:g.a):null;
  const choose=(h,a,p,mode,rng=Math.random)=>mode==='favorite'?(p>=.5?h:a):mode==='underdog'?(p<.5?h:a):(rng()<(mode==='coin'?.5:p)?h:a);
  const z=values=>{const n=values.length,m=values.reduce((a,b)=>a+b,0)/n,sd=Math.sqrt(values.reduce((s,v)=>s+(v-m)**2,0)/Math.max(1,n-1));return values.map(v=>sd?(v-m)/sd:0);};
  function evaluate({schedule,meta,ratings,committee,playoff,picks={},titles={}}){
    const names=ratings.map(r=>r.team),fbs=new Set(names),conf=t=>meta[t]?.conference;
    const records=Object.fromEntries(names.map(t=>[t,{wins:0,losses:0,ties:0,played:0,cWins:0,cGames:0}]));
    const rows=schedule.filter(g=>(fbs.has(g.h)||fbs.has(g.a))&&g.st!=='postseason').map(g=>({g,key:key(g),locked:final(g),winner:final(g)?winner(g):[g.h,g.a].includes(picks[key(g)])?picks[key(g)]:null}));
    const add=(h,a,w,c,tie=false)=>{
      for(const t of [h,a].filter(t=>fbs.has(t))){const r=records[t];r.played++;if(tie){r.ties++;}else if(t===w)r.wins++;else r.losses++;if(c){r.cGames++;r.cWins+=tie?.5:t===w?1:0;}}
    };
    for(const r of rows)if(r.winner||r.locked)add(r.g.h,r.g.a,r.winner,r.g.c&&conf(r.g.h)===conf(r.g.a),r.locked&&r.g.hp===r.g.ap);
    const remaining=rows.filter(r=>!r.locked&&!r.winner).length;
    const power=Object.fromEntries(ratings.map(r=>[r.team,r.power]));
    const meetings=rows.filter(r=>r.winner).map(r=>[r.winner,r.winner===r.g.h?r.g.a:r.g.h]);
    const titleGames=[];
    if(!remaining)for(const c of [...P4,...G6]){
      const members=names.filter(t=>conf(t)===c);
      members.sort((a,b)=>{const A=records[a],B=records[b],diff=B.cWins/Math.max(1,B.cGames)-A.cWins/Math.max(1,A.cGames);if(Math.abs(diff)>1e-10)return diff;const h2h=meetings.filter(([w,l])=>w===a&&l===b).length-meetings.filter(([w,l])=>w===b&&l===a).length;return -h2h||(power[b]-power[a])||a.localeCompare(b);});
      if(members.length<2)continue;
      const [h,a]=members,w=[h,a].includes(titles[c])?titles[c]:null;
      titleGames.push({conf:c,h,a,winner:w});if(w)add(h,a,w,false);
    }
    const titleRemaining=titleGames.filter(t=>!t.winner).length;
    const complete=!remaining&&titleGames.length>0&&!titleRemaining;
    let field=[],out=[],order=[];
    if(complete){
      const quality=z(names.map(t=>power[t])),q=Object.fromEntries(names.map((t,i)=>[t,quality[i]]));
      const sos=names.map(t=>{const opp=rows.flatMap(({g})=>g.h===t?[g.a]:g.a===t?[g.h]:[]);return opp.reduce((s,o)=>s+(q[o]??-2),0)/Math.max(1,opp.length);}),sosZ=z(sos);
      const w={win_pct:10.3397,rating:.2262,sos:.7362,power_conf:1.0864,...playoff?.provisional_weights};
      const stat=Object.fromEntries(names.map((t,i)=>[t,w.rating*q[t]+w.sos*sosZ[i]+w.power_conf*(P4.includes(conf(t))||t==='Notre Dame'?1:0)]));
      const wp=t=>(records[t].wins+records[t].ties*.5)/Math.max(1,records[t].played);
      let scores=Object.fromEntries(names.map(t=>[t,w.win_pct*wp(t)+stat[t]]));
      if(committee?.season===playoff?.season&&committee.rankings?.length===25){
        const base=Object.fromEntries(names.map(t=>[t,{wins:0,games:0}]));
        for(const {g,locked} of rows)if(locked&&g.d?.slice(0,10)<=committee.games_through){for(const t of [g.h,g.a].filter(t=>fbs.has(t))){base[t].games++;base[t].wins+=g.hp===g.ap?.5:t===winner(g)?1:0;}}
        const baseline=Object.fromEntries(names.map(t=>[t,w.win_pct*(base[t].games?base[t].wins/base[t].games:.5)+stat[t]]));
        const ranked=committee.rankings,meanX=ranked.reduce((s,r)=>s+r.rank,0)/ranked.length,meanY=ranked.reduce((s,r)=>s+baseline[r.team],0)/ranked.length;
        const slope=ranked.reduce((s,r)=>s+(r.rank-meanX)*(baseline[r.team]-meanY),0)/ranked.reduce((s,r)=>s+(r.rank-meanX)**2,0),spacing=Math.max(Math.abs(slope),Math.sqrt(Number.EPSILON));
        const official=new Map(ranked.map(r=>[r.team,r.rank]));
        for(const t of names){const anchor=official.has(t)?meanY+spacing*(meanX-official.get(t)):Math.min(baseline[t],meanY+spacing*(meanX-26));scores[t]+=anchor-baseline[t];}
      }
      order=names.slice().sort((a,b)=>scores[b]-scores[a]||a.localeCompare(b));
      const auto=new Map(titleGames.filter(t=>P4.includes(t.conf)).map(t=>[t.winner,`${t.conf} champion`]));
      const g6=order.find(t=>G6.includes(conf(t)));if(g6)auto.set(g6,'Group of 6 bid');
      if(order.slice(0,12).includes('Notre Dame'))auto.set('Notre Dame','Notre Dame Top 12');
      const selected=new Set(auto.keys());for(const t of order){if(selected.size>=12)break;selected.add(t);}
      field=order.filter(t=>selected.has(t)).map((team,i)=>({team,seed:i+1,auto:auto.has(team),bid:auto.get(team)||'At-large',...records[team]}));
      out=order.filter(t=>!selected.has(t)).slice(0,2);
    }
    return {rows,records,remaining,titleGames,titleRemaining,complete,field,out,order};
  }
  function resolvePicks(games,feeds,picks={}){
    const result=games.map(g=>({round:g.round,top:g.top||null,bottom:g.bottom||null,site:g.site||null,winner:null}));
    for(let i=0;i<result.length;i++){
      const g=result[i],src=feeds[i]||feeds[String(i)];
      if(src){const advanced=src.map(k=>result[k].winner);if(!g.top)g.top=advanced[0]||null;if(!g.bottom)g.bottom=advanced.length>1?advanced[1]||null:advanced[0]||null;}
      if(g.top&&g.bottom&&[g.top,g.bottom].includes(picks[i]))g.winner=picks[i];
    }return result;
  }
  const api={key,final,winner,choose,evaluate,resolvePicks};root.ScenarioLab=api;if(typeof module!=='undefined')module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
