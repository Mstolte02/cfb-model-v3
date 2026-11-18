/* Freeze Sunday/Monday Committee Watch predictions before the Tuesday CFP release.
   On the next official poll, attach the actual ranking and accuracy. */
'use strict';
const fs=require('node:fs'),path=require('node:path');
const C=require('../viz/committee-watch.js');
const root=path.resolve(__dirname,'../viz/data');
const read=name=>JSON.parse(fs.readFileSync(path.join(root,name),'utf8'));
const shift=(date,days)=>{const d=new Date(date+'T12:00:00Z');d.setUTCDate(d.getUTCDate()+days);return d.toISOString().slice(0,10);};

function update({archive,schedule,teams,poll,model,now}){
  const entries=structuredClone(archive?.entries||[]);
  for(const entry of entries){
    if(entry.actual||entry.target_release!==poll.released)continue;
    const actual=poll.rankings.map(x=>x.team),predicted=entry.predicted;
    const ranks=new Map(predicted.map((team,i)=>[team,i+1]));
    entry.actual=actual;
    entry.metrics={rank_mae:Number((actual.reduce((s,team,i)=>s+Math.abs((ranks.get(team)||30)-(i+1)),0)/25).toFixed(3)),
      top12_overlap:actual.slice(0,12).filter(t=>predicted.slice(0,12).includes(t)).length,
      top25_overlap:actual.filter(t=>predicted.includes(t)).length};
  }
  const res=C.project({schedule,teams,poll,model,now});
  if(res.status==='projection'&&res.newResults){
    const day=C.weeklyWindow(now).today;
    const weekday=new Date(day+'T12:00:00Z').getUTCDay();
    const target=shift(day,weekday===0?2:1);
    const forecast={target_release:target,captured_at:new Date(now).toISOString(),
      input_release:poll.released,results_through:res.through,
      predicted:res.rows.slice(0,25).map(x=>x.team),actual:null,metrics:null};
    const index=entries.findIndex(e=>e.target_release===target);
    if(index<0)entries.push(forecast);
    else if(!entries[index].actual)entries[index]=forecast;
  }
  return {version:1,entries:entries.sort((a,b)=>a.target_release.localeCompare(b.target_release))};
}
if(require.main===module){
  const p=path.join(root,'committee_forecast_archive.json');
  const archive=fs.existsSync(p)?JSON.parse(fs.readFileSync(p,'utf8')):{entries:[]};
  const poll=read('cfp_rankings.json'),schedule=read('schedule.json'),model=read('committee_timeline_model.json');
  const playoff=read('playoff_current.json');
  const teams=playoff.teams.map(x=>({team:x.team,conference:x.conference,rating:x.rating}));
  const next=update({archive,schedule,teams,poll,model,now:process.env.CFP_TIMESTAMP||new Date()});
  if(JSON.stringify(next)!==JSON.stringify(archive))fs.writeFileSync(p,JSON.stringify(next,null,2)+'\n');
  console.log('Committee forecast archive:',next.entries.length,'releases');
}
module.exports={update};
