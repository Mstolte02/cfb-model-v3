const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {execFileSync}=require('node:child_process');
const path=require('node:path');
const X=require('../viz/power-xray/engine.js');
const root=path.resolve(__dirname,'..');
const pythonExecutable=process.env.PYTHON || (process.platform==='win32' ? 'python' : 'python3');
const e=JSON.parse(fs.readFileSync(path.join(root,'viz/data/model_v4.json'))).ensemble;
const rows=JSON.parse(fs.readFileSync(path.join(root,'viz/data/ratings.json'))).teams;
const names=rows.map(r=>r.team);
const close=(a,b,tol=1e-11)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test('all team ratings reconcile and Shapley contributions sum to the exact result',()=>{
  const before=JSON.stringify(e);
  for(const r of rows){
    const v=X.explain(e,r.team,names);
    close(v.power,r.power,0.000002);
    close(.5+v.contributions.reduce((s,c)=>s+c.value,0),v.power);
    close(v.values[0],.5);
    close(v.members.reduce((s,m)=>s+m.power,0)/e.members.length,v.power);
    for(const c of v.contributions)if(!c.active)close(c.value,0);
    assert.ok(v.values.every(v=>v>=0 && v<=1));
  }
  assert.equal(JSON.stringify(e),before,'explanations must not mutate the model');
});
test('matches canonical Python runtime for every neutral matchup',()=>{
  const python=`import json\nfrom scripts.ensemble_replay import probability\ne=json.load(open('viz/data/model_v4.json'))['ensemble']\nn=[r['team'] for r in json.load(open('viz/data/ratings.json'))['teams']]\nprint(json.dumps([[a,b,probability(e,e['state'],a,b,0)] for i,a in enumerate(n) for b in n[i+1:]]))`;
  const pairs=JSON.parse(execFileSync(pythonExecutable,['-c',python],{cwd:root,maxBuffer:8*1024*1024}));
  for(const [a,b,p] of pairs){close(X.matchup(e,a,b),p);close(X.matchup(e,b,a),1-p);}
});
test('stack availability, empty payloads, legacy slates and insufficient form use canonical rules',()=>{
  const variants=[];
  for(const kind of ['base','pff','empty','legacy','missing-form']){
    const v=structuredClone(e);
    if(kind==='base'){v.state.pff=null;v.state.war=null;}
    if(kind==='pff')v.state.war=null;
    if(kind==='empty'){v.state.pff={};v.state.war={};}
    if(kind==='legacy')v.state.slate=1;
    if(kind==='missing-form')for(const m of v.members)v.state.form[m.name]={[names[0]]:[10,10,1]};
    variants.push(v);
  }
  const python=`import json,sys\nfrom scripts.ensemble_replay import probability\nv=json.load(sys.stdin)\nprint(json.dumps([probability(e,e['state'],${JSON.stringify(names[0])},${JSON.stringify(names[1])},0) for e in v]))`;
  const expected=JSON.parse(execFileSync(pythonExecutable,['-c',python],{cwd:root,input:JSON.stringify(variants),maxBuffer:8*1024*1024}));
  variants.forEach((v,i)=>close(X.matchup(v,names[0],names[1]),expected[i]));
});
