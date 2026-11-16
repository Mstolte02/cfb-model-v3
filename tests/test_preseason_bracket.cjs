'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('viz/app.js','utf8');
const render=source.slice(source.indexOf('  function renderPlayoff() {'),source.indexOf('  /* ---------- how each team in the bracket got there'));
function harness(version,lock=true){
  const elements={};
  const games=[{top:'A',bottom:'B',p:.8,winner:'A'}];
  let resolved=0,shown;
  const playoff={n_sims:2000,teams:[],basis:'Week-zero',snapshot_lock:lock?{version:1}:null,bracket:{games,seeds:['A','B'],feeds:{}}};
  const context={playoffVersion:version,cur:()=>({playoff}),document:{getElementById:id=>elements[id]??=( {innerHTML:''})},
    esc:String,crestOf:()=>({}),meta:{},renderSelection:()=>{},renderResumes:()=>{},wireTeamLinks:()=>{},
    resolveBracket:()=>{resolved++;return [{winner:'B',p:.1}];},
    MarksPlayoff:{projectedHTML:input=>{shown=input.games;return 'bracket';}}};
  vm.runInNewContext(render+';renderPlayoff();',context);
  return {elements,games,shown,resolved};
}
test('preseason renders its stored winners and probabilities without live resolution',()=>{
  const r=harness('preseason');assert.equal(r.resolved,0);assert.equal(r.shown,r.games);
  assert.equal(r.shown[0].winner,'A');assert.equal(r.shown[0].p,.8);
});
test('current bracket still resolves with the current model',()=>{
  const r=harness('current');assert.equal(r.resolved,1);assert.equal(r.shown[0].winner,'B');
});
test('an incomplete preseason snapshot never falls back to live matchups',()=>{
  const r=harness('preseason',false);assert.equal(r.resolved,0);assert.equal(r.shown,undefined);
  assert.match(r.elements.bracket.innerHTML,/being rebuilt/);
});
