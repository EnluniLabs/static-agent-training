function puzzleSize(env){return ['switches-2x2','lights-out-2x2'].includes(env)?2:['lights-out','switchboard'].includes(env)?3:0;}
'use strict';
const $ = id => document.getElementById(id);
const state = {token: '', runs: [], populations: [], selected: null, detail: null, view: 'overview', snapshots: [], snapshotEpoch: 0, attemptEpoch: 0, detailEpoch: 0, busy: false, attemptBusy: false};
const labels = {'agent-q-training':'Agent Q evolution','agent-z-training':'Agent Z evolution',assessment:'Frozen population assessment','starting-population-benchmark':'Evolutionary starting-population benchmark','population-benchmark':'Saved population benchmark',training: 'Categorical evolution', 'sensory-training': 'Fixed-senses evolution', senses: 'Fixed-senses adaptation', interfaces: 'Lifetime model benchmark', comparison: 'Treatment comparison', 'model-comparison': 'Evolutionary model comparison'};
const colors = ['#83b5ff', '#71dbb5', '#ffc780', '#a89ae8'];
const pct = n => typeof n === 'number' && Number.isFinite(n) ? (100*n).toFixed(2)+'%' : '—';
const pp = n => typeof n === 'number' ? (n>=0?'+':'')+n.toFixed(2)+' pp' : '—';
function node(tag, text, cls) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; }
async function api(path, data) {
  const options=data===undefined?{}:{method:'POST', headers:{'Content-Type':'application/json','X-Workspace-Token':state.token},body:JSON.stringify(data)};
  const response=await fetch(path, options), result=await response.json();
  if(!response.ok)throw Error(result.error||'Request failed'); return result;
}
function view(name) {
  state.view=name; document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!=='view-'+name);
  document.querySelectorAll('[data-view]').forEach(e=>{e.classList.toggle('active',e.dataset.view===name);e.setAttribute('aria-current',e.dataset.view===name?'page':'false');});
  if(name==='playground')updatePlayRuns();
  if(name==='assessments'){assessmentChoices();assessmentHistory();loadAssessment();}
  if(name==='overview'&&state.detail)chart(state.detail);
}
document.querySelectorAll('[data-view]').forEach(e=>e.onclick=()=>view(e.dataset.view));
function currentRuns(){return state.runs.filter(r=>['agent-x','agent-y','agent-z','agent-q'].includes(r.model_family)||['assessment','starting-population-benchmark'].includes(r.kind));}
function familyLabel(r){return r.model_family==='agent-q'?'Agent Q':r.model_family==='agent-z'?'Agent Z':r.model_family==='agent-y'?'Agent Y':r.model_family==='agent-x'?'Agent X':labels[r.kind]||r.kind;}
function renderLibrary() {
  const filter=$('filter').value, search=$('search').value.trim().toLowerCase(), scroll=$('session-list').scrollTop;
  const rows=currentRuns().filter(r=>(filter==='all'||r.kind===filter||r.model_family===filter)&&((r.name+' '+(r.session?.name||'')).toLowerCase().includes(search)));
  const cards=rows.map(r=>{const b=node('button',undefined,'session-card kind-'+r.kind);b.classList.toggle('selected',r.run===state.selected);b.setAttribute('aria-current',r.run===state.selected?'true':'false');b.append(node('span',familyLabel(r),'kind-tag'),node('strong',r.session?.name||r.name),node('small',r.state+(r.progress?' · '+r.progress:'')));b.onclick=()=>selectRun(r.run);return b;});
  $('session-list').replaceChildren(...(cards.length?cards:[node('p','No matching sessions.','muted')]));$('session-list').scrollTop=scroll;
}
$('filter').onchange=renderLibrary;$('search').oninput=renderLibrary;
async function selectRun(run) {
  state.selected=run;state.detailEpoch++;view('overview');renderLibrary();await loadDetail();
}
function metric(label,value){const d=node('div',undefined,'metric-card');d.append(node('strong',value),node('span',label));return d;}
function svg(tag,attrs={},text) {const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e;}
function chart(detail) {
  const training=['training','sensory-training'].includes(detail.kind),complexity=training&&$('chart-metric').value==='complexity',structure=training&&['repertoire','complexity'].includes($('chart-metric').value);
  $('chart-metric-control').hidden=!training;$('chart-title').textContent=complexity?'Structural growth':structure?'Initial pairs':'Performance';
  const format=structure?(n=>Number.isFinite(n)?n.toFixed(2):'—'):pct;
  const points=detail.series||[], defs=complexity?[['mean_repertoire','Mean developed states'],['mean_convergence_states','Mean convergence states'],['mean_seed_term_nodes','Mean seed AST nodes']]:structure?[['mean_unique_initial_pairs','Mean unique initial pairs per animal']]:detail.chart?.series||[], target=$('trend');target.replaceChildren();$('chart-legend').replaceChildren();
  const width=Math.max(340,target.clientWidth||1000);target.setAttribute('viewBox',`0 0 ${width} 320`);
  for(const [i,[key,label]] of defs.entries()){const entry=node('span');const dot=node('i');dot.style.background=colors[i%colors.length];entry.append(dot,document.createTextNode(label));$('chart-legend').append(entry);}
  const values=points.flatMap(p=>defs.map(([k])=>p[k])).filter(v=>typeof v==='number'&&Number.isFinite(v));
  const ymax=(structure?Math.max(1,Math.ceil(Math.max(0,...values)*1.12)):Math.min(1,Math.ceil(Math.max(.1,...values)*1.12*10)/10)), xmin=points[0]?.x||0, xmax=points.at(-1)?.x||1;
  const x=n=>60+(n-xmin)*(width-85)/Math.max(1,xmax-xmin), y=n=>270-230*n/ymax;
  for(let i=0;i<=4;i++){const value=ymax*i/4;target.append(svg('line',{x1:60,x2:width-25,y1:y(value),y2:y(value),stroke:'#29374b','stroke-dasharray':i?'3 5':''}),svg('text',{x:48,y:y(value)+4,fill:'#9aaac0','font-size':12,'text-anchor':'end'},structure?Number(value.toFixed(1)):Math.round(value*100)+'%'));}
  if(!points.length){target.append(svg('text',{x:width/2,y:150,fill:'#9aaac0','text-anchor':'middle','font-size':12},'Waiting for a saved result'));}
  else {
    for(let i=0;i<Math.min(5,points.length);i++){const p=points[Math.round(i*(points.length-1)/Math.max(1,Math.min(5,points.length)-1))];target.append(svg('text',{x:x(p.x),y:294,fill:'#9aaac0','font-size':11,'text-anchor':'middle'},p.label?p.label.replaceAll('-',' '):p.x));}
    for(const [i,[key,label]] of defs.entries()) {let d='';for(const p of points){if(typeof p[key]!=='number'||!Number.isFinite(p[key]))continue;d+=(d?' L':'M')+x(p.x)+' '+y(Math.max(0,p[key]));}target.append(svg('path',{d,fill:'none',stroke:colors[i%colors.length],'stroke-width':2.5,'stroke-linecap':'round'}));
      const shown=points.length<60?points:points.filter((p,j)=>j===points.length-1||j%Math.ceil(points.length/40)===0);
      for(const p of shown){if(typeof p[key]!=='number')continue;const dot=svg('circle',{cx:x(p.x),cy:y(p[key]),r:points.length<10?4:2,fill:colors[i%colors.length]});dot.append(svg('title',{},`${label}: ${format(p[key])} · ${p.label||p.x}`));target.append(dot);}}
  }
  $('trend').setAttribute('aria-label',complexity?'Structural growth by generation':structure?'Average unique initial pairs per animal by generation':'Task success over session progress');
  $('chart-caption').textContent=complexity?'Generation · mean counts per animal':structure?'Generation · mean unique initial pairs per animal':detail.chart?.x||'Task success';
  $('chart-note').textContent=complexity?'Developed states, states with at least two unique incoming edges, and seed term AST nodes, averaged per animal. Missing historical measurements are unavailable. These counts do not establish better performance.':structure?'Count distinct ordered seed pairs within each animal, then average across the evaluated population. Developed successor states are excluded. Older generations without this metric are unavailable.':detail.kind==='interfaces'?'Cumulative means across completed pairs. Frozen task success; no evolution.':detail.kind==='senses'?'Frozen scores by condition. These points are not a learning curve.':detail.kind==='starting-population-benchmark'?'Both arms evolve through these generations. Training and held-out curves show population means; the final test compares frozen training-selected champions.':detail.kind==='model-comparison'?'Frozen training-selected champions. Open each training population for its generation-by-generation means.':'Only completed, saved evaluations are shown.';
}
function table(headings,rows){const t=node('table'),head=node('tr');headings.forEach(h=>head.append(node('th',h)));const thead=node('thead');thead.append(head);t.append(thead);const body=node('tbody');rows.forEach(values=>{const tr=node('tr');values.forEach(v=>{const cell=node('td');if(v instanceof Node)cell.append(v);else cell.textContent=v;tr.append(cell);});body.append(tr);});t.append(body);return t;}
function renderDetail(d) {
  state.detail=d;const row=state.runs.find(r=>r.run===state.selected), report=d.report||{}, latest=d.series?.at(-1)||{};
  $('run-title').textContent=row?.session?.name||row?.name||d.run;$('run-subtitle').textContent=(familyLabel(row||{})||'Session')+' · '+d.run+((d.config?.environment||report.config?.environment)==='switchboard'?' · Three-panel switchboard':(d.config?.environment||report.config?.environment)==='switches-2x2'?' · Simple switches 2×2':(d.config?.environment||report.config?.environment)==='lights-out-2x2'?' · Lights Out 2×2':(d.config?.environment||report.config?.environment)==='lights-out'?' · Lights Out 3×3':(d.config?.environment||report.config?.environment)==='shifted'?' · Shifted [30, 70]':'');
  if(row?.elapsed_seconds!=null){const minutes=Math.floor(row.elapsed_seconds/60);$('run-subtitle').textContent+=` · ${minutes} min elapsed`+(row.remaining_seconds!=null?` · about ${Math.ceil(row.remaining_seconds/60)} min remaining`:'');}
  const evaluation=d.evaluation_progress;
  if(row?.state==='running'&&evaluation?.phase==='evaluating')$('run-subtitle').textContent+=` · Generation ${evaluation.generation+1}: ${evaluation.completed}/${evaluation.total} agents evaluated`;
  $('run-purpose').textContent=d.session?.config?.notes||d.run_metadata?.notes||'';$('run-purpose').hidden=!$('run-purpose').textContent;
  $('run-state').textContent=row?.state||report.status||'Saved';$('run-state').className='badge '+(row?.state||'');
  let metrics=[];
  if(['training','sensory-training','agent-z-training','agent-q-training'].includes(d.kind))metrics=[['Mean training',pct(latest.mean)],['Mean held-out',pct(latest.assessment_mean)],['Random baseline',pct(latest.random_assessment)],['Generations',row?.progress||'0'],['Mean developed states / animal',Number.isFinite(latest.mean_repertoire)?latest.mean_repertoire.toFixed(2):'—'],['Mean convergence states / animal',Number.isFinite(latest.mean_convergence_states)?latest.mean_convergence_states.toFixed(2):'—'],['Mean unique initial pairs / animal',Number.isFinite(latest.mean_unique_initial_pairs)?latest.mean_unique_initial_pairs.toFixed(2):'—']];
  if(['agent-z-training','agent-q-training'].includes(d.kind)){
    metrics=[['Mean training',pct(latest.mean)],['Champion held-out',pct(latest.held_out)],
             ['Generations',row?.progress||'0'],['Champion lambda pairs',String(latest.repertoire??'—')]];
    if(d.config?.speciation)metrics.push(['Species',String(latest.species_count??'—')]);
  }
  if(d.kind==='assessment')metrics=[['Frozen candidates',`${report.progress?.completed_animals||0} / ${report.progress?.total_animals||0}`],['Populations',String(report.populations?.length||0)],['Episodes per candidate',String(report.config?.episodes||0)],['Random baseline',pct(report.random_success)]];
  if(d.kind==='population-benchmark')metrics=[['Completed populations',String(report.populations?.length||0)],['Random baseline',pct(report.random_assessment)],['Training episodes',String(report.config?.episodes||0)],['Held-out episodes',String(report.config?.assessment_episodes||0)]];
  if(d.kind==='interfaces')metrics=[['Old categorical mean',pct(report.population_mean?.old)],['Fixed-senses mean',pct(report.population_mean?.senses)],['Fixed senses − old',pp(report.paired_effect?.mean_percentage_points)],['Completed pairs',`${report.completed||0} / ${report.target||0}`]];
  if(d.kind==='senses')metrics=[['Prior adapted · expanded',pct(report.results?.['prior-adapting']?.frozen_success_by_region?.expanded)],['Fresh adapted · expanded',pct(report.results?.['fresh-adapting']?.frozen_success_by_region?.expanded)],['Original learning',pct(report.training?.success_rate)],['Fixed receptors',String(report.senses?.receptors||'—')]];
  if(['comparison','model-comparison','starting-population-benchmark'].includes(d.kind))metrics=[[d.kind==='starting-population-benchmark'?'Final B − A':'Treatment effect',pp(report.paired_effect?.mean_percentage_points)],['Completed seeds',String(report.paired_effect?.seeds||0)],['Population per arm',String(report.population||'—')],['Generations',String(report.generations||'—')]];
  $('delete-session').disabled=['running','stopping','finishing','recorded-running','recorded running'].includes(row?.state);
  $('metric-cards').replaceChildren(...metrics.map(([l,v])=>metric(l,v)));chart(d);
  const effect=report.paired_effect;if(effect?.interval95)$('chart-note').textContent+=` Seed-paired 95% interval: ${pp(effect.interval95[0])} to ${pp(effect.interval95[1])}.`;
  $('results-panel').hidden=d.kind==='assessment'?!report.populations?.length:d.kind==='population-benchmark'?!report.populations?.length:!d.results?.length;
  if(d.kind==='assessment')$('results-table').replaceChildren(frozenResultsTable(report));
  if(d.kind==='population-benchmark')$('results-table').replaceChildren(table(['Population','Model','Animals','Lifetime training','Frozen assessment'],(report.populations||[]).map(p=>[p.name,p.model,p.count,pct(p.training_mean),pct(p.assessment_mean)])));
  if(d.results?.length){const rows=d.results.map(r=>[r.seed,pct(r.old??r.control),pct(r.senses??r[report.treatment||'speciation']),pp(r.difference_percentage_points)]);$('results-table').replaceChildren(table(['Seed',d.kind==='interfaces'?'Old categorical':report.control_label||'Control',d.kind==='interfaces'?'Fixed senses':report.treatment_label||'Treatment','Difference'],rows));}
  $('arms-panel').hidden=!report.arms?.length;if(report.arms?.length){const rows=report.arms.map(arm=>{const source=state.runs.find(r=>r.name===arm.run_name);const link=node('button',(arm.enabled?report.treatment_label:report.control_label)||arm.run_name,'secondary');link.disabled=!source;link.onclick=()=>selectRun(source.run);return[arm.seed,link,`${arm.completed}/${arm.target}`,pct(arm.latest?.mean),pct(arm.latest?.assessment_mean),pct(arm.final_benchmark?.animal?.success_rate)];});$('arms-table').replaceChildren(table(['Seed','Model / arm','Generations','Population training','Population held-out','Selected champion final'],rows));}
  $('config-output').textContent=JSON.stringify(d.config||report.plan||d.run_metadata||report.spec||{},null,2);
  $('artifacts').replaceChildren(...(d.artifacts||[]).map(file=>{const a=node('a',file);a.href='/api/artifact?'+new URLSearchParams({run:state.selected,file});a.download=file;return a;}));
  $('limitations').replaceChildren(...(d.limitations||[]).map(text=>node('p',text)));
  const job=d.session;const active=job&&['running','stopping'].includes(job.state);
  $('stop-session').hidden=!active;$('stop-session').disabled=job?.state==='stopping';
  $('stop-session').textContent=job?.state==='stopping'?'Stopping…':'Stop session';
  const generational=['training','sensory-training','agent-z-training','agent-q-training'].includes(d.kind);
  const resumable=!['running','stopping','finishing'].includes(row?.state)&&((generational&&d.files.includes('checkpoint.json.gz'))||(['interfaces','comparison','model-comparison','starting-population-benchmark'].includes(d.kind)&&['stopped','failed','interrupted'].includes(row?.state)));
  $('resume-session').textContent=d.config?.agent_y&&d.config.backend==='python'?'Resume with native CPU':'Resume session';
  $('resume-session').hidden=!resumable;$('resume-options').hidden=!(resumable&&generational);
  if(generational&&document.activeElement!==$('resume-generations')){const current=d.config?.generations||50;if($('resume-generations').dataset.run!==d.run||(row?.state==='completed'&&Number($('resume-generations').value)<=current)){$('resume-generations').value=current+(row?.state==='completed'?100:0);$('resume-generations').dataset.run=d.run;}$('resume-generations').min=current;}
  $('resume-session').textContent=row?.state==='completed'?'Continue training':'Resume session';
  $('ownership').textContent=job?'Stopping preserves the last completed checkpoint.':'Saved session · external live processes remain view-only.';
  $('save-population').disabled=!['training','sensory-training','agent-z-training','agent-q-training'].includes(d.kind)||!d.series.length;$('candidate-save-options').hidden=!['training','sensory-training','agent-z-training','agent-q-training'].includes(d.kind);if($('candidate-save-count').dataset.run!==d.run){$('candidate-save-count').value=d.config?.population_save_count||10;$('candidate-save-count').dataset.run=d.run;$('candidate-set-name').value='';}
  $('open-model').disabled=!['sensory-training','agent-z-training','agent-q-training'].includes(d.kind)||!currentRuns().some(r=>r.run===d.run);
}
async function loadDetail(){const run=state.selected,epoch=++state.detailEpoch;if(!run)return;try{const d=await api('/api/run?'+new URLSearchParams({run}));if(run===state.selected&&epoch===state.detailEpoch){renderDetail(d);$('overview-error').textContent='';await loadLog();}}catch(e){if(run===state.selected)$('overview-error').textContent=e.message;}}
async function loadLog(){const run=state.selected;if(!run)return;try{const d=await api('/api/log?'+new URLSearchParams({run,source:$('log-source').value}));if(run===state.selected){const p=$('log-output'),follow=p.scrollHeight-p.scrollTop-p.clientHeight<40;p.textContent=d.text;if(follow)p.scrollTop=p.scrollHeight;}}catch(e){$('log-output').textContent=e.message;}}
$('log-source').onchange=loadLog;
$('stop-session').onclick=async()=>{try{await api('/api/stop',{id:state.detail.session.id});await refresh();}catch(e){$('overview-error').textContent=e.message;}};
$('resume-session').onclick=async()=>{try{const data={run:state.selected};if(state.detail.config?.agent_y&&state.detail.config.backend==='python')data.backend='native';if(['training','sensory-training','agent-z-training','agent-q-training'].includes(state.detail.kind))data.generations=Number($('resume-generations').value);const job=await api('/api/resume-run',data);await refresh();await selectRun(job.run);}catch(e){$('overview-error').textContent=e.message;}};
$('save-population').onclick=async()=>{try{$('save-population').disabled=true;const result=await api('/api/save-population',{run:state.selected,count:Number($('candidate-save-count').value),name:$('candidate-set-name').value});state.populations=await api('/api/populations');$('overview-error').textContent=`Saved ${result.count} candidates as “${result.name}”. Choose this set under Load saved candidates in a new training session.`;}catch(e){$('overview-error').textContent=e.message;}finally{$('save-population').disabled=false;}};
$('open-model').onclick=()=>{const kind=state.detail?.kind;$('play-model').value=kind==='agent-q-training'?'agent-q':kind==='agent-z-training'?'agent-z':state.detail?.config?.agent_y?'agent-y':'agent-x';view('playground');$('play-run').value=state.selected;loadSnapshots();};

const common=[['episodes','Learning episodes',100],['assessment_episodes','Assessment episodes',200]];
function currentPopulations(){return state.populations.filter(p=>['agent-z','agent-q'].includes(p.model)||p.model==='senses'&&(p.agent_y||p.sensory_partitions>1||p.sensory_interface==='signals'));}
const numeric=(name,label,value)=>({name,label,value,type:'number'});
// Scientific starting settings, not evidence of improved fitness.
const structureKeys=['initial_pairs','chain_budget','seed_term_max_nodes','seed_term_max_depth','combinator_probability','development_max_nodes','development_max_transitions','subtree_probability','term_probability','insertion_probability','deletion_probability','new_pair_budget'];
const structurePresets={
  baseline:Object.fromEntries(structureKeys.map((key,i)=>[key,[4,64,31,6,.5,256,1000,.15,.10,.05,.05,64][i]])),
  exploratory:Object.fromEntries(structureKeys.map((key,i)=>[key,[16,256,127,10,.15,2048,2000,.35,.20,.25,.02,256][i]])),
};
const structureFields=[{name:'structure_preset',label:'Structure profile',type:'select',options:[['baseline','Baseline'],['exploratory','Exploratory'],['custom','Custom']]},
  ...structureKeys.map((key,i)=>numeric(key,['Starting pairs','Initial chain reduction budget','Seed term maximum nodes','Seed term maximum depth','Combinator probability','Development maximum nodes','Development maximum transitions','Subtree mutation probability','Term mutation probability','Insertion probability','Deletion probability','New pair reduction budget'][i],structurePresets.baseline[key]))];
function applyStructurePreset(){const preset=$('config-structure_preset').value;if(structurePresets[preset])for(const key of structureKeys)$('config-'+key).value=structurePresets[preset][key];setupSummary();}
function structureSummary(c){return (c.sensory_interface?` Apparatus fixed at birth: ${c.sensory_interface==='signals'?`${c.retina_width}×${c.retina_height} signal receptor layout · tiles · indexing only`:c.sensory_interface==='retina'?`${c.retina_width}×${c.retina_height} retina · tiles`:'legacy bytes'} · ${c.receptors} receptors · ${c.sensory_partitions} partitions. `:'')+` Starting pairs: ${c.initial_pairs}; seed terms: ${c.seed_term_max_nodes} nodes / depth ${c.seed_term_max_depth}; development: ${c.development_max_nodes} nodes / ${c.development_max_transitions} transitions; reductions: ${c.chain_budget} initial / ${c.new_pair_budget} new pairs.`;}
const environmentField={name:'environment',label:'Environment',type:'select',options:[['original','Original [−20, 20]'],['switchboard','Three-panel switchboard'],['shifted','Shifted [30, 70]'],['switches-2x2','Simple switches · 2×2'],['lights-out-2x2','Lights Out · 2×2 puzzle'],['lights-out','Lights Out · 3×3 puzzle']]};
const fields={
  training:[environmentField,{name:'initial_population',label:'Load saved candidates',type:'population'},numeric('population_save_count','Top candidates to retain',10),numeric('population','Animals',32),numeric('generations','Generations',50),...common.map(x=>numeric(...x)),numeric('seed','Seed',48),{name:'backend',label:'Evaluator',type:'select',options:[['native','Native CPU'],['python','Python reference']]},numeric('workers','CPU workers',4),numeric('reproduction_workers','Offspring processes · 0 automatic, 1 serial',0),numeric('batch_size','Batch size',256),numeric('champion_count','Generation champions to keep',10),numeric('memory_limit_mb','Memory ceiling (MiB)',4096),{name:'learning',label:'Lifetime learning',type:'checkbox',value:true},{name:'inherit_lifetime',label:'Inherit learned weights',type:'checkbox',value:false},{name:'speciation',label:'Structural speciation',type:'checkbox',value:false}],
};
fields['sensory-training']=[{name:'sensory_interface',label:'Sensory interface · immutable after birth',type:'select',options:[['signals','Signals · universal source bytes'],['legacy-bytes','Legacy bytes · historical controls'],['retina','Retina · historical packed binary pixels']]},numeric('retina_width','Receptor layout width',16),numeric('retina_height','Receptor layout height',16),{name:'embedding_selection',label:'Embedding selection',type:'select',options:[['argmax','Highest probability · random exact ties'],['sample','Probability-weighted sampling']]},numeric('receptors','Resolution · receptor count',256),numeric('sensory_partitions','Parallel sensory partitions',1),{name:'action_policy',label:'Parallel action policy',type:'select',options:[['agreement','Agreement · all proposals must agree'],['vote','Vote · most proposed action wins']]},...fields.training.map(f=>({...f}))];
fields['agent-z-training']=[
  {name:'initial_population',label:'Load saved Agent Z candidates',type:'population'},
  numeric('population','Animals',32),numeric('generations','Generations',50),numeric('episodes','Learning episodes per animal',500),
  numeric('regions','Parallel sensory regions',2),numeric('seed','Seed',7),
  {name:'structure_preset',label:'Structure profile',type:'select',options:[['baseline','Baseline'],['exploratory','Exploratory']]},
  {name:'learning',label:'Lifetime learning',type:'select',options:[['both','Empirical and structural'],['empirical','Empirical only'],['structural','Structural only'],['off','Off']]},
  {name:'speciation', label:'Structural speciation', type:'checkbox', value:false},
];
fields.training.push(...structureFields);
fields['sensory-training'].push(...structureFields);
fields['agent-q-training']=[
  ...fields['agent-z-training'].map(field=>field.name==='initial_population'?{...field,label:'Load saved Agent Q candidates'}:field),
  numeric('exploration','Uniform exploration probability',.05),
];
const descriptions={'agent-q-training':'Agent Q derives actions from predicted outcomes and their compatibility with the fixed successor. Native CPU · original number line.','agent-z-training':'Agent Z runs on the native CPU in the original −20 to 20 number line. The environment interprets universal output signals.','starting-population-benchmark':'Two full evolutionary sessions with matched tasks and budgets. Each arm can start fresh or load any saved candidate set for this model. Compare population curves across generations and a final frozen champion test.',training:'Evolutionary training with categorical mappings. Start fresh or load saved candidates, choose an environment, then evolve through selection and mutation.','sensory-training':'Evolutionary training with fixed senses. Start fresh or load saved candidates, choose an environment, then evolve through selection and mutation.'};
function setupKind(){const previous=$('session-kind').value,kind=$('model-family').value==='agent-q'?'agent-q-training':$('model-family').value==='agent-z'?'agent-z-training':['senses','agent-x','agent-y'].includes($('model-family').value)?'sensory-training':'training';fields['starting-population-benchmark']=[environmentField,{name:'population_a',label:'Arm A — starting candidates',type:'population'},{name:'population_b',label:'Arm B — starting candidates',type:'population'},...fields[kind].filter(f=>!['environment','initial_population'].includes(f.name)),numeric('benchmark_episodes','Final shared test episodes',2000)];$('session-kind').replaceChildren(...(['agent-y','agent-z','agent-q'].includes($('model-family').value)?[[kind,'Evolutionary training']]:[[kind,'Evolutionary training'],['starting-population-benchmark','Compare starting populations']]).map(([value,label])=>{const option=node('option',label);option.value=value;return option;}));if(previous==='starting-population-benchmark'&&!['agent-y','agent-z','agent-q'].includes($('model-family').value))$('session-kind').value=previous;renderFields();}
function renderFields(){ $('parallel-senses-help').hidden=!['senses','agent-x','agent-y'].includes($('model-family').value);const kind=$('session-kind').value;$('model-description').textContent=$('model-family').value==='agent-x'&&kind!=='starting-population-benchmark'?'Agent X activates several sensory regions independently and runs their chains in parallel. Configure how their action proposals are resolved.':descriptions[kind];$('session-fields').replaceChildren(...fields[kind].map(field=>{const label=node('label',field.label),input=node(['select','candidate','population','populations'].includes(field.type)?'select':'input');input.dataset.config=field.name;input.id='config-'+field.name;
  if(field.type==='select'){input.replaceChildren(...field.options.map(([v,l])=>{const o=node('option',l);o.value=v;return o;}));if(field.name==='sensory_interface')input.value=['agent-x','agent-y'].includes($('model-family').value)?'signals':'legacy-bytes';}
  else if(['population','populations'].includes(field.type)){input.multiple=field.type==='populations';if(!input.multiple){const empty=node('option','Start fresh');empty.value='';input.append(empty);}else input.size=5;for(const p of currentPopulations().filter(p=>['agent-z-training','agent-q-training'].includes(kind)?p.model===kind.replace('-training','')&&p.can_seed:p.model===(['senses','agent-x','agent-y'].includes($('model-family').value)?'senses':'categorical')&&Boolean(p.agent_y)===($('model-family').value==='agent-y'))){const option=node('option',`${p.name} · ${p.count} animals · ${p.model==='agent-q'?'Agent Q':p.model==='agent-z'?'Agent Z':p.agent_y?'Agent Y':'Agent X'} · ${p.sensory_interface||'legacy-bytes'} · ${p.environment}`);option.value=p.id;input.append(option);}}
  else if(field.type==='candidate'){const empty=node('option','Develop a fresh repertoire');empty.value='';input.append(empty);state.runs.filter(r=>r.kind==='training'&&r.summary).forEach(r=>{const o=node('option',r.name);o.value=r.run;input.append(o);});}
  else if(field.type==='checkbox'){input.type='checkbox';input.checked=field.value;label.className='checkbox-label';label.prepend(input);return label;}
  else{input.type=field.type==='seeds'?'text':'number';input.value=field.name==='sensory_partitions'&&['agent-x','agent-y'].includes($('model-family').value)?4:field.value;if(input.type==='number'){input.min=['seed','reproduction_workers'].includes(field.name)?0:field.name==='memory_limit_mb'?256:field.name==='receptors'?16:1;input.step=field.name==='receptors'?16:1;}input.required=true;}
  if(structureKeys.includes(field.name)){label.className='structure-control';if(field.name.endsWith('_probability')){input.min=0;input.max=1;input.step=.01;}else{input.min=['chain_budget','new_pair_budget'].includes(field.name)?0:['seed_term_max_nodes','seed_term_max_depth'].includes(field.name)?2:1;input.max=2147483647;}}
  input.oninput=()=>{if(['initial_population','population_a','population_b'].includes(field.name))adoptSourceApparatus();if(structureKeys.includes(field.name))$('config-structure_preset').value='custom';setupSummary();};label.append(input);return label;}));
  if(['agent-z-training','agent-q-training'].includes(kind)){$('config-regions').max=8;if(kind==='agent-q-training'){const e=$('config-exploration');e.min=0;e.max=1;e.step='any';}setupSummary();return;}
  if($('model-family').value==='agent-y'){
    $('model-description').textContent='Agent Y learns receptor grouping from prediction feedback, using shared signal-to-pair mappings. Native CPU evaluation; receptor membership and group count are learned.';
    $('config-sensory_partitions').value=1;$('config-sensory_partitions').closest('label').hidden=true;
    $('config-backend').value='native';
    $('config-embedding_selection').value='argmax';$('config-embedding_selection').disabled=true;
  }
  const profile=$('config-structure_preset');profile.value=['agent-x','agent-y'].includes($('model-family').value)?'exploratory':'baseline';profile.onchange=applyStructurePreset;applyStructurePreset();}
function configValues(){const result={};for(const field of fields[$('session-kind').value]){const e=$('config-'+field.name);result[field.name]=field.type==='populations'?Array.from(e.selectedOptions,o=>o.value):field.type==='checkbox'?e.checked:field.type==='seeds'?e.value.split(/[\s,]+/).filter(Boolean).map(Number):field.type==='number'?Number(e.value):e.value;}if($('session-kind').value==='starting-population-benchmark')result.model=['senses','agent-x','agent-y'].includes($('model-family').value)?'senses':'categorical';if($('model-family').value==='agent-y')result.agent_y=true;result.notes=$('session-notes').value;return result;}
function adoptSourceApparatus(){
  const c=configValues(),source=state.populations.find(p=>p.id===(c.initial_population||c.population_a||c.population_b));
  if(['agent-z-training','agent-q-training'].includes($('session-kind').value)){if(source)$('config-regions').value=source.sensory_partitions;return;}
  for(const key of ['receptors','sensory_partitions','sensory_interface','retina_width','retina_height']){
    const input=$('config-'+key);if(!input)continue;
    input.disabled=Boolean(source);
    if(source)input.value=source[key]??({sensory_interface:'legacy-bytes',sensory_partitions:1,retina_width:16,retina_height:16,receptors:256}[key]);
  }
}
function setupSummary(){if(['agent-z-training','agent-q-training'].includes($('session-kind').value)){const c=configValues();$('setup-summary').textContent=`Native CPU · [−20, 20] · ${c.population} animals × ${c.generations} generations · ${c.episodes} episodes per animal · ${c.regions} sensory regions.`;return;}const c=configValues();
  if(c.sensory_interface){
    const fixed=['signals','retina'].includes(c.sensory_interface),r=$('config-receptors');
    if(fixed){c.receptors=c.retina_width*c.retina_height;r.value=c.receptors;}
    r.disabled=fixed||Boolean(c.initial_population||c.population_a||c.population_b);
    r.min=fixed?1:16;r.step=fixed?1:16;
    for(const key of ['retina_width','retina_height'])$('config-'+key).hidden=!fixed;
  }
  const partitions=$('config-sensory_partitions'),selection=$('config-embedding_selection');if(partitions){const minimum=1;partitions.min=minimum;partitions.max=c.receptors;partitions.setCustomValidity(Number.isInteger(c.sensory_partitions)&&c.sensory_partitions>=minimum&&c.sensory_partitions<=c.receptors?'':`Choose an integer from ${minimum} to the receptor count.`);selection.setCustomValidity(c.sensory_partitions>1&&c.embedding_selection!=='argmax'?'Parallel sensory partitions require highest-probability embedding selection.':'');}if($('session-kind').value==='starting-population-benchmark'){const label=id=>state.populations.find(p=>p.id===id)?.name||'Start fresh';$('setup-summary').textContent=`A: ${label(c.population_a)} · B: ${label(c.population_b)}. Each arm: ${c.population} animals × ${c.generations} evolutionary generations in ${c.environment==='switchboard'?'Three-panel switchboard':c.environment==='switches-2x2'?'Simple switches 2×2':c.environment==='lights-out-2x2'?'Lights Out 2×2':c.environment==='lights-out'?'Lights Out 3×3':c.environment==='shifted'?'[30, 70]':'[−20, 20]'}. Identical training and assessment tasks; ${c.benchmark_episodes} shared final test episodes.`+structureSummary(c);return;}const source=state.populations.find(p=>p.id===c.initial_population);$('setup-summary').textContent=(c.sensory_partitions>1?`Agent X · ${c.sensory_partitions} parallel regions. `:'')+`${c.environment==='switchboard'?'Three-panel switchboard':c.environment==='switches-2x2'?'Simple switches 2×2':c.environment==='lights-out-2x2'?'Lights Out 2×2':c.environment==='lights-out'?'Lights Out 3×3':c.environment==='shifted'?'Shifted [30, 70]':'Original [−20, 20]'} · ${c.initial_population?'Loaded candidates seed the population; extra slots are mutated descendants. ':'Fresh population. '}${c.population} animals × ${c.generations} generations. Each animal: ${c.episodes} learning + ${c.assessment_episodes} assessment episodes per generation.`+(source?.model==='categorical'&&source.environment!==c.environment?' New categorical input and prediction tokens will be initialized for this environment.':'')+structureSummary(c);}
$('model-family').onchange=setupKind;$('session-kind').onchange=renderFields;
$('launch-form').onsubmit=async event=>{event.preventDefault();$('launch').disabled=true;$('setup-error').textContent='';$('launch-status').textContent='Starting…';try{const job=await api('/api/sessions',{kind:$('session-kind').value,name:$('session-name').value||($('model-family').value==='agent-q'?'Agent Q evolution':$('model-family').value==='agent-z'?'Agent Z evolution':$('model-family').value==='agent-y'?'Agent Y evolution':'Agent X evolution'),config:configValues()});$('launch-status').textContent='Session started.';await refresh();await selectRun(job.run);}catch(e){$('setup-error').textContent=e.message;$('launch-status').textContent='';}finally{$('launch').disabled=false;}};

let assessmentRun=null,assessmentJob=null,assessmentBusy=false,assessmentEpoch=0,assessmentOptionsKey='';
function assessmentSelectionSummary(){
  const ids=new Set(Array.from(document.querySelectorAll('#assessment-populations input:checked'),e=>e.value));
  const populations=currentPopulations().filter(p=>ids.has(p.id)),count=populations.reduce((n,p)=>n+p.count,0),episodes=Number($('assessment-episodes').value);
  $('assessment-plan-count').textContent=populations.length?`${populations.length} population${populations.length===1?'':'s'} · ${count.toLocaleString()} candidates`:'No populations selected';
  $('assessment-plan-detail').textContent=count&&Number.isInteger(episodes)&&episodes>0?`${(count*episodes).toLocaleString()} total episodes · ${episodes.toLocaleString()} per candidate`:'Choose candidates and an episode count to build your assessment.';
  $('assessment-launch').disabled=assessmentBusy||!count;
}
function filterAssessmentPopulations(){
  const query=$('assessment-search').value.trim().toLowerCase();let count=0;
  for(const card of document.querySelectorAll('.population-card')){card.hidden=!card.dataset.search.includes(query);if(!card.hidden)count++;}
  $('assessment-no-match').hidden=Boolean(count)||!currentPopulations().length;
}
function assessmentChoices(){
  const key=JSON.stringify(state.populations);if(key===assessmentOptionsKey){assessmentSelectionSummary();return;}assessmentOptionsKey=key;
  const selected=new Set(Array.from(document.querySelectorAll('#assessment-populations input:checked'),e=>e.value));
  $('assessment-populations').replaceChildren(...currentPopulations().map(p=>{
    const card=node('div',undefined,'population-card'),label=node('label'),input=node('input'),body=node('div'),tags=node('div',undefined,'population-tags');
    card.dataset.search=`${p.name} ${p.model} ${p.environment}`.toLowerCase();
    input.type='checkbox';input.value=p.id;input.checked=selected.has(p.id);input.setAttribute('aria-label',p.name);input.onchange=assessmentSelectionSummary;
    tags.append(node('span',p.model==='agent-q'?'Agent Q':p.model==='agent-z'?'Agent Z':p.agent_y?'Agent Y':'Agent X','population-tag '+p.model),node('span',environmentField.options.find(([id])=>id===p.environment)?.[1]||p.environment,'population-tag'));
    body.append(node('strong',p.name),tags);label.append(input,body);
    const footer=node('div',undefined,'population-card-footer'),remove=node('button','Delete');remove.type='button';remove.setAttribute('aria-label',`Delete candidates ${p.name}`);remove.onclick=()=>deletePopulation(p);
    footer.append(node('span',`${p.count.toLocaleString()} saved candidates`),remove);card.append(label,footer);return card;
  }));
  if(!currentPopulations().length){const empty=node('div',undefined,'assessment-empty');empty.append(node('h3','Your candidates belong here.'),node('p','Save candidates from a training session, then return to put them to the test.'));$('assessment-populations').append(empty);}
  const missing=node('p','No populations match your search.','muted');missing.id='assessment-no-match';missing.hidden=true;$('assessment-populations').append(missing);
  filterAssessmentPopulations();assessmentSelectionSummary();
}
$('assessment-search').oninput=filterAssessmentPopulations;
$('assessment-episodes').oninput=assessmentSelectionSummary;
$('assessment-select-all').onclick=()=>{document.querySelectorAll('.population-card:not([hidden]) input').forEach(e=>e.checked=true);assessmentSelectionSummary();};
$('assessment-clear').onclick=()=>{document.querySelectorAll('#assessment-populations input').forEach(e=>e.checked=false);assessmentSelectionSummary();};
function assessmentReport(report){
  const populations=report?.populations||[],progress=report?.progress||{},done=progress.completed_animals||0,total=progress.total_animals||0;
  $('assessment-empty').hidden=populations.length>0;
  $('assessment-progress').hidden=!total;
  const percent=total?Math.min(100,100*done/total):0;$('assessment-progress').setAttribute('aria-valuenow',String(percent));$('assessment-progress').firstElementChild.style.width=percent+'%';
  $('assessment-summary').hidden=!populations.length;
  $('assessment-summary').replaceChildren(...[[`${done} / ${total}`,'Candidates assessed'],[String(report?.config?.episodes||0),'Episodes per candidate'],[pct(report?.random_success),'Random baseline · amber marker']].map(([value,label])=>{const item=node('div');item.append(node('strong',value),node('span',label));return item;}));
  const ranked=populations.map((p,index)=>({...p,order:index})).sort((a,b)=>(b.mean_success??-1)-(a.mean_success??-1)||a.order-b.order);
  $('assessment-results').replaceChildren(...ranked.map((pop,i)=>{
    const card=node('article',undefined,'assessment-result'),body=node('div'),bar=node('div',undefined,'assessment-bar'),fill=node('span',undefined,'assessment-bar-fill'),baseline=node('span',undefined,'assessment-bar-baseline'),score=node('div',pct(pop.mean_success),'assessment-score');
    const valid=Number.isFinite(pop.mean_success);fill.style.width=(valid?Math.max(0,Math.min(100,pop.mean_success*100)):0)+'%';
    baseline.hidden=!Number.isFinite(report.random_success);baseline.style.left=(Math.max(0,Math.min(1,report.random_success||0))*100)+'%';baseline.title='Random baseline: '+pct(report.random_success);bar.append(fill,baseline);
    body.append(node('div',pop.name,'assessment-result-name'),node('span',`${pop.model==='agent-q'?'Agent Q':p.model==='agent-z'?'Agent Z':pop.agent_y?'Agent Y':'Agent X'} · ${pop.completed}/${pop.count} assessed · Best ${pct(pop.best_success)} · Worst ${pct(pop.worst_success)}`,'assessment-result-meta'),bar);
    let caption='Mean success';if(valid&&Number.isFinite(report.random_success))caption+=` · ${pp(100*(pop.mean_success-report.random_success))} vs random`;
    score.append(node('small',caption));card.append(node('span',String(i+1).padStart(2,'0'),'assessment-rank'),body,score);return card;
  }));
}
function assessmentHistory(){const rows=state.runs.filter(r=>r.kind==='assessment');const previous=assessmentRun||$('assessment-run').value;$('assessment-run').replaceChildren(...rows.map(row=>{const option=node('option',`${row.session?.name||row.name} · ${row.state}`);option.value=row.run;return option;}));assessmentRun=rows.some(r=>r.run===previous)?previous:rows[0]?.run||null;if(assessmentRun)$('assessment-run').value=assessmentRun;}
function frozenResultsTable(report){return table(['Population','Model','Candidates assessed','Mean success','Best','Worst'],(report.populations||[]).map(p=>[p.name,p.model,`${p.completed}/${p.count}`,pct(p.mean_success),pct(p.best_success),pct(p.worst_success)]));}
async function loadAssessment(){const run=assessmentRun,epoch=++assessmentEpoch;if(!run){assessmentReport(null);$('assessment-status').textContent='Select populations and run an assessment.';$('assessment-individuals').replaceChildren();$('assessment-download').hidden=true;$('assessment-stop').hidden=true;return;}try{const detail=await api('/api/run?'+new URLSearchParams({run}));if(epoch!==assessmentEpoch||run!==assessmentRun)return;const report=detail.report||{},progress=report.progress||{};assessmentJob=detail.session;const current=state.runs.find(r=>r.run===run);$('assessment-status').textContent=`${current?.state||report.status||'Preparing'} · ${progress.completed_animals||0}/${progress.total_animals||0} candidates · ${report.config?.episodes||0} episodes each · Random baseline ${pct(report.random_success)} · Learning off`+(report.error?' · '+report.error:'');assessmentReport(report);const openIndividuals=new Set(Array.from(document.querySelectorAll('#assessment-individuals details[open]'),e=>e.dataset.population));$('assessment-individuals').replaceChildren(...(report.populations||[]).map(pop=>{const details=node('details'),summary=node('summary',`${pop.name} — individual candidates`);details.dataset.population=pop.snapshot||pop.name;details.open=openIndividuals.has(details.dataset.population);const scroll=node('div',undefined,'table-scroll');scroll.append(table(['Candidate','Source generation','Success','Actions','Predictions tested','Exact prediction accuracy'],(pop.individuals||[]).map(row=>[row.index+1,row.source_generation+1,pct(row.assessment.fitness),row.assessment.actions,row.assessment.tested,pct(row.assessment.prediction_accuracy)])));details.append(summary,scroll);return details;}));$('assessment-download').hidden=!detail.files.includes('report.json');$('assessment-download').href='/api/artifact?'+new URLSearchParams({run,file:'report.json'});$('assessment-stop').hidden=!assessmentJob||!['running','stopping'].includes(assessmentJob.state);$('assessment-stop').disabled=assessmentJob?.state==='stopping';}catch(e){if(epoch===assessmentEpoch){const row=state.runs.find(r=>r.run===run);$('assessment-status').textContent=['running','stopping'].includes(row?.state)?'Preparing assessment results…':e.message;}}}
$('assessment-environment').onchange=()=>{$('assessment-boundary').disabled=Boolean(puzzleSize($('assessment-environment').value));};
$('assessment-run').onchange=()=>{assessmentRun=$('assessment-run').value;assessmentJob=null;$('assessment-stop').hidden=true;$('assessment-download').hidden=true;$('assessment-status').textContent='Loading assessment…';loadAssessment();};
$('assessment-refresh').onclick=async()=>{try{state.populations=await api('/api/populations');assessmentChoices();}catch(e){$('assessment-error').textContent=e.message;}};
$('assessment-form').onsubmit=async event=>{event.preventDefault();if(assessmentBusy)return;assessmentBusy=true;$('assessment-launch').disabled=true;$('assessment-error').textContent='';try{const populations=Array.from(document.querySelectorAll('#assessment-populations input:checked'),e=>e.value);if(!populations.length)throw Error('Select at least one saved population.');const job=await api('/api/sessions',{kind:'assessment',name:$('assessment-name').value,config:{populations,environment:$('assessment-environment').value,episodes:Number($('assessment-episodes').value),seed:Number($('assessment-seed').value),workers:Number($('assessment-workers').value),boundary_mode:$('assessment-boundary').value}});assessmentRun=job.run;assessmentJob=job;$('assessment-stop').hidden=true;$('assessment-download').hidden=true;$('assessment-results').replaceChildren();$('assessment-individuals').replaceChildren();assessmentReport(null);$('assessment-status').textContent='Starting frozen assessment…';await refresh();await loadAssessment();}catch(e){$('assessment-error').textContent=e.message;}finally{assessmentBusy=false;assessmentSelectionSummary();}};
$('assessment-stop').onclick=async()=>{try{await api('/api/stop',{id:assessmentJob.id});await refresh();await loadAssessment();}catch(e){$('assessment-error').textContent=e.message;}};

let playback=null,frameIndex=0,timer=null,worldWidth=1000;
let playgroundLifetime=crypto.randomUUID(), lifetimeStats=null, lifetimeSnapshot=null;
function learningStatus(){const random=$('policy').value==='random';$('play-learning').disabled=random;const mode=random?'Random movement does not train the animal.':$('play-learning').checked?'Learning on. Each attempt builds on this animal’s learned state.':lifetimeStats?'Learning off. Using this animal’s current learned state, frozen.':'Learning off. Attempts use frozen saved weights.';$('learning-status').textContent=mode+(lifetimeStats?` ${lifetimeStats.attempts} attempts · ${lifetimeStats.total_empirical_updates} prediction updates · ${lifetimeStats.total_structural_corrections} structural corrections.`:'');}
function resetPlayLifetime(){const old=playgroundLifetime;playgroundLifetime=crypto.randomUUID();lifetimeStats=null;learningStatus();api('/api/reset-lifetime',{lifetime_id:old}).catch(e=>{$('play-error').textContent=e.message;});}
$('play-learning').onchange=learningStatus;
$('reset-learning').onclick=()=>{if(state.attemptBusy)return;resetPlayLifetime();clearPlayback();$('frame-label').textContent='Learned state reset to the saved snapshot.';};
function pause(){clearInterval(timer);timer=null;$('play').textContent='Play';}
function clearPlayback(){clearStreamPreview();clearBrain();state.attemptEpoch++;pause();playback=null;frameIndex=0;$('world-scale').replaceChildren();$('puzzle-current').replaceChildren();$('puzzle-view').hidden=true;$('animal').setAttribute('visibility','hidden');$('target').setAttribute('visibility','hidden');$('position').textContent='';$('sensory-panel').hidden=true;for(const id of ['play','step','restart'])$(id).disabled=true;$('scrub').max=0;$('frame-detail').textContent='';}
function updatePlayRuns(){const model=$('play-model').value,previous=$('play-run').value;const rows=currentRuns().filter(r=>r.model_family===model);$('play-run').replaceChildren(...rows.map(r=>{const o=node('option',r.name);o.value=r.run;return o;}));if(rows.some(r=>r.run===previous))$('play-run').value=previous;else if(rows.some(r=>r.run===state.selected))$('play-run').value=state.selected;loadSnapshots();}
async function loadSnapshots(){const run=$('play-run').value,previous=$('play-snapshot').value,epoch=++state.snapshotEpoch;clearPlayback();state.snapshots=[];streamAvailability();$('attempt').disabled=true;$('again').disabled=true;$('snapshot-info').textContent='Loading snapshots…';$('play-snapshot').replaceChildren();if($('q-play-learning-label'))$('q-play-learning-label').hidden=$('play-model').value!=='agent-q';if(!run){$('snapshot-info').textContent='No saved snapshots for this model yet.';return;}try{const options=await api('/api/models?'+new URLSearchParams({run,model:['agent-z','agent-q'].includes($('play-model').value)?$('play-model').value:'senses'}));if(epoch!==state.snapshotEpoch)return;state.snapshots=options.filter(o=>Boolean(o.agent_y)===($('play-model').value==='agent-y'));$('play-snapshot').replaceChildren(...state.snapshots.map(o=>{const e=node('option',o.label+(o.model==='agent-q'?' · Agent Q · '+o.sensory_partitions+' regions':o.model==='agent-z'?' · Agent Z · '+o.sensory_partitions+' regions':o.agent_y?' · Agent Y · learned grouping':o.sensory_partitions>1?' · Agent X · '+o.sensory_partitions+' regions':''));e.value=o.id;return e;}));if(options.some(o=>o.id===previous))$('play-snapshot').value=previous;snapshotInfo();}catch(e){if(epoch===state.snapshotEpoch)$('snapshot-info').textContent=e.message;}}
function snapshotInfo(){if($('q-play-learning-label'))$('q-play-learning-label').hidden=$('play-model').value!=='agent-q';clearPlayback();const selected=state.snapshots.find(o=>o.id===$('play-snapshot').value);if((selected?.id||null)!==lifetimeSnapshot){resetPlayLifetime();lifetimeSnapshot=selected?.id||null;}$('attempt').disabled=!selected;$('again').disabled=!selected;$('snapshot-info').textContent=selected?`${selected.label}${selected.model==='agent-q'?' · action-conditioned expectations':selected.model==='agent-z'?' · universal byte signals':selected.sensory_interface==='signals'?` · ${selected.retina_width}×${selected.retina_height} receptor signals · fixed layout`:selected.sensory_interface==='retina'?` · native ${selected.retina_width}×${selected.retina_height} retina`:' · legacy bytes'}${selected.sensory_partitions>1?' · '+selected.sensory_partitions+' regions':''}${selected.training_success!=null?' · training '+pct(selected.training_success):''}${selected.assessment_success!=null?' · saved held-out '+pct(selected.assessment_success):''}. Saved snapshot; optional learning uses a separate playground copy.`+(!puzzleSize(selected.environment)&&selected.boundary_mode==='terminate'?' Saved scores used out-of-bounds failure.':''):'No completed model snapshot saved yet.';$('frame-label').textContent=selected?'Snapshot ready. Choose a task.':'Select a saved snapshot.';const senses=['agent-x','agent-y'].includes($('play-model').value);$('play-world').disabled=!senses;if(!senses||selected?.environment)$('play-world').value=selected?.environment||'original';$('export-model').hidden=!selected;if(selected){$('export-model').href='/api/export?'+new URLSearchParams({snapshot:selected.id});$('export-model').download=selected.model+'-snapshot'+((selected.id.startsWith('candidate:')||['agent-z','agent-q'].includes(selected.model))?'.json':'.json.gz');}worldLimits();goalMode();$('reset-learning').disabled=!selected;streamAvailability();}
function goalMode(){const selected=state.snapshots.find(o=>o.id===$('play-snapshot').value),puzzle=puzzleSize($('play-world').value);$('goal').disabled=puzzle||selected?.varying_goals===false&&$('policy').value==='animal';if($('goal').disabled)$('goal').value=puzzle?0:selected?.environment==='shifted'?50:0;}
function groupPuzzlePanels(board,environment){
  const switchboard=environment==='switchboard';board.classList.toggle('switchboard',switchboard);
  board.setAttribute('aria-label',switchboard?'Three-panel switchboard':'Lights board');
  if(!switchboard)return;
  const cells=Array.from(board.children);board.replaceChildren();
  for(let panel=0;panel<3;panel++){
    const row=node('div',undefined,'switchboard-panel');row.setAttribute('role','group');row.setAttribute('aria-label',`Panel ${panel+1}`);
    row.append(node('span',`Panel ${panel+1}`,'switchboard-panel-label'));
    for(let cell=0;cell<3;cell++){
      const light=cells[panel*3+cell];light.setAttribute('aria-label',`Panel ${panel+1}, switch ${cell+1}`+(light.tagName==='BUTTON'?'':`: ${light.classList.contains('on')?'on':'off'}`));row.append(light);
    }
    board.append(row);
  }
}
function puzzleStart(){const board=Number($('start').value),size=puzzleSize($('play-world').value);$('puzzle-start').style.gridTemplateColumns=`repeat(${size},64px)`;$('puzzle-start').replaceChildren(...Array.from({length:size*size},(_,i)=>{const button=node('button',undefined,'puzzle-cell'+((board>>i)&1?' on':''));button.type='button';button.setAttribute('aria-label',`Starting light ${i+1}`);button.setAttribute('aria-pressed',String(Boolean((board>>i)&1)));button.onclick=()=>{$('start').value=Number($('start').value)^(1<<i);puzzleStart();clearPlayback();};return button;}));groupPuzzlePanels($('puzzle-start'),$('play-world').value);}
function worldLimits(){const mode=$('play-world').value,puzzle=puzzleSize(mode),lower=puzzle?0:mode==='shifted'?30:mode==='expanded'?-30:-20,upper=puzzle?(1<<(puzzle*puzzle))-1:mode==='shifted'?70:-lower;for(const id of ['start','goal']){$(id).min=lower;$(id).max=upper;$(id).value=Math.max(lower,Math.min(upper,Number($(id).value)));}if(puzzle){$('goal').value=0;$('puzzle-editor').hidden=false;$('puzzle-help').textContent='Click lights to set the starting board. '+(['switches-2x2','switchboard'].includes(mode)?(mode==='switchboard'?'Three panels, three independent switches each. Each press toggles only its own light.':'Each button toggles only its own light.'):'A press toggles a cell and its orthogonal neighbors.')+' Goal: all lights off.';puzzleStart();}else{$('puzzle-editor').hidden=true;if($('start').value===$('goal').value)$('start').value=Math.min(upper,Number($('goal').value)+1);}goalMode();}
$('start').oninput=()=>{if(puzzleSize($('play-world').value))puzzleStart();};
$('play-model').onchange=updatePlayRuns;$('play-run').onchange=loadSnapshots;$('play-snapshot').onchange=snapshotInfo;$('refresh-models').onclick=loadSnapshots;$('play-world').onchange=()=>{worldLimits();clearPlayback();};
$('policy').onchange=()=>{goalMode();learningStatus();};
function worldX(n){const margin=worldWidth<500?35:65;return margin+(n-playback.lower)*(worldWidth-2*margin)/(playback.upper-playback.lower);}
function drawWorld(){const puzzle=puzzleSize(playback.environment);$('world-view').toggleAttribute('hidden',puzzle);$('puzzle-view').hidden=!puzzle;if(puzzle)return;worldWidth=Math.max(340,$('world-view').clientWidth||1000);$('world-view').setAttribute('viewBox',`0 0 ${worldWidth} 180`);const g=$('world-scale');g.replaceChildren(svg('line',{x1:worldX(playback.lower),x2:worldX(playback.upper),y1:80,y2:80,stroke:'#60748f','stroke-width':2}));const increment=worldWidth<500?10:5;for(let n=playback.lower;n<=playback.upper;n+=increment){g.append(svg('line',{x1:worldX(n),x2:worldX(n),y1:73,y2:88,stroke:'#60748f'}),svg('text',{x:worldX(n),y:119,fill:'#9aaac0','text-anchor':'middle','font-size':13},n));}g.append(svg('text',{x:worldX(playback.goal),y:151,fill:'#71dbb5','text-anchor':'middle','font-size':13},'Goal '+playback.goal));$('target').setAttribute('cx',worldX(playback.goal));$('target').setAttribute('visibility','visible');}
function bits(hex){return hex?Array.from(hex).flatMap(c=>Number.parseInt(c,16).toString(2).padStart(4,'0').split('').map(Number)):[];}
function drawRetinaView(frame){
  const camera=$('retina-camera');if(!camera)return;
  const visible=playback?.sensory_interface==='retina';camera.hidden=!visible;
  if(!visible)return;
  const canvas=$('retina-surface'),w=playback.retina_width,h=playback.retina_height,observed=bits(frame.sensory);
  canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');
  observed.forEach((v,i)=>{ctx.fillStyle=v?'#a9d5ff':'#080e1b';ctx.fillRect(i%w,Math.floor(i/w),1,1);});
}
function renderFrame(){if(!playback)return;const frame=playback.frames[frameIndex],frameLabel=playback.environment==='switchboard'?frame.label.replace(/Press row (\d+), column (\d+)/g,'Press panel $1, switch $2'):frame.label;if(puzzleSize(playback.environment)){const size=puzzleSize(playback.environment);$('puzzle-current').style.gridTemplateColumns=`repeat(${size},64px)`;$('puzzle-current').replaceChildren(...(frame.board||Array.from({length:size*size},(_,i)=>(frame.position>>i)&1)).map((bit,i)=>{const cell=node('span',undefined,'puzzle-cell'+(bit?' on':''));cell.setAttribute('aria-label',`Light ${i+1}: ${bit?'on':'off'}`);return cell;}));groupPuzzlePanels($('puzzle-current'),playback.environment);}$('animal').setAttribute('visibility','visible');$('animal').setAttribute('cx',worldX(frame.position));$('position').setAttribute('x',worldX(frame.position));$('position').textContent=frame.position;$('frame-label').textContent=frame.done?(frame.success?(puzzleSize(playback.environment)?'Solved · all lights off':'Solved · goal reached'):'Stopped · '+(frame.reason||'finished').replaceAll('_',' ')):frameLabel;$('frame-detail').textContent=`${puzzleSize(playback.environment)?'Presses':'Moves'} ${frame.actions}/12 · Output ${frameIndex}/${playback.frames.length-1} · ${frameLabel}`+(frame.tested?` · Exact predictions ${frame.correct}/${frame.tested}`:'')+(playback.random_expected_success!=null?` · Random chance ${pct(playback.random_expected_success)}`:'');$('scrub').max=playback.frames.length-1;$('scrub').value=frameIndex;
  drawRetinaView(frame);const observed=bits(frame.sensory),predicted=bits(frame.prediction);$('sensory-panel').hidden=!observed.length;if(observed.length){const geometry=playback||{},width=frame.retina_width||geometry.retina_width;$('receptors').style.gridTemplateColumns=['signals','retina'].includes(frame.sensory_interface||geometry.sensory_interface)&&width?`repeat(${width}, minmax(0, 1fr))`:'';$('receptor-count').textContent=(frame.sensory_interface||geometry.sensory_interface)==='signals'?`RECEPTOR SIGNALS · ${width}×${frame.retina_height||geometry.retina_height} · fixed layout`:(frame.sensory_interface||geometry.sensory_interface)==='retina'?`RETINAL VIEW · ${width}×${frame.retina_height||geometry.retina_height} · fixed eye across environments`:observed.length+' LEGACY RECEPTORS';$('receptors').replaceChildren(...observed.map((value,i)=>{const cell=node('span',undefined,'receptor '+(value&&predicted[i]?'shared':value?'observed':predicted[i]?'predicted':''));cell.title=`Receptor ${i+1}: observed ${value}`+(predicted.length?`, predicted ${predicted[i]}`:'');return cell;}));const active=frame.active_pairs;const details=[Array.isArray(active)?(active.length?'Active pairs: '+active.map(({region,pair})=>`region ${region}: ${pair}`).join(' · '):'No active internal state.'):(frame.pair?'Internal state: '+frame.pair:'No active internal state.')];if((frame.predictions||[]).length>1)details.push(`Receptor overlay: first of ${frame.predictions.length} separate predictions`);if(frame.action_resolution){const r=frame.action_resolution;details.push(`Action policy: ${r.policy} · ${r.conflict?'conflict · ':''}${r.action==null?'no action':`action ${r.action}`}`);if(r.counts!=null)details.push('Proposal counts: '+JSON.stringify(r.counts));}$('pair-detail').textContent=details.join(' · ');}if(frame.emissions){$('pair-detail').textContent=frame.emissions.map(e=>`Region ${e.chain.region} · ${e.action?'action '+e.action+' · expected '+e.signal:'emitted '+e.signal} · ${e.observation===null?'no feedback':'observed '+e.observation+' · '+e.mismatched_bits+' differing bits'}`).join(' / ');$('sensory-panel').hidden=false;$('receptors').replaceChildren();$('receptor-count').textContent='UNIVERSAL SIGNALS';}if(frame.done)pause();drawBrain();}
function step(){if(playback&&frameIndex<playback.frames.length-1)frameIndex++;renderFrame();}
function play(){if(!playback)return;if(timer){pause();return;}if(frameIndex===playback.frames.length-1)frameIndex=0;timer=setInterval(step,Number($('speed').value));$('play').textContent='Pause';renderFrame();}
async function runAttempt(event){event?.preventDefault();if(state.attemptBusy||!$('play-snapshot').value)return;clearStreamPreview();state.attemptBusy=true;streamAvailability();const epoch=state.attemptEpoch,snapshot=$('play-snapshot').value;$('attempt').disabled=true;$('again').disabled=true;$('reset-learning').disabled=true;$('play-error').textContent='';pause();try{const result=await api('/api/attempt',{snapshot,lifetime_id:playgroundLifetime,learning:$('play-learning').checked&&$('policy').value==='animal',...($('play-model').value==='agent-q'?{learning_mode:$('q-play-learning').value}:{}),start:Number($('start').value),goal:Number($('goal').value),seed:Number($('attempt-seed').value),policy:$('policy').value,world:$('play-world').value});if(epoch!==state.attemptEpoch)return;playback=result;if(result.learning)lifetimeStats=result.learning;learningStatus();frameIndex=0;drawWorld();$('scrub').disabled=false;['play','step','restart'].forEach(id=>$(id).disabled=false);renderFrame();play();}catch(e){if(epoch===state.attemptEpoch)$('play-error').textContent=e.message;}finally{state.attemptBusy=false;streamAvailability();$('attempt').disabled=!state.snapshots.length;$('again').disabled=!state.snapshots.length;$('reset-learning').disabled=!state.snapshots.length;}}
$('attempt-form').onsubmit=runAttempt;$('again').onclick=()=>{$('attempt-seed').value=(Number($('attempt-seed').value)+1)%4294967296;runAttempt();};$('play').onclick=play;$('step').onclick=()=>{pause();step();};$('restart').onclick=()=>{pause();frameIndex=0;renderFrame();};$('scrub').oninput=()=>{pause();frameIndex=Number($('scrub').value);renderFrame();};$('speed').onchange=()=>{if(timer){pause();play();}};

async function refresh(){if(state.busy)return;state.busy=true;try{state.runs=await api('/api/runs');state.populations=await api('/api/populations');renderLibrary();assessmentChoices();assessmentHistory();if(state.view==='assessments')await loadAssessment();$('connection').textContent='Updated '+new Date().toLocaleTimeString();$('connection').className='connection online';if(!state.selected&&currentRuns().length)state.selected=currentRuns()[0].run;if(state.selected)await loadDetail();}catch(e){$('connection').textContent='Connection lost · retrying';$('connection').className='connection';}finally{state.busy=false;}}
$('refresh').onclick=refresh;
async function init(){try{const boot=await api('/api/bootstrap');state.token=boot.token;state.runs=boot.runs;state.populations=await api('/api/populations');renderLibrary();assessmentChoices();assessmentHistory();if(state.view==='assessments')await loadAssessment();setupKind();const requested=new URLSearchParams(location.search).get('run');state.selected=state.runs.find(r=>r.run===requested||r.name===requested)?.run||currentRuns()[0]?.run||null;await refresh();}catch(e){$('connection').textContent=e.message;setupKind();}setInterval(refresh,3000);}
init();
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(state.view==='overview'&&state.detail)chart(state.detail);if(state.view==='playground'&&playback){drawWorld();renderFrame();}},150);});

$('chart-metric').onchange=()=>{if(state.detail)chart(state.detail);};

async function deletePopulation(population){
  if(!confirm(`Delete candidate set “${population.name}” (${population.count} candidates)? It will be moved to local trash. Existing sessions and their retained copies stay available.`))return;
  try{await api('/api/delete-population',{population:population.id});assessmentOptionsKey='';await refresh();if(state.view==='setup')renderFields();}
  catch(e){$('assessment-error').textContent=e.message;}
}
$('delete-session').onclick=async()=>{
  const run=state.selected,row=state.runs.find(r=>r.run===run);
  if(!run||!confirm(`Delete session “${row?.session?.name||row?.name||run}”, including its saved results, models, logs and diagnostics? It will be moved to local trash. Exported candidate sets and separate comparison arms are kept.`))return;
  try{
    await api('/api/delete-session',{run});state.detailEpoch++;state.selected=null;state.detail=null;
    clearPlayback();state.snapshots=[];resetPlayLifetime();
    $('run-title').textContent='Session deleted';$('run-subtitle').textContent='Select another saved session.';$('metric-cards').replaceChildren();
    $('delete-session').disabled=true;$('save-population').disabled=true;$('open-model').disabled=true;$('resume-session').hidden=true;$('resume-options').hidden=true;$('stop-session').hidden=true;$('candidate-save-options').hidden=true;$('results-panel').hidden=true;$('run-state').textContent='Deleted';$('log-output').textContent='';chart({});await refresh();
  }catch(e){$('overview-error').textContent=e.message;}
};

// Stable brain-shaped coordinates, relaxed toward actual graph connections.
function brainHash(value,salt=0){let h=(2166136261^salt)>>>0;for(let i=0;i<value.length;i++){h^=value.charCodeAt(i);h=Math.imul(h,16777619);}h^=h>>>16;h=Math.imul(h,0x7feb352d);h^=h>>>15;h=Math.imul(h,0x846ca68b);return (h^(h>>>16))>>>0;}
function brainSurface(u,v,side,layer=1){
  const theta=Math.acos(1-2*u),phi=(v-.5)*Math.PI;
  const fold=1+.035*Math.sin(theta*19+phi*8)+.022*Math.cos(phi*23-theta*7);
  const ring=Math.sin(theta),front=1+.09*Math.sin(phi);
  let y=.12+.79*Math.cos(theta)*fold;
  if(y<-.37)y=-.37+(y+.37)*.45;
  return {x:side*(.065+.77*ring*Math.cos(phi)*fold*front*layer),
    y:y*layer,z:1.12*ring*Math.sin(phi)*fold*layer};
}
// Compute once per loaded graph, never during camera movement or playback.
const brainLayoutCache=new WeakMap();
function brainLocalize(graph,positions){
  const mobile=[...positions.values()].filter(p=>p.n.kind==='state'||p.n.kind==='prediction')
    .sort((a,b)=>a.n.id.localeCompare(b.n.id));
  if(!mobile.length)return;
  const links=new Map(mobile.map(p=>[p,[]])),degree=new Map(),springs=[];
  const strengthByKind={successor:1,prediction:.42,embedding:.09,output:.09};
  for(const e of [...(graph.edges||[])].sort((a,b)=>(a.source+'>'+a.target+':'+a.kind).localeCompare(b.source+'>'+b.target+':'+b.kind))){
    const a=positions.get(e.source),b=positions.get(e.target),base=strengthByKind[e.kind];
    if(!a||!b||a===b||!base||!Number.isFinite(e.value)||e.value===0)continue;
    const strength=base*(e.kind==='successor'?1:Math.min(1,Math.abs(e.value)));
    const link={a,b,strength};springs.push(link);
    degree.set(a,(degree.get(a)||0)+strength);degree.set(b,(degree.get(b)||0)+strength);
    links.get(a)?.push(link);links.get(b)?.push(link);
  }
  if(!springs.length)return;
  for(const link of springs)link.strength/=Math.sqrt(Math.max(1,degree.get(link.a),degree.get(link.b)));
  const cost=(p,location,other=null,otherLocation=null)=>{
    let total=0;
    for(const link of links.get(p)){
      const neighbor=link.a===p?link.b:link.a,target=neighbor===other?otherLocation:neighbor;
      const dx=location.x-target.x,dy=location.y-target.y,dz=location.z-target.z;
      total+=link.strength*(dx*dx+dy*dy+dz*dz);
    }
    return total;
  };
  // Keep every original brain-shaped site occupied. Only exchange which node
  // occupies a site, so connection localization cannot collapse the repertoire.
  for(const kind of ['state','prediction']){
    const pool=mobile.filter(p=>p.n.kind===kind),count=pool.length;
    for(let pass=0;pass<12;pass++){
      let swaps=0;
      for(let i=0;i<count;i++){
        const p=pool[i],before=cost(p,p);let best=null,bestDelta=-1e-10;
        const samples=Math.min(count-1,48);
        for(let sample=0;sample<samples;sample++){
          const j=count<=49?(i+sample+1)%count:brainHash(p.n.id,pass*65537+sample*131+17)%count;
          const q=pool[j];if(p===q)continue;
          const delta=cost(p,q,q,p)+cost(q,p,p,q)-before-cost(q,q);
          if(delta<bestDelta){best=q;bestDelta=delta;}
        }
        if(best){
          const {x,y,z}=p;p.x=best.x;p.y=best.y;p.z=best.z;best.x=x;best.y=y;best.z=z;swaps++;
        }
      }
      if(!swaps)break;
    }
  }
}
function brainLayout(graph){
  if(brainLayoutCache.has(graph))return brainLayoutCache.get(graph);
  const positions=new Map(),groups={input:[],state:[],output:[],prediction:[]};
  graph.nodes.forEach(n=>groups[n.kind].push(n));
  for(const kind of ['state','prediction'])for(const n of groups[kind]){
    const u=(brainHash(n.id,11)+.5)/4294967296,v=(brainHash(n.id,53)+.5)/4294967296;
    const side=brainHash(n.id,97)&1?1:-1;
    const layer=kind==='state'?.55+.43*(brainHash(n.id,131)/4294967296):.98;
    positions.set(n.id,{...brainSurface(u,v,side,layer),n});
  }
  const columns=Math.max(1,Math.ceil(Math.sqrt(groups.input.length)*.65)),rows=Math.ceil(groups.input.length/columns);
  const geometry=graph.sensory_geometry;
  groups.input.forEach((n,i)=>{
    if(['signals','retina'].includes(geometry?.interface)){
      const cell=n.receptor??n.pixel??(geometry.interface==='signals'&&Number.isInteger(n.feature)&&n.feature>0?[(n.feature-1)%geometry.width,Math.floor((n.feature-1)/geometry.width)]:null);
      if(geometry.interface==='signals')n.label=Number.isInteger(n.feature)&&n.feature===0?'Receptor channel presence':`Receptor channel ${n.feature??n.id}`;
      positions.set(n.id,Array.isArray(cell)?{n,x:-1.93+.38*cell[0]/Math.max(1,geometry.width-1),y:.74-1.34*cell[1]/Math.max(1,geometry.height-1),z:-.28}:{n,x:-2.08,y:.91,z:-.28});
    }else positions.set(n.id,{n,x:-1.93+.38*(i%columns)/Math.max(1,columns-1),
      y:.74-1.34*Math.floor(i/columns)/Math.max(1,rows-1),z:-.28});
  });
  groups.output.forEach((n,i)=>positions.set(n.id,{n,x:1.66+.1*(i%2),
    y:.66-1.22*i/Math.max(1,groups.output.length-1),z:.12}));
  brainLocalize(graph,positions);brainLayoutCache.set(graph,positions);
  return positions;
}
const brain = {graph:null, yaw:.28, pitch:.85, zoom:1, selected:null, points:[], epoch:0};
const brainColors={input:'#83b5ff',state:'#b79aff',output:'#71dbb5',prediction:'#ffc780'};
function clearBrain(){brain.epoch++;brain.graph=null;brain.selected=null;brain.points=[];$('brain-status').textContent='Mappings changed. Load / refresh to inspect the current agent.';$('brain-inspect').textContent='Select a node to inspect its connections.';drawBrain();}
function drawBrain(){
  const canvas=$('brain-canvas'),width=canvas.clientWidth;if(!width)return;
  const height=canvas.clientHeight||560,dpr=Math.min(2,window.devicePixelRatio||1);canvas.width=width*dpr;canvas.height=height*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,height);
  if(!brain.graph)return;
  const project=p=>{
    const rx=p.x*Math.cos(brain.yaw)+p.z*Math.sin(brain.yaw),rz=-p.x*Math.sin(brain.yaw)+p.z*Math.cos(brain.yaw);
    const ry=p.y*Math.cos(brain.pitch)-rz*Math.sin(brain.pitch),depth=p.y*Math.sin(brain.pitch)+rz*Math.cos(brain.pitch);
    const scale=Math.min(width/4.7,height/3.15)*brain.zoom*4.8/(4.8+depth);
    return {x:width/2+rx*scale,y:height/2-ry*scale,depth,scale};
  };
  const center=project({x:0,y:.12,z:0});
  const halo=ctx.createRadialGradient(center.x,center.y,10,center.x,center.y,Math.min(width*.37,height*.62)*brain.zoom);
  halo.addColorStop(0,'rgba(128,103,204,.12)');halo.addColorStop(1,'rgba(128,103,204,0)');
  ctx.fillStyle=halo;ctx.fillRect(0,0,width,height);
  // Faint cortical folds are a layout guide, not additional agent connections.
  function guide(points,alpha){
    ctx.strokeStyle=`rgba(170,148,215,${alpha})`;ctx.lineWidth=.8;ctx.beginPath();
    points.forEach((p,i)=>{const s=project(p);if(i)ctx.lineTo(s.x,s.y);else ctx.moveTo(s.x,s.y);});ctx.stroke();
  }
  function silhouette(side){
    const points=[];
    for(let i=0;i<=16;i++)for(let j=0;j<=16;j++)points.push(project(brainSurface(i/16,j/16,side)));
    points.sort((a,b)=>a.x-b.x||a.y-b.y);
    const cross=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
    const lower=[],upper=[];
    for(const p of points){while(lower.length>1&&cross(lower.at(-2),lower.at(-1),p)<=0)lower.pop();lower.push(p);}
    for(const p of [...points].reverse()){while(upper.length>1&&cross(upper.at(-2),upper.at(-1),p)<=0)upper.pop();upper.push(p);}
    const hull=lower.slice(0,-1).concat(upper.slice(0,-1));
    ctx.beginPath();hull.forEach((p,i)=>{if(i)ctx.lineTo(p.x,p.y);else ctx.moveTo(p.x,p.y);});ctx.closePath();
    ctx.fillStyle='rgba(139,111,200,.035)';ctx.fill();ctx.strokeStyle='rgba(165,140,211,.24)';ctx.lineWidth=.9;ctx.stroke();
  }
  for(const side of [-1,1])silhouette(side);
  for(const side of [-1,1]){
    for(let band=1;band<13;band++){
      const points=[];
      for(let j=0;j<=72;j++){
        const v=j/72,u=band/14+.012*Math.sin(v*25+band*1.8)*Math.sin(Math.PI*v);
        points.push(brainSurface(u,v,side));
      }
      guide(points,.19);
    }
    for(let fold=0;fold<=9;fold++){
      const points=[];
      for(let j=0;j<=72;j++){
        const u=.012+.976*j/72,v=fold/9+.017*Math.sin(u*35+fold)*Math.sin(Math.PI*u);
        points.push(brainSurface(u,Math.max(0,Math.min(1,v)),side));
      }
      guide(points,fold===0||fold===9?.3:.12);
    }
  }
  const positions=brainLayout(brain.graph),projected=new Map();
  positions.forEach((p,id)=>projected.set(id,{...project(p),n:p.n,world:p}));
  const filter=$('brain-filter').value;
  const frame=streamPreview?.frames?.[streamFrameIndex]??playback?.frames?.[frameIndex];
  const activeRegions=new Map();
  for(const a of frame?.current_pairs??frame?.active_pairs??[]){if(!activeRegions.has(a.pair))activeRegions.set(a.pair,[]);activeRegions.get(a.pair).push(a.region);}
  if(!frame?.active_pairs&&frame?.pair)activeRegions.set(frame.pair,[]);
  const traversed=new Set((frame?.traversals||[]).map(e=>e.source+'>'+e.target));
  brain.graph.edges.forEach(e=>{
    if(filter!=='all'&&e.kind!==filter)return;
    const a=projected.get(e.source),b=projected.get(e.target);if(!a||!b)return;
    const active=e.kind==='successor'&&traversed.has(e.source+'>'+e.target);
    const incident=e.source===brain.selected||e.target===brain.selected;
    const focused=!brain.selected||incident;
    ctx.globalAlpha=active?.95:incident?.88:focused?(e.kind==='successor'?.38:e.kind==='embedding'?.095:.12):.018;
    ctx.strokeStyle=active?'#ffffff':e.kind==='successor'?'#b79aff':e.value<0?'#ff738c':brainColors[b.n.kind];
    ctx.lineWidth=active?2.6:incident?1.8:e.kind==='successor'?1.2:.45+Math.min(.9,Math.abs(e.value)*.8);
    if(e.source===e.target){
      const radius=9,cx=a.x+radius,cy=a.y-radius,r=radius*Math.SQRT2;
      ctx.beginPath();ctx.arc(cx,cy,r,.8*Math.PI,2.7*Math.PI);ctx.stroke();return;
    }
    const aw=a.world,bw=b.world;
    let c1,c2;
    if(e.kind==='embedding'){
      c1=project({x:-1.15,y:aw.y*.65,z:aw.z});
      c2=project({x:bw.x-.18,y:bw.y+.08,z:bw.z-.12});
    }else if(b.n.kind==='output'){
      c1=project({x:aw.x+.28,y:aw.y+.06,z:aw.z});
      c2=project({x:1.12,y:bw.y*.7,z:bw.z});
    }else{
      const arc=(brainHash(e.source+e.target,17)&1?1:-1)*.14;
      c1=project({x:aw.x*.65+bw.x*.35,y:aw.y*.65+bw.y*.35+arc,z:aw.z*.65+bw.z*.35});
      c2=project({x:aw.x*.3+bw.x*.7,y:aw.y*.3+bw.y*.7+arc,z:aw.z*.3+bw.z*.7});
    }
    ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.bezierCurveTo(c1.x,c1.y,c2.x,c2.y,b.x,b.y);ctx.stroke();
    if(incident||active){
      const t=.82,u=1-t,x=u*u*u*a.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*b.x,
        y=u*u*u*a.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*b.y,
        dx=3*u*u*(c1.x-a.x)+6*u*t*(c2.x-c1.x)+3*t*t*(b.x-c2.x),
        dy=3*u*u*(c1.y-a.y)+6*u*t*(c2.y-c1.y)+3*t*t*(b.y-c2.y),len=Math.hypot(dx,dy);
      if(len>0){ctx.beginPath();ctx.moveTo(x-dx/len*5-dy/len*3,y-dy/len*5+dx/len*3);ctx.lineTo(x,y);ctx.lineTo(x-dx/len*5+dy/len*3,y-dy/len*5-dx/len*3);ctx.stroke();}
    }
  });
  ctx.globalAlpha=1;brain.points=[...projected.values()].sort((a,b)=>b.depth-a.depth);
  brain.points.forEach(p=>{
    const selected=p.n.id===brain.selected,active=activeRegions.has(p.n.id),kind=p.n.kind;
    ctx.globalAlpha=selected||active?1:kind==='prediction'?.73:kind==='input'?.85:1;
    ctx.fillStyle=brainColors[kind];ctx.shadowColor=ctx.fillStyle;
    ctx.shadowBlur=selected?18:kind==='state'?11:4;
    const radius=selected?7:kind==='state'?4.2:kind==='output'?4.6:kind==='input'?2.1:1.9;
    ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
    if(active){ctx.globalAlpha=1;ctx.strokeStyle='#ffffff';ctx.lineWidth=1.7;ctx.beginPath();ctx.arc(p.x,p.y,8,0,Math.PI*2);ctx.stroke();const regions=activeRegions.get(p.n.id);if(regions.length){ctx.fillStyle='#ffffff';ctx.font='11px system-ui';ctx.fillText('R'+regions.join(', R'),p.x+11,p.y-8);}}
    if(selected){ctx.globalAlpha=1;ctx.font='12px system-ui';ctx.fillStyle='#edf1ff';ctx.fillText(p.n.label,p.x+12,p.y+17);}
  });
  ctx.globalAlpha=.8;ctx.font='10px system-ui';ctx.textAlign='center';ctx.letterSpacing='1px';
  for(const [label,where,color] of [['SENSORY INPUT',{x:-1.75,y:-.82,z:-.28},'#83b5ff'],['OUTPUT SIGNALS',{x:1.73,y:-.82,z:.12},'#71dbb5']]){
    const p=project(where);ctx.fillStyle=color;ctx.fillText(label,p.x,p.y);
  }
  ctx.fillStyle='#c7b5ed';ctx.fillText('REPERTOIRE + PREDICTIONS',width/2,27);
  ctx.globalAlpha=1;ctx.textAlign='start';ctx.letterSpacing='0px';
}
$('brain-refresh').onclick=async()=>{
  const snapshot=$('play-snapshot').value;if(!snapshot){$('brain-status').textContent='Choose a saved snapshot first.';return;}
  const epoch=++brain.epoch;$('brain-status').textContent='Loading actual agent mappings…';
  try{const graph=await api('/api/brain-scan',{snapshot,lifetime_id:playgroundLifetime});if(epoch!==brain.epoch||snapshot!==$('play-snapshot').value)return;brain.graph=graph;brain.selected=null;$('brain-status').textContent=`${graph.source}${graph.sensory_partitions!=null?` · ${graph.sensory_partitions} sensory partition(s) · ${graph.action_policy??'agreement'} policy`:''} · ${graph.attempts} playground attempts · ${graph.nodes.length} nodes · ${graph.edges.length} displayed connections`;$('brain-inspect').textContent='Select a node to inspect its connections.';drawBrain();}catch(e){if(epoch===brain.epoch)$('brain-status').textContent=e.message;}
};
$('brain-home').onclick=()=>{brain.yaw=.28;brain.pitch=.85;brain.zoom=1;drawBrain();};
$('brain-filter').onchange=drawBrain;
$('brain-panel').addEventListener('toggle',drawBrain);
let brainDrag=null;
$('brain-canvas').onpointerdown=e=>{brainDrag={x:e.clientX,y:e.clientY,moved:false};e.currentTarget.setPointerCapture(e.pointerId);};
$('brain-canvas').onpointermove=e=>{if(!brainDrag)return;const dx=e.clientX-brainDrag.x,dy=e.clientY-brainDrag.y;if(Math.abs(dx)+Math.abs(dy)>2)brainDrag.moved=true;brain.yaw+=dx*.008;brain.pitch=Math.max(-1.5,Math.min(1.5,brain.pitch+dy*.008));brainDrag.x=e.clientX;brainDrag.y=e.clientY;drawBrain();};
$('brain-canvas').onpointerup=e=>{
  if(brainDrag&&!brainDrag.moved){const rect=e.currentTarget.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const p=[...brain.points].reverse().find(p=>Math.hypot(p.x-x,p.y-y)<10);brain.selected=p?.n.id||null;
    if(p){const links=brain.graph.edges.filter(e=>e.source===p.n.id||e.target===p.n.id);const names=new Map(brain.graph.nodes.map(n=>[n.id,n.label]));$('brain-inspect').textContent=`${p.n.label}${p.n.pair_id?' · '+p.n.pair_id:''}${p.n.bias!==undefined?' · bias '+p.n.bias.toPrecision(5):''} — ${links.length} displayed connections. `+links.slice(0,12).map(e=>`${names.get(e.source)} → ${names.get(e.target)}: ${e.value.toPrecision(5)} (${e.measure})`).join('; ');}else $('brain-inspect').textContent='Select a node to inspect its connections.';
  }brainDrag=null;drawBrain();
};
$('brain-canvas').onpointercancel=()=>{brainDrag=null;};
$('brain-canvas').addEventListener('wheel',e=>{e.preventDefault();brain.zoom=Math.max(.3,Math.min(4,brain.zoom*Math.exp(-e.deltaY*.001)));drawBrain();},{passive:false});
window.addEventListener('resize',drawBrain);
$('attempt-form').addEventListener('submit',clearBrain);
['play-model','play-run','play-snapshot','reset-learning','attempt','again'].forEach(id=>$(id).addEventListener(id.startsWith('play-')?'change':'click',clearBrain));


// A stream frame overlays the graph only while its preview is active.
let streamPreview=null,streamFrameIndex=0,streamEpoch=0,streamBusy=false;
function streamAvailability(){
  const enabled=['agent-x','agent-y','agent-z'].includes($('play-model').value)&&Boolean($('play-snapshot').value)&&state.snapshots.length>0;
  for(const id of ['stream-file','stream-max','stream-window'])$(id).disabled=!enabled||streamBusy;
  $('stream-inspect').disabled=!enabled||streamBusy||state.attemptBusy||!$('stream-file').files?.length;
  $('stream-clear').disabled=!streamPreview&&!streamBusy;
}
function clearStreamPreview(){
  streamEpoch++;streamPreview=null;streamFrameIndex=0;streamBusy=false;
  for(const id of ['play','step','restart','scrub'])$(id).disabled=!playback;
  $('stream-result').hidden=true;$('stream-output').textContent='';$('stream-regions').replaceChildren();
  $('stream-file').value='';$('stream-status').textContent='Choose a local file to inspect with fixed senses.';
  streamAvailability();
}
function renderStreamFrame(){
  const frame=streamPreview?.frames[streamFrameIndex];if(!frame)return;
  $('stream-scrub').value=streamFrameIndex;
  $('stream-output').textContent=JSON.stringify({frame:streamFrameIndex+1,sensory:frame.sensory,
    input_window_size:frame.input_window_size,actions:frame.actions,predictions:frame.predictions,emissions:frame.emissions,
    arbitration:frame.action_resolution??frame.arbitration,feedback:(frame.events||[]).filter(e=>e.kind==='FEEDBACK'),expired:(frame.events||[]).filter(e=>e.kind==='EXPIRED')},null,2);
  $('stream-regions').replaceChildren(...(frame.active_pairs||[]).map(a=>{const current=(frame.current_pairs||frame.active_pairs).find(c=>c.region===a.region);return node('p',`Region ${a.region} · entry: ${a.pair??'none'} · current: ${current?.pair??'none'}`);}));
  drawBrain();
}
$('stream-file').onchange=()=>{streamEpoch++;streamPreview=null;$('stream-result').hidden=true;streamAvailability();clearBrain();if(playback){for(const id of ['play','step','restart','scrub'])$(id).disabled=false;drawWorld();renderFrame();}};
$('stream-clear').onclick=()=>{clearStreamPreview();clearBrain();if(playback){drawWorld();renderFrame();}};
$('stream-scrub').oninput=()=>{streamFrameIndex=Number($('stream-scrub').value);renderStreamFrame();};
$('stream-inspect').onclick=async()=>{
  const file=$('stream-file').files?.[0],snapshot=$('play-snapshot').value;if(!file||streamBusy||state.attemptBusy)return;
  const epoch=++streamEpoch;streamBusy=true;streamAvailability();pause();$('stream-status').textContent='Inspecting frozen stream copy…';
  try{
    if(file.size>256*1024)throw Error('File exceeds the 256 KiB limit');
    const max_frames=Number($('stream-max').value),windowValue=$('stream-window').value;
    if(!Number.isInteger(max_frames)||max_frames<1||max_frames>1024)throw Error('Maximum frames must be an integer from 1 to 1024');
    const payload={snapshot,lifetime_id:playgroundLifetime,max_frames,seed:42};
    if(windowValue!==''){payload.window_bytes=Number(windowValue);if(!Number.isInteger(payload.window_bytes)||payload.window_bytes<1)throw Error('Window bytes must be a positive integer');}
    const bytes=new Uint8Array(await file.arrayBuffer());let binary='';
    for(let offset=0;offset<bytes.length;offset+=8192){let chunk='';for(const byte of bytes.subarray(offset,offset+8192))chunk+=String.fromCharCode(byte);binary+=chunk;}
    payload.base64_data=btoa(binary);
    if(epoch!==streamEpoch)return;
    const result=await api('/api/stream-preview',payload);if(epoch!==streamEpoch||snapshot!==$('play-snapshot').value)return;
    streamPreview=result;streamFrameIndex=0;brain.epoch++;brain.graph=result.graph;brain.selected=null;
    $('world-view').setAttribute('hidden','');$('puzzle-view').hidden=true;$('sensory-panel').hidden=true;
    for(const id of ['play','step','restart','scrub'])$(id).disabled=true;
    $('frame-label').textContent='Frozen stream preview';$('frame-detail').textContent='';
    $('brain-status').textContent=`Frozen stream copy · ${result.sensory_partitions} regions · ${result.action_policy} policy · ${result.geometry_policy}`;
    const meta=result.metadata;$('stream-status').textContent=`${meta.bytes_consumed} / ${meta.bytes} bytes consumed · ${meta.windows} windows · ${result.frames.length} frames · ${meta.stop_reason}`;
    $('stream-result').hidden=!result.frames.length;$('stream-scrub').max=Math.max(0,result.frames.length-1);
    $('brain-panel').open=true;renderStreamFrame();drawBrain();
  }catch(e){if(epoch===streamEpoch)$('stream-status').textContent=e.message;}
  finally{if(epoch===streamEpoch){streamBusy=false;streamAvailability();}}
};
