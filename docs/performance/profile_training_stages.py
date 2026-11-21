import gzip,json,time,tempfile,hashlib,shutil
from pathlib import Path
from dataclasses import asdict
from static_agent_training import experiment as exp
from static_agent_training.evolution import TrainingConfig,reproduce
from static_agent_training.genome import Species
from static_agent_training.populations import update_best
from static_agent_training.experimental import sensory_evolution as se, sensory_accelerated as sa
source=Path('runs/ui-20261104-220724-Model_A_1-eb173e')
raw=(source/'checkpoint.json.gz').read_bytes();state=json.loads(gzip.decompress(raw));cfg=TrainingConfig.from_data(state['config'])
pop=[Species.from_data(d) for d in state['population']];gen=state['next_generation'];times={}
def measure(name,fn,*args,**kwargs):
 t=time.perf_counter();out=fn(*args,**kwargs);times[name]=times.get(name,0)+time.perf_counter()-t;return out
def wrap(module,name,label):
 fn=getattr(module,name)
 def call(*args,**kwargs):return measure(label,fn,*args,**kwargs)
 setattr(module,name,call)
wrap(sa,'_pack','nested_pack');wrap(sa,'_phase','nested_native_phases');wrap(se,'evaluate','nested_champion_reference')
with tempfile.TemporaryDirectory(prefix='training-stage-review-') as tmp:
 dest=Path(tmp)
 for name in ('best-population.json.gz','champions.json'):
  if (source/name).exists():shutil.copyfile(source/name,dest/name)
 t=time.perf_counter()
 record,detail,lifetime=measure('evaluation',se.evaluate_generation,pop,cfg,gen,state['partitions'],return_lifetime=True)
 measure('best_candidates',update_best,dest,pop,record['individuals'],cfg,gen)
 children,ancestry,report,sp=measure('reproduction',reproduce,pop,record['individuals'],cfg,gen,speciation_state=state.get('speciation_state'),return_speciation=True,lifetime_population=lifetime)
 record.update(reproduction=ancestry,speciation=report)
 measure('generation_record_write',exp.write_json,dest/'generations'/f'{gen:05d}.json',record)
 champion=pop[record['champion_index']]
 measure('champion_trace_write',exp.write_json,dest/'traces'/f'{gen:05d}-champion.json.gz',dict(generation=gen,individual=champion.to_data(),result=record['individuals'][record['champion_index']],**detail))
 measure('champion_index',exp.update_champion_collection,dest,5,record)
 nextpop=measure('population_to_data',lambda:[p.to_data() for p in children])
 checkpoint=dict(schema_version='crt-training-v1',config=asdict(cfg),next_generation=gen+1,history=state['history']+[record['summary']],partitions=state['partitions'],population=nextpop,speciation_state=sp)
 measure('checkpoint_write',exp.write_json,dest/'checkpoint.json.gz',checkpoint)
 times['total_measured']=time.perf_counter()-t
 result=dict(source=str(source),snapshot_sha256=hashlib.sha256(raw).hexdigest(),generation=gen,config={k:asdict(cfg)[k] for k in ('environment','model','workers','population','episodes','assessment_episodes','inherit_lifetime','speciation')},timings_seconds=times,checkpoint_bytes=(dest/'checkpoint.json.gz').stat().st_size)
 Path('/tmp/training-stage-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
