"""Reusable top-performing inherited genomes and matched population benchmarks."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
from . import serialization as json
from pathlib import Path
import statistics
import threading
from functools import lru_cache

from .environments import environment_spec, categorical_world
from .genome import Species
from .seeds import derive



def sensory_apparatus(source):
    """Historical sources are byte transducers with one partition by default."""
    return dict(receptors=source.get('receptors',256),
        sensory_partitions=source.get('sensory_partitions',1),
        sensory_interface=source.get('sensory_interface','legacy-bytes'),
        retina_width=source.get('retina_width',16),
        retina_height=source.get('retina_height',16))


def apparatus_arguments(config):
    return {key:getattr(config,key,default) for key,default in sensory_apparatus({}).items()}

def read_population(path):
    path = Path(path)
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as handle:
        value = json.load(handle)
    schema = value.get('schema_version')
    if schema not in ('crt-population-v1', 'crt-agent-z-population-v1') or not value.get('members'):
        raise ValueError('Choose a nonempty saved top-performer population')
    if schema == 'crt-agent-z-population-v1':
        from .experimental.agent_z import AgentZ
        for row in value['members']:
            AgentZ.from_data(row['individual'])
    else:
        for row in value['members']:
            Species.from_data(row['individual'])
    return value


def update_best(directory, population, results, config, generation):
    from .experiment import write_json
    path = Path(directory) / 'best-population.json.gz'
    old = read_population(path)['members'] if path.exists() else []
    rows = {Species.from_data(r['individual']).genotype_id:r for r in old}
    selected = set()
    for index in sorted(range(len(results)),key=lambda i:(-results[i]['training']['fitness'],results[i]['index'])):
        animal, result = population[index], results[index]
        identity = result['genotype_id']
        if identity in selected:
            continue
        selected.add(identity)
        previous = rows.get(identity)
        if previous is None or result['training']['fitness'] > previous['training_fitness']:
            rows[identity] = dict(individual=animal.to_data(encoded_development=True), training_fitness=result['training']['fitness'],
                assessment_fitness=result['assessment']['fitness'], generation=generation, index=result['index'])
        if len(selected) >= config.population_save_count:
            break
    members = sorted(rows.values(), key=lambda r:(-r['training_fitness'], r['generation'], r['index']))[:config.population_save_count]
    write_json(path, dict(schema_version='crt-population-v1', name=Path(directory).name,
        source=str(Path(directory).resolve()), config=asdict(config), weights='inherited-genotype',
        selection='highest observed training fitness; ties use earliest generation and index; no held-out selection',
        source_type='best-evaluated-individuals', members=members))


def save_population(directory, output, *, count=10, name=None):
    from .experiment import write_json
    directory, output = Path(directory), Path(output)
    if type(count) is not int or count < 1:
        raise ValueError('population count must be positive')
    if output.exists():
        raise ValueError('Choose a new population filename; existing populations are preserved')
    source = directory / 'best-population.json.gz'
    if source.exists():
        data = read_population(source)
    else:
        # Older runs only retained whole species for generation champions.
        config = json.loads((directory/'config.json').read_text())['config']
        rows = {}
        for path in sorted((directory/'traces').glob('*-champion.json.gz')):
            with gzip.open(path,'rt') as handle:
                trace = json.load(handle)
            individual = Species.from_data(trace['individual'])
            result = trace['result']
            row = dict(individual=trace['individual'],training_fitness=result['training']['fitness'],
                assessment_fitness=result['assessment']['fitness'],generation=trace['generation'],index=result.get('index',0))
            if individual.genotype_id not in rows or row['training_fitness'] > rows[individual.genotype_id]['training_fitness']:
                rows[individual.genotype_id] = row
        if not rows:
            raise ValueError('This session has no saved evaluated individuals yet')
        data = dict(schema_version='crt-population-v1',config=config,source=str(directory.resolve()),
            source_type='generation-champions-only',weights='inherited-genotype',
            selection='training-selected saved generation champions; earlier runs did not retain every individual',
            members=sorted(rows.values(),key=lambda r:(-r['training_fitness'],r['generation'],r['index'])))
    stamp = datetime.now(timezone.utc)
    data = dict(data, name=name or directory.name+' · '+stamp.strftime('%Y-%m-%d %H:%M:%S UTC'), saved_at=stamp.isoformat(),
                members=data['members'][:count])
    write_json(output,data)
    return dict(path=str(output.resolve()),name=data['name'],count=len(data['members']),source_type=data['source_type'])


_population_summary_lock = threading.RLock()


@lru_cache(maxsize=256)
def _population_summary(filename, device, inode, modified, changed, size):
    # Cache only this small row, never genomes, decoded histories or mappings.
    data = read_population(filename)
    config = data['config']
    return dict(name=data['name'],count=len(data['members']),
        model=config.get('model','categorical'),agent_y=config.get('agent_y',False),
        receptors=config.get('receptors',256),sensory_interface=config.get('sensory_interface','legacy-bytes'),
        retina_width=config.get('retina_width',16),retina_height=config.get('retina_height',16),
        sensory_partitions=config.get('sensory_partitions',1),action_policy=config.get('action_policy','agreement'),
        environment=config.get('environment','original'),source_type=data['source_type'],
        **({'can_seed': all('seeds' in member and 'budgets' in member for member in data['members'])}
           if config.get('model') == 'agent-z' else {}))


def list_populations(root):
    root = Path(root).resolve()
    rows = []
    # Concurrent browser polls must not all decode the same large cold archive.
    with _population_summary_lock:
        for path in sorted((root/'runs/populations').glob('*.json.gz')):
            stat = path.stat()
            summary = _population_summary(str(path),stat.st_dev,stat.st_ino,
                stat.st_mtime_ns,stat.st_ctime_ns,stat.st_size)
            rows.append(dict(summary,id=path.relative_to(root/'runs').as_posix()))
    return rows


def validate_compatibility(data, config, *, allow_new_environment=False):
    source = data['config']
    if source.get('agent_y', False) != getattr(config, 'agent_y', False):
        raise ValueError('Saved population and session must use the same Agent X/Y architecture')
    if source.get('model','categorical') != config.model:
        raise ValueError('Saved population and new session must use the same model')
    if source.get('encoding_seed',0) != config.encoding_seed:
        raise ValueError('Saved population uses a different action encoding seed')
    if source.get('varying_goals',False) != config.varying_goals:
        raise ValueError('Saved population and new session must use the same goal-input mode')
    if config.model == 'senses':
        saved, requested = sensory_apparatus(source), apparatus_arguments(config)
        keys = ['receptors','sensory_partitions','sensory_interface']
        if saved['sensory_interface'] in ('retina','signals') or requested['sensory_interface'] in ('retina','signals'):
            keys += ['retina_width','retina_height']
        for key in keys:
            if saved[key] != requested[key]:
                raise ValueError(f'Saved population {key} must match the new session')
    elif not allow_new_environment and source.get('environment','original') != config.environment:
        raise ValueError('Categorical populations have fixed vocabularies; choose their original environment')


def initialize_saved(config):
    from .mutation import mutate
    data = read_population(config.initial_population)
    validate_compatibility(data,config,allow_new_environment=True)
    parents = [Species.from_data(row['individual']) for row in data['members']]
    if config.model == 'senses':
        for parent in parents:
            senses = parent.inherited.senses
            expected = apparatus_arguments(config)
            actual_interface = getattr(senses,'interface','legacy-bytes')
            if senses.receptors != expected['receptors'] or actual_interface != expected['sensory_interface']:
                raise ValueError('Parent senses do not match the saved apparatus')
            count = getattr(senses,'partitions',None)
            if count is not None and count != expected['sensory_partitions']:
                raise ValueError('Parent senses partitions do not match the saved apparatus')
            if actual_interface in ('retina', 'signals') and (senses.width != expected['retina_width'] or senses.height != expected['retina_height']):
                raise ValueError('Parent retina dimensions do not match the saved apparatus')
    if config.model == 'categorical':
        from .environment import NumberLine
        from .mappings import InheritedMappings
        spec = environment_spec(config.environment)
        world = categorical_world(config.environment,encoding_seed=config.encoding_seed,goal_conditioned=config.varying_goals)
        observations = tuple(o.token for o in world.observation_tokens)
        outputs = tuple(o.token for o in world.output_tokens)
        adapted = []
        for index, parent in enumerate(parents):
            old = parent.inherited
            if old.observations == observations and old.outputs == outputs:
                adapted.append(parent)
                continue
            fresh = InheritedMappings.initialize(parent.repertoire, observations, outputs,
                seed=derive(config.seed,'candidate-environment',index))
            embedding = {s:{q:old.embedding[s][q] if s in old.embedding else fresh.embedding[s][q]
                            for q in old.pair_ids} for s in observations}
            output = {q:{o:old.output[q][o] if o in old.output[q] else fresh.output[q][o]
                         for o in outputs} for q in old.pair_ids}
            adapted.append(replace(parent,inherited=InheritedMappings(observations,outputs,old.pair_ids,embedding,output)))
        parents = adapted
    result = parents[:config.population]
    while len(result) < config.population:
        i = len(result)
        child,_ = mutate(parents[(i-len(parents)) % len(parents)], seed=derive(config.seed,'population-import',i),config=config.mutation)
        result.append(child)
    return result


def benchmark_populations(paths, output, *, environment='original', episodes=100, assessment_episodes=200,
                          seed=48, backend='native', workers=4):
    """Common tasks and learning budget, fixed structure, no reproduction or selection."""
    from .evolution import TrainingConfig, generation_tasks, partitions
    from .accelerated import evaluate_batch
    from .experimental.sensory_accelerated import evaluate_batch as sensory_batch
    from .experiment import write_json
    from .baseline import random_task_mean
    output = Path(output)
    if (output/'report.json').exists():
        raise ValueError('Choose a new benchmark output directory')
    if len(paths) < 2:
        raise ValueError('Choose at least two saved populations to compare')
    populations = [read_population(path) for path in paths]
    modes = {p['config'].get('varying_goals',False) for p in populations}
    if len(modes) != 1:
        raise ValueError('Compared populations must use the same goal-input mode')
    config = TrainingConfig(seed=seed,episodes=episodes,assessment_episodes=assessment_episodes,
                            environment=environment,varying_goals=modes.pop())
    split = partitions(seed,include_zero=config.varying_goals,environment=environment)
    starts,assess,goals,assess_goals = generation_tasks(config,0,split)
    spec = environment_spec(environment)
    report = dict(schema_version='crt-population-benchmark-v1',status='running',config=dict(environment=environment,boundary_mode='clamp',prediction_rate=.05,embedding_rate=.05,compatibility_threshold=.1,
        episodes=episodes,assessment_episodes=assessment_episodes,seed=seed,backend=backend,workers=workers),
        task_source='common sampled training tasks followed by frozen held-out assessment; no evolutionary generations',
        weights='inherited-genotype',learning=True,partitions=split,
        training_starts=starts,training_goals=goals,assessment_starts=assess,assessment_goals=assess_goals,
        random_assessment=random_task_mean(assess,assess_goals,**spec.baseline_kwargs),populations=[])
    write_json(output/'report.json',report)
    for path,data in zip(paths,populations):
        cfg = replace(config,model=data['config'].get('model','categorical'),agent_y=data['config'].get('agent_y',False),**sensory_apparatus(data['config']),
                      encoding_seed=data['config'].get('encoding_seed',0),embedding_selection=data['config'].get('embedding_selection','sample'),
                      action_policy=data['config'].get('action_policy','agreement'))
        validate_compatibility(data,cfg)
        members = [Species.from_data(row['individual']) for row in data['members']]
        evaluator = sensory_batch if cfg.model=='senses' else evaluate_batch
        kwargs = dict(goal_conditioned=config.varying_goals,embedding_selection=cfg.embedding_selection,
                      sensory_partitions=cfg.sensory_partitions,action_policy=cfg.action_policy) if cfg.model=='senses' else {}
        rows = []
        for first in range(0,len(members),64):
            batch = members[first:first+64]
            rows.extend(evaluator(batch,starts,assess,training_goals=goals,assessment_goals=assess_goals,
                seeds=[derive(seed,'population-benchmark',i) for i in range(first,first+len(batch))],
                encoding_seed=cfg.encoding_seed,environment=environment,backend=backend,workers=workers,**kwargs))
        row = dict(name=data['name'],source=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            **sensory_apparatus(data['config']),model=cfg.model,embedding_selection=cfg.embedding_selection if cfg.model=='senses' else None,action_policy=cfg.action_policy,count=len(rows),training_mean=statistics.mean(r['training']['fitness'] for r in rows),
            assessment_mean=statistics.mean(r['assessment']['fitness'] for r in rows),individuals=rows)
        report['populations'].append(row)
        write_json(output/'report.json',report)
        print(f"{row['name']}: {row['count']} animals; frozen assessment {row['assessment_mean']:.2%}",flush=True)
    report['status']='completed'
    write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    save=commands.add_parser('save');save.add_argument('run',type=Path);save.add_argument('--output',type=Path,required=True)
    save.add_argument('--count',type=int,default=10);save.add_argument('--name')
    bench=commands.add_parser('benchmark');bench.add_argument('--populations',nargs='+',type=Path,required=True)
    bench.add_argument('--output',type=Path,required=True);bench.add_argument('--environment',choices=('original','shifted','lights-out','switches-2x2','lights-out-2x2','switchboard'),default='original')
    bench.add_argument('--episodes',type=int,default=100);bench.add_argument('--assessment-episodes',type=int,default=200)
    bench.add_argument('--seed',type=int,default=48);bench.add_argument('--backend',choices=('native',),default='native')
    bench.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); values=vars(args);command=values.pop('command')
    if command=='save':print(json.dumps(save_population(values.pop('run'),**values)))
    else:benchmark_populations(values.pop('populations'),**values)


if __name__=='__main__':
    main()
