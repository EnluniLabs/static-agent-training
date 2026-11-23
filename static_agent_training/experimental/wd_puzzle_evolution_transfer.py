"""Evolve the original W/D transfer candidates on Lights Out and test retention."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from statistics import mean

from ..experiment import write_json
from ..serialization import dumps
from .agent_d import AgentD
from .agent_w import AgentW
from .agent_w_experiment import run
from .agent_z_experiment import read_z_artifact
from .wd_learning_ablation import puzzle_frequency

SOURCE = Path('runs/agent-w-d-frozen-transfer-20261123-003957/report.json.gz')
BASELINE = Path('runs/agent-w-d-large-frozen-transfer-20261123-004251/report.json.gz')
FROZEN = dict(action_rate=0, structural_rate=0)


def identity(data):
    return sha256(dumps(data, sort_keys=True).encode()).hexdigest()


def execute(output, generations=100):
    selected, baseline = map(read_z_artifact, (SOURCE, BASELINE))
    tasks = {'original': [(s - 50, g - 50) for s, g in baseline['tasks']['shifted']],
             'lights-out-2x2': baseline['tasks']['puzzle']}
    probe_seeds = {'original':'large-transfer:shifted:7',
                   'lights-out-2x2':'large-transfer:puzzle:7'}
    settings = dict(seed=7, population=19, generations=generations, episodes=500,
                    regions=2, structure_preset='baseline', speciation=True,
                    **FROZEN, environment='lights-out-2x2')
    output.mkdir(parents=True, exist_ok=True)
    protected = [SOURCE, BASELINE]
    founders = {}
    for m in ('w', 'd'):
        parent = selected['models'][m]
        checkpoint = Path(parent['source'])/'checkpoint.json.gz'
        protected.append(checkpoint)
        source = read_z_artifact(checkpoint)
        assert len(source['history']) == parent['checkpoint_generations']
        founders[m] = [source['population'][row['checkpoint_index']] for row in parent['candidates']]
        assert len(founders[m]) == 19
        assert [identity(member['individual']) for member in founders[m]] == [
            row['identity'] for row in parent['candidates']]
    hashes = {str(p):sha256(p.read_bytes()).hexdigest() for p in protected}
    config_path = output/'config.json.gz'
    configuration = dict(settings=settings, sources=hashes, tasks=tasks, probe_seeds=probe_seeds,
                         founders={m:[identity(row['individual']) for row in rows]
                                   for m,rows in founders.items()})
    if config_path.exists():
        assert dumps(read_z_artifact(config_path), sort_keys=True) == dumps(configuration, sort_keys=True)
    else:
        write_json(config_path, configuration)

    def probe(cls, individual):
        agent = cls.from_data(individual)
        result = {}
        for env, batch in tasks.items():
            stats = agent.evaluate(batch, seed=probe_seeds[env], environment=env, diagnostics=True, **FROZEN)
            assert stats['embedding'] == agent.embedding and stats['continuation'] == agent.continuation
            result[env] = {k:stats[k] for k in ('successes','episodes','actions')}
            if env == 'lights-out-2x2':
                result[env]['frequency_matched'] = puzzle_frequency(stats['diagnostics']['action_counts'])
        return result

    prior = {}
    for m,cls in [('w',AgentW),('d',AgentD)]:
        prior[m] = [dict(identity=identity(member['individual']), **probe(cls,member['individual']))
                    for member in founders[m]]
        for expected, observed in zip(baseline['models'][m]['candidates'],prior[m]):
            assert expected['identity'] == observed['identity']
            assert expected['shifted']['successes'] == observed['original']['successes']
            assert expected['puzzle']['successes'] == observed['lights-out-2x2']['successes']
    print('Reproduced all 38 frozen baselines.', flush=True)

    def train(m):
        path = output/f'agent-{m}'
        marker = path/'result.json.gz'
        if not marker.exists():
            print(f'Agent {m.upper()} started.',flush=True)
            run(path,model=f'agent-{m}',initial_members=founders[m],**settings)
        return path
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = dict(zip(('w','d'),pool.map(train,('w','d'))))

    report = dict(schema_version='crt-agent-w-d-puzzle-evolution-transfer-v1',
                  config=str(config_path), models={})
    for m,cls in [('w',AgentW),('d',AgentD)]:
        path = paths[m]
        result = read_z_artifact(path/'result.json.gz')
        final_population = read_z_artifact(path/'final-population.json.gz')
        assert final_population['generation'] == generations and len(final_population['members']) == 19
        after = [dict(identity=identity(row['individual']), training_success=row['training_success'],
                      **probe(cls,row['individual'])) for row in final_population['members']]
        data = dict(before=prior[m], after=after, final_candidates=str(path/'final-population.json.gz'),
                    final_training=result['history'][-1])
        for label in ('before','after'):
            print(m,label,*[
                (env,mean(x[env]['successes']/x[env]['episodes'] for x in data[label]),
                 max(x[env]['successes']/x[env]['episodes'] for x in data[label]))
                for env in tasks],mean(x['lights-out-2x2']['frequency_matched'] for x in data[label]),flush=True)
        report['models'][m] = data
    assert all(sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items())
    write_json(output/'report.json.gz',report)
    assert set(read_z_artifact(output/'report.json.gz')['models']) == {'w','d'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--generations',type=int,default=100)
    args=parser.parse_args()
    execute(args.output,args.generations)
