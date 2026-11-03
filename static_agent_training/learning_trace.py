"""Reproduce one controlled learning event in both model implementations.

Run: ./project trace-learning --output docs/records/learning-event
Scripted sampling selects the same semantic path without changing parameters.
This is a mechanism audit, not a performance or naturally sampled-run claim.
"""
import argparse
import copy
import json
import math
from pathlib import Path

from .agent import Agent
from .engine import run
from .environment import NumberLine
from .initialize import I, K, pair_id
from .learning import LearningConfig
from .mappings import InheritedMappings, softmax
from .repertoire import Repertoire
from .experimental.bit_model import BitModel, sigmoid, pack_bits
from .experimental.number_line import StimulusNumberLine
from .experimental.senses import FixedSenses, SensoryAdapter


class Draws:
    def __init__(self, *values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def midpoint(probabilities, index):
    return math.fsum(probabilities[:index]) + probabilities[index]/2


def build_traces():
    rep = Repertoire.from_development(run([(I, K)], 20))
    ids = tuple(rep.pairs)
    entry = pair_id(I,K); source = rep.successors[entry]; successor = rep.successors[source]
    config = LearningConfig()
    old_world = NumberLine(goal_conditioned=True)
    raw = StimulusNumberLine(); senses = FixedSenses()
    world = SensoryAdapter(raw,senses)
    inherited = InheritedMappings.initialize(rep,[o.token for o in old_world.observation_tokens],
        [o.token for o in old_world.output_tokens],scale=0)
    maps = inherited.lifetime()
    # Both maps give the same initial entry distribution and weak successor support.
    for row in maps.embedding.values():
        row[successor] = -2.0  # position + goal give a combined successor logit -4.
    maps.output[source]['expect_+6'] = .01
    old = Agent(rep,maps,learning_config=config)
    model = BitModel(rep,senses=senses,learning_config=config)
    model.bias = [-4.0 if q==successor else 0.0 for q in ids]
    model.weights = [[0.0]*len(ids) for _ in range(model.spec.features)]
    model.gates = {q:[0.0]*10 for q in ids}
    model.gates[source][9] = math.log(40+math.exp(.01))
    raw.position,raw.goal = 6,0
    predicted = senses.frame(raw.stimulus())
    predicted_bits = model.spec.encode(predicted)[1:]
    model.prediction = {q:[[.01 if bit else -.01 for bit in predicted_bits]] for q in ids}
    move = next(t for t,v in old_world.diagnostic_state()['move_decoding'].items() if v==0)
    move_index = int(move.split('_')[1])
    # Next repertoire state emits the zero movement under controlled draws.
    common = dict(schema_version='crt-learning-event-trace-v1',
        fixture=dict(start=5,goal=0,predicted_position=6,outcome_position=5,action=0,
            receptors=256,learning_config=vars(config),sampling='scripted inverse-CDF draws; no forced logit changes during event',
            calibration='initial embedding distributions matched; sensory gate prediction mass matched to categorical; prediction objectives cannot be matched',
            default_zero_logit_prediction_mass=dict(categorical=41/50,senses=1/10)),
        pair_aliases=dict(entry=entry,source=source,successor=successor),repertoire=rep.to_data())
    old_trace=dict(copy.deepcopy(common),model='categorical',stages=[])
    new_trace=dict(copy.deepcopy(common),model='senses',stages=[])
    old_context='goal_+0'
    probe_tokens=['obs_+5','obs_+6','obs_+7']
    def old_snapshot():
        conditional=softmax(maps.output[source][t] for t in old.learner.expectations)
        full=maps.output_probabilities(source)
        modal=max(old.learner.expectations,key=lambda t:maps.output[source][t])
        return dict(input_embedding={t:list(maps.embedding_probabilities(t,old_context)) for t in probe_tokens},
            modal_signal='obs_'+modal[len('expect_'):],
            modal_embedding=list(maps.embedding_probabilities('obs_'+modal[len('expect_'):],old_context)),
            conditional_prediction=dict(zip(old.learner.expectations,conditional)),
            output_distribution=dict(zip(maps.outputs,full)),prediction_mass=sum(p for t,p in zip(maps.outputs,full) if t.startswith('expect_')),
            parameters=dict(embedding={s:dict(row) for s,row in maps.embedding.items()},output={q:dict(row) for q,row in maps.output.items()}),
            revised=source in old.learner.revised)
    raw.position,raw.goal=5,0
    actual=senses.frame(raw.stimulus())
    probes={}
    for x in (5,6,7):
        raw.position=x;probes[str(x)]=senses.frame(raw.stimulus())
    def new_snapshot():
        row=model.prediction[source][0]
        modal=(pack_bits([int(v>=0) for v in row]),)
        probabilities=[sigmoid(v) for v in row]
        actual_bits=model.spec.encode(actual)[1:]
        exact=math.exp(math.fsum(math.log(p if b else 1-p) for p,b in zip(probabilities,actual_bits)))
        return dict(input_embedding={x:list(model.probabilities(f)) for x,f in probes.items()},
            modal_signal=modal[0].hex(),modal_embedding=list(model.probabilities(modal)),
            bit_probabilities=probabilities,gate_distribution=list(softmax(model.gates[source])),
            prediction_mass=softmax(model.gates[source])[9],observed_pattern_probability=exact,
            parameters=model.parameters(),revised=source in model.revised)
    def stage(trace,name,snapshot,operation,events):
        before=snapshot();offset=len(events);extra=operation();after=snapshot()
        trace['stages'].append(dict(stage=name,before=before,after=after,events=copy.deepcopy(events[offset:]),detail=extra))
    old_reset=old_world.reset(start=5,goal=0);new_reset=world.reset((5,0))
    def old_observe():
        old._embedding_rng=Draws(midpoint(maps.embedding_probabilities('obs_+5',old_context),ids.index(entry)))
        old.observe(old_reset.observation)
    def new_observe():
        model.embedding_rng=Draws(midpoint(model.probabilities(new_reset.frame),ids.index(entry)))
        model.observe(new_reset.frame)
    stage(old_trace,'observation',old_snapshot,old_observe,old.events)
    stage(new_trace,'observation',new_snapshot,new_observe,model.events)
    def old_predict():
        old._output_rng=Draws(midpoint(maps.output_probabilities(source),maps.outputs.index('expect_+6')))
        emission=old.next_output();old.receive(old_world.emit(emission.output,source=emission.source))
        return dict(emitted=emission.output.token,source=emission.source)
    def new_predict():
        model.output_rng=Draws(midpoint(softmax(model.gates[source]),9),*([.5]*256))
        emission=model.emit();return dict(emitted=emission.signal.hex(),source=model.pair)
    stage(old_trace,'expectation',old_snapshot,old_predict,old.events)
    stage(new_trace,'expectation',new_snapshot,new_predict,model.events)
    outcomes={}
    def old_outcome():
        old._output_rng=Draws(midpoint(maps.output_probabilities(successor),maps.outputs.index(move)))
        emission=old.next_output();outcomes['old']=old_world.emit(emission.output,source=emission.source)
        return dict(action=move,observed=outcomes['old'].observation.token,feedback=[vars(f) for f in outcomes['old'].feedback])
    def new_outcome():
        model.output_rng=Draws(midpoint(softmax(model.gates[successor]),move_index))
        emission=model.emit();outcomes['new']=world.step(emission)
        return dict(action=emission.index,observed=outcomes['new'].frame[0].hex())
    stage(old_trace,'outcome',old_snapshot,old_outcome,old.events)
    stage(new_trace,'outcome',new_snapshot,new_outcome,model.events)
    stage(old_trace,'empirical_update',old_snapshot,lambda:old.receive(outcomes['old']),old.events)
    stage(new_trace,'empirical_update',new_snapshot,lambda:model.receive(outcomes['new'].frame),model.events)
    # Revisit the same source through the same real successor traversal. Separate
    # the encounter hook from sampling to expose both distributions in the trace.
    old._embedding_rng=Draws(midpoint(maps.embedding_probabilities('obs_+5',old_context),ids.index(entry)))
    old.observe(outcomes['old'].observation);old.pair=rep.successors[old.pair];old.path.append(old.pair)
    model.embedding_rng=Draws(midpoint(model.probabilities(actual),ids.index(entry)))
    model.observe(actual);model.pair=rep.successors[model.pair]
    stage(old_trace,'structural_correction',old_snapshot,lambda:old.learner.encounter(source,context=old_context),old.events)
    stage(new_trace,'structural_correction',new_snapshot,lambda:model.structural_update(source,enabled=True),model.events)
    # Sample from the corrected state without invoking another encounter/update.
    def old_next():
        token=maps.emit(source,Draws(midpoint(maps.output_probabilities(source),maps.outputs.index('expect_+5'))))
        old.learner.register(token,source,old.signal,old.initial_pair,old.path,old.timestamp+1,old.observation_index+1)
        return dict(emitted=token,source=source)
    def new_next():
        gate=9
        # emit() follows an edge first; entry -> source; disable a second correction.
        model.pair=entry;model.output_rng=Draws(midpoint(softmax(model.gates[source]),gate),*([.5]*256))
        emission=model.emit(learn=False)
        return dict(emitted=emission.signal.hex(),source=model.pair)
    stage(old_trace,'next_prediction',old_snapshot,old_next,old.events)
    stage(new_trace,'next_prediction',new_snapshot,new_next,model.events)
    return old_trace,new_trace


def write_record(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    traces=build_traces()
    for trace in traces:
        (directory/(trace['model']+'.json')).write_text(json.dumps(trace,indent=2)+'\n')
    return traces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/records/learning-event'))
    args=parser.parse_args();write_record(args.output);print(f'Traces saved: {args.output}')


if __name__=='__main__':
    main()
