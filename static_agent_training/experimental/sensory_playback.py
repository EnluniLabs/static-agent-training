"""Frozen sensory replay shared by standalone exports and interactive playback."""
import random

from .bit_model import BitModel
from .number_line import StimulusNumberLine
from .senses import SensoryAdapter
from .signals import Action, Prediction, Step
from ..seeds import derive


def record_episode(model, start, goal, *, encoding_seed=0, boundary_mode='clamp', goal_conditioned=True,
                   lower=-20, upper=20, random_rng=None):
    raw = StimulusNumberLine(lower=lower,upper=upper,encoding_seed=encoding_seed,
                             boundary_mode=boundary_mode,goal_conditioned=goal_conditioned)
    world = SensoryAdapter(raw,model.senses)
    frames = []
    def record(label, transition, prediction=None, events=()):
        feedback = [e for e in events if e['kind']=='FEEDBACK']
        frames.append(dict(position=raw.position,actions=raw.actions,label=label,done=transition.done,
            success=transition.success,reason=transition.reason,correct=sum(e['correct'] for e in feedback),
            tested=len(feedback),sensory=model.senses.frame(raw.stimulus())[0].hex(),
            prediction=prediction.hex() if prediction is not None else None,pair=model.pair))
    if start==goal:
        return [dict(position=start,actions=0,label='Already at goal',done=True,success=True,reason='target',correct=0,tested=0)]
    model.reset_episode()
    transition=world.reset((start,goal));record('Start',transition);internal=0
    while not transition.done:
        if transition.frame is not None:
            model.observe(transition.frame)
        offset=len(model.events)
        emission=Action(random_rng.randrange(9)) if random_rng is not None else model.emit(learn=False)
        internal+=1
        if emission is None:
            transition=Step(None,done=True,reason='chain_terminated');record('Chain ended',transition)
        elif isinstance(emission,Prediction):
            transition=Step(None,done=True,reason='internal_limit') if internal>=16 else Step(None)
            record('Predict next sensory pattern',transition,emission.signal)
        else:
            old=raw.position;move=raw.moves[emission.index]
            transition=world.step(emission);model.receive(transition.frame,learn=False)
            label=f'Move {move:+d}'
            if boundary_mode=='clamp' and not lower<=old+move<=upper:
                label+=' (stopped at wall)'
            record(label,transition,events=model.events[offset:]);internal=0
    model.receive(None,learn=False);model.reset_episode()
    return frames


def attempt(bundle,start,goal,seed,policy='animal',*,lower=-20,upper=20):
    from ..baseline import random_success
    model=BitModel.from_data(bundle['learned_model'])
    model.events.clear()
    model.embedding_rng.seed(derive(seed,'interactive','embedding'))
    model.output_rng.seed(derive(seed,'interactive','output'))
    boundary=bundle.get('boundary_mode','clamp')
    frames=record_episode(model,start,goal,encoding_seed=bundle.get('encoding_seed',0),
        boundary_mode=boundary,goal_conditioned=bundle.get('varying_goals',True),lower=lower,upper=upper,
        random_rng=random.Random(derive(seed,'interactive','random')) if policy=='random' else None)
    if model.parameters()!=bundle['learned_model']['parameters']:
        raise AssertionError('Playback modified learned parameters')
    return dict(start=start,goal=goal,seed=seed,policy=policy,model='senses',frames=frames,lower=lower,upper=upper,
                boundary_mode=boundary,random_expected_success=random_success(start,goal,boundary_mode=boundary) if lower==-20 and upper==20 else None)
