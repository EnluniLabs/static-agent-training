"""Frozen sensory replay shared by standalone exports and interactive playback."""
import random

from .bit_model import BitModel, step_action_batch
from .number_line import StimulusNumberLine
from .senses import SensoryAdapter, stimulus_payload
from .signals import Action, Prediction, Step
from ..environments import environment_spec, stimulus_world
from ..seeds import derive
from ..lights_out import board_bits as raw_board


def record_episode(model, start, goal, *, encoding_seed=0, boundary_mode='clamp', goal_conditioned=True,
                   lower=-20, upper=20, random_rng=None, learn=False, environment="original"):
    puzzle=environment_spec(environment).kind=='lights-out'
    raw = stimulus_world(environment,lower=lower,upper=upper,encoding_seed=encoding_seed,
                             boundary_mode=boundary_mode,goal_conditioned=goal_conditioned)
    world = SensoryAdapter(raw,model.senses)
    frames = []
    def record(label, transition, prediction=None, events=()):
        feedback = [e for e in events if e['kind']=='FEEDBACK']
        frames.append(dict(position=raw.position,actions=raw.actions,label=label,done=transition.done,
            success=transition.success,reason=transition.reason,correct=sum(e['correct'] for e in feedback),
            tested=len(feedback),sensory=model.senses.frame(stimulus_payload(raw, model.senses))[0].hex(),
            prediction=prediction.hex() if prediction is not None else None,pair=model.pair,
            **({'active_pairs':[dict(a) for a in model.activations]} if model.sensory_partitions>1 else {})))
        batch = next((e for e in reversed(events) if e['kind']=='BATCH_ACTION'), None)
        if batch is not None:
            frames[-1]['action_resolution'] = {k:batch[k] for k in ('policy','action','conflict','counts')}
        frames[-1]['traversals'] = [dict(source=e['source'], target=e['target'], region=e.get('region'))
                                   for e in events if e['kind']=='EDGE']
        frames[-1]['predictions'] = [dict(region=e.get('region'), pair=e['pair'], signal=e['signal'])
                                     for e in events if e['kind']=='EXPECTATION']
        if puzzle:frames[-1]['board']=raw.board
    if start==goal:
        return [dict(position=start,actions=0,label='Already at goal',done=True,success=True,reason='target',correct=0,tested=0,**({'board':raw_board(start,environment)} if puzzle else {}))]
    model.reset_episode()
    transition=world.reset((start,goal));record('Start',transition);internal=0
    while not transition.done:
        if transition.frame is not None:
            model.observe(transition.frame)
        offset=len(model.events)
        if model.sensory_partitions > 1 and random_rng is None:
            emissions = model.emit_round(learn=learn)
            actions = [e for e in emissions if isinstance(e, Action)]
            predictions = [e for e in emissions if isinstance(e, Prediction)]
            internal += 1
            if actions:
                old = raw.position
                transition = step_action_batch(model, world, actions, learn=learn)
                resolution = world.last_resolution
                action = getattr(resolution.action, 'index', None)
                if action is None:
                    label = 'Conflicting actions · no-op'
                elif puzzle:
                    cell = raw.cells[action]
                    label = f'Press row {cell//raw.grid_size+1}, column {cell%raw.grid_size+1}' if cell is not None else 'Idle'
                else:
                    move = raw.moves[action]
                    label = f'Move {move:+d}'
                    if boundary_mode=='clamp' and not lower<=old+move<=upper:
                        label += ' (stopped at wall)'
                label += f' · {model.action_policy}'
                internal = 0
            elif not model.activations:
                transition = Step(None, done=True, reason='chain_terminated')
                label = 'All chains ended'
            else:
                transition = Step(None, done=True, reason='internal_limit') if internal>=16 else Step(None)
                label = f'{len(predictions)} parallel predictions'
            record(label, transition, predictions[0].signal if predictions else None, model.events[offset:])
            continue
        emission=Action(random_rng.randrange(9)) if random_rng is not None else model.emit(learn=learn)
        internal+=1
        if emission is None:
            transition=Step(None,done=True,reason='chain_terminated');record('Chain ended',transition)
        elif isinstance(emission,Prediction):
            transition=Step(None,done=True,reason='internal_limit') if internal>=16 else Step(None)
            record('Predict next sensory pattern',transition,emission.signal)
        else:
            old=raw.position;move=raw.moves[emission.index]
            transition=world.step(emission);model.receive(transition.frame,learn=learn)
            cell=raw.cells[emission.index] if puzzle else None
            label=f'Press row {cell//raw.grid_size+1}, column {cell%raw.grid_size+1}' if cell is not None else 'Idle' if puzzle else f'Move {move:+d}'
            if not puzzle and boundary_mode=='clamp' and not lower<=old+move<=upper:
                label+=' (stopped at wall)'
            record(label,transition,events=model.events[offset:]);internal=0
    model.receive(None,learn=learn);model.reset_episode()
    return frames


def attempt(bundle,start,goal,seed,policy='animal',*,lower=None,upper=None):
    spec = environment_spec(bundle.get('environment','original'))
    lower = spec.lower if lower is None else lower
    upper = spec.upper if upper is None else upper
    from ..baseline import random_success
    model=BitModel.from_data(bundle['learned_model'])
    model.events.clear()
    model.embedding_rng.seed(derive(seed,'interactive','embedding'))
    model.output_rng.seed(derive(seed,'interactive','output'))
    boundary=bundle.get('boundary_mode','clamp')
    frames=record_episode(model,start,goal,encoding_seed=bundle.get('encoding_seed',0),
        boundary_mode=boundary,goal_conditioned=bundle.get('varying_goals',True),lower=lower,upper=upper,
        random_rng=random.Random(derive(seed,'interactive','random')) if policy=='random' else None,environment=bundle.get('environment','original'))
    if model.parameters()!=bundle['learned_model']['parameters']:
        raise AssertionError('Playback modified learned parameters')
    return dict(start=start,goal=goal,seed=seed,policy=policy,model='senses',embedding_selection=model.embedding_selection,sensory_partitions=model.sensory_partitions,action_policy=model.action_policy,
                sensory_interface=getattr(model.senses,'interface','legacy-bytes'),
                retina_width=getattr(model.senses,'width',None),retina_height=getattr(model.senses,'height',None),frames=frames,lower=lower,upper=upper,
                environment=bundle.get('environment','original'),boundary_mode=boundary,random_expected_success=random_success(start,goal,boundary_mode=boundary,**({'environment':bundle.get('environment','original')} if spec.kind=='lights-out' else dict(lower=lower,upper=upper))))
