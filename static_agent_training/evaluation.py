"""One lifetime of local learning, followed by independent frozen assessment."""

from collections import Counter
import math

from .environments import environment_spec, categorical_world
from .agent import Agent, run_episode
from .environment import NumberLine
from .mappings import LifetimeMappings
from .seeds import derive


def _block(agent, environment, starts, *, learn, goals=None):
    episodes = []
    if goals is not None and len(goals) != len(starts):
        raise ValueError("one goal is required per episode")
    for index, start in enumerate(starts):
        offset = len(agent.events)
        transitions = run_episode(agent, environment, start=start, learn=learn, goal=None if goals is None else goals[index])
        events = agent.events[offset:]
        commitments = {e['expectation_id']: e for e in events if e['kind'] == 'EXPECTATION'}
        feedback = [f for t in transitions for f in t.feedback]
        loss = 0.0
        for f in feedback:
            logits = {o: v for o, v in commitments[f.expectation_id]['logits'] if o.startswith('expect_')}
            target = 'expect_' + f.observed[len('obs_'):]
            largest = max(logits.values())
            loss += largest - logits[target] + math.log(math.fsum(math.exp(v - largest) for v in logits.values()))
        outputs = [e['output'] for e in events if e['kind'] == 'OUTPUT']
        episodes.append(dict(start=start, goal=environment.default_goal if goals is None else goals[index], success=transitions[-1].success,
                             reason=transitions[-1].reason, outputs=len(outputs),
                             actions=sum(o.startswith('move_') for o in outputs),
                             expectations=sum(o.startswith('expect_') for o in outputs),
                             tested=len(feedback), correct=sum(f.correct for f in feedback),
                             loss_sum=loss, expired=sum(len(t.expired_expectations) for t in transitions)))
    counts = Counter(e['kind'] for e in agent.events)
    tested = sum(e['tested'] for e in episodes)
    correct = sum(e['correct'] for e in episodes)
    return dict(fitness=sum(e['success'] for e in episodes) / len(episodes), episodes=episodes,
                outputs=sum(e['outputs'] for e in episodes), actions=sum(e['actions'] for e in episodes),
                expectations=sum(e['expectations'] for e in episodes), tested=tested, correct=correct,
                prediction_accuracy=correct / tested if tested else None,
                prediction_nll=sum(e['loss_sum'] for e in episodes) / tested if tested else None,
                expired=sum(e['expired'] for e in episodes),
                empirical_updates=counts['EMPIRICAL_UPDATE'], conflicts=counts['CONFLICT'])


def evaluate(individual, training_starts, assessment_starts, *, seed, encoding_seed=0,
             learning=True, learning_config=None, training_goals=None, assessment_goals=None, boundary_mode="clamp", environment="original"):
    if not training_starts or not assessment_starts:
        raise ValueError('training and assessment blocks must be nonempty')
    if set(training_starts) & set(assessment_starts):
        raise ValueError('assessment start positions must be disjoint')
    conditioned = any(s.startswith('goal_') for s in individual.inherited.observations)
    agent = Agent(individual.repertoire, individual.inherited.lifetime(),
                  embedding_seed=derive(seed, 'training', 'embedding'),
                  output_seed=derive(seed, 'training', 'output'),
                  learning=learning, learning_config=learning_config)
    training = _block(agent, categorical_world(environment,encoding_seed=encoding_seed,goal_conditioned=conditioned,boundary_mode=boundary_mode), training_starts,
                      learn=learning, goals=training_goals)
    # Independent RNG and weight copies prevent assessment changing the training lifetime.
    assessor = Agent(individual.repertoire, LifetimeMappings(agent.mappings),
                     embedding_seed=derive(seed, 'assessment', 'embedding'),
                     output_seed=derive(seed, 'assessment', 'output'), learning=False,
                     learning_config=learning_config)
    assessment = _block(assessor, categorical_world(environment,encoding_seed=encoding_seed,goal_conditioned=conditioned,boundary_mode=boundary_mode), assessment_starts,
                        learn=False, goals=assessment_goals)
    result = dict(genotype_id=individual.genotype_id, seed=seed, training=training, assessment=assessment,
                  repertoire_size=len(individual.repertoire.pairs),
                  successor_count=len(individual.repertoire.successors))
    detail = dict(training_events=agent.events, assessment_events=assessor.events,
                  learned_embedding=agent.mappings.embedding, learned_output=agent.mappings.output)
    return result, detail
