"""Agent W number-line adapter with immediate, individually attributed feedback."""
from .agent_z_environment import NumberLine as ZNumberLine


class NumberLine:
    def __init__(self, *, encoding_seed=0, action_limit=12):
        self._world = ZNumberLine(encoding_seed=encoding_seed, action_limit=action_limit)
        self.action_signals = tuple(sorted(self._world.actions))

    @property
    def done(self):
        return self._world.done

    @property
    def success(self):
        return self._world.position == self._world.goal

    def reset(self, task):
        return self._world.reset(task)

    def order(self, regions, round_index):
        """Rotate first access each round, with no learned region priority."""
        offset = round_index % len(regions) if regions else 0
        return regions[offset:] + regions[:offset]

    def _native_moves(self):
        return tuple(self._world.actions[signal] for signal in self.action_signals)

    def execute(self, signal):
        world = self._world
        if world.done:
            raise RuntimeError('episode ended')
        if signal not in world.actions:
            raise ValueError('unknown physical action signal')
        world.position = max(-20, min(20, world.position + world.actions[signal]))
        world.remaining -= 1
        world.done = world.position == world.goal or world.remaining == 0
        return world.measurement()
