"""Named worlds; coordinates remain physical at the model boundary."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentSpec:
    lower: int
    upper: int
    kind: str = "number-line"
    variant: str = "lights-out"

    @property
    def center(self):
        return 0 if self.kind=="lights-out" else (self.lower + self.upper) // 2

    @property
    def positions(self):
        return tuple(range(self.lower, self.upper + 1))

    @property
    def world_kwargs(self):
        return {} if self.kind=="lights-out" else dict(lower=self.lower, upper=self.upper)


    @property
    def baseline_kwargs(self):
        return {'environment':self.variant} if self.kind=='lights-out' else self.world_kwargs


ENVIRONMENTS = {
    'original': EnvironmentSpec(-20, 20),
    'shifted': EnvironmentSpec(30, 70),
    'lights-out': EnvironmentSpec(0,511,'lights-out'),
    'lights-out-2x2': EnvironmentSpec(0,15,'lights-out','lights-out-2x2'),
    'switchboard': EnvironmentSpec(0,511,'lights-out','switchboard'),
    'switches-2x2': EnvironmentSpec(0,15,'lights-out','switches-2x2'),
}


def environment_spec(name='original'):
    if not isinstance(name, str) or name not in ENVIRONMENTS:
        raise ValueError('environment must be original, shifted, lights-out, lights-out-2x2, switches-2x2, or switchboard')
    return ENVIRONMENTS[name]


def categorical_world(environment='original', **kwargs):
    spec=environment_spec(environment)
    if spec.kind=='lights-out':
        from .lights_out import LightsOutWorld
        return LightsOutWorld(environment=environment,**kwargs)
    from .environment import NumberLine
    return NumberLine(**{**spec.world_kwargs,**kwargs})


def stimulus_world(environment='original', **kwargs):
    spec=environment_spec(environment)
    if spec.kind=='lights-out':
        from .lights_out import StimulusLightsOut
        return StimulusLightsOut(environment=environment,**kwargs)
    from .experimental.number_line import StimulusNumberLine
    return StimulusNumberLine(**{**spec.world_kwargs,**kwargs})
