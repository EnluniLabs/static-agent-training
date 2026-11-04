"""Named number-line worlds; coordinates remain physical at the model boundary."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentSpec:
    lower: int
    upper: int

    @property
    def center(self):
        return (self.lower + self.upper) // 2

    @property
    def positions(self):
        return tuple(range(self.lower, self.upper + 1))

    @property
    def world_kwargs(self):
        return dict(lower=self.lower, upper=self.upper)


ENVIRONMENTS = {
    'original': EnvironmentSpec(-20, 20),
    'shifted': EnvironmentSpec(30, 70),
}


def environment_spec(name='original'):
    if not isinstance(name, str) or name not in ENVIRONMENTS:
        raise ValueError('environment must be original or shifted')
    return ENVIRONMENTS[name]
