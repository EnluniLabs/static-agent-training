"""Opt-in structural controls shared by experiment and comparison frontends.

Flat keys are also the workspace API. Presets resolve into existing config fields;
no preset label is persisted and no learning or weight settings are changed.
"""

from dataclasses import replace
from collections.abc import Mapping


_MUTATION_FIELDS = {
    'seed_term_max_nodes': 'max_nodes',
    'seed_term_max_depth': 'max_depth',
    'combinator_probability': 'combinator_probability',
    'subtree_probability': 'subtree_probability',
    'term_probability': 'term_probability',
    'insertion_probability': 'insertion_probability',
    'deletion_probability': 'deletion_probability',
    'new_pair_budget': 'new_pair_budget',
}
_DEVELOPMENT_FIELDS = {
    'development_max_nodes': 'max_nodes',
    'development_max_transitions': 'max_transitions',
}
STRUCTURE_PRESETS = {
    'baseline': dict(initial_pairs=4, chain_budget=64, seed_term_max_nodes=31,
                     seed_term_max_depth=6, combinator_probability=.5,
                     development_max_nodes=256, development_max_transitions=1000,
                     subtree_probability=.15, term_probability=.10,
                     insertion_probability=.05, deletion_probability=.05,
                     new_pair_budget=64),
    'exploratory': dict(initial_pairs=16, chain_budget=256, seed_term_max_nodes=127,
                        seed_term_max_depth=10, combinator_probability=.15,
                        development_max_nodes=2048, development_max_transitions=2000,
                        subtree_probability=.35, term_probability=.20,
                        insertion_probability=.25, deletion_probability=.02,
                        new_pair_budget=256),
    'custom': {},
}
STRUCTURE_ARGUMENTS = ('structure_preset', *STRUCTURE_PRESETS['baseline'])


def add_structure_arguments(parser):
    """Register optional flat controls; omission always preserves current values."""
    parser.add_argument('--structure-preset', choices=tuple(STRUCTURE_PRESETS), default=None,
                        help='structural base profile for a new run; explicit flags override it')
    for name in STRUCTURE_PRESETS['baseline']:
        parser.add_argument('--' + name.replace('_', '-'), default=None,
                            type=float if name.endswith('_probability') else int,
                            help='override ' + name.replace('_', ' '))


def apply_structure_arguments(config, args):
    """Resolve a namespace or flat mapping without mutating either input."""
    supplied = args if isinstance(args, Mapping) else vars(args)
    preset = supplied.get('structure_preset')
    if preset is not None and preset not in STRUCTURE_PRESETS:
        raise ValueError('unknown structure preset')
    values = dict(STRUCTURE_PRESETS[preset]) if preset is not None else {}
    values.update((name, supplied[name]) for name in STRUCTURE_PRESETS['baseline']
                  if supplied.get(name) is not None)
    if not values:
        return config
    # bool is a Python number but is not a meaningful numeric UI/API control.
    if any(isinstance(value, bool) for value in values.values()):
        raise ValueError('structural controls must be numeric, not boolean')
    changes = {name: values[name] for name in ('initial_pairs', 'chain_budget') if name in values}
    for field, fields in (('mutation', _MUTATION_FIELDS), ('development', _DEVELOPMENT_FIELDS)):
        nested = {target: values[name] for name, target in fields.items() if name in values}
        if nested:
            changes[field] = replace(getattr(config, field), **nested)
    return replace(config, **changes)
