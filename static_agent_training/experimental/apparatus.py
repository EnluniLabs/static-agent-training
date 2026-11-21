"""Construct an agent's fixed input apparatus before environment interaction."""
from .senses import FixedSenses


def make_senses(*, receptors=None, sensory_partitions=1, sensory_interface='legacy-bytes',
                retina_width=16, retina_height=16):
    if sensory_interface in ('retina', 'signals'):
        from .senses import RetinaSenses, SignalSenses
        apparatus = SignalSenses if sensory_interface == 'signals' else RetinaSenses
        retina = apparatus(width=retina_width, height=retina_height, partitions=sensory_partitions)
        if receptors is not None and receptors != retina.receptors:
            raise ValueError('receptor resolution must match the saved layout width and height')
        return retina
    if sensory_interface != 'legacy-bytes':
        raise ValueError('unknown sensory interface')
    return FixedSenses(receptors=256 if receptors is None else receptors, partitions=sensory_partitions)
