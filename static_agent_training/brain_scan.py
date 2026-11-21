"""Read-only graph export of actual agent parameters for the workspace."""
from .mappings import softmax
from .experimental.bit_model import sigmoid
from .experimental.senses import receptor_regions


def mapping_graph(repertoire, *, parameters=None, mappings=None, sensory_partitions=None, senses=None):
    if sensory_partitions is None:
        sensory_partitions = getattr(senses, 'partitions', None) or 1
    nodes, edges = [], []
    def node(identity, label, kind, **extra):
        nodes.append(dict(id=identity, label=label, kind=kind, **extra))
    def strongest(source, targets, values, kind, probability=False):
        # Bound the display, while preserving exact values for displayed edges.
        ranked = sorted(zip(targets, values), key=lambda item: abs(item[1]), reverse=True)
        for target, value in ranked[:4]:
            edges.append(dict(source=source, target=target, value=value, kind=kind,
                              measure='probability' if probability else 'weight'))
    ids = list(repertoire.pairs)
    for i, q in enumerate(ids):
        node(q, f'State {i + 1}', 'state', pair_id=q)
    for source, target in repertoire.successors.items():
        edges.append(dict(source=source, target=target, value=1, kind='successor', measure='structural'))
    if parameters is not None:
        retina = senses is not None and getattr(senses, 'width', None) is not None
        if senses is not None and hasattr(senses, 'partition_indices'):
            indices = senses.partition_indices(sensory_partitions)
        else:
            indices = [range(start, end) for start, end in
                       receptor_regions(len(parameters['weights']) - 1, sensory_partitions)]
        regions = {pixel + 1: region for region, pixels in enumerate(indices) for pixel in pixels}
        for i, row in enumerate(parameters['weights']):
            key = f'input:{i}'
            if retina and i > 0:
                x, y = (i - 1) % senses.width, (i - 1) // senses.width
                region = regions[i]
                label = f'Pixel {x},{y}' if senses.interface == 'retina' else f'Receptor {i-1} · layout {x},{y}'
                node(key, f'{label} · region {region}', 'input', region=region, pixel=[x, y])
            elif sensory_partitions > 1 or retina:
                region = regions.get(i)
                label = 'Presence · shared across regions' if i == 0 else f'Region {region} · receptor {i-1}'
                node(key, label, 'input', region=region)
            else:
                node(key, f'Input feature {i}', 'input')
            strongest(key, ids, row, 'embedding')
        for j, q in enumerate(ids):
            nodes[j]['bias'] = parameters['bias'][j]
        gates = parameters['gates']
        predictions = parameters['prediction']
        if ids:
            channels = len(predictions[ids[0]])
            actions = len(gates[ids[0]]) - channels
            targets = []
            for i in range(actions + channels):
                key = f'output:{i}'
                node(key, f'Action {i}' if i < actions else f'Predict channel {i-actions}', 'output')
                targets.append(key)
            for q in ids:
                strongest(q, targets, softmax(gates[q]), 'output', True)
                for c, row in enumerate(predictions[q]):
                    # Each Bernoulli probability is conditional on its prediction channel.
                    for i, value in enumerate(row):
                        key = f'prediction:{c}:{i}'
                        if q == ids[0]:
                            node(key, f'Channel {c} · prediction bit {i}', 'prediction')
                    strongest(q, [f'prediction:{c}:{i}' for i in range(len(row))],
                              [sigmoid(v) for v in row], 'prediction', True)
    else:
        for symbol, row in mappings.embedding.items():
            key = 'input:' + symbol
            node(key, symbol, 'input')
            strongest(key, list(row), softmax(row.values()), 'embedding', True)
        for symbol in mappings.outputs:
            node('output:' + symbol, symbol, 'output')
        for q, row in mappings.output.items():
            strongest(q, ['output:' + s for s in row], softmax(row.values()), 'output', True)
    graph = dict(nodes=nodes, edges=edges, connection_limit=4)
    if senses is not None:
        graph['sensory_geometry'] = sensory_geometry(senses, sensory_partitions)
    return graph


def sensory_geometry(senses, partitions):
    """Saved receptor geometry, independent of the selected data source."""
    retina = getattr(senses, 'width', None) is not None
    return dict(interface=getattr(senses, 'interface', 'retina' if retina else 'legacy-bytes'),
                width=getattr(senses, 'width', None), height=getattr(senses, 'height', None),
                resolution=senses.receptors, partitions=partitions,
                scheme=getattr(senses, 'scheme', 'contiguous'))
