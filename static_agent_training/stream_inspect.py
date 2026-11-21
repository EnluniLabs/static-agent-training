"""Inspect local opaque byte streams, optionally through a saved BitModel.

Shared streaming imports are deliberately lazy so CLI help is read-only and
available independently of the experimental stream implementation.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import gzip
from itertools import islice
import json
from pathlib import Path
import sys


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def parser():
    result = argparse.ArgumentParser(
        prog='./project stream', description=__doc__.split('\n\n')[0])
    result.add_argument('--input', required=True, help="local binary file or '-' for stdin")
    result.add_argument('--output', type=Path, help='new JSON report file (never overwritten)')
    result.add_argument('--snapshot', type=Path, help='local BitModel JSON or JSON.gz, or bundle with learned_model')
    result.add_argument('--receptors', type=positive, help='default: 256, or saved geometry')
    result.add_argument('--partitions', type=positive, help='default: 4, or saved geometry')
    result.add_argument('--window-bytes', type=positive, help='default: saved raw input capacity')
    result.add_argument('--sensory-interface', choices=('legacy-bytes', 'retina', 'signals'))
    result.add_argument('--retina-width', '--receptor-width', type=positive)
    result.add_argument('--retina-height', '--receptor-height', type=positive)
    result.add_argument('--max-frames', type=positive, default=16)
    result.add_argument('--rounds-per-frame', type=positive, default=1,
                        help='model emission rounds per incoming window (default: 1)')
    result.add_argument('--learn', action='store_true', help='learn only in memory; never update snapshot')
    return result


def _load_model(path):
    from .experimental.bit_model import BitModel

    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError('snapshot must be a BitModel object or bundle with learned_model')
    return BitModel.from_data(data.get('learned_model', data))


def _validate_output(options):
    output = options.output
    if output is None:
        return
    for source in (None if options.input == '-' else Path(options.input), options.snapshot):
        if source is not None and output.resolve() == source.resolve():
            raise ValueError('output must differ from input and snapshot')
    if output.exists() or output.is_symlink():
        raise ValueError('output already exists; refusing to overwrite')


def inspect(options):
    """Return a JSON-compatible report; own files, but never own stdin."""
    _validate_output(options)
    if options.learn and options.snapshot is None:
        raise ValueError('--learn requires --snapshot')
    from .experimental.senses import FixedSenses

    model = _load_model(options.snapshot) if options.snapshot is not None else None
    if model is not None:
        if model.senses is None:
            raise ValueError('snapshot must contain saved senses')
        senses = model.senses
        partitions = model.sensory_partitions
        if options.receptors is not None and options.receptors != senses.receptors:
            raise ValueError('--receptors conflicts with snapshot')
        if options.partitions is not None and options.partitions != partitions:
            raise ValueError('--partitions conflicts with snapshot')
    else:
        partitions = options.partitions if options.partitions is not None else 4
        if options.sensory_interface in ('retina', 'signals'):
            from .experimental.senses import RetinaSenses, SignalSenses
            surface = SignalSenses if options.sensory_interface == 'signals' else RetinaSenses
            senses = surface(width=options.retina_width or 16, height=options.retina_height or 16,
                                  partitions=partitions)
            if options.receptors is not None and options.receptors != senses.receptors:
                raise ValueError('--receptors conflicts with retina dimensions')
        else:
            if options.retina_width is not None or options.retina_height is not None:
                raise ValueError('retina dimensions require --sensory-interface retina')
            senses = FixedSenses(receptors=options.receptors if options.receptors is not None else 256)
    if partitions > senses.receptors:
        raise ValueError('partitions cannot exceed receptor count')
    from .brain_scan import sensory_geometry
    geometry = sensory_geometry(senses, partitions)
    if model is not None:
        for option, saved in ((options.sensory_interface, geometry['interface']),
                              (options.retina_width, geometry['width']),
                              (options.retina_height, geometry['height'])):
            if option is not None and option != saved:
                raise ValueError('sensory geometry conflicts with snapshot')
    indices = (senses.partition_indices(partitions) if hasattr(senses, 'partition_indices') else
               tuple(tuple(range(start, end)) for start, end in senses.regions(partitions)))
    bounds = None if geometry['interface'] in ('retina','signals') else senses.regions(partitions)
    window_bytes = options.window_bytes if options.window_bytes is not None else getattr(senses, 'raw_window_bytes', senses.receptors // 16)
    from .experimental.stream_input import ReceptorStream, iter_stimuli

    stream = ReceptorStream(senses, window_bytes)

    report = dict(model_mode='conversion' if model is None else 'learn' if options.learn else 'frozen',
                  receptors=senses.receptors, partitions=partitions,
                  window_bytes=window_bytes, max_frames=options.max_frames,
                  region_bounds=[list(bound) for bound in bounds] if bounds is not None else None,
                  region_indices=[list(region) for region in indices], sensory_geometry=geometry, frames=[])
    if hasattr(senses, 'partition_rectangles') and geometry['interface'] in ('retina','signals'):
        report['partition_rectangles'] = senses.partition_rectangles
    report.update(rounds_per_frame=options.rounds_per_frame,
                  input_clock='incoming byte windows; model actions do not advance input')

    def annotate(record, receptor_hex):
        bits = tuple((byte >> bit) & 1 for byte in bytes.fromhex(receptor_hex)
                     for bit in range(7, -1, -1))
        record.update(receptor_hex=receptor_hex,
                      region_active_counts=[sum(bits[pixel] for pixel in region) for region in indices])
        return record

    context = nullcontext(sys.stdin.buffer) if options.input == '-' else open(options.input, 'rb', buffering=0)
    with context as source:
        if model is None:
            # islice stops without requesting a sentinel beyond the frame budget.
            windows = iter_stimuli(source, window_bytes)
            try:
                for stimulus in islice(windows, options.max_frames):
                    report['frames'].append(annotate(
                        dict(input_window_size=len(stimulus)), stream.convert(stimulus)[0].hex()))
            finally:
                windows.close()
            reason = 'frame_limit' if len(report['frames']) == options.max_frames else 'end_of_stream'
            report['bytes_consumed'] = sum(frame['input_window_size'] for frame in report['frames'])
        else:
            from .experimental.stream_input import run_stream

            before = model.parameters()
            event_offset = len(model.events)
            stream_report = run_stream(model, source, window_bytes=window_bytes,
                                       frame_limit=options.max_frames, learn=options.learn,
                                       rounds_per_frame=options.rounds_per_frame)
            report.update(stream_report)
            for record in report['frames']:
                annotate(record, record['sensory'])
            events = model.events[event_offset:]
            reason = stream_report['stop_reason']
            report.update(action_policy=model.action_policy,
                          embedding_selection=model.embedding_selection,
                          parameter_changed=model.parameters() != before,
                          actions=[action for frame in report['frames'] for action in frame['actions']],
                          outputs=[event for event in events if event['kind'] in
                                   ('ACTION', 'ACTION_SIGNAL', 'EXPECTATION', 'BATCH_ACTION')],
                          events=events,
                          event_summary=dict(Counter(event['kind'] for event in events)))
    report.update(frame_count=len(report['frames']),
                  input_window_sizes=[frame['input_window_size'] for frame in report['frames']],
                  stop_reason=reason)
    return report


def main(argv=None):
    arguments = parser()
    options = arguments.parse_args(argv)
    try:
        report = inspect(options)
        encoded = json.dumps(report, indent=2, sort_keys=True) + '\n'
        if options.output is None:
            print(encoded, end='')
        else:
            # Exclusive creation also protects against a collision after preflight.
            with options.output.open('x', encoding='utf-8') as destination:
                destination.write(encoded)
            print(f"{report['frame_count']} frames; {report['stop_reason']}; {report['model_mode']}; {options.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        arguments.error(str(error))
    return report


if __name__ == '__main__':
    main()
