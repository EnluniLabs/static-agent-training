"""Dependency-free live terminal fitness plot, with plain log fallback."""

import math
import shutil
import sys
import time


def plot(history, *, width=60, height=10):
    """Zero-based percentage scale; sample the full history to fit the terminal."""
    width, height = max(2, width), max(2, height)
    if not history:
        return 'Waiting for the first completed generation...'
    count = min(width, len(history))
    indices = ([len(history) - 1] if count == 1 else
               [round(i * (len(history) - 1) / (count - 1)) for i in range(count)])
    peak = max(point[key] for point in history for key in ('mean', 'best', 'assessment_mean'))
    ceiling = min(1.0, max(.1, math.ceil(peak * 10) / 10))
    grid = [[' ' for _ in range(count)] for _ in range(height)]
    for column, index in enumerate(indices):
        for key, symbol in (('mean', 'M'), ('best', 'B'), ('assessment_mean', 'H')):
            value = max(0.0, min(1.0, history[index][key]))
            row = height - 1 - round(value / ceiling * (height - 1))
            previous = grid[row][column]
            grid[row][column] = symbol if previous == ' ' else '*'
    lines = [f'{round(100 * ceiling * (height - 1 - row) / (height - 1)):3d}% |' + ''.join(cells)
             for row, cells in enumerate(grid)]
    lines.append('     +' + '-' * count)
    lines.append(f"      generations {history[indices[0]]['generation'] + 1} to {history[indices[-1]]['generation'] + 1}")
    lines.append('M mean training  B best training  H mean held-out  * overlap')
    lines.append('Y axis auto-scales; zero baseline.')
    return '\n'.join(lines)


class Dashboard:
    def __init__(self, config, *, live=True, stream=None):
        self.config = config
        self.stream = stream or sys.stdout
        self.live = live and self.stream.isatty()
        self.history = []
        self.last_draw = 0.0
        self.started = time.monotonic()
        self.last_status = {}

    def __call__(self, status):
        self.last_status = status
        if 'history' in status:
            self.history = status['history']
        phase = status['phase']
        force = phase in ('ready', 'generation-complete', 'finished', 'interrupted')
        if not self.live:
            if phase == 'generation-complete':
                s = status['summary']
                self.stream.write(f"gen {s['generation'] + 1:4d} | mean {s['mean']:6.1%} | "
                                  f"best {s['best']:6.1%} | held-out {s['assessment_mean']:6.1%}\n")
                self.stream.flush()
            return
        now = time.monotonic()
        if not force and now - self.last_draw < .2:
            return
        self.last_draw = now
        columns, rows = shutil.get_terminal_size(fallback=(90, 24))
        elapsed = now - self.started
        lines = ['Static agent training | one population | ' + ('learning' if self.config.learning else 'no learning'),
                 f'Population {self.config.population} | seed {self.config.seed} | elapsed {elapsed:.1f}s']
        generation = status.get('generation', len(self.history))
        if phase == 'evaluating':
            lines.append(f"Generation {generation + 1}/{self.config.generations}: evaluating "
                         f"{status['completed']}/{status['total']} | "
                         f"running mean {status['mean']:.1%}, best {status['best']:.1%}")
        elif 'completed' in status:
            lines.append(f"{phase.capitalize()} {status['completed']}/{status['total']}")
        else:
            lines.append(phase.replace('-', ' ').capitalize())
        lines.append(plot(self.history, width=max(2, columns - 9), height=max(2, min(10, rows - 11))))
        if self.history:
            s = self.history[-1]
            lines.append(f"Latest: mean {s['mean']:.1%} | best {s['best']:.1%} | held-out {s['assessment_mean']:.1%}")
        lines.append('Fitness = episodes reaching target. Held-out assessment has learning disabled.')
        self.stream.write('\x1b[H\x1b[2J' + '\n'.join(lines) + '\n')
        self.stream.flush()
