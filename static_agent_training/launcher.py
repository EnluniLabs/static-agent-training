"""Short launcher for fresh experiments and the latest saved run."""

from datetime import datetime
import os
from pathlib import Path
import sys


HELP = '''Usage:
  ./train                         Start a new run with a live graph
  ./train compare                 Run matched treatment benchmarks
  ./train compare --treatment learning
  ./train status                  Show running and saved runs
  ./train status --watch          Refresh progress every 5 seconds
  ./train resume                  Resume the most recently checkpointed run
  ./train resume runs/my-run      Resume a specific run
  ./train --population 64 --generations 100
  ./train --backend native --workers 8 --population 1024
  ./train --distance-mean 10 --distance-stddev 5

Defaults: 32 animals, 50 generations, seed 42; automatic unique run folder.
Trials: uniform start/goal pairs; hard walls at -20 and +20.
Press Ctrl-C to stop. Checkpoints save after every completed generation.
Extra options are passed to the experiment runner.
'''


def main(root, argv=None):
    root = Path(root)
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ('-h', '--help', 'help'):
        print(HELP)
        return
    if args and args[0] == 'compare':
        os.chdir(root)
        environment_python = root / '.venv' / 'bin' / 'python'
        executable = str(environment_python) if environment_python.exists() else sys.executable
        os.execv(executable, [executable, '-m', 'static_agent_training.comparison', *args[1:]])
        return
    if args and args[0] == 'status':
        from .status import main as status_main
        status_main(args[1:])
        return
    if args and args[0] == 'resume':
        args.pop(0)
        if args and not args[0].startswith('-'):
            directory = args.pop(0)
        else:
            checkpoints = list((root / 'runs').glob('*/checkpoint.json.gz'))
            if not checkpoints:
                raise SystemExit('No saved runs found. Start one with ./train')
            checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
            directory = str(checkpoint.parent)
        options = ['--resume', directory, *args]
    elif any(arg == '--resume' or arg.startswith('--resume=') for arg in args):
        options = args
    else:
        directory = Path('runs') / datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
        options = ['--population', '32', '--generations', '50', '--seed', '42',
                   '--output', str(directory), *args]
    os.chdir(root)
    environment_python = root / '.venv' / 'bin' / 'python'
    executable = str(environment_python) if environment_python.exists() else sys.executable
    os.execv(executable, [executable, '-m', 'static_agent_training.experiment', *options])
