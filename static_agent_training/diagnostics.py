"""Persistent process diagnostics and an address-space ceiling for CLI runs."""
from contextlib import contextmanager
import faulthandler
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback


@contextmanager
def diagnostics(directory, memory_limit_mb=4096):
    """Log failures and resource use; SIGKILL/host shutdown cannot run handlers."""
    if memory_limit_mb < 256:
        raise ValueError('memory limit must be at least 256 MiB')
    directory = Path(directory)
    directory = directory.parent / (directory.name + "-diagnostics")
    directory.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    old_limit = resource.getrlimit(resource.RLIMIT_AS)
    limit = memory_limit_mb * 1024 * 1024
    hard = old_limit[1]
    limit = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
    if old_limit[0] != resource.RLIM_INFINITY:
        limit = min(limit, old_limit[0])
    with (directory / 'crash.log').open('a', buffering=1) as crash, (directory / 'resources.jsonl').open('a', buffering=1) as usage:
        def record(event, **extra):
            # Current resident and virtual size make growth visible before a crash.
            try:
                extra['memory'] = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                                   for line in Path('/proc/self/status').read_text().splitlines()
                                   if line.startswith(('VmRSS:', 'VmSize:'))}
            except OSError:
                pass
            usage.write(json.dumps(dict(time=time.time(), pid=os.getpid(), event=event,
                                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                        **extra)) + '\n')
            usage.flush()
            os.fsync(usage.fileno())
        def monitor():
            while not stop.wait(5):
                record('sample')
        enabled = faulthandler.is_enabled()
        faulthandler.enable(file=crash, all_threads=True)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        record('start', argv=sys.argv, memory_limit_bytes=limit)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            yield
        except BaseException:
            traceback.print_exc(file=crash)
            crash.flush()
            os.fsync(crash.fileno())
            record('failed')
            raise
        else:
            record('completed')
        finally:
            stop.set()
            thread.join()
            resource.setrlimit(resource.RLIMIT_AS, old_limit)
            faulthandler.disable()
            if enabled:
                faulthandler.enable()
