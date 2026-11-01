"""Read small generation summaries without decoding large episode/mutation logs."""
from functools import lru_cache
import json
import mmap
from pathlib import Path


@lru_cache(maxsize=4096)
def _read(filename, modified, size):
    # Native generation records have one 'summary' field. mmap.find scans in C;
    # decode only that small object rather than tens of MB of episode records.
    with open(filename, 'rb') as handle:
        if size:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                marker = b'"summary":'
                index = data.find(marker)
                if index >= 0:
                    start = index + len(marker)
                    summary, _ = json.JSONDecoder().raw_decode(data[start:start+65536].decode().lstrip())
                    if isinstance(summary, dict) and 'generation' in summary and 'best' in summary:
                        return summary
    return json.loads(Path(filename).read_text())['summary']


def read_summary(path):
    path = Path(path)
    stat = path.stat()
    return _read(str(path), stat.st_mtime_ns, stat.st_size)
