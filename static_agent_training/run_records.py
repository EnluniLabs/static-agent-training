"""Read small generation summaries without decoding large episode/mutation logs."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=4096)
def _read(filename, modified, size):
    # Bounded reads tolerate concurrent truncation without mmap's SIGBUS risk.
    marker = b'"summary":'
    decoder = json.JSONDecoder()
    with open(filename, 'rb') as handle:
        # Atomic generation records normally place the small summary near the
        # end. Seeking avoids scanning gigabytes of episode data for a chart.
        handle.seek(max(0, size - 262144))
        tail = handle.read()
        index = tail.rfind(marker)
        if index >= 0:
            try:
                summary, _ = decoder.raw_decode(tail[index + len(marker):].decode().lstrip())
                if isinstance(summary, dict) and 'generation' in summary and 'best' in summary:
                    return summary
            except (ValueError, UnicodeError):
                pass
        handle.seek(0)
        tail = b''
        while chunk := handle.read(65536):
            data = tail + chunk
            index = data.find(marker)
            if index >= 0:
                payload = data[index + len(marker):] + handle.read(65536)
                summary, _ = decoder.raw_decode(payload.decode().lstrip())
                if isinstance(summary, dict) and 'generation' in summary and 'best' in summary:
                    return summary
                raise ValueError(f'Invalid summary in {filename}')
            tail = data[-(len(marker)-1):]
    raise ValueError(f'Missing summary in {filename}')


def read_summary(path):
    path = Path(path)
    stat = path.stat()
    return _read(str(path), stat.st_mtime_ns, stat.st_size)
