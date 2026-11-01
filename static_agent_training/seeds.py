"""Independent reproducible streams derived from experiment coordinates."""

import hashlib
import json


def derive(seed, *coordinates):
    payload = json.dumps(['crt-training-v1', seed, *coordinates], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), 'big')
