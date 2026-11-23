"""Stack-safe JSON for AST-bearing artifacts, preserving the standard JSON format."""

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodedJSON:
    """Validated immutable JSON fragment for streaming artifact writes only.

    Construction validates with the stack-safe reader. Ordinary strings remain
    quoted strings; only this explicit wrapper is emitted as JSON by dump().
    """
    document: str

    def __post_init__(self):
        if not isinstance(self.document, str):
            raise TypeError('encoded JSON requires text')
        loads(self.document)


def dumps(value, *, sort_keys=False, separators=None, indent=None):
    # Keep ordinary artifacts byte-for-byte compatible with the standard encoder.
    try:
        return json.dumps(value, sort_keys=sort_keys, separators=separators, indent=indent)
    except RecursionError:
        pass
    return ''.join(_iterencode(value, sort_keys=sort_keys, separators=separators, indent=indent))


def _iterencode(value, *, sort_keys=False, separators=None, indent=None):
    item_sep, key_sep = separators or ((',', ': ') if indent is not None else (', ', ': '))
    active = set()
    pending = [('value', value)]
    while pending:
        kind, item = pending.pop()
        if kind == 'text':
            yield item
        elif kind == 'end':
            active.remove(item)
        elif isinstance(item, EncodedJSON):
            for offset in range(0, len(item.document), 65536):
                yield item.document[offset:offset + 65536]
        elif isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError('Circular reference detected')
            active.add(identity)
            pending.append(('end', identity))
            if isinstance(item, dict):
                entries = sorted(item.items()) if sort_keys else list(item.items())
                yield '{'
                pending.append(('text', '}'))
                for index in range(len(entries) - 1, -1, -1):
                    key, child = entries[index]
                    # Delegate key coercion and validation to the standard encoder.
                    encoded_key = json.dumps({key: None}, separators=(',', ':'))[1:-6]
                    if index < len(entries) - 1:
                        pending.append(('text', item_sep))
                    pending.extend([('value', child), ('text', key_sep), ('text', encoded_key)])
            else:
                yield '['
                pending.append(('text', ']'))
                for index in range(len(item) - 1, -1, -1):
                    if index < len(item) - 1:
                        pending.append(('text', item_sep))
                    pending.append(('value', item[index]))
        else:
            yield json.dumps(item)


def dump(value, handle, **kwargs):
    if kwargs.get('indent') is not None:
        handle.write(dumps(value, **kwargs))
    else:
        # Bounded batches avoid a TextIO/gzip write for every scalar or bracket.
        # Keep iterative encoding so deeply developed terms remain stack-safe.
        chunks, size = [], 0
        for chunk in _iterencode(value, **kwargs):
            chunks.append(chunk)
            size += len(chunk)
            if size >= 65536:
                handle.write(''.join(chunks))
                chunks.clear()
                size = 0
        if chunks:
            handle.write(''.join(chunks))


_WHITESPACE = re.compile(r'[ \t\n\r]*')


def loads(document):
    try:
        return json.loads(document)
    except RecursionError:
        pass
    if isinstance(document, (bytes, bytearray)):
        document = document.decode(json.detect_encoding(document), 'surrogatepass')
    # Containers use explicit frames; raw_decode only ever sees scalar tokens.
    decoder = json.JSONDecoder()
    frames = []
    position = 0
    root = None
    have_root = False

    def error(message):
        raise json.JSONDecodeError(message, document, position)

    def attach(value):
        nonlocal root, have_root
        if not frames:
            root, have_root = value, True
        else:
            container, state, key = frames[-1]
            if isinstance(container, list):
                container.append(value)
            else:
                container[key] = value
            frames[-1][1] = 'comma'

    while True:
        position = _WHITESPACE.match(document, position).end()
        if frames:
            container, state, key = frames[-1]
            closing = ']' if isinstance(container, list) else '}'
            char = document[position:position + 1]
            if state in ('first', 'comma') and char == closing:
                frames.pop()
                position += 1
                continue
            if state == 'comma':
                if char != ',':
                    error("Expecting ',' delimiter")
                position += 1
                frames[-1][1] = 'value' if isinstance(container, list) else 'key'
                continue
            if isinstance(container, dict) and state in ('first', 'key'):
                if char != '"':
                    error('Expecting property name enclosed in double quotes')
                key, position = decoder.raw_decode(document, position)
                position = _WHITESPACE.match(document, position).end()
                if document[position:position + 1] != ':':
                    error("Expecting ':' delimiter")
                position += 1
                frames[-1][1:] = ['value', key]
                continue
        elif have_root:
            if position != len(document):
                error('Extra data')
            return root
        char = document[position:position + 1]
        if char in ('[', '{'):
            container = [] if char == '[' else {}
            attach(container)
            frames.append([container, 'first', None])
            position += 1
        else:
            value, position = decoder.raw_decode(document, position)
            attach(value)


def load(handle):
    return loads(handle.read())
