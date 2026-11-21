"""Stack safety and compatibility with the original recursive AST algorithms."""
import hashlib
import random
import unittest

from static_agent_training.initialize import (
    I, canonical_bytes, depth, display, is_closed, node_count, term_id,
)
from static_agent_training.reduction import contract, normalize, redexes, shift, substitute
from static_agent_training.repertoire import freeze
from static_agent_training.engine import run


def wrap(term, count, tag='abs'):
    for _ in range(count):
        term = (tag, term) if tag == 'abs' else ('app', term, I)
    return term


def ref_bytes(t):
    if t[0] == 'var':
        n, out = t[1], bytearray(b'v')
        while n >= 128:
            out.append((n & 127) | 128)
            n >>= 7
        out.append(n)
        return bytes(out)
    return (b'l' if t[0] == 'abs' else b'a') + b''.join(map(ref_bytes, t[1:]))


def ref_shift(t, amount, cutoff=0):
    if t[0] == 'var':
        return ('var', t[1] + amount if t[1] >= cutoff else t[1])
    return (t[0], *(ref_shift(c, amount, cutoff + (t[0] == 'abs')) for c in t[1:]))


def ref_sub(t, index, replacement, level=0):
    if t[0] == 'var':
        return ref_shift(replacement, level) if t[1] == index + level else t
    return (t[0], *(ref_sub(c, index, replacement, level + (t[0] == 'abs')) for c in t[1:]))


def ref_redexes(t, path=()):
    if t[0] == 'app' and t[1][0] == 'abs':
        yield path
    if t[0] != 'var':
        for i, child in enumerate(t[1:], 1):
            yield from ref_redexes(child, path + (i,))


def ref_contract(t, path):
    if path:
        parts = list(t)
        parts[path[0]] = ref_contract(t[path[0]], path[1:])
        return tuple(parts)
    return ref_shift(ref_sub(t[1][1], 0, ref_shift(t[2], 1)), -1)


def ref_count(t):
    return 1 if t[0] == 'var' else 1 + sum(map(ref_count, t[1:]))


def ref_depth(t):
    return 1 if t[0] == 'var' else 1 + max(map(ref_depth, t[1:]))


def ref_closed(t, bound=0):
    if t[0] == 'var':
        return 0 <= t[1] < bound
    return all(ref_closed(c, bound + (t[0] == 'abs')) for c in t[1:])


def ref_display(t):
    if t[0] == 'var':
        return str(t[1])
    if t[0] == 'abs':
        return 'λ.' + ref_display(t[1])
    return '(' + ref_display(t[1]) + ' ' + ref_display(t[2]) + ')'


class DeepTermsTests(unittest.TestCase):
    def test_deep_core_and_json_freeze(self):
        term = wrap(('var', 1999), 2000)
        expected = b'l' * 2000 + b'v\xcf\x0f'
        self.assertEqual(node_count(term), 2001)
        self.assertEqual(depth(term), 2001)
        self.assertTrue(is_closed(term))
        self.assertFalse(is_closed(wrap(('var', 2000), 2000)))
        self.assertEqual(canonical_bytes(term), expected)
        self.assertEqual(term_id(term), hashlib.sha256(b'crt-term-v1' + expected).hexdigest())
        self.assertEqual(display(term), 'λ.' * 2000 + '1999')
        json_ast = ['var', 1999]
        for _ in range(2000):
            json_ast = ['abs', json_ast]
        self.assertEqual(canonical_bytes(freeze(json_ast)), expected)
        self.assertEqual(canonical_bytes(freeze(term)), expected)
        for bad in (['var', True], ['var', -1], ['abs'], ['app', I], ['unknown', I]):
            with self.assertRaises(ValueError):
                freeze(wrap(bad, 2000))
        app = wrap(I, 2000, 'app')
        self.assertEqual(node_count(app), 6002)
        self.assertEqual(depth(app), 2002)
        self.assertEqual(canonical_bytes(freeze(app)), canonical_bytes(app))

    def test_deep_capture_avoidance_and_normalization(self):
        body = wrap(('var', 2000), 2000)
        self.assertEqual(canonical_bytes(shift(body, 3)), b'l' * 2000 + ref_bytes(('var', 2003)))
        self.assertEqual(canonical_bytes(substitute(body, 0, ('var', 1))),
                         b'l' * 2000 + ref_bytes(('var', 2001)))
        term = wrap(('app', I, ('var', 0)), 2000)
        path = (1,) * 2000
        self.assertEqual(tuple(redexes(term)), (path,))
        expected = canonical_bytes(wrap(('var', 0), 2000))
        self.assertEqual(canonical_bytes(contract(term, path)), expected)
        result = normalize(term, 1, max_nodes=10000)
        self.assertEqual((result.status, result.trace), ('NORMAL', (path,)))
        self.assertEqual(canonical_bytes(result.term), expected)
        root = ('app', ('abs', body), I)
        self.assertEqual(canonical_bytes(contract(root, ())), b'l' * 2000 + canonical_bytes(I))
        engine = run([(I, wrap(I, 2000))], 1, max_nodes=10000, max_transitions=1)
        self.assertEqual(engine['groups'][0]['status'], 'DEPTH')

    def test_seeded_recursive_reference_and_rng_witnesses(self):
        rng = random.Random(57123)

        def build(level, bound):
            tag = rng.choice(('var', 'abs', 'app')) if level else 'var'
            if tag == 'var':
                return ('var', rng.randrange(bound + 3))
            if tag == 'abs':
                return ('abs', build(level - 1, bound + 1))
            return ('app', build(level - 1, bound), build(level - 1, bound))

        for case in range(250):
            term, replacement = build(5, 0), build(3, 0)
            bound, index, amount = rng.randrange(4), rng.randrange(4), rng.randrange(-2, 4)
            self.assertEqual(canonical_bytes(term), ref_bytes(term))
            self.assertEqual(node_count(term), ref_count(term))
            self.assertEqual(depth(term), ref_depth(term))
            self.assertEqual(is_closed(term, bound), ref_closed(term, bound))
            self.assertEqual(display(term), ref_display(term))
            self.assertEqual(freeze(term), term)
            self.assertEqual(shift(term, amount, bound), ref_shift(term, amount, bound))
            self.assertEqual(substitute(term, index, replacement, bound), ref_sub(term, index, replacement, bound))
            self.assertEqual(tuple(redexes(term, (2,))), tuple(ref_redexes(term, (2,))))
            actual_rng, reference_rng = random.Random(case), random.Random(case)
            for _ in range(8):
                positions = tuple(ref_redexes(term))
                self.assertEqual(tuple(redexes(term)), positions)
                if not positions:
                    break
                selected = actual_rng.choice(tuple(redexes(term)))
                self.assertEqual(selected, reference_rng.choice(positions))
                self.assertEqual(actual_rng.getstate(), reference_rng.getstate())
                reduced = ref_contract(term, selected)
                self.assertEqual(contract(term, selected), reduced)
                term = reduced
