# Fixed-Sense Error Propagation Audit

Scope: dated measurement, audit or research design; its results and proposals are retained as historical evidence. Current implemented behavior: [architecture reference](../ARCHITECTURE.md), reviewed 2026-11-22.

Audited commit: `594951dc7de4687c5e2539ea6f3a881f59189416`.
Executed 2026-11-03 UTC (2026-11-02 America/Chicago).

**Result: local correction and delayed timing pass; transfer is selective only in
part. Shared components carry the expected structural correction, but the
unrelated control also changes. Prediction learning through a shared repertoire
state harms the control's distinguishing components, even while its full-pattern
loss improves. These tests establish mechanical behavior, not generally useful
semantic generalization.**

No production source, learning algorithm, saved animal, or running session was
changed. No evolutionary training or performance benchmark was run. The only
repository addition for this audit is this file.

## Execution and controls

All learning ran in the existing C++ `sensory_one` implementation from
[`evaluate_senses.cpp`](../../static_agent_training/native/evaluate_senses.cpp),
using its production MT19937 sampling, empirical update, revised-state flags,
structural check, and episode-reset behavior. Python prepared arrays and read
snapshots; it did not apply learning updates. There was no Python-model replay.

A temporary copy of the native source added read-only observation hooks:
after embedding, at a state encounter, after expectation sampling, before
feedback updates, after an empirical update, and before/after an eligible
structural correction. A read-only probe called the existing native
`sensory_logits` function. It did not replace a learning expression, branch,
random draw, or update. The instrumented one-episode and two-episode executions
matched the **unmodified native kernel exactly** for every final array, metric,
and revision flag. Temporary instrumentation was not installed in the project.

The complete harness is embedded below so this remains a one-file record.
Compilation used g++ 15.2.0 with `-O3 -std=c++17 -shared -fPIC -fopenmp
-ffp-contract=off -DMAX_Q=4 -DMAX_R=256`; native evaluation used one worker.
The full audit was run twice: both serialized result records were byte-identical.
Every assertion in the embedded harness passed on both executions.

Source SHA-256 values:

```text
evaluate_senses.cpp e1c44d6f62d00d45ce2ca8ce3c5f5de22faf413bf160a27e446015a5dfddc8a3
evaluate.cpp        5b8d6ec86a8928515c376058f6f69d61eb87e9e2a6409e5d33c06a8348eb3b54
```

### Common deterministic fixture

- One coherent four-state repertoire, developed once from `(I,K)` with budget 20;
  no mutation, selection, or redevelopment during testing.
- Sorted state columns are `q0,q1,q2,q3`; entry is `q1`, prediction source is
  `q3`, and that source's fixed successor is `q0`. Edges are
  `[2,3,0,0]`: `q0→q2`, `q1→q3`, `q2→q0`, `q3→q0`.
- 256 fixed dual-rail receptors; one presence feature; nine actions plus one
  prediction gate. The standard converter processes the bytes below unchanged.
- Native NumberLine motion/termination is retained: start 0, goal 1, action +1
  ends each episode. The harness supplies synthetic byte-derived sensory packets
  through the native frame table instead of JSON number-line packets. This
  isolates component sharing from number serialization and boundary effects.
- Initial biases `[−4,4,0,0]`; all input weights zero. The source strongly selects
  prediction and its successor strongly selects +1, using gate logits ±40.
  The native seeded draws actually select those outputs; no output is substituted.
- Source prediction logits are +0.01 for active receptors of B, −0.01 otherwise;
  other states' prediction logits are zero. Parameters are set only before a test.
- Seed 0, phase `training`, standard derived embedding/output RNG streams;
  prediction rate 0.05, embedding rate 0.05, compatibility threshold 0.1.

| Input | Raw bytes | Meaning in this synthetic fixture | Active receptors shared with A |
|---|---|---|---:|
| A | `a0 01` | Family `a0`, instance 01; expected next sensory packet A | 16 / 16 |
| B | `a0 02` | Same family, different instance; expected next packet A | 14 / 16 |
| C | `5f fe` | Unrelated family; expected next packet C | 0 / 16 |

The family is an externally assigned test meaning, not a decoder or label given
to the animal. A and B share the whole family byte plus six instance bits.
C complements A's source bits and shares no active receptors. All inputs still
share the presence feature, bias, and 224 inactive trailing receptor bits.
There is no folding or collision for these two-byte inputs.

Only A is observed during learning. B and C are evaluated with learning disabled.
For the structural mapping probe, A/B's declared desired continuation is `q0`;
C's desired mapping remains `q1`. This deliberately simple criterion tests the
direction of credit transfer; it does not establish that a state assignment is
useful in a learned environmental task. Movement success is predetermined and is
not counted as evidence of learning or generalization.

## 1. Local Error Correction

**Setup.** Run one native episode exposing A at both start and outcome. The
source emits one sampled prediction, then the successor performs +1. The sampled
prediction differs from A in **107 of 256 bits**, so this is an actual incorrect
prediction. The initial modal prediction is B, differing from A in four receptor
bits. Terminal feedback is processed before the episode ends.

| Measurement | Before feedback | After one empirical update |
|---|---:|---:|
| NLL of observed full pattern A, conditional on prediction | 176.208878210013 | 173.059602903936 |
| Probability of observed full pattern A | 2.97479149411e-77 | 6.93696364472e-76 |
| NLL on the 32 receptors representing the actual bytes | 22.061109776252 | 21.666580861353 |
| Modal receptor prediction | B | A |
| Embedding probability of the checked successor `q0` | 0.000323503748800442 | 0.000323503748800442 |
| Empirical updates / structural corrections | 0 / 0 | 1 / 0 |

All four groups of source prediction-logit changes are listed below. The counts
sum to 256; probabilities in parentheses are Bernoulli probabilities of bit 1.

| Count | Outcome bit | Logit before | Logit after | Direction |
|---:|---:|---:|---:|---|
| 14 | 1 | +0.010000000000 (0.502499979167) | +0.034875001042 (0.508717866675) | Toward 1 |
| 238 | 0 | −0.010000000000 (0.497500020833) | −0.034875001042 (0.491282133325) | Toward 0 |
| 2 | 1 | −0.010000000000 (0.497500020833) | +0.015124998958 (0.503781177656) | Corrects modal bit |
| 2 | 0 | +0.010000000000 (0.502499979167) | −0.015124998958 (0.496218822344) | Corrects modal bit |

**Affected parameters.** Exactly 256 prediction logits in source row `q3`;
zero prediction logits in other rows; zero bias, input-weight, or gate changes.
The update matches `v -= 0.05 * (sigmoid(v) - observed_bit)` exactly under scalar
native-compatible arithmetic. The source's revised flag changes 0→1. Pending
feedback is consumed; no structural update is performed at feedback time.

**Assessment: matches the implemented local mechanism.** Probability moves toward
the outcome, and empirical credit stays at the emitting prediction row. Exact
sampled correctness and prediction probability are distinct: this one update
does not make subsequent stochastic full-pattern predictions reliably correct.

## 2. Error Transfer

**Setup.** Inspect three snapshots: original parameters; immediately after the
single A empirical update; and immediately after the first later structural
correction caused by that update, before any second empirical feedback. Probe B
and C without learning. All three snapshots receive frozen native forward passes;
all parameter arrays stay byte-identical during those passes, and empirical and
structural update counts are zero. A is the only training input.

### Empirical transfer through a shared prediction state

| Held-out expected outcome | Full-pattern NLL before | After empirical correction on A | NLL on meaningful 32 receptors before → after |
|---|---:|---:|---:|
| A after A | 176.208878210013 | 173.059602903936 | 22.061109776252 → 21.666580861353 |
| A after B | 176.208878210013 | 173.059602903936 | 22.061109776252 → 21.666580861353 |
| C after C | 176.448878210013 | 174.096602928936 | **22.301109776252 → 22.703580886352** |

The frozen native losses match independently read prediction-logit losses within
3e-14. All inputs embed with the same distribution before structural correction
and reach the same prediction state under this seed. B benefits because this
fixture expects the same outcome A. This empirical transfer occurs through
**state reuse**, not receptor-weight changes: the empirical step leaves every
input weight and embedding distribution unchanged. C inherits that same source
prediction update even though it shares no active sensory receptors with A.

**Unexpected/important result:** C's full-pattern score improves while its
meaningful-component prediction gets worse. The 224 inactive trailing bits have
NLL 154.147768433761→151.393022042584 for every input, masking the increased
error on C's 32 informative receptors. Full-pattern probability for C rises
2.34005387013e-77→2.45926896876e-76 despite that interference. A full-pattern
loss improvement alone would incorrectly suggest useful transfer to this control.

### Structural transfer through shared input components

Initial embedding distribution for every probe, in `[q0,q1,q2,q3]` order:

```text
[0.000323503748800442, 0.9643510838246173,
 0.017662706213291114, 0.017662706213291114]
```

After the A empirical update this distribution is unchanged for every probe.
After its delayed structural correction:

| Probe | `q0` logit before → after | `P(q0)` before → after | Other probabilities after `[q1,q2,q3]` |
|---|---|---|---|
| A | −4 → −3.950000000000 | 0.000323503748800442 → 0.000340084499908405 | `[0.9643350889849251, 0.01766241325758323, 0.01766241325758323]` |
| B | −4 → −3.955555555556 | 0.000323503748800442 → 0.000338201017298974 | `[0.9643369059112032, 0.01766244653574883, 0.01766244653574883]` |
| C | −4 → −3.994444444444 | 0.000323503748800442 → 0.000325305407169138 | `[0.9643493458311694, 0.01766267438083073, 0.01766267438083073]` |

**Affected parameters.** One shared bias and 17 input weights, all in the `q0`
column; the exact coordinates and deltas are in test 3. No additional learning
occurs for B or C. For a probe x, the target-logit increment is
`delta * (1 + shared_active_features(x,A))`, including the presence feature in
that overlap. Thus A gets 0.05, B gets 0.0444444444444, and C still gets
0.00555555555556 from bias plus presence. All other state logits stay unchanged;
their probabilities change through softmax normalization.

**Assessment: mixed.** The B mapping change has the intended direction and
strength ordering for this declared shared-family fixture. C is not isolated:
its desired `P(q1)` falls 0.9643510838246173→0.9643493458311694. The effect is small
but systematic, not random measurement noise. The native mechanism implements
overlap-weighted propagation plus global spillover; it contains no learned
semantic test that decides whether this spillover is appropriate. No claim of
useful generalization beyond this controlled fixture follows.

## 3. Delayed Structural Correction

**Setup.** Start fresh from the same initial parameters, then run two A episodes
inside one native lifetime call. The first produces one empirical error. The
second re-embeds A and traverses the same coherent `q1→q3→q0` structure. Read
snapshots around the real structural check, before the second prediction and its
later feedback. Revision history remains in the lifetime across episode reset.

| Actual order | State/event | Empirical updates so far | Structural corrections so far |
|---:|---|---:|---:|
| 1 | Episode 0: encounter `q3`, revised flag 0 | 0 | 0 |
| 2 | Emit incorrect prediction from `q3` | 0 | 0 |
| 3 | Encounter `q0`; execute +1; observe A | 0 | 0 |
| 4 | Apply feedback to `q3`; revised flag becomes 1 | 1 | 0 |
| 5 | Episode 1: observe A, embed `q1`, follow edge to `q3` | 1 | 0 |
| 6 | Revised `q3` checks its modal A against fixed successor `q0` | 1 | 0 |
| 7 | Conflict correction completes | 1 | 1 |
| 8 | Emit next prediction; later execute action and receive feedback | 2 after feedback | 1 |

The modal pattern at the check is exactly A after the empirical update. Native
compatibility is 0.000323503748800442, below 0.1. There is no correction on the
first unrevised encounter, no embedding correction at feedback, and exactly one
correction on the later source encounter. The target probability rises to
0.000340084499908405; it remains below threshold, so the rule is a small local
increment rather than a one-step repair.

Before/after parameter effects of that **structural step alone**:

| Parameter group | Changed cells | Before → after |
|---|---:|---|
| Bias | 1: `bias[q0]` | −4 → −3.997222222222222 |
| Input weights | 17: `W[f,q0]` for f listed below | Each 0 → 0.002777777777777778 |
| Other state columns | 0 | Unchanged |
| Prediction logits | 0 | Unchanged from first empirical update |
| Action/prediction gates | 0 | Unchanged |
| Repertoire nodes and successor edges | 0 | Same nodes; `[2,3,0,0]` before and after |

Feature indices, where feature 0 is presence and feature `i+1` is receptor i:

```text
[0, 2, 3, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32]
```

There are 17 active features, so each cell and the bias receives
`0.05/(1+17) = 0.002777777777777778`. The checked A logit gains exactly 0.05;
B/C receive the overlap-dependent changes reported above. Both native episodes
had one expectation, one tested incorrect prediction, and one empirical update;
conflict counts were `[0,1]`.

**Assessment: matches the intended delayed encounter-local mechanism.** Credit
uses the revised source's fixed successor, touches only that successor's bias and
active-feature weights, and leaves the coherent repertoire intact. This verifies
correction timing and locality, not that this chosen successor is environmentally
optimal or that repeated corrections would remain beneficial.

## Answer to the central question

Error reaches the emitting prediction row and later the appropriate fixed
successor column at the intended times. Shared components produce measurable,
ordered transfer to a related input. However, propagation is not exclusively to
useful places: shared-state prediction learning harms an unrelated input's
informative components, and global bias/presence updates leak structural credit
into that input too. Inactive receptor learning can mask this harm in the aggregate
loss. The tests therefore support **mechanical correctness with demonstrated
interference**, not a blanket claim that error propagation produces useful
changes. No unexpected behavior was corrected during this audit.

## Reproduce this audit

From the repository root, extract the single Python block below to a temporary
file and run it with the existing NumPy-enabled environment:

```sh
python3 - <<'PY'
from pathlib import Path
text = Path('docs/audits/ERROR_PROPAGATION_AUDIT.md').read_text()
code = text.split('<!-- AUDIT_HARNESS_BEGIN -->\n```python\n', 1)[1].split('\n```\n<!-- AUDIT_HARNESS_END -->', 1)[0]
Path('/tmp/run_error_propagation_audit.py').write_text(code + '\n')
PY
PYTHONPATH="$PWD" .venv/bin/python /tmp/run_error_propagation_audit.py
```

It writes detailed measurements only to `/tmp/error_propagation_results.json`,
compiles instrumentation in a temporary directory, and asserts exact parity with
the unmodified native evaluator. The source hashes above identify the audited
implementation. Running against a later revision would audit that later code.

<!-- AUDIT_HARNESS_BEGIN -->
```python
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from static_agent_training.accelerated import SOURCE as SHARED, _random_states
from static_agent_training.experimental.sensory_accelerated import SOURCE, _kernel, _pack
from static_agent_training.experimental.senses import FixedSenses
from static_agent_training.genome import develop
from static_agent_training.initialize import I, K, pair_id

ROOT = Path.cwd()
R, Q = 256, 4
senses = FixedSenses(R)
animal = develop([(I, K)], (), (), budgets=20, senses=senses)
ids = animal.inherited.pair_ids
entry = ids.index(pair_id(I, K))
source = ids.index(animal.repertoire.successors[ids[entry]])
successor = ids.index(animal.repertoire.successors[ids[source]])
A, B, C = b'\xa0\x01', b'\xa0\x02', b'\x5f\xfe'
features = {name: np.asarray(senses.spec.encode(senses.frame(raw)), dtype=np.uint8)
            for name, raw in [('A', A), ('B', B), ('C', C)]}
base = _pack([animal], 0, True)
base = [a.copy() for a in base]
base[0][:] = 0; base[0][0, entry] = 4; base[0][0, successor] = -4
base[1][:] = 0; base[2][:] = -40; base[3][:] = 0
move = list(base[7]).index(1)
for j in range(Q):
    base[2][0,j,move] = 40
base[2][0,source,:] = -40; base[2][0,source,9] = 40
base[3][0,source] = np.where(features['B'][1:], .01, -.01)
# Input A is observed at start and after the action; B/C are held-out probes.
base[6][:] = features['A']

original = SOURCE.read_text().replace('#include "evaluate.cpp"', SHARED.read_text())
# These hooks only copy values. No branch, update, RNG, or arithmetic is replaced.
header = '''
extern "C" {
typedef void (*audit_callback)(int,int,int,const double*,const double*,const double*,const double*,const unsigned char*,const double*,const unsigned char*);
static audit_callback audit_hook=nullptr;
void audit_set_hook(audit_callback callback){audit_hook=callback;}
}
#define AUDIT(stage,who,bits) if(audit_hook)audit_hook(stage,e,who,b,w,g,p,rev,stats,bits)
'''
source_text = SOURCE.read_text().replace('DEVICE void sensory_one(', header+'\nDEVICE void sensory_one(', 1)
insertions = [
 ('pair=sample(logits,count,random_uniform(erng,stride));observe=false;',
  'pair=sample(logits,count,random_uniform(erng,stride));observe=false;AUDIT(0,pair,nullptr);'),
 ('if(learning&&rev[pair]&&edges[pair]>=0){',
  'AUDIT(6,pair,nullptr);\n            if(learning&&rev[pair]&&edges[pair]>=0){AUDIT(4,pair,nullptr);'),
 ('for(int i=0;i<n;++i)w[active[i]*Q+target]+=delta;stats[10]+=1;',
  'for(int i=0;i<n;++i)w[active[i]*Q+target]+=delta;stats[10]+=1;AUDIT(5,pair,nullptr);'),
 ('++pending;stats[4]+=1;', 'AUDIT(1,pair,pending_bits[pending]);++pending;stats[4]+=1;'),
 ('// Capture emission losses before sequential updates to repeated source rows.',
  'AUDIT(2,pair,observed);\n            // Capture emission losses before sequential updates to repeated source rows.'),
 ('stats[11]+=loss+c;rev[sources[k]]=1;stats[9]+=1;',
  'stats[11]+=loss+c;rev[sources[k]]=1;stats[9]+=1;AUDIT(3,sources[k],observed);'),
]
for old, new in insertions:
    assert source_text.count(old) == 1, old
    source_text = source_text.replace(old, new, 1)
# Read-only probe invokes the production sensory_logits function.
source_text += '''
extern "C" void audit_probe(const double* b,const double* w,const unsigned char* frame,int Q,int R,double* logits,double* probs){
    int active[MAX_R+1],n=0;for(int i=0;i<=R;++i)if(frame[i])active[n++]=i;
    sensory_logits(logits,b,w,active,n,Q,Q);
    double largest=logits[0],total=0;for(int j=1;j<Q;++j)if(logits[j]>largest)largest=logits[j];
    for(int j=0;j<Q;++j){probs[j]=exp(logits[j]-largest);total+=probs[j];}
    for(int j=0;j<Q;++j)probs[j]/=total;
}
'''
source_text = source_text.replace('#include "evaluate.cpp"', SHARED.read_text())
CALLBACK = ct.CFUNCTYPE(None, ct.c_int, ct.c_int, ct.c_int,
                       *([ct.POINTER(ct.c_double)]*4), ct.POINTER(ct.c_ubyte),
                       ct.POINTER(ct.c_double), ct.POINTER(ct.c_ubyte))


def snapshot(b, w, g, p):
    return dict(bias=np.array(b, copy=True).reshape(Q), weights=np.array(w, copy=True).reshape(R+1,Q),
                gates=np.array(g, copy=True).reshape(Q,10), prediction=np.array(p, copy=True).reshape(Q,R))


def initial_snapshot(arrays):
    return snapshot(*[a[0] for a in arrays[:4]])


def invoke(arrays, episodes, learning, function, revised=None):
    metrics = np.zeros((1,episodes,12)); rev = np.zeros((1,Q),dtype=np.uint8) if revised is None else revised
    inputs = [*arrays,np.zeros(episodes,dtype=np.int32),np.ones(episodes,dtype=np.int32),
              _random_states([0],'training','embedding'),_random_states([0],'training','output')]
    function(1,Q,R,episodes,*[ct.c_void_p(a.ctypes.data) for a in inputs],
             int(learning),1,.05,.05,.1,ct.c_void_p(metrics.ctypes.data),ct.c_void_p(rev.ctypes.data),1)
    return metrics,rev


def differences(before, after):
    return {name: dict(count=int(np.count_nonzero(after[name]!=before[name])),
                coordinates=np.argwhere(after[name]!=before[name]).tolist()) for name in before}


def loss(row, target):
    return float(np.sum(np.maximum(row,0)-row*target+np.log1p(np.exp(-np.abs(row)))))


def probability(row,target):
    return math.exp(-loss(row,target))


with tempfile.TemporaryDirectory(prefix='native-error-audit-') as temporary:
    libpath=Path(temporary)/'audit.so'
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-ffp-contract=off',
                    '-DMAX_Q=4','-DMAX_R=256','-x','c++','-','-o',str(libpath)],
                   input=source_text,text=True,capture_output=True,check=True)
    lib=ct.CDLL(str(libpath)); function=lib.evaluate_senses_cpu
    function.argtypes=_kernel(Q,R,'native').argtypes; function.restype=None
    lib.audit_set_hook.argtypes=[CALLBACK]
    lib.audit_probe.argtypes=[ct.c_void_p]*3+[ct.c_int]*2+[ct.c_void_p]*2
    lib.audit_probe.restype=None
    events=[]
    def receive(stage,episode,pair,b,w,g,p,rev,stats,bits):
        value=snapshot(np.ctypeslib.as_array(b,(Q,)),np.ctypeslib.as_array(w,((R+1)*Q,)),
                       np.ctypeslib.as_array(g,(Q*10,)),np.ctypeslib.as_array(p,(Q*R,)))
        events.append(dict(stage=stage,episode=episode,pair=pair,parameters=value,
            revised=np.ctypeslib.as_array(rev,(Q,)).copy(),metrics=np.ctypeslib.as_array(stats,(12,)).copy(),
            bits=np.ctypeslib.as_array(bits,(R,)).copy() if bits else None))
    callback=CALLBACK(receive);lib.audit_set_hook(callback)
    # Test 1: one terminal action, one wrong prediction and one empirical update.
    first=[a.copy() for a in base];m1,r1=invoke(first,1,True,function);events1=list(events)
    assert m1[0,0,4]==m1[0,0,5]==m1[0,0,9]==1 and m1[0,0,6]==m1[0,0,10]==0
    outcome=next(e for e in events1 if e['stage']==2)
    empirical=next(e for e in events1 if e['stage']==3)
    before=outcome['parameters'];after=empirical['parameters']
    assert all(np.array_equal(before[k],initial_snapshot(base)[k]) for k in before)
    expected=np.array([v-.05*((1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v)))-int(bit))
                       for v,bit in zip(before['prediction'][source],features['A'][1:])])
    assert np.array_equal(expected,after['prediction'][source])
    assert np.array_equal(after['prediction'][source]>=0,features['A'][1:].astype(bool))
    assert loss(after['prediction'][source],features['A'][1:]) < loss(before['prediction'][source],features['A'][1:])
    assert differences(before,after)['prediction']['count']==R
    assert all(differences(before,after)[k]['count']==0 for k in ('bias','weights','gates'))
    assert np.all(after['prediction'][np.arange(Q)!=source]==before['prediction'][np.arange(Q)!=source])
    sampled=next(e['bits'] for e in events1 if e['stage']==1)
    assert not np.array_equal(sampled,features['A'][1:])
    # Test 3: two episodes in one native call; revision persists across reset.
    events.clear();twice=[a.copy() for a in base];m2,r2=invoke(twice,2,True,function);events2=list(events)
    assert m2[0,0,10]==0 and m2[0,1,10]==1
    pre=next(e for e in events2 if e['stage']==4)
    post=next(e for e in events2 if e['stage']==5)
    assert pre['episode']==1 and pre['pair']==source and pre['revised'][source]==1
    assert np.array_equal(pre['parameters']['prediction'],after['prediction'])
    assert np.array_equal(pre['parameters']['weights'],before['weights'])
    assert np.array_equal(base[4],twice[4]) and np.array_equal(base[5],twice[5])
    active=np.flatnonzero(features['A']);delta=.05/(1+len(active))
    changes=differences(pre['parameters'],post['parameters'])
    assert changes['bias']['coordinates']==[[successor]]
    assert changes['weights']['coordinates']==[[int(i),successor] for i in active]
    assert changes['prediction']['count']==changes['gates']['count']==0
    assert math.isclose(post['parameters']['bias'][successor]-pre['parameters']['bias'][successor],delta,abs_tol=5e-16)
    assert np.all(post['parameters']['weights'][active,successor]-pre['parameters']['weights'][active,successor]==delta)
    # Minimal instrumentation must agree exactly with the original cached native kernel.
    for count,instrumented,metrics in ((1,first,m1),(2,twice,m2)):
        untouched=[a.copy() for a in base]
        original_metrics,original_rev=invoke(untouched,count,True,_kernel(Q,R,'native'))
        assert np.array_equal(metrics,original_metrics)
        assert np.array_equal(original_rev,r1 if count==1 else r2)
        assert all(np.array_equal(a,b) for a,b in zip(untouched,instrumented))
    # Test 2: frozen probes only; A was the only frame exposed during training.
    probe_rows=[]
    for name,target in [('A','A'),('B','A'),('C','C')]:
        item=dict(input=name,raw_hex={'A':A,'B':B,'C':C}[name].hex(),expected_outcome=target,
                  shared_active_receptors_with_A=int(np.sum(features[name][1:]*features['A'][1:])))
        for label,state in [('before',initial_snapshot(base)),('after_empirical',after),('after_structural',post['parameters'])]:
            logits=np.empty(Q);probs=np.empty(Q)
            lib.audit_probe(ct.c_void_p(state['bias'].ctypes.data),ct.c_void_p(state['weights'].ctypes.data),
                            ct.c_void_p(features[name].ctypes.data),Q,R,ct.c_void_p(logits.ctypes.data),ct.c_void_p(probs.ctypes.data))
            item[label]=dict(logits=logits.copy(),embedding=probs.copy(),
                desired_mapping_probability=float(probs[successor if name!='C' else entry]),
                conditional_expected_pattern_probability=probability(state['prediction'][source],features[target][1:]),
                expected_pattern_nll=loss(state['prediction'][source],features[target][1:]),
                informative_32bit_nll=loss(state['prediction'][source,:32],features[target][1:33]),
                inactive_224bit_nll=loss(state['prediction'][source,32:],features[target][33:]))
        # Verify the native frozen forward pass at each stage, including the structural-only snapshot.
        for label,state in [('before',initial_snapshot(base)),('after_empirical',after),('after_structural',post['parameters'])]:
            frozen=[a.copy() for a in base]
            for i,key in enumerate(('bias','weights','gates','prediction')):frozen[i][0]=state[key]
            frozen[6][:]=features[name]
            frozen[6][(1+20)*41+(1+20)]=features[target]
            saved=[a.copy() for a in frozen];events.clear()
            mm,_=invoke(frozen,1,False,function)
            assert all(np.array_equal(a,b) for a,b in zip(saved,frozen))
            assert mm[0,0,9]==mm[0,0,10]==0 and mm[0,0,5]==1
            assert math.isclose(mm[0,0,7],item[label]['expected_pattern_nll'],rel_tol=1e-13)
            item[label]['frozen_native_nll']=float(mm[0,0,7])
        probe_rows.append(item)
    assert probe_rows[1]['after_structural']['desired_mapping_probability'] > probe_rows[1]['before']['desired_mapping_probability']
    assert probe_rows[2]['after_structural']['desired_mapping_probability'] < probe_rows[2]['before']['desired_mapping_probability']
    assert probe_rows[2]['after_empirical']['informative_32bit_nll'] > probe_rows[2]['before']['informative_32bit_nll']
    def serial(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
        if isinstance(value,list):return [serial(v) for v in value]
        return value
    output=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,SHARED)},
        instrumented_source_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        ids=list(ids),entry=entry,source=source,successor=successor,
        first_metrics=m1,second_metrics=m2,empirical_changes=differences(before,after),structural_changes=changes,
        sampled_bit_errors=int(np.count_nonzero(sampled!=features['A'][1:])),
        prediction_before=before['prediction'][source],prediction_after=after['prediction'][source],
        empirical_loss_before=loss(before['prediction'][source],features['A'][1:]),
        empirical_loss_after=loss(after['prediction'][source],features['A'][1:]),
        first_events=[dict(stage=e['stage'],episode=e['episode'],pair=e['pair'],revised=e['revised']) for e in events1],
        second_events=[dict(stage=e['stage'],episode=e['episode'],pair=e['pair'],revised=e['revised']) for e in events2],
        active_features=active,component_delta=delta,probes=probe_rows,
        edge_arrays_before=base[4],edge_arrays_after=twice[4],repertoire=animal.repertoire.to_data())
    Path('/tmp/error_propagation_results.json').write_text(json.dumps(serial(output),indent=2)+'\n')
    print(json.dumps(serial({k:v for k,v in output.items() if k not in ('prediction_before','prediction_after','empirical_changes','repertoire')}),indent=2))
    print('PASS: native-only learning, exact uninstrumented parity, frozen probes, and parameter locality assertions')
```
<!-- AUDIT_HARNESS_END -->
