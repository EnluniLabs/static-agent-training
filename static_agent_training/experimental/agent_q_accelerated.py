"""Native Agent Q lifetime evaluation. No Python callbacks in the lifetime loop."""
import ctypes as c
from functools import lru_cache
import hashlib
from pathlib import Path
import random
import subprocess
import tempfile

from .agent_q import Event, MODES, event_data
from .agent_w_environment import NumberLine
from .agent_z import Chain


class Trace(c.Structure):
    _fields_ = [(name,c.c_int) for name in
                ('round','region','source','successor','action','sample','observation','position','actions')] + [
                    ('empirical_error',c.c_double),('structural_error',c.c_double)]


@lru_cache(maxsize=1)
def _kernel():
    source=Path(__file__).resolve().parents[1]/'native'/'evaluate_agent_q.cpp'
    digest=hashlib.sha256(b''.join(source.with_name(f'evaluate_agent_{m}.cpp').read_bytes()
                                   for m in ('q','w','z'))).hexdigest()[:24]
    cache=Path.home()/'.cache/static-agent-training'
    cache.mkdir(parents=True,exist_ok=True)
    library=cache/f'agent-q-{digest}.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            target=Path(temp)/library.name
            try:
                subprocess.run(['g++','-std=c++17','-O3','-ffp-contract=off','-shared','-fPIC',
                                str(source),'-o',str(target)],check=True,capture_output=True,text=True)
            except (OSError,subprocess.CalledProcessError) as e:
                raise ValueError(f'Agent Q native build failed: {getattr(e,"stderr",e)}') from e
            target.replace(library)
    call=c.CDLL(str(library)).q_evaluate
    call.argtypes=([c.c_int]*4+[c.POINTER(c.c_int)]*3+[c.POINTER(c.c_uint32)]+
                   [c.POINTER(c.c_double)]*2+[c.c_double]*3+[c.POINTER(c.c_int),c.POINTER(Trace)])
    call.restype=c.c_int
    return call


class NativeQ:
    def evaluate(self,agent,tasks,*,seed,learning='both',round_limit=64,trace=False,return_weights=False):
        tasks=tuple(tasks)
        if any(len(t)!=2 or any(type(x) is not int or not -20<=x<=20 for x in t) for t in tasks):
            raise ValueError('Agent Q requires tasks within the original number line')
        if type(round_limit) is not int or not 1<=round_limit<=2**31-1:
            raise ValueError('invalid Agent Q round limit')
        n=len(agent.ids)
        if max(n*72,len(tasks)*12)>2**31-1:
            raise ValueError('Agent Q evaluation exceeds native index limits')
        edges=(c.c_int*n)(*(agent.index.get(agent.structure.successors.get(p),-1) for p in agent.ids))
        positions=(c.c_int*(len(tasks)*2))(*(x for t in tasks for x in t))
        moves=(c.c_int*9)(*NumberLine()._native_moves())
        rng=(c.c_uint32*625)(*random.Random(seed).getstate()[1])
        embedding=(c.c_double*(n*9))(*(v for row in agent.embedding for v in row))
        expectations=(c.c_double*(n*72))(*(v for pair in agent.expectations for row in pair for v in row))
        stats=(c.c_int*4)()
        records=(Trace*(len(tasks)*min(12,round_limit*agent.regions)))() if trace else None
        if _kernel()(n,agent.regions,len(tasks),round_limit,edges,positions,moves,rng,embedding,
                     expectations,agent.exploration,*MODES[learning],stats,records):
            raise MemoryError('native Agent Q could not allocate evaluation buffers')
        successes,rounds,actions,correct=stats
        result=dict(successes=successes,episodes=len(tasks),rounds=rounds,actions=actions,tested=actions,correct=correct)
        if trace:
            values=[[] for _ in range(rounds)]
            frames=[{} for _ in range(rounds)]
            for r in records[:actions]:
                event=Event(Chain(r.region,agent.ids[r.source]),agent.ids[r.successor],r.action,
                            r.sample,r.observation,r.empirical_error,r.structural_error)
                values[r.round].append(event_data(event,agent))
                frames[r.round]=dict(position=r.position,actions=r.actions)
            result.update(trace=values,world_frames=frames)
        if return_weights:
            result.update(embedding=[embedding[i:i+9] for i in range(0,len(embedding),9)],
                          expectations=[[expectations[i+j:i+j+8] for j in range(0,72,8)]
                                        for i in range(0,len(expectations),72)])
        return result
