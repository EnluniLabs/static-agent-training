"""Binary toggle puzzles with an animal-independent, nine-signal motor interface."""
from functools import lru_cache
import random

from .environment import NumberLine, Observation, Output
from .experimental.signals import Action, NoOp
from .experimental.senses import StimulusStep


def grid_size(environment='lights-out'):
    if environment in ('lights-out', 'switchboard'): return 3
    if environment in ('switches-2x2','lights-out-2x2'): return 2
    raise ValueError('unknown toggle puzzle')


def board_bits(board, environment='lights-out'):
    cells=grid_size(environment)**2
    if type(board) is not int or not 0 <= board < 1<<cells:
        raise ValueError(f'board must be an integer mask from 0 to {(1<<cells)-1}')
    return [(board >> cell) & 1 for cell in range(cells)]


MASKS = tuple(sum(1 << (rr*3+cc) for rr,cc in ((r,c),(r-1,c),(r+1,c),(r,c-1),(r,c+1))
                  if 0 <= rr < 3 and 0 <= cc < 3) for r in range(3) for c in range(3))


MASKS_2X2 = (7, 11, 13, 14)

def action_cells(environment='lights-out'):
    grid_size(environment)
    # Two equivalent signals per cell on either 2×2 board, plus one idle signal.
    return list(range(9)) if environment in ('lights-out', 'switchboard') else [0,0,1,1,2,2,3,3,None]


def press_mask(cell, environment='lights-out'):
    if cell is None:return 0
    grid_size(environment)
    if environment=='lights-out':return MASKS[cell]
    if environment=='lights-out-2x2':return MASKS_2X2[cell]
    return 1<<cell


@lru_cache(maxsize=4)
def solutions(environment='lights-out'):
    """Enumerate actual scrambles without exposing solutions to the animal."""
    cells=grid_size(environment)**2
    result={}
    for presses in range(1<<cells):
        board=0
        for cell in range(cells):
            if presses & (1<<cell):board ^= press_mask(cell,environment)
        path=tuple(cell for cell in range(cells) if presses & (1<<cell))
        if board not in result or len(path)<len(result[board]):result[board]=path
    return result


def sample_tasks(seed, count, allowed=None, environment='lights-out'):
    if type(count) is not int or count<1:raise ValueError('task count must be positive')
    boards=sorted(set(solutions(environment))-{0} if allowed is None else set(allowed)-{0})
    if not boards or any(board not in solutions(environment) for board in boards):
        raise ValueError('tasks require nonzero solvable boards')
    rng=random.Random(seed)
    return [(rng.choice(boards),0) for _ in range(count)]


def stimulus(board, environment='lights-out'):
    return bytes(board_bits(board,environment))


@lru_cache(maxsize=None)
def random_success(board, moves=12, environment='lights-out'):
    board_bits(board,environment)
    if board==0:return 1.0
    if moves==0:return 0.0
    return sum(random_success(board^press_mask(cell,environment),moves-1,environment)
               for cell in action_cells(environment))/9


class LightsOutWorld(NumberLine):
    """Categorical adapter sharing output/feedback timing with NumberLine."""
    def __init__(self, *, encoding_seed=0, goal_conditioned=True, action_limit=12,
                 internal_limit=16, boundary_mode='clamp', environment='lights-out', **unused):
        self.environment=environment;self.grid_size=grid_size(environment)
        states=1<<(self.grid_size**2)
        super().__init__(encoding_seed=encoding_seed,goal_conditioned=goal_conditioned,
            action_limit=action_limit,internal_limit=internal_limit,boundary_mode=boundary_mode,lower=0,upper=states-1)
        self.default_goal = self._goal = 0
        order = list(range(states));random.Random(encoding_seed).shuffle(order)
        prefix={'lights-out':'board','switches-2x2':'switches','lights-out-2x2':'board2','switchboard':'panel'}[environment]
        self._signals = {board:f'obs_{prefix}_{index:03d}' for index,board in enumerate(order)}
        self._expects = {'expect_'+token[len('obs_'):]:board for board,token in self._signals.items()}
        self.cells = action_cells(environment);random.Random(f'moves:{encoding_seed}').shuffle(self.cells)
        self._moves = {f'move_{i:02d}':press_mask(cell,environment) for i,cell in enumerate(self.cells)}
        self.observation_tokens = tuple(Observation(t) for t in self._signals.values())
        if goal_conditioned:self.observation_tokens += (Observation('goal_+0'),)
        self.output_tokens = tuple(Output(t) for t in (*self._moves,*self._expects))

    def reset(self, *, start=None, goal=None):
        if goal not in (None,0): raise ValueError('Lights Out always targets the all-off board (0)')
        if start is None: start=self._rng.choice(sorted(set(solutions(self.environment))-{0}))
        if type(start) is not int or start not in solutions(self.environment) or start==0:
            raise ValueError('choose a nonzero solvable Lights Out board')
        return super().reset(start=start,goal=0)

    def _advance(self, mask):
        self._position ^= mask
        return True

    def diagnostic_state(self):
        state = super().diagnostic_state()
        if state['position'] is not None:state['board']=board_bits(state['position'],self.environment)
        state.update(environment=self.environment,grid_size=self.grid_size,button_cells=dict(zip(self._moves,self.cells)))
        return state


class StimulusLightsOut:
    def __init__(self, *, encoding_seed=0, action_limit=12, goal_conditioned=True,
                 boundary_mode='clamp', environment='lights-out', **unused):
        self.environment=environment;self.grid_size=grid_size(environment)
        self.cells=action_cells(environment);random.Random(f'moves:{encoding_seed}').shuffle(self.cells)
        self.moves=[press_mask(cell,environment) for cell in self.cells]
        self.action_limit=action_limit
        if type(action_limit) is not int or action_limit<1:raise ValueError('action limit must be positive')
        self.done=True

    @property
    def board(self):return board_bits(self.position,self.environment)

    def stimulus(self):return stimulus(self.position,self.environment)

    def render_retina(self, senses):
        """Fixed physical lamp geometry, independent of sensory partitions.

        Cells have an outline and lit interiors, with dark right/bottom gutters.
        Geometry is identical across states and puzzles of the same board size.
        """
        width, height, size = senses.width, senses.height, self.grid_size
        pixels = bytearray(width * height // 8)
        for y in range(height):
            row = min(size - 1, ((y + 1) * size - 1) // height)
            top, bottom = row * height // size, (row + 1) * height // size
            for x in range(width):
                col = min(size - 1, ((x + 1) * size - 1) // width)
                left, right = col * width // size, (col + 1) * width // size
                border = x == left or y == top
                interior = x < right - 1 and y < bottom - 1
                lit = self.position & (1 << (row * size + col))
                if (lit if right - left < 3 or bottom - top < 3 else interior and (border or lit)):
                    i = y * width + x
                    pixels[i // 8] |= 1 << (7 - i % 8)
        return bytes(pixels)

    def reset(self,task):
        start,goal=task
        if type(start) is not int or start not in solutions(self.environment) or start==0 or goal!=0:
            raise ValueError('Lights Out tasks require a nonzero solvable board and goal 0')
        self.position,self.goal,self.actions,self.done=start,0,0,False
        return StimulusStep(self.stimulus())

    def step(self,action):
        if self.done:raise RuntimeError('reset required')
        if not isinstance(action,NoOp) and (not isinstance(action,Action) or type(action.index) is not int or not 0<=action.index<9):
            raise ValueError('unknown button')
        self.position ^= 0 if isinstance(action,NoOp) else self.moves[action.index];self.actions+=1
        success=self.position==0;self.done=success or self.actions>=self.action_limit
        return StimulusStep(self.stimulus(),self.done,success,'target' if success else 'action_limit' if self.done else None)
