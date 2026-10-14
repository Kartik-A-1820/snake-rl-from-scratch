# A non-RL reference agent: follow a fixed Hamiltonian cycle of the board. It visits every cell once and returns to its
# start, so a snake that follows it can never trap itself and always eats every food: it fills the board (a win) on every
# seed, in at most about (cells x cells) steps. It is the upper bound and a teaching contrast: planning with full knowledge
# of the rules wins every time, a learned policy has to discover something like it from reward alone.
# Works on even square boards of at least 6x6 (the standard construction needs an even side; the starting body must lie
# on the cycle, which needs side >= 6).
from .core import DIRS

_CACHE = {}


def cycle(size):
    """The cell sequence [(x, y), ...] of a Hamiltonian cycle, oriented so that the game's starting snake (head at the centre
    moving right, body behind it) moves along it."""
    if size in _CACHE:
        return _CACHE[size]
    if size % 2 or size < 6:
        raise ValueError("the Hamiltonian-cycle agent needs an even board of at least 6x6")
    S = size
    cyc = [(x, 0) for x in range(S)]
    for r in range(1, S):
        xs = range(S - 1, 0, -1) if r % 2 else range(1, S)
        cyc += [(x, r) for x in xs]
    cyc += [(0, y) for y in range(S - 1, 0, -1)]
    assert len(cyc) == S * S and len(set(cyc)) == S * S
    c = S // 2
    k = cyc.index((c, c))
    if cyc[(k + 1) % len(cyc)] != (c + 1, c):   # the snake starts heading right: make the cycle run the same way
        cyc.reverse()
    pos = {cell: i for i, cell in enumerate(cyc)}
    _CACHE[size] = (cyc, pos)
    return _CACHE[size]


def ham_action(g):
    """Relative action (0 straight, 1 right, 2 left) that steps to the next cell of the cycle."""
    cyc, pos = cycle(g.w)
    hx, hy = g.body[0]
    nx, ny = cyc[(pos[(hx, hy)] + 1) % len(cyc)]
    d = DIRS.index((nx - hx, ny - hy))
    return {0: 0, 1: 1, 3: 2}[(d - g.dir) % 4]
