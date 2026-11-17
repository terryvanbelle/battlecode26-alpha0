"""scripts/zdn_margins.py: the margin columns on hand-worked states, and the read hook that appends them to pre-margin
records (the Java parity check is tests/test_zdn_java.py::test_state_margins)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import zdn_data  # noqa: E402
import zdn_margins as zm  # noqa: E402


def state(r, w, h, g, c, b, surplus):
    s = np.zeros(zm.OLD_NS, np.int32)
    s[[zm.ROUND, zm.W, zm.H, zm.GLOBAL, zm.COST, zm.BABIES]] = r, w, h, g, c, b
    s[zm.SURPLUS] = surplus
    return s


def test_margins_by_hand():
    S = np.stack([
        state(100, 20, 20, 150, 30, 8, 200),     # floor 300 (8 babies), core 12 (area 400)
        state(1990, 30, 30, 3500, 10, 0, 0),     # last-rat floor 20, end-of-game cap 2*(2000-1990) = 20, core 18
        state(0, 21, 21, 125, 20, 10, 130),      # area 441 is still small: core 12
    ])
    assert zm.margins(S).tolist() == [
        [50, 50, 20, -50, -50, 50, 0, -3, -3, -3, -3, 0, 0],
        [50, 50, 50, 50, 50, -50, -3, -3, -3, -3, -3, 1, 1],
        [50, 50, 5, -50, -50, 10, 2, -3, -3, -3, -2, 0, 0],
    ]


def test_floor_switches_at_eight_babies():
    lo, hi = zm.margins(np.stack([state(0, 30, 30, 230, 30, 7, 0), state(0, 30, 30, 230, 30, 8, 0)]))[:, 4]
    assert (lo, hi) == (50, -50)    # 200 over cost: clears the 100 floor below 8 babies, not the 300 floor from 8


def test_read_hook_appends_once(monkeypatch):
    S = np.stack([state(5, 20, 20, 400, 10, 2, 50)] * 3)
    fake = lambda path, kinds=('baby', 'king'): {u: {'reg': {'S': S.copy()}, 'step': None} for u in kinds}
    monkeypatch.setattr(zdn_data, 'read', fake)
    zm.install()
    zm.install()                     # idempotent: the margins are appended once
    d = zdn_data.read('x', kinds=('king',))
    assert d['king']['reg']['S'].shape == (3, zm.OLD_NS + zm.N) and d['king']['reg']['S'].dtype == S.dtype
    assert np.array_equal(d['king']['reg']['S'][:, -zm.N:], zm.margins(S))
    assert np.array_equal(d['king']['reg']['S'][:, :zm.OLD_NS], S)


def test_refuses_states_that_already_have_margins():
    with pytest.raises(ValueError):
        zm.with_margins(np.zeros((2, zm.OLD_NS + zm.N), np.int32))
