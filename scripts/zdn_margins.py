"""Spawn-economy margins (docs/zdn.md, "State margins"): the last 13 columns of zdn's state (ZO.state), as a function of
the columns before them. Records written before the margins existed (runs/teacher-zdn1, state width 158) can be
trained on as if they carried them:

    python scripts/zdn_margins.py scripts/zdn_train.py --games 'runs/teacher-zdn1/games/*' --unit king --out ...

runs the script with zdn_data.read appending the margins to every record's state. The Java side computes the same
columns live (tests/test_zdn_java.py::test_state_margins checks the two agree)."""
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zdn_data  # noqa: E402

OLD_NS = 158          # state width before the margins
N = 13                # margin columns
ROUND, W, H, GLOBAL, COST, BABIES = 0, 5, 6, 9, 11, 12
SURPLUS = -12         # zdn's surplus estimate, 12th column from the end of the pre-margin state


def margins(S):
    """The 13 margin columns [n, 13] for pre-margin states S [n, k]: global cheese minus the next rat's cost against
    0, 20, 100, 300 and zdb's spawn floor, the surplus against cost + 100 (all clipped to +-50); the baby count
    against 8, 18, 40, 80 and the core count (clipped to +-3); global >= 1000 and >= 3000."""
    S = np.asarray(S).astype(np.int64)
    r, g, c, b, surplus = S[:, ROUND], S[:, GLOBAL], S[:, COST], S[:, BABIES], S[:, SURPLUS]
    m = g - c
    floor = np.where(c == 10, 20, np.where(b < 8, 100, 300))
    core = np.where(S[:, W] * S[:, H] <= 441, 12, 18)
    cols = [np.clip(m - t, -50, 50) for t in (0, 20, 100, 300)]
    cols += [np.clip(m - np.minimum(floor, 2 * (2000 - r)), -50, 50), np.clip(surplus - c - 100, -50, 50)]
    cols += [np.clip(b - t, -3, 3) for t in (8, 18, 40, 80)] + [np.clip(b - core, -3, 3)]
    cols += [(g >= 1000).astype(np.int64), (g >= 3000).astype(np.int64)]
    return np.stack(cols, 1)


def with_margins(S):
    """Pre-margin states [n, OLD_NS] with the margins appended [n, OLD_NS + N], in the records' dtype."""
    if S.shape[1] != OLD_NS:
        raise ValueError(f'state width {S.shape[1]}, expected the pre-margin width {OLD_NS}')
    return np.concatenate([S, margins(S).astype(S.dtype)], 1)


def install():
    """Make zdn_data.read append the margins to every record's state (pre-margin records only)."""
    read = zdn_data.read
    if getattr(read, 'with_margins', False):
        return

    def read_with_margins(path, kinds=('baby', 'king')):
        out = read(path, kinds)
        for d in out.values():
            if d is not None and d.get('reg') is not None:
                d['reg']['S'] = with_margins(d['reg']['S'])
        return out
    read_with_margins.with_margins = True
    zdn_data.read = read_with_margins


if __name__ == '__main__':
    install()
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
