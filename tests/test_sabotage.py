"""scripts/sabotage.py self-test: +DELTA lands on exactly one output bias (logit shifted by DELTA on that option only),
other points, hidden layers and last-layer weights unchanged, norm.npz copied, the probability report sees the shift.

    python3 tests/test_sabotage.py      (also runs under pytest)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import sabotage  # noqa: E402


def test_sabotage():
    sabotage.self_test()


if __name__ == '__main__':
    test_sabotage()
