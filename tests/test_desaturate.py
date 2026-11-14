"""scripts/desaturate.py self-test: last layer scaled by 1/tau equals sampling at temperature tau, other points and
hidden layers unchanged, norm.npz copied, saturation report sees lower saturation (numpy; no games).

    python3 tests/test_desaturate.py      (also runs under pytest)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import desaturate  # noqa: E402


def test_desaturate():
    desaturate.self_test()


if __name__ == '__main__':
    test_desaturate()
