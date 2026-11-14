"""pytest wrapper: scripts/reinit_new_points.py --self-test (logits identical, hidden activations O(1))."""
import subprocess, sys, pathlib

def test_reinit_self_test():
    s = pathlib.Path(__file__).resolve().parent.parent / 'scripts' / 'reinit_new_points.py'
    r = subprocess.run([sys.executable, str(s), '--self-test'], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
