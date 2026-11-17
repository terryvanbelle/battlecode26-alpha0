"""scripts/helper_lib.sh: the remote job-state check of box_helper.sh and vm_helper.sh, run locally in a sandbox."""
import os, signal, subprocess, time
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / 'scripts/helper_lib.sh'


def state(d, tag):
    cmd = subprocess.run(['bash', '-c', f'source {LIB}; job_state {d} {tag}'], capture_output=True, text=True, check=True).stdout
    return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True).stdout.strip()   # as ssh would run it


def test_job_state(tmp_path):
    (tmp_path / 'runs/box').mkdir(parents=True)
    tag = f'rl-test-{os.getpid()}'
    log = tmp_path / f'runs/box/{tag}.log'
    assert state(tmp_path, tag) == 'DIED'      # nothing running: the check must not count its own shell
    p = subprocess.Popen(['bash', '-c', f'cd {tmp_path} && (echo 3 of 22 done; sleep 30) > runs/box/{tag}.log'], start_new_session=True)
    try:
        for _ in range(100):
            if log.exists() and log.read_text():
                break
            time.sleep(0.05)
        assert state(tmp_path, tag) == 'RUNNING 3 of 22 done'
    finally:
        os.killpg(p.pid, signal.SIGKILL); p.wait()
    assert state(tmp_path, tag) == 'DIED'      # the job died without an rc
    (tmp_path / f'runs/box/{tag}.rc').write_text('0\n')
    assert state(tmp_path, tag) == 'rc=0'
