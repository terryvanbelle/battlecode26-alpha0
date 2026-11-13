"""Java Net vs PyTorch StudentNet parity on real records (needs build/student and a records file)."""
import os, subprocess, tempfile
from pathlib import Path
import numpy as np, pytest, torch
from alpha0.student.data import read_records, unpack_mask
from alpha0.student.spec import N_ACTIONS, GROUP_OF, N_GROUPS
from alpha0.student.train import StudentNet, export

ROOT = Path(__file__).resolve().parents[1]
REC = os.environ.get('STUDENT_RECORDS', str(ROOT / 'runs/smoke/zdb-mirror/records-A.bin.gz'))


@pytest.mark.skipif(not (ROOT / 'build/student/classes').exists() or not Path(REC).exists(), reason='needs student build and records')
def test_java_matches_torch():
    torch.manual_seed(1)
    net = StudentNet(64, 32, 16, 16)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(3)   # make logits spread so argmax is well defined
    k = 200
    with tempfile.TemporaryDirectory() as d:
        w = Path(d) / 'w.bin'
        export(net, w)
        cp = f"{ROOT/'build/student/classes'}:{ROOT/'build/java'}:{ROOT/'vendor/engine/battlecode26-java-1.2.6.jar'}"
        out = subprocess.run([str(ROOT / 'vendor/jdk/bin/java'), '-cp', cp, 'alpha0.student.NetCheck', str(w), REC, str(k)],
                             capture_output=True, text=True, check=True).stdout.split('\n')
    h, o, m = read_records(REC)
    legal = torch.from_numpy(unpack_mask(m[:k], N_ACTIONS))
    with torch.no_grad():
        logits, value, hid = net(torch.from_numpy(np.ascontiguousarray(o[:k])))
        plog = net.param_logits(hid, torch.from_numpy(h['label'][:k].astype(np.int64)))
    probs = torch.softmax(logits.masked_fill(~legal, -1e9), 1)
    gid = torch.tensor(GROUP_OF)
    mass = torch.zeros(k, N_GROUPS).index_add_(1, gid, probs)
    best_group = mass.argmax(1)
    within = logits.masked_fill(~legal | (gid[None, :] != best_group[:, None]), -1e9)
    pred = within.argmax(1).numpy()
    jpred = np.array([int(l.split()[0]) for l in out if l])
    jval = np.array([float(l.split()[1]) for l in out if l])
    assert (jpred == pred).mean() > 0.99
    assert np.abs(jval - value.numpy()).max() < 1e-3
    jp = np.array([[float(x) for x in l.split()[2:4]] for l in out if l])
    assert np.abs(jp - plog[:, [0, 31]].numpy()).max() < 1e-2
