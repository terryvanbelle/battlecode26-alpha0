"""Cross-language transport checks; these do not run teacher games."""
import json
import subprocess
from pathlib import Path
import pytest
from alpha0.imitation.trace import read_trace, compare_complete, digest

ROOT = Path(__file__).resolve().parents[1]

def test_java_journal_python_equivalence(tmp_path):
    classes = tmp_path / 'classes'
    classes.mkdir()
    jar = ROOT / 'vendor/engine/battlecode26-java-1.2.6.jar'
    sources = [ROOT / 'java/imitation/alpha0/imitation' / name for name in
               ('PrimitiveRecorder.java', 'RobotThreadRecorder.java', 'TraceJournal.java',
                'JournalObserver.java', 'PublicObservation.java', 'PauseReason.java')]
    sources.append(ROOT / 'tests/java/TraceJournalTest.java')
    sources.append(ROOT / 'tests/java/PauseReasonTest.java')
    subprocess.run([str(ROOT / 'vendor/jdk/bin/javac'), '-cp', str(jar), '-d', str(classes),
                    *map(str, sources)], check=True, timeout=30, capture_output=True)
    subprocess.run([str(ROOT / 'vendor/jdk/bin/java'), '-cp', f'{classes}:{jar}',
                    'TraceJournalTest', str(tmp_path)], check=True, timeout=30, capture_output=True)
    subprocess.run([str(ROOT / 'vendor/jdk/bin/java'), '-cp', f'{classes}:{jar}', 'PauseReasonTest'], check=True, timeout=30)
    rows = read_trace(tmp_path / 'trace.jsonl')
    seal = json.loads((tmp_path / 'seal.json').read_text())
    assert compare_complete(rows, rows, seal, seal)['training_approved'] is False
    assert rows[0]['payload']['return_boolean'] is False
    assert (tmp_path / 'unicode-hash.txt').read_text() == digest({'s': 'é😀\n', 'n': -1})
    assert not (tmp_path / 'bad-seal.json').exists()
    assert (tmp_path / 'capped.jsonl').read_bytes() == b''
    altered = json.loads(json.dumps(rows))
    altered[0]['payload']['bytecodes_after'] += 1
    with pytest.raises(ValueError, match='hash mismatch'):
        compare_complete(rows, altered, seal, seal)
