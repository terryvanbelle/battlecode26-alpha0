from pathlib import Path
from scripts.run_target_recheck_v9r1 import R,CFG,paired_decision

def test_revision_is_isolated_and_replay_task_supplied():
    assert R.name=='target-recheck-v9r1' and CFG.name=='target-recheck-v9r1.json'
    source=Path('scripts/run_target_recheck_v9r1.py').read_text()
    assert "'worker_task_path':str(R/'replay-task.json')" in source
    assert not paired_decision([{'complete':True,'source_json':str(i//2),'fresh_seed_advantage':0} for i in range(32)])['review_for_expansion']
