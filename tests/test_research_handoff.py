import json
import pytest
from scripts.continue_after_baseline import inspect_baseline

def test_missing_baseline_waits(tmp_path):
    assert inspect_baseline(tmp_path,{}, {}, {}) is None

def test_exhausted_incomplete_baseline_requires_review(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps({'complete':False,'unresolved_errors':4}))
    with pytest.raises(ValueError,match='review required'):inspect_baseline(tmp_path,{}, {}, {})

def test_complete_flag_alone_cannot_launch(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps({'complete':True,'unresolved_errors':0}))
    (tmp_path/'evaluations.jsonl').write_text('')
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/improvement-v4.json').read_text())
    panel=json.loads((root/cfg['manifest']).read_text())
    with pytest.raises(ValueError,match='incomplete'):inspect_baseline(tmp_path,cfg,panel,{})
