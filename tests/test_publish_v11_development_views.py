import json
from pathlib import Path
from scripts import publish_v11_development_views as V


def test_no_benchmark_is_not_zero_strength(tmp_path):
    out=tmp_path/'out'
    assert V.publish(tmp_path/'missing',out)['status']=='benchmark_not_started'
    assert not out.exists()


def test_saved_full_panel_conversion(tmp_path,monkeypatch):
    # Read-only regression fixture; never rerun these completed games.
    original=V.ROOT/'progress/terminal-target-head-v8'
    fixture=tmp_path/'fixture';fixture.mkdir()
    for name in ('identity.json','benchmarks'):
        (fixture/name).symlink_to(original/name)
    (tmp_path/'configs').symlink_to(V.ROOT/'configs')
    actions=json.loads((original/'validation-actions.json').read_text())
    actions['checkpoints']={a:{'sha256':sha} for a,sha in actions.pop('checkpoint_sha256_by_arm').items()}
    (fixture/'validation-actions.json').write_text(json.dumps(actions))
    monkeypatch.setattr(V,'ROOT',tmp_path)
    result=V.publish(fixture,tmp_path)
    assert {arm:(r['valid'],r['wins'],r['losses']) for arm,r in result.items()} == {
        'cp2':(256,82,174),'terminal_head':(256,81,175),'puct_head':(256,79,177)}
    for arm in result:
        source=json.loads((tmp_path/f'v11-{arm}'/'source.json').read_text())
        assert source['derived_dashboard_view'] and not source['automatic_promotion']
        assert len((tmp_path/f'v11-{arm}'/'evaluations.jsonl').read_text().splitlines())==256
    import pytest
    actions['checkpoints']['terminal_head']['sha256']='0'*64
    (fixture/'validation-actions.json').write_text(json.dumps(actions))
    with pytest.raises(ValueError,match='changed candidate'):
        V.publish(fixture,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
