from copy import deepcopy
import pytest
from scripts import recover_expanded_v11_chunks as X


def test_exact_remaining_schedule_preserves_completed_cells():
    review=X.S.read(X.REVIEW);chunks=review['chunks']
    identity=X.S.read(X.S.P/'identity.json')
    roots=X.S.read(X.S.C.ROOT/identity['input_plans']['training']['path'])['roots']
    expected={(r['root_id'],a,s) for r in roots for a in r['legal_actions'] for s in identity['config']['selection_seeds']}
    for name in review['preserved_files']:
        if '/cells/' not in name or not name.endswith('.json') or name.endswith('-start.json'):continue
        row=X.S.read(X.S.C.ROOT/name)
        if row['status']=='ok':expected.discard((row['root_id'],row['action'],row['seed']))
    assert {(c['root']['root_id'],a,s) for c in chunks for a,s in c['cells']}==expected
    cells=[(c['root']['root_id'],a,s) for c in chunks for a,s in c['cells']]
    assert len(cells)==len(set(cells))==162
    assert all(0<len(c['cells'])<=8 for c in chunks)
    assert 1800+8*180<3600
    assert len(review['interrupted_cells'])==2


def test_changed_runner_or_identity_refused():
    review=deepcopy(X.S.read(X.REVIEW));review['runner_sha256']='0'*64
    with pytest.raises(ValueError,match='runner changed'):X.verify(review)
    review['identity_sha256']='0'*64
    with pytest.raises(ValueError,match='identity changed'):X.verify(review)
