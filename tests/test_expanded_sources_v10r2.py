from scripts import collect_expanded_sources_v10r2 as S

def test_real_backend_loading_and_storage_isolation():
    b=S.backend()
    assert b.C.RUN == S.R and b.C.PROGRESS == S.P
    assert callable(b.source_game)
    assert b.C.check_stop is S.check
    assert b.source_spec is S.source_spec

def test_all_inherited_sources_validate():
    r=S.report(S.identity())
    assert r['completed_sources']==38
    assert len(S.jobs(S.read(S.CFG)))==64
