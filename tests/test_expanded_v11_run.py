import sys
import pytest
from alpha0 import expanded_v11_run as R

def test_draft_launch_is_blocked(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'R',tmp_path/'run');monkeypatch.setattr(R,'P',tmp_path/'progress');monkeypatch.setattr(R,'LOCKS',());monkeypatch.setattr(R,'STOPS',())
    monkeypatch.setattr(R.V,'active_actors',lambda:[]);monkeypatch.setattr(R,'identity',lambda:{'config':{'launch_status':'draft'}})
    monkeypatch.setattr(sys,'argv',['expanded_v11_run'])
    with pytest.raises(ValueError,match='launch forbidden'):R.main()
    assert not (tmp_path/'progress/identity.json').exists()

def test_unclosed_job_cannot_silently_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'R',tmp_path/'run');monkeypatch.setattr(R,'P',tmp_path/'progress')
    for path in (R.R/'jobs',R.R/'logs',R.P/'jobs'):path.mkdir(parents=True)
    expected={'config':{}};payload={'root_id':'example'};key=R.C.fingerprint({'kind':'selection','payload':payload,'identity':R.C.fingerprint(expected)})
    (R.R/'logs'/f'{key}.log').write_text('interrupted')
    with pytest.raises(RuntimeError,match='unclosed job'):R.run_jobs('selection',[payload],expected,float('inf'))

def test_changed_completed_job_cannot_be_skipped(tmp_path,monkeypatch):
    monkeypatch.setattr(R,'R',tmp_path/'run');monkeypatch.setattr(R,'P',tmp_path/'progress')
    for path in (R.R/'jobs',R.R/'logs',R.P/'jobs'):path.mkdir(parents=True)
    expected={'config':{}};payload={'root_id':'example'};key=R.C.fingerprint({'kind':'selection','payload':payload,'identity':R.C.fingerprint(expected)})
    R.C.immutable_json(R.P/'jobs'/f'{key}.json',{'job_sha256':'wrong','status':'ok'})
    with pytest.raises(RuntimeError,match='saved job failed or changed'):R.run_jobs('selection',[payload],expected,float('inf'))

def test_completed_job_output_tampering_blocks_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(R.C,'ROOT',tmp_path);monkeypatch.setattr(R,'R',tmp_path/'run');monkeypatch.setattr(R,'P',tmp_path/'progress')
    for path in (R.R/'jobs',R.R/'logs',R.P/'jobs',R.P/'comparators'):path.mkdir(parents=True,exist_ok=True)
    expected={'config':{}};payload={'root_id':'example'};key=R.C.fingerprint({'kind':'comparator','payload':payload,'identity':R.C.fingerprint(expected)})
    job={'id':key,'kind':'comparator','payload':payload};path=R.R/'jobs'/f'{key}.json';R.C.immutable_json(path,job)
    output=R.P/'comparators/example.json';output.write_text('{}')
    R.C.immutable_json(R.P/'jobs'/f'{key}.json',{'job_sha256':R.C.sha256(path),'status':'ok','output_hashes':R.job_outputs(job)})
    output.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='outputs changed'):R.run_jobs('comparator',[payload],expected,float('inf'))
