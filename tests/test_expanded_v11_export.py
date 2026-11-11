from pathlib import Path
from scripts import export_expanded_v11 as E

def test_chunked_reassembly_and_hashes(tmp_path,monkeypatch):
    monkeypatch.setattr(E.C,'ROOT',tmp_path)
    p=tmp_path/'archive';data=b'0123456789'*13;p.write_bytes(data)
    parts=E.chunks(p,limit=37)
    assert len(parts)==4 and b''.join((tmp_path/x['path']).read_bytes() for x in parts)==data
    assert all(E.C.sha256(tmp_path/x['path'])==x['sha256'] for x in parts)
