import hashlib,json,zlib
import pytest
from alpha0.imitation.audit_store import read_snapshot

def fixture(tmp_path):
    data=b'abcd'*2500;sha=hashlib.sha256(data).hexdigest();(tmp_path/'chunks').mkdir();keys=[]
    for i in range(0,len(data),4096):
        c=data[i:i+4096];key=hashlib.sha256(c).hexdigest();keys.append(key);(tmp_path/'chunks'/(key+'.zlib')).write_bytes(zlib.compress(c))
    obj=dict(schema='audit-chunks-v1',sha256=sha,length=len(data),chunk_size=4096,chunks=keys)
    p=tmp_path/(sha+'.index.json');p.write_text(json.dumps(obj));return data,sha,p,obj

def test_exact_reconstruction(tmp_path):
    data,sha,_,_=fixture(tmp_path);assert read_snapshot(tmp_path,sha)==data

@pytest.mark.parametrize('mutation',['length','order','truncated','extra_stream','bomb','missing','hash'])
def test_reject_corruption(tmp_path,mutation):
    data,sha,p,obj=fixture(tmp_path);chunk=tmp_path/'chunks'/(obj['chunks'][0]+'.zlib')
    if mutation=='length':obj['length']=True
    if mutation=='order':obj['chunks'].reverse()
    if mutation=='truncated':chunk.write_bytes(chunk.read_bytes()[:-2])
    if mutation=='extra_stream':chunk.write_bytes(chunk.read_bytes()+zlib.compress(b'x'))
    if mutation=='bomb':chunk.write_bytes(zlib.compress(b'x'*1000000))
    if mutation=='missing':chunk.unlink()
    if mutation=='hash':chunk.write_bytes(zlib.compress(b'x'*4096))
    p.write_text(json.dumps(obj))
    with pytest.raises((ValueError,OSError,zlib.error)):read_snapshot(tmp_path,sha)
