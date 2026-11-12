"""Bounded independent reconstruction of lossless audit snapshots."""
import hashlib,json,zlib
from pathlib import Path
from .trace import hash_value

def read_snapshot(root,sha):
    hash_value(sha);root=Path(root)
    index=root/(sha+'.index.json')
    if index.stat().st_size>1024*1024:raise ValueError('audit index cap')
    obj=json.loads(index.read_text())
    if set(obj)!={'schema','sha256','length','chunk_size','chunks'} or obj['schema']!='audit-chunks-v1' or obj['sha256']!=sha or obj['chunk_size']!=4096:raise ValueError('invalid audit index')
    n=obj['length']
    if type(n) is not int or not 0<n<=16*1024*1024 or not isinstance(obj['chunks'],list) or len(obj['chunks'])!=(n+4095)//4096:raise ValueError('invalid audit length')
    data=bytearray()
    for i,key in enumerate(obj['chunks']):
        hash_value(key);p=root/'chunks'/(key+'.zlib')
        if p.stat().st_size>8192:raise ValueError('compressed chunk cap')
        z=zlib.decompressobj();chunk=z.decompress(p.read_bytes(),4097)
        expected=min(4096,n-i*4096)
        if len(chunk)!=expected or not z.eof or z.unused_data or z.unconsumed_tail or hashlib.sha256(chunk).hexdigest()!=key:raise ValueError('invalid audit chunk')
        data.extend(chunk)
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('audit reconstruction hash differs')
    return bytes(data)
