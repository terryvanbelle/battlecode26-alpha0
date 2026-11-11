"""Quiescent V11 recovery export, chunked publication and isolated byte restore."""
import fcntl,hashlib,json,tarfile,tempfile,sys
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0.expanded_v11_run import C,R,P,LOCKS,identity,read

def chunks(path,limit=48*1024*1024):
    if path.stat().st_size<=limit:return [{'path':str(path.relative_to(C.ROOT)),'sha256':C.sha256(path),'bytes':path.stat().st_size}]
    result=[]
    with path.open('rb') as stream:
        i=0
        while data:=stream.read(limit):
            out=path.with_name(path.name+f'.part{i:03d}');C.immutable_bytes(out,data)
            result.append({'path':str(out.relative_to(C.ROOT)),'sha256':C.sha256(out),'bytes':len(data)});i+=1
    return result

def export():
    with ExitStack() as stack:
        for path in LOCKS:
            h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected=read(P/'identity.json')
        if identity()!=expected:raise ValueError('runtime changed')
        dependencies=[read(C.ROOT/p/'recovery.json') for p in ('progress/general-roots-v11','progress/validation-v11','progress/terminal-target-head-v8')]
        for dep in dependencies:
            if dep['partial'] or C.sha256(C.ROOT/dep['archive'])!=dep['sha256']:raise ValueError('unverified dependency')
        files=[p for directory in (R,P) for p in directory.rglob('*') if p.is_file() and p.suffix not in ('.lock','.tmp') and p.name!='recovery.json']
        files += [C.ROOT/p for p in expected['sources']]+[Path(__file__).resolve()]
        files=sorted(set(files));hashes={str(p.relative_to(C.ROOT)):C.sha256(p) for p in files}
        directory=C.ROOT/'artifacts/expanded-terminal-v11';directory.mkdir(parents=True,exist_ok=True)
        archive=directory/('recovery-'+C.now().replace(':','').replace('-','').replace('+','_')+'.tar.gz')
        with tarfile.open(archive,'x:gz') as tar:
            for p in files:tar.add(p,arcname=str(p.relative_to(C.ROOT)),recursive=False)
        assert hashes=={str(p.relative_to(C.ROOT)):C.sha256(p) for p in files}
        archive_hash=C.sha256(archive);parts=chunks(archive)
        with tempfile.TemporaryDirectory(prefix='v11-restore-') as tmp:
            temp=Path(tmp);assembled=temp/'assembled.tar.gz'
            with assembled.open('wb') as dst:
                for part in parts:
                    path=C.ROOT/part['path'];assert C.sha256(path)==part['sha256']
                    with path.open('rb') as src:
                        while block:=src.read(1024*1024):dst.write(block)
            assert C.sha256(assembled)==archive_hash
            with tarfile.open(assembled) as tar:
                assert {m.name for m in tar.getmembers()}==set(hashes)
                for member in tar:
                    assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                    data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==hashes[member.name]
                    out=temp/'state'/member.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
            assert all(C.sha256(temp/'state'/name)==digest for name,digest in hashes.items())
        manifest={'archive_sha256':archive_hash,'parts':parts,'files':hashes,'dependencies':dependencies,'isolated_restore_verified':True,'identity_sha256':C.fingerprint(expected),'complete':read(R/'status.json')['phase']=='complete_review_required'}
        side=archive.with_name(archive.name+'.json');C.immutable_json(side,manifest)
        C.atomic_json(P/'recovery.json',{'manifest':str(side.relative_to(C.ROOT)),'archive_sha256':archive_hash,'parts':parts,'dependencies':dependencies,'partial':not manifest['complete']})
        print(json.dumps({'manifest':str(side.relative_to(C.ROOT)),'parts':parts,'files_verified':len(files)}))
if __name__=='__main__':export()
