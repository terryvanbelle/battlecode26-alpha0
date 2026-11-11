"""Quiescent V10 source export with byte verification and isolated own-state restore."""
import fcntl,hashlib,json,tarfile,tempfile,sys
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_general_roots_v11 import C,R,P,identity,report,LOCKS

def export():
    with ExitStack() as stack:
        for p in LOCKS:
            h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected=json.loads((P/'identity.json').read_text());assert identity()==expected
        measured=report(expected)
        dependency=json.loads((C.PROGRESS/'recovery.json').read_text())
        assert not dependency['partial'] and C.sha256(C.ROOT/dependency['archive'])==dependency['sha256']
        files=[]
        for d in (R,P,C.ROOT/'runs/expanded-corpus-v10',C.ROOT/'progress/expanded-corpus-v10',C.ROOT/'runs/expanded-corpus-v10r2',C.ROOT/'progress/expanded-corpus-v10r2',C.ROOT/'runs/expanded-corpus-v10r1',C.ROOT/'progress/expanded-corpus-v10r1'):
            files += [p for p in d.rglob('*') if p.is_file() and p.suffix not in ('.lock','.tmp') and p.name!='recovery.json']
        files += [C.ROOT/name for name in expected['files']]
        files.append(Path(__file__).resolve())
        files=sorted(set(files));hashes={str(p.relative_to(C.ROOT)):C.sha256(p) for p in files}
        manifest={'schema':'general-roots-v11-recovery-v1','files':hashes,'dependency_v8':dependency,'complete':measured['complete'],'completed_sources':measured['completed_sources'],'identity':C.fingerprint(expected)}
        directory=C.ROOT/'artifacts/general-roots-v11';directory.mkdir(parents=True,exist_ok=True)
        stamp=C.now().replace(':','').replace('-','').replace('+','_')
        archive=directory/f'recovery-{stamp}.tar.gz'
        with tarfile.open(archive,'x:gz') as tar:
            for p in files:tar.add(p,arcname=str(p.relative_to(C.ROOT)),recursive=False)
        assert hashes=={str(p.relative_to(C.ROOT)):C.sha256(p) for p in files}
        with tempfile.TemporaryDirectory(prefix='v10-source-restore-') as tmp:
            restored=Path(tmp)
            with tarfile.open(archive,'r:gz') as tar:
                members=tar.getmembers();assert {m.name for m in members}==set(hashes)
                for member in members:
                    assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                    data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==hashes[member.name]
                    out=restored/member.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
                assert all(C.sha256(restored/name)==digest for name,digest in hashes.items())
        manifest.update(archive=str(archive.relative_to(C.ROOT)),sha256=C.sha256(archive),isolated_own_state_restore_verified=True,files_verified=len(files))
        side=archive.with_name(archive.name+'.json');C.immutable_json(side,manifest)
        C.atomic_json(P/'recovery.json',{'archive':manifest['archive'],'sha256':manifest['sha256'],'manifest':str(side.relative_to(C.ROOT)),'v8_dependency':dependency,'partial':not measured['complete']})
        print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
if __name__=='__main__':export()
