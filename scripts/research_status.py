"""Cheap read-only liveness/lock/STOP preflight. Never launches or retries work."""
import fcntl,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from alpha0.expanded_v11_run import LOCKS,STOPS

def status():
    actors=[];uncertain=[];held=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0');exe=os.path.basename(os.fsdecode(args[0]))
        except FileNotFoundError:continue
        except (OSError,IndexError) as e:uncertain.append({'pid':p.name,'error':type(e).__name__});continue
        command=b' '.join(args).decode(errors='replace')
        if ('python' in exe or 'java' in exe) and any(x in command for x in ('alpha0','imitation','battlecode','finalize_v11')):
            try:cwd=os.readlink(p/'cwd')
            except FileNotFoundError:
                if not p.exists():continue
                cwd=None
            except OSError:cwd=None
            row={'pid':int(p.name),'cwd':cwd,'command':command[:1000]}
            if cwd is None:uncertain.append(row)
            elif Path(cwd)==ROOT or str(ROOT) in command:actors.append(row)
    for p in LOCKS:
        if not p.exists():continue
        with p.open('r') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:held.append(str(p.relative_to(ROOT)))
    stops=[str(p.relative_to(ROOT)) for p in STOPS if p.exists()]
    return {'actors':actors,'unknown_processes':uncertain,'held_locks':held,'stops':stops,
            'launch_blocked':bool(actors or uncertain or held or stops),
            'zdb_stop_preserved':(ROOT/'runs/zdb-allmaps-cp2/STOP').exists(),
            'handoff':'progress/research-continuation.json'}
if __name__=='__main__':print(json.dumps(status(),separators=(',',':')))
