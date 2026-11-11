"""Read-only fixed-transcript eligibility probe; no training or new terminal outcomes."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import collect_expanded_sources_v10r2 as S
from alpha0.env import EngineEnv
from alpha0.terminal_v8_collect import actor_info,state_hash
R=S.C.ROOT/'runs/v10-eligibility-probe';P=S.C.ROOT/'progress/v10-eligibility-probe'
LOCKS=(*S.LOCKS,R/'experiment.lock');STOPS=(*S.STOPS,R/'STOP')
def check():
    if any(p.exists() for p in STOPS):raise RuntimeError('STOP')
def worker(name):
    import ctypes
    ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    expected=S.read(S.P/'identity.json');job=next(j for j in S.jobs(expected['config']) if j['source_id']==name)
    row,path=S.source_row(job,expected);found={};env=EngineEnv(action_version='macro_v2',response_timeout=30)
    try:
        state=env.reset(job['map'],job['seed'])
        for index in range(len(row['actions'])+1):
            check()
            if state.terminal:break
            actor=actor_info(state);key=f"{actor['type']}:{state.player}"
            if key not in found and len(state.legal_actions)>=2:
                found[key]={'decision_index':index,'round':state.raw['round'],'legal_actions':state.legal_actions,'state_sha256':state_hash(state.raw)}
            if len(found)==4:break
            if index==len(row['actions']):raise RuntimeError('transcript exhausted before terminal')
            action=row['actions'][index]
            if action not in state.legal_actions:raise RuntimeError('recorded action illegal during replay')
            state=env.transition(state,action);S.C.release(env,state)
        result={'source_id':name,'map':job['map'],'source_sha256':S.C.sha256(path),'eligible_roles':found,'complete':True,'replayed_actions':index,'new_terminal_games':0,'sgd_updates':0}
        S.C.immutable_json(P/'sources'/f'{name}.json',result)
    finally:env.close()
def main():
    a=argparse.ArgumentParser();a.add_argument('--worker');args=a.parse_args()
    if args.worker:return worker(args.worker)
    for p in (R,P,P/'sources',R/'logs'):p.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for p in LOCKS:
            h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check();assert not S.active_actors()
        expected=S.read(S.P/'identity.json');assert S.identity()==expected
        identity={'source_identity':S.C.fingerprint(expected),'script_sha256':S.C.sha256(Path(__file__)),'protocol_sha256':S.C.sha256(S.C.ROOT/'docs/v10-eligibility-probe.md')}
        S.C.immutable_json(P/'identity.json',identity)
        deadline=time.monotonic()+10800
        def status(phase,**kw):S.C.atomic_json(R/'status.json',dict(phase=phase,updated_at=S.C.now(),pid=os.getpid(),**kw))
        try:
            for j in S.jobs(expected['config']):
                check();name=j['source_id'];out=P/'sources'/f'{name}.json'
                if out.exists():continue
                log=R/'logs'/f'{name}.log'
                if log.exists():raise RuntimeError('prior interrupted/failed probe requires review')
                remaining=deadline-time.monotonic()
                if remaining<=0:status('deadline_review_required');return
                status('replaying',source=name)
                with log.open('x') as f:
                    child=subprocess.Popen([sys.executable,'-u',__file__,'--worker',name],stdout=f,stderr=f,start_new_session=True)
                    try:
                        until=time.monotonic()+min(remaining,1800)
                        while child.poll() is None:
                            check()
                            if time.monotonic()>=until:raise RuntimeError('probe watchdog')
                            time.sleep(.5)
                        if child.returncode:raise RuntimeError('probe worker failure')
                    finally:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait()
            rows=[S.read(p) for p in sorted((P/'sources').glob('*.json'))];assert len(rows)==64
            S.C.immutable_json(P/'summary.json',{'complete':True,'sources':64,'all_four_roles':sum(len(r['eligible_roles'])==4 for r in rows),'new_terminal_games':0,'sgd_updates':0})
            status('complete_review_required')
        except BaseException as e:status('review_required',reason=str(e));raise
if __name__=='__main__':main()
