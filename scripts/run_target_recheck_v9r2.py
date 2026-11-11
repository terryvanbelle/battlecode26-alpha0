"""Versioned, no-SGD target-reliability experiment using frozen V8 training roots."""
import argparse,ctypes,fcntl,json,os,signal,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alpha0 import terminal_v8_common as C
from alpha0.terminal_v8 import active_actors
from alpha0.terminal_v8_collect import RootWorker,choose_policy_action,terminal_proof,verify_terminal
R=C.ROOT/'runs/target-recheck-v9r2';P=C.ROOT/'progress/target-recheck-v9r2'
CFG=C.ROOT/'configs/target-recheck-v9r2.json'
def read(p):return json.loads(p.read_text())
def stopped():return any(p.exists() for p in (*C.STOPS,R/'STOP'))
def identity():
    return {'config':read(CFG),'source_identity':read(C.PROGRESS/'identity.json'),
        'source_runtime_current':C.identity(C.load_config()),
        'files':{str(p.relative_to(C.ROOT)):C.sha256(p) for p in (Path(__file__).resolve(),CFG,C.ROOT/'docs/target-recheck-v9r2.md',C.ROOT/'diagnostics/target-recheck-planning/action-inventory.json',C.PROGRESS/'roots-training.json',C.ROOT/'configs/target-recheck-v9r2-inherited.json')}}
def paths(root,action,seed):return P/'cells'/f'{root}-a{action}-s{seed}'
def valid_records(d):
    rows=[read(p) for p in sorted(d.glob('attempt-[0-9][0-9][0-9].json'))]
    if not rows:
        pin=read(C.ROOT/'configs/target-recheck-v9r2-inherited.json').get(d.name)
        if pin:
            p=C.ROOT/pin['path'];assert C.sha256(p)==pin['sha256']
            row=read(p);assert row['experiment_identity']==pin['experiment_identity']
            rows=[{**row,'decisions':len(row['actions']),'inherited_verified':True}]
    ok=[r for r in rows if r['status']=='ok']
    assert len(ok)<=1,'duplicate successful cell'
    for r in ok:verify_terminal(r,r['root_player'])
    return rows,ok

def worker(root_id,deadline,parent):
    import torch
    torch.set_num_threads(1)
    ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    if os.getppid()!=parent:raise RuntimeError('parent changed')
    def interrupt(*_):raise C.Interrupted('supervisor interrupt')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    frozen=read(P/'identity.json');assert identity()==frozen
    root=next(r for r in read(C.PROGRESS/'roots-training.json')['roots'] if r['root_id']==root_id)
    root={**root,'worker_task_path':str(R/'replay-task.json')}
    pair=next(r for r in read(P/'actions.json')['roots'] if r['root_id']==root_id)
    config=C.load_config();w=RootWorker(root,config,frozen['source_identity'],deadline)
    current=None
    try:
        for a in sorted({pair['selected_action'],pair['cp2_action']}):
            for seed in frozen['config']['seeds']:
                d=paths(root_id,a,seed);d.mkdir(parents=True,exist_ok=True)
                rows,ok=valid_records(d)
                if ok:continue
                if rows:raise RuntimeError('incomplete/failed cell requires explicit review before retry')
                if stopped() or time.monotonic()>=deadline:raise C.Interrupted('STOP or deadline')
                C.atomic_json(R/'task.json',{'deadline':min(deadline,time.monotonic()+1800),'phase':'root_restore'})
                state=w.restore();actions=[]
                spec={'experiment_identity':C.fingerprint(frozen),'root_id':root_id,'root_player':root['player'],
                      'input_hash':root['input_hash'],'action':a,'seed':seed,'temperature':.35}
                current=d/'attempt-001.json'
                row={**spec,'status':'incomplete','attempt':1,'started_at':C.now(),'score':None}
                C.immutable_json(d/'attempt-001-start.json',row)
                cell_deadline=min(deadline,time.monotonic()+180)
                C.atomic_json(R/'task.json',{'deadline':cell_deadline,'phase':'cell','output':str(current),'start':row})
                rng=np.random.default_rng(seed)
                try:
                    while not state.terminal:
                        if stopped() or time.monotonic()>=cell_deadline:raise C.Interrupted('STOP or deadline')
                        if len(actions)>=100000:raise RuntimeError('decision cap')
                        selected=a if not actions else choose_policy_action(w.infer(state)[0],state.legal_actions,rng,.35)
                        actions.append(selected);state=w.env.transition(state,selected);C.release(w.env,w.state,state)
                    row.update(status='ok',**terminal_proof(state,root['player']))
                except BaseException as e:
                    row.update(status='incomplete' if isinstance(e,C.Interrupted) else 'error',error=f'{type(e).__name__}: {e}')
                    raise
                finally:
                    row.update(actions=actions,decisions=len(actions),actions_sha256=C.fingerprint(actions),ended_at=C.now())
                    C.immutable_json(current,row);current=None
        assert identity()==frozen
    finally:
        w.close();(R/'task.json').unlink(missing_ok=True)

def write_summary(result):
    C.atomic_json(P/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,4))
    roots=result['roots'];x=np.arange(len(roots))
    ax.bar(x,[r['selection_seed_advantage'] for r in roots],width=.7,alpha=.35,label='Original selection seeds (biased)')
    done=[(i,r) for i,r in enumerate(roots) if r['complete']]
    ax.scatter([i for i,r in done],[r['fresh_seed_advantage'] for i,r in done],color='black',label='Fresh seeds; complete roots only')
    ax.axhline(0,color='gray');ax.set(xlabel='Frozen training root',ylabel='Selected minus cp2 game score',title=f"V9 target reliability: {result['completed_cells']}/184 continuations; no SGD")
    ax.legend();fig.tight_layout();fig.savefig(P/'target-reliability.png');plt.close(fig)

def paired_decision(views):
    assert len(views)==32 and all(v['complete'] for v in views)
    groups=sorted({v['source_json'] for v in views});rng=np.random.default_rng(219101)
    sums=np.array([sum(v['fresh_seed_advantage'] for v in views if v['source_json']==g) for g in groups]);counts=np.array([sum(v['source_json']==g for v in views) for g in groups])
    draw=rng.integers(len(groups),size=(10000,len(groups)));boot=sums[draw].sum(1)/counts[draw].sum(1)
    mean=float(np.mean([v['fresh_seed_advantage'] for v in views]));interval=np.quantile(boot,[.025,.975]).tolist()
    return dict(mean_advantage=mean,source_cluster_bootstrap_95=interval,review_for_expansion=mean>=.1 and interval[0]>0)

def summary():
    frozen=read(P/'identity.json');pairs=read(P/'actions.json')['roots'];views=[];total=0
    for pair in pairs:
        scores={}
        for a in sorted({pair['selected_action'],pair['cp2_action']}):
            for seed in frozen['config']['seeds']:
                _,ok=valid_records(paths(pair['root_id'],a,seed))
                if ok:
                    r=ok[0];assert (r['experiment_identity']==C.fingerprint(frozen) or r.get('inherited_verified') is True) and r['action']==a and r['seed']==seed and r['input_hash']==pair['input_hash']
                    scores[a,seed]=r['score'];total+=1
        complete=len(scores)==pair['distinct_actions']*4
        delta=float(np.mean([scores[pair['selected_action'],s]-scores[pair['cp2_action'],s] for s in frozen['config']['seeds']])) if complete else None
        views.append({**pair,'complete':complete,'fresh_seed_advantage':delta})
    complete=all(v['complete'] for v in views);result={'complete':complete,'completed_cells':total,'expected_cells':184,'roots':views,'no_sgd':True,'benchmark_improvement':False}
    if complete:
        result.update(paired_decision(views))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,default=10800);p.add_argument('--worker');p.add_argument('--deadline',type=float);p.add_argument('--parent',type=int);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.worker:return worker(a.worker,a.deadline,a.parent)
    assert 0<a.seconds<=10800
    R.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for path in (*C.LOCKS,R/'experiment.lock'):
            h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not stopped() and not active_actors()
        expected=identity();assert expected['source_identity']==expected['source_runtime_current']
        assert read(C.RUN/'status.json')['phase']=='complete_review_required'
        actions=read(C.ROOT/'diagnostics/target-recheck-planning/action-inventory.json')
        assert len(actions['roots'])==32 and actions['distinct_cells_for_four_new_seeds']==184
        C.immutable_json(P/'identity.json',expected);C.immutable_json(P/'actions.json',actions)
        if a.preflight:print('preflight passed');return
        deadline=time.monotonic()+a.seconds
        def status(phase,**kw):C.atomic_json(R/'status.json',{'phase':phase,'pid':os.getpid(),'updated_at':C.now(),'identity':C.fingerprint(expected),**kw})
        status('running')
        try:
            for root in actions['roots']:
                if stopped() or time.monotonic()>=deadline:raise C.Interrupted('STOP or deadline')
                if all(valid_records(paths(root['root_id'],ac,s))[1] for ac in {root['selected_action'],root['cp2_action']} for s in expected['config']['seeds']):continue
                with (R/f"{root['root_id']}.log").open('a') as log:
                    proc=subprocess.Popen([sys.executable,'-u',__file__,'--worker',root['root_id'],'--deadline',str(deadline),'--parent',str(os.getpid())],cwd=C.ROOT,start_new_session=True,stdout=log,stderr=log)
                    try:
                        while proc.poll() is None:
                            task=read(R/'task.json') if (R/'task.json').exists() else {'deadline':deadline}
                            if stopped() or time.monotonic()>min(deadline,task['deadline']+5):raise C.Interrupted('watchdog/STOP/deadline')
                            time.sleep(.5)
                    finally:
                        if proc.poll() is None:
                            os.killpg(proc.pid,signal.SIGTERM)
                            try:proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        if (R/'task.json').exists():
                            task=read(R/'task.json')
                            if task.get('output') and not Path(task['output']).exists():C.immutable_json(Path(task['output']),{**task['start'],'status':'incomplete','error':'supervisor finalized missing worker result'})
                            (R/'task.json').unlink()
                    if proc.returncode:raise RuntimeError('worker failed; inspect journal/log; no automatic retry')
                assert identity()==expected
                write_summary(summary())
            result=summary();assert result['complete'];write_summary(result);status('complete_review_required')
        except BaseException as e:
            status('review_required',reason=f'{type(e).__name__}: {e}')
            write_summary(summary());raise
if __name__=='__main__':main()
