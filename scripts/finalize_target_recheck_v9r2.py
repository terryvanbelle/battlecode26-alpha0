"""Bounded completion monitor: independent audit, verified export, truthful publication."""
import fcntl,json,subprocess,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_target_recheck_v9r2 import C,R,P,read,summary,write_summary,stopped
from scripts.audit_target_recheck_v9r2 import audit
from scripts.export_target_recheck_v9r2 import export

def main():
    with (R/'finalizer.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def state(phase,**kw):C.atomic_json(R/'finalizer-status.json',{'phase':phase,'updated_at':C.now(),**kw})
        end=time.monotonic()+10800;state('waiting_for_completion')
        while time.monotonic()<end:
            if stopped():state('stopped');return
            s=read(R/'status.json')
            if s['phase']=='complete_review_required':break
            if s['phase']=='review_required':state('collector_review_required',reason=s.get('reason'));return
            time.sleep(5)
        else:state('wait_deadline');return
        try:
            # Acquire/release the eight locks to ensure the last actor has exited.
            from contextlib import ExitStack
            quiescent_deadline=time.monotonic()+60
            while True:
                try:
                    with ExitStack() as stack:
                        for path in (*C.LOCKS,R/'experiment.lock'):
                            h=stack.enter_context(path.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if stopped() or time.monotonic()>quiescent_deadline:raise
                    time.sleep(1)
            state('auditing')
            result=audit();s=summary();assert result['complete'] and s['complete'] and result['verified_cells']==184
            for key in ('mean_advantage','source_cluster_bootstrap_95','review_for_expansion'):assert result[key]==s[key]
            C.immutable_json(P/'independent-final-audit.json',result);write_summary(s)
            state('exporting');export()
            proof=read(P/'recovery.json');assert not proof['partial']
            verdict='PASSED' if result['review_for_expansion'] else 'FAILED'
            text=f'''# V9r2 complete — target reliability only

All **184/184** planned continuations are valid, covering all32 roots. Eight inherited games were preserved and independently reverified without reruns. No SGD or benchmark games.

- Fresh-seed mean paired advantage: **{result['mean_advantage']:.6f}**.
- Source-game-cluster bootstrap95% interval: **{result['source_cluster_bootstrap_95']}**.
- Precommitted expansion rule: **{verdict}**.

This is a training-target mechanism result, not stronger playing performance, model promotion or benchmark confirmation. Review the full per-root/source evidence before choosing a new learning experiment.

![All root comparisons](target-reliability.png)

[Summary](summary.json) · [Independent final audit](independent-final-audit.json) · [Verified recovery](recovery.json) · [Frozen protocol](../../docs/target-recheck-v9r2.md)
'''
            (P/'README.md').write_text(text)
            paths=[str(P.relative_to(C.ROOT)),proof['archive'],proof['manifest']]
            subprocess.run(['git','add','--',*paths],cwd=C.ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Publish independently audited V9r2 target-reliability result','--',*paths],cwd=C.ROOT,check=True)
            subprocess.run(['git','push'],cwd=C.ROOT,check=True)
            state('published_review_required',mean_advantage=result['mean_advantage'],expansion_rule=result['review_for_expansion'],recovery=proof)
        except BaseException as e:state('review_required',reason=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':main()
