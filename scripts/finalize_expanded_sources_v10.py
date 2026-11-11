"""Wait for V10 source completion, freeze coverage plan and publish verified recovery."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_expanded_sources_v10 import C,R,P,STOPS,read
from scripts.review_expanded_sources_v10 import main as review
from scripts.export_expanded_sources_v10 import export

def main():
    with (R/'finalizer.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def state(phase,**kw):C.atomic_json(R/'finalizer-status.json',{'phase':phase,'updated_at':C.now(),**kw})
        deadline=time.monotonic()+10800;state('waiting_for_sources')
        while time.monotonic()<deadline:
            if any(p.exists() for p in STOPS):state('stopped');return
            status=read(R/'status.json')
            if status['phase']=='complete_review_required':break
            if status['phase']=='review_required':state('collector_review_required',reason=status.get('reason'));return
            time.sleep(5)
        else:state('wait_deadline');return
        try:
            state('reviewing_coverage')
            for attempt in range(60):
                try:review();break
                except BlockingIOError:time.sleep(1)
            else:raise RuntimeError('collector lock still held')
            coverage=read(P/'coverage-review.json');state('exporting');export();pointer=read(P/'recovery.json')
            (P/'README.md').write_text(f'''# V10 source phase complete

All64 scheduled self-play source games completed. No target scoring, SGD, validation or benchmark games were run in this phase.

Coverage complete: **{coverage['complete']}**. Planned selection cells: **{coverage['planned_selection_cells']} / 3200 ceiling**. Missing root slots: {len(coverage['missing_root_slots'])}; anchor shortfalls: {len(coverage['anchor_shortfalls'])}.

[Coverage review](coverage-review.json) · [Frozen input plan](training-input-plan.json) · [Verified recovery](recovery.json) · [Protocol](../../docs/expanded-corpus-v10.md)

A complete tested downstream implementation and a fresh validation-source preregistration are required before target outcomes or gradients. Source completion is not playing-strength improvement.
''')
            if (C.ROOT/pointer['archive']).stat().st_size>=95*1024*1024:raise RuntimeError('verified archive needs chunked publication review')
            paths=[str(P.relative_to(C.ROOT)),pointer['archive'],pointer['manifest']]
            subprocess.run(['git','add','--',*paths],cwd=C.ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Publish complete V10 source coverage and verified recovery','--',*paths],cwd=C.ROOT,check=True)
            subprocess.run(['git','push'],cwd=C.ROOT,check=True);state('published_review_required',coverage=coverage,recovery=pointer)
        except BaseException as e:state('review_required',reason=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':main()
