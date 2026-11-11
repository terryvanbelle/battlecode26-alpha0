"""Bounded no-game completion audit and publication for the prefix diagnostic."""
import fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.probe_v10_root_eligibility import R,P,S,LOCKS,STOPS

def main():
 with (R/'finalizer.lock').open('a+') as h:
  fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def state(phase,**kw):S.C.atomic_json(R/'finalizer-status.json',dict(phase=phase,updated_at=S.C.now(),**kw))
  state('waiting');deadline=time.monotonic()+10800
  while time.monotonic()<deadline:
   if any(p.exists() for p in STOPS):state('stopped');return
   current=S.read(R/'status.json')
   if current['phase']=='complete_review_required':break
   if current['phase'] in ('review_required','deadline_review_required'):state('collector_review_required',collector=current);return
   time.sleep(5)
  else:state('wait_deadline');return
  try:
   with ExitStack() as stack:
    for p in LOCKS:
     h2=stack.enter_context(p.open('a+'));fcntl.flock(h2,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check=S.read(P/'identity.json');assert check['script_sha256']==S.C.sha256(S.C.ROOT/'scripts/probe_v10_root_eligibility.py')
    assert check['protocol_sha256']==S.C.sha256(S.C.ROOT/'docs/v10-eligibility-probe.md')
    source_summary=S.read(S.P/'summary.json');pins={r['source_id']:r for r in source_summary['sources']};rows=[]
    assert {p.stem for p in (P/'sources').glob('*.json')}==set(pins)
    for name,pin in pins.items():
     row=S.read(P/'sources'/f'{name}.json');source_path=S.C.ROOT/pin['source_json'];source=S.read(source_path)
     assert row['source_id']==name and row['source_sha256']==pin['source_sha256']==S.C.sha256(source_path)
     assert row['complete'] and row['new_terminal_games']==0 and row['sgd_updates']==0
     assert row['map']==source['map'] and 0<=row['replayed_actions']<=len(source['actions'])
     for role,pos in row['eligible_roles'].items():
      assert role in ['BABY_RAT:0','BABY_RAT:1','RAT_KING:0','RAT_KING:1']
      assert len(set(pos['legal_actions']))>=2 and 0<=pos['decision_index']<=row['replayed_actions']
     rows.append(row)
    audit={'complete':True,'sources':64,'all_four_roles':sum(len(r['eligible_roles'])==4 for r in rows),'missing_roles':{r['source_id']:sorted(set(['BABY_RAT:0','BABY_RAT:1','RAT_KING:0','RAT_KING:1'])-set(r['eligible_roles'])) for r in rows if len(r['eligible_roles'])<4},'limitations':'Saved-record and provenance audit; not independent engine replay or terminal equivalence. No training authorization from this audit alone.'}
    S.C.immutable_json(P/'completion-audit.json',audit)
    (P/'RESULTS.md').write_text(f"# Prefix eligibility result\n\nAll64 prefixes completed; {audit['all_four_roles']}/64 expose all four role/team slots with at least two legal actions. No new terminal games, target outcomes or SGD.\n\n[Audit](completion-audit.json) · [Records](sources)\n\nThis supports reviewing a new general-decision extraction protocol, not retroactively passing V10 coverage or claiming stronger play. Complete input/terminal replay verification, new validation preregistration and tested downstream code remain required before learning.\n")
    hashes={str(p.relative_to(S.C.ROOT)):S.C.sha256(p) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='publication-manifest.json'}
    S.C.immutable_json(P/'publication-manifest.json',{'files':hashes,'source_recovery':S.read(S.P/'recovery.json'),'complete':True})
    subprocess.run(['git','add','--',str(P.relative_to(S.C.ROOT))],cwd=S.C.ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','Publish complete fixed-prefix eligibility diagnostic','--',str(P.relative_to(S.C.ROOT))],cwd=S.C.ROOT,check=True)
    subprocess.run(['git','push'],cwd=S.C.ROOT,check=True)
   state('published_review_required',audit=audit)
  except BaseException as e:state('review_required',reason=str(e));raise
if __name__=='__main__':main()
