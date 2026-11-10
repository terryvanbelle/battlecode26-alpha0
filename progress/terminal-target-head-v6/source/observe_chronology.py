#!/usr/bin/env python3
"""Capture immutable local provenance observations while v6 is collecting cells."""
from datetime import datetime,timezone
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path('/workspace/battlecode26-alpha0');PROGRESS=ROOT/'progress/terminal-target-head-v6';RUN=ROOT/'runs/terminal-target-head-v6'
def digest(data):return hashlib.sha256(data).hexdigest()
def observe(path):
 before=path.stat();data=path.read_bytes();after=path.stat()
 if (before.st_ino,before.st_mtime_ns,before.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size):raise RuntimeError('artifact changed during observation')
 return {'sha256':digest(data),'mtime_ns':after.st_mtime_ns,'observed_at':datetime.now(timezone.utc).isoformat()},json.loads(data)
def publish(path,data):
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.link(tmp,path)
 finally:os.unlink(tmp)
identityobs,identity=observe(PROGRESS/'identity.json');statusobs,status=observe(RUN/'status.json')
observations={};sources=[];comparators=[]
paths=[PROGRESS/f for f in ('roots-training.json','roots-validation.json','anchors.json','target-budget.json')]
paths+=sorted((RUN/'sources').glob('*.json'))+sorted((PROGRESS/'comparators').glob('*.json'))
for path in paths:
 observation,row=observe(path);name=str(path.relative_to(ROOT));observations[name]=observation
 if path.parent==RUN/'sources':sources.append({'path':name,'source_id':row['source_id'],'started_at':row['started_at'],'ended_at':row['ended_at'],'sha256':observation['sha256']})
 elif path.parent==PROGRESS/'comparators':comparators.append({'path':name,'root_id':row['root_id'],'completed_at':row['completed_at'],'sha256':observation['sha256']})
starts=[]
for path in sorted((PROGRESS/'cells').glob('selection-*/attempt-*-start.json')):
 observation,row=observe(path);starts.append((row['started_at'],str(path.relative_to(ROOT)),observation,row))
first=min(starts,key=lambda x:datetime.fromisoformat(x[0]))
observations[first[1]]=first[2]
source=Path(__file__).read_bytes()
report={'schema_version':1,'experiment_id':'terminal-target-head-v6','created_at':datetime.now(timezone.utc).isoformat(),'experiment_identity_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'identity_file':identityobs,'observed_actor_status':status,'actor_status_file':statusobs,'observations':observations,'source_journals':sources,'comparator_journals':comparators,'first_observed_selection_start':{'path':first[1],'sha256':first[2]['sha256'],'record':first[3]},'completed_source_count':len(sources),'frozen_comparator_count':len(comparators),'auditor_source_sha256':digest(source),'limitations':'These are hash-bound observations of original-run local filesystem mtimes and embedded timestamps captured after selection started. They are not cryptographic proof of creation time. Restore audits must reuse these observations, never Git checkout mtimes. All roots/G/allocation were independently checked earlier in this session, but that earlier interactive check has no separately published timestamp attestation.'}
archive=PROGRESS/'source/observe_chronology.py';publish(archive,source)
output=PROGRESS/'chronology-observation.json';publish(output,(json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps({'path':str(output),'sha256':digest(output.read_bytes()),'created_at':report['created_at'],'source_games':len(sources),'comparators':len(comparators),'first_selection_start':first[0],'observations':len(observations)},indent=2))
