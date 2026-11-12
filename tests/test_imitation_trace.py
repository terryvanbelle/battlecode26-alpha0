import copy,json,tempfile,unittest
from pathlib import Path
from alpha0.imitation.trace import TraceWriter,read_trace,compare_complete,digest,arguments,validate
H='a'*64

def action(op='move',args=None):return {'operation':op,'arguments':['NORTH'] if args is None else args,'outcome':'returned','return_boolean':None,'exception':None,'public_observation_sha256':H,'bytecodes_before':10,'bytecodes_after':20}
def end():return {'public_observation_sha256':H,'physical_state_sha256':H,'bytecodes_used':30,'termination':'yield'}
def seal(rows):return {'game_id':H,'complete':True,'record_count':len(rows),'last_sha256':rows[-1]['sha256'],'terminal_sha256':H,'runtime_identity':H,'teacher_identity':H,'source_spec_sha256':H}
class TraceTests(unittest.TestCase):
 def fixture(self,p):
  w=TraceWriter(p,H);w.append(1,'A',1,0,'action',action());w.append(1,'A',1,0,'turn_end',end());w.close();return read_trace(p)
 def test_roundtrip_and_no_training_authorization(self):
  with tempfile.TemporaryDirectory() as t:
   rows=self.fixture(Path(t)/'trace.jsonl');r=compare_complete(rows,rows,seal(rows),seal(rows));self.assertTrue(r['equivalent']);self.assertFalse(r['training_approved'])
 def test_budget_perturbation_fails_even_with_rehashed_trace(self):
  with tempfile.TemporaryDirectory() as t:
   rows=self.fixture(Path(t)/'trace.jsonl');other=copy.deepcopy(rows);other[-1]['payload']['bytecodes_used']+=1;other[-1]['sha256']=digest({k:v for k,v in other[-1].items() if k!='sha256'})
   with self.assertRaises(ValueError):compare_complete(rows,other,seal(rows),seal(other))
 def test_truncation_and_reordering_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'trace.jsonl';self.fixture(p);text=p.read_text();p.write_text(text[:-1])
   with self.assertRaises(ValueError):read_trace(p)
   p.write_text(''.join(reversed(text.splitlines(keepends=True))))
   with self.assertRaises(ValueError):read_trace(p)
 def test_incomplete_seal_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   rows=self.fixture(Path(t)/'trace.jsonl');s=seal(rows);s['complete']=False
   with self.assertRaises(ValueError):compare_complete(rows,rows,s,s)
 def test_overloads_communication_and_typed_arguments(self):
  arguments('attack',[[2,3]]);arguments('attackAmount',[[2,3],5]);arguments('writeSharedArray',[0,1023])
  for op,args in [('attack',[[2,3],5]),('writeSharedArray',[True,5]),('move',['BAD']),('squeak',[2**31])]:
   with self.assertRaises(ValueError):arguments(op,args)
 def test_unknown_privileged_field_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   rows=self.fixture(Path(t)/'trace.jsonl');row=rows[0];row['teacher_memory']={}
   with self.assertRaises(ValueError):validate(row)
 def test_exclusive_write_and_robot_history(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'trace.jsonl';w=TraceWriter(p,H);w.append(1,'A',2,1,'turn_end',end())
   with self.assertRaises(ValueError):w.append(1,'A',1,0,'turn_end',end())
   w.close()
   with self.assertRaises(FileExistsError):TraceWriter(p,H)
if __name__=='__main__':unittest.main()
