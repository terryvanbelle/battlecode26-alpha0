import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from alpha0 import expanded_v11_benchmark_deadline as D
class RecoveryTests(unittest.TestCase):
 def test_actual_exact_remaining_inventory(self):
  r=D.verify();specs=D.S.read(D.S.P/'benchmarks/specs.json')['specs'];remaining=[s for s in specs if s['cell_id'] in r['remaining_cell_ids']]
  self.assertEqual(len(remaining),22);self.assertEqual(sum(D.attempt_number(s,r)==2 for s in remaining),2)
  with self.assertRaises(ValueError):D.attempt_number(next(s for s in specs if s not in remaining),r)
 def test_corruption_and_already_attempted_rejected(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(D.S,'P',Path('/unused')):
   with patch.object(D.S,'P',Path(tmp)):
    cell=Path(tmp)/'benchmarks/cells/x';cell.mkdir(parents=True)
    D.S.C.immutable_json(cell/'attempt-001.json',{'status':'error','evaluation':{'error':'SIGTERM','score':None,'rounds':None}})
    r={'remaining_cell_ids':['x'],'interrupted':{'x':D.S.C.sha256(cell/'attempt-001.json')}}
    self.assertEqual(D.attempt_number({'cell_id':'x'},r),2)
    (cell/'attempt-001.json').write_text('{}')
    with self.assertRaises(ValueError):D.attempt_number({'cell_id':'x'},r)
 def test_no_automatic_retry_for_new_cells(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(D.S,'P',Path('/unused')):
   with patch.object(D.S,'P',Path(tmp)):
    r={'remaining_cell_ids':['x'],'interrupted':{}};self.assertEqual(D.attempt_number({'cell_id':'x'},r),1)
    cell=Path(tmp)/'benchmarks/cells/x';cell.mkdir(parents=True);(cell/'attempt-001.json').write_text('{}')
    with self.assertRaises(ValueError):D.attempt_number({'cell_id':'x'},r)
if __name__=='__main__':unittest.main()
