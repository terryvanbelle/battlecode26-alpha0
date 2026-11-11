import tempfile,unittest
from pathlib import Path
from scripts.inventory_zdb_imitation_interface import calls,inventory
class InventoryTests(unittest.TestCase):
 def test_comments_and_strings_are_not_calls(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'X.java';p.write_text('// rc.fake();\n/* G.rc.fake2(); */\nString s="rc.fake3()";\nG.rc.squeak(1); rc.move(d);')
   self.assertEqual(calls(p),[{'method':'squeak','line':4},{'method':'move','line':4}])
 def test_actual_communication_gap_and_no_dynamic_claim(self):
  r=inventory();self.assertTrue(r['not_dynamic_coverage']);self.assertEqual(r['teacher_games'],0)
  for name in ('squeak','writeSharedArray'):
   self.assertGreater(r['communication_methods'][name]['teacher_call_sites'],0);self.assertEqual(r['communication_methods'][name]['macro_call_sites'],0)
if __name__=='__main__':unittest.main()
