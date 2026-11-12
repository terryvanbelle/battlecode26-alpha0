import unittest
from scripts.build_imitation_hooks import transform,SOURCE
from alpha0.imitation.trace import OPERATIONS
class HookBuildTests(unittest.TestCase):
 def test_exact_primitive_inventory_and_missing_method_rejected(self):
  source=SOURCE.read_text();generated,found=transform(source)
  self.assertEqual(sorted(found),sorted(OPERATIONS));self.assertEqual(len(found),23)
  self.assertEqual(generated.count('PrimitiveRecorder.begin('),23)
  with self.assertRaises(ValueError):transform(source.replace('public void resign()', 'public void missingResign()'))
 def test_overloads_keep_original_bodies(self):
  generated,_=transform(SOURCE.read_text())
  self.assertIn('private void _imitationOriginal_attack(MapLocation loc)',generated)
  self.assertIn('private void _imitationOriginal_attackAmount(MapLocation loc, int cheese)',generated)
  self.assertIn('Boolean result = _imitationOriginal_squeak(messageContent)',generated)
if __name__=='__main__':unittest.main()
