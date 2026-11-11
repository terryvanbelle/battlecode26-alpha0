import tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from alpha0 import expanded_v11_benchmark_recovery as B
from alpha0 import terminal_v8_report as T

class AdapterTests(unittest.TestCase):
 def test_one_physical_attempt_with_required_helper_contract(self):
  with tempfile.TemporaryDirectory() as tmp:
   with patch.object(T,'validate_benchmark_spec'),patch.object(T,'select_benchmark_result',return_value=None),patch.object(T,'benchmark_worker',return_value={'attempt':1,'error_kind':'test'}) as work:
    result=T.record_benchmark_attempt({},Path(tmp),attempt=1,max_attempts=2)
    self.assertEqual(result['attempt'],1);work.assert_called_once_with({},1)
 def test_worker_single_attempt_and_failed_result_stops(self):
  ns=B.runtime()
  with tempfile.TemporaryDirectory() as tmp:
   ns.update(P=Path(tmp),R=Path(tmp),identity=lambda:{'config':{'evaluation_timeout_seconds':1800}})
   calls=[]
   def record(*a,**kw):calls.append(kw);return {'error':'test'}
   ns['record_benchmark_attempt']=record;ns['valid_benchmark_result']=lambda *a:False
   with self.assertRaisesRegex(RuntimeError,'benchmark failure requires review'):
    ns['worker']({'id':'test','kind':'benchmark','payload':{'cell_id':'test'}},{'config':{'evaluation_timeout_seconds':1800}},time.monotonic()+60,__import__('os').getppid())
   self.assertEqual(calls,[{'attempt':1,'max_attempts':2}])
 def test_only_benchmark_execute(self):
  ns=B.runtime();self.assertIs(ns['execute'],B.execute)
  self.assertIn('benchmark-recovery',ns['run_jobs'].__code__.co_consts.__repr__())
if __name__=='__main__':unittest.main()
