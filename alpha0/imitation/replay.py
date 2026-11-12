"""Independent replay callback comparison, explicitly narrower than API tracing."""
import json
from pathlib import Path
from .trace import digest

def compare_replay(baseline, recording):
    def load(path):
        if Path(path).stat().st_size > 9*1024*1024:
            raise ValueError('replay audit exceeds cap')
        rows=json.loads(Path(path).read_text())
        if not isinstance(rows,list) or not rows:
            raise ValueError('empty replay audit')
        for i,row in enumerate(rows):
            if not isinstance(row,dict) or set(row)!={'sequence','method','arguments'}:
                raise ValueError('invalid replay callback')
            if type(row['sequence']) is not int or row['sequence']!=i or not isinstance(row['method'],str) or not isinstance(row['arguments'],list):
                raise ValueError('invalid replay sequence')
        return rows
    left,right=load(baseline),load(recording)
    if left!=right:raise ValueError('independent replay callbacks differ')
    return {'schema':'replay-callback-comparison-v1','callbacks':len(left),
            'ordered_callbacks_sha256':digest(left),'equal':True,
            'complete_primitive_trace':False,'teacher_equivalence':False,'training_approved':False,
            'known_omissions':['shared-array writes','squeak payload/return value','failed API calls','intermediate move/turn calls','match header/footer']}
