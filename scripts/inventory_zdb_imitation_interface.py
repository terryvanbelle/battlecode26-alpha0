"""Static teacher/interface feasibility inventory; never executes bots or produces labels."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def code_only(text):
    # Preserve line numbers when suppressing comments/strings. This is a lexical
    # inventory, not Java call graph analysis or proof that a branch executes.
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',lambda m:''.join('\n' if c=='\n' else ' ' for c in m.group()),text)
def calls(path):
    text=code_only(path.read_text());return [{'method':m.group(1),'line':text.count('\n',0,m.start())+1} for m in re.finditer(r'\b(?:G\.)?rc\.([A-Za-z_]\w*)\s*\(',text)]
def inventory():
    teacher=sorted((ROOT/'benchmarks/bots/vibe26b_zdb').glob('*.java'));macro=ROOT/'java/alpha0/MacroV2.java'
    records={str(p.relative_to(ROOT)):calls(p) for p in teacher};macro_calls=calls(macro)
    methods=sorted({r['method'] for rs in records.values() for r in rs});macro_methods={r['method'] for r in macro_calls}
    communication=['readSharedArray','writeSharedArray','readSqueaks','squeak']
    return {'kind':'static_interface_inventory','not_dynamic_coverage':True,'teacher_games':0,'training_labels':0,'gradient_updates':0,'pins':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in teacher+[macro,ROOT/'alpha0/features_v2.py',ROOT/'alpha0/network_v2.py']},'teacher_calls':records,'teacher_method_names':methods,'macro_method_names':sorted(macro_methods),'communication_methods':{m:{'teacher_call_sites':sum(r['method']==m for rs in records.values() for r in rs),'macro_call_sites':sum(r['method']==m for r in macro_calls)} for m in communication},'teacher_methods_absent_from_macro':sorted(set(methods)-macro_methods),'limits':['Lexical receiver-name scan; aliases/reflection/call reachability not resolved.','Call-site counts are not action frequencies or measured behavioral coverage.','Missing communication operations establish that macro_v2 cannot express arbitrary complete teacher controller traces; actual unmatched-turn fraction is unmeasured.','Teacher private memory must not be exposed as student inputs.']}
if __name__=='__main__':
    out=ROOT/'diagnostics/zdb-imitation-planning/static-interface-inventory.json';out.write_text(json.dumps(inventory(),indent=2)+'\n');print(out)
