"""zdp decision-point catalogue from the Java constants (no numpy/torch needed).

    option_counts()   {point id: option count}  (NAME = id; NAME_NET_OPTIONS preferred over NAME_OPTIONS)
    feature_counts()  {point id: feature count} (NAME_FEATS: a record's n minus its RL trailer)
    SECONDARY         points recorded from D's secondary slot (RL trailer flags bit1)

    python3 scripts/dp_points.py     # print the table
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'java/bots/zdp'
SECONDARY = (6, 7, 9, 15)   # BABY_COMMS, MINE_PICK, BABY_VAL, KING_VAL
NAMES = {1: 'BABY_MOVE', 2: 'BABY_ACT_PRE', 3: 'BABY_ACT_POST', 4: 'BABY_FACE', 5: 'BABY_ROLE', 6: 'BABY_COMMS',
         7: 'MINE_PICK', 8: 'BABY_FREE', 9: 'BABY_VAL', 10: 'KING_MOVE', 11: 'KING_SPAWN', 12: 'KING_ACT',
         13: 'KING_FORM', 14: 'KING_COMMS', 15: 'KING_VAL', 16: 'KING_TURN'}


def constants(src=SRC):
    """{NAME: first int value} over D*.java (NAME = int, including comma-separated declarations)."""
    consts = {}
    for f in sorted(Path(src).glob('D*.java')):
        for m in re.finditer(r'\b([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\b', f.read_text()):
            consts.setdefault(m.group(1), int(m.group(2)))
    return consts


def option_counts(src=SRC):
    """{point id: option count}: NAME_NET_OPTIONS (net-mode options, e.g. BABY_MOVE's primitives) when present, else
    NAME_OPTIONS."""
    c = constants(src)
    out = {c[k[:-8]]: v for k, v in c.items() if k.endswith('_OPTIONS') and not k.endswith('_NET_OPTIONS') and k[:-8] in c}
    out.update({c[k[:-12]]: v for k, v in c.items() if k.endswith('_NET_OPTIONS') and k[:-12] in c})
    return out


def feature_counts(src=SRC):
    """{point id: feature count} from NAME_FEATS."""
    c = constants(src)
    return {c[k[:-6]]: v for k, v in c.items() if k.endswith('_FEATS') and k[:-6] in c}


def main():
    o, f = option_counts(), feature_counts()
    for p in sorted(set(o) | set(f)):
        print(f'{p:3d} {NAMES.get(p, "?"):14s} options {o.get(p)}  features {f.get(p)}'
              f'{"  (secondary)" if p in SECONDARY else ""}')


if __name__ == '__main__':
    main()
