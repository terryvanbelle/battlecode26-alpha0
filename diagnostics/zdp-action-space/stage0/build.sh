#!/bin/bash
# box: build + compile each stage-0 variant (plain zdp, no net) from variants.tsv on stdin
cd ~/rltest/bc26a0
while IFS=$'\t' read -r name grp label cfg; do
  .venv/bin/python scripts/build_dpnet.py --name $name ${cfg:+--cfg $cfg} | tail -1 || exit 1
done < runs/aspace/s0/variants.tsv
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, 'scripts')
from concurrent.futures import ThreadPoolExecutor
from eval_student import compile_local
names = ['zdpc%02d' % i for i in range(25)]
with ThreadPoolExecutor(4) as ex:
    list(ex.map(compile_local, names))
print('compiled', len(names))
PY
