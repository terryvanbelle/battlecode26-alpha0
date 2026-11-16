#!/bin/bash
# Stage 0 (design C's T2 family slack): each plain zdb variant zdpc01-24 vs zdpc00 (= zdb), 64 games on row 3 (line 4),
# seed 3801401, all variants on the same maps and seed. Runs after the section-18 g tests (T5 ALL DONE).
S=/tmp/claude-1000/-home-terryvanbelle-projects-vibe-bc26a0/31310128-fb77-4058-8e53-1da41e55d47b/scratchpad
H=/tmp/claude-1000/-home-terryvanbelle-projects-vibe-bc26a0/31310128-fb77-4058-8e53-1da41e55d47b/tasks/bf4yhan8m.output
until grep -q "T5 ALL DONE" $H; do sleep 60; done
R=$(sed -n 4p $S/judge_maps.txt | cut -d' ' -f2)
tail -n +2 $S/aspace/s0/variants.tsv | while IFS=$'\t' read -r name grp label cfg; do
  n=s0${name#zdpc}
  bash $S/judge.sh $n $name zdpc00 3801401 "$(echo $R | cut -d, -f1-28)" "$(echo $R | cut -d, -f29-32)" > $S/aspace/s0/$n.log 2>&1 < /dev/null
  echo "T6 $n $grp $label $(TZ=Etc/GMT+7 date +%H:%M) $(tail -1 $S/aspace/s0/$n.log | grep -o '"points.*')"
done
echo "T6 ALL DONE"
