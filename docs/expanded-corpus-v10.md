# V10 expanded self-play corpus — phase A

V9r2 completed184/184 valid continuations and passed its fixed reliability rule: fresh-seed mean advantage0.1484375, source-cluster95% interval[0.016129,0.296875]. The independently audited result supports useful selected actions on the training roots, despite selection-seed optimism (original0.328125). V8 nevertheless failed broad evaluation (terminal81/256, freshcp2 82/256). The next hypothesis is insufficient state/map coverage and transfer, not insufficient fitting updates on the same32 roots.

## Phase A frozen before collection

Collect **64 complete own-self-play source games**: eight maps in this exact order: tiny, DefaultSmall, closeup, knifefight, DefaultMedium, DefaultLarge, cheesefarm, wallsofparadis; map index m=0..7, j=1..8. Every scheduled source is retained, including games not eventually supplying a selected root. This is not opponent benchmarking: both teams use frozen cp2, temperature1.0, no league or benchmark policies. The four original excluded training maps remain excluded.

Engine seed `20265200+100*m+j`; team policy seeds `2*(430000+100*m+j)+team`. Anchor reservoir seed223151 uses the existing per-source/team hash derivation and capacity16. Source seeds are disjoint from prior explicit lists and formula ranges; the prelaunch JSON scan is retained separately. Every source gets an immutable identity, terminal proof, complete actions, root candidates and canonical feature cache. Winners and terminal scores never choose source inclusion.

The implementation reuses the frozen V8 source collector in a separate module instance, replacing only its source-schedule validator and storage/STOP context. The V8 files and runtime remain unchanged. The V10 identity pins the adapter, config, protocol, backend and the full V8 runtime. There is no SGD, search-target scoring, model validation or benchmark evaluation in this phase.

Use three one-thread collectors at most, each owning an isolated Java engine. The supervisor owns the seven shared V8 locks plus V9r2 and its own lock. Respect all corresponding STOPs. A cycle is bounded to10800seconds; each source has1800seconds and a parent process-group watchdog. Completed sources are never rerun. Any interrupted or failed attempt stops for evidence review; no automatic seed substitution or retry. A source returning no required role is still a valid retained source, not an excuse to replace its outcome.

All64 games are required before the next phase. Report coverage by map, role/team and anchor availability. Deadline means preserved incomplete work, not a reduced corpus. The collector's completion means source data are available, not that a learned model improved.

## Precommitted downstream plan; separate implementation freeze required

After phase A, select64 training roots: four primary groups j=1..4 per map, odd groups(A baby,B king), even(B baby,A king). Use the first source turn with macro37 legal for each assigned role/team. Matching-parity j=5..8 reserves fill missing slots in increasing order, exclusively assigned to the earliest still-missing group; never replace a filled slot. All64 root slots are required. Their outcomes must not affect selection.

Use1024 anchors, exactly8 per team from every one of the64 source games. Uniformly sample8 without replacement from each complete16-capacity reservoir with RNG223153 and fixed source/team order; a source with fewer than8 positions for either team causes a coverage review, not substitution. Freeze roots/anchors and their hashes before validation discovery. Report exact input overlap rather than claiming universal independence.

The next implementation must generate32 fresh validation roots, four per map, excluded from the full union of V6 gradient/validation inputs, V8 gradient/validation inputs, and all V10 gradient inputs. V9 adds no new gradient inputs. Validation source seeds/reserves will be explicitly preregistered and checked before those games, with role/team allocation and chronological exclusion rules fixed, without consulting training target outcomes.

Only after complete input coverage, independently tested target/learner/validation/evaluation implementations and their runtime freeze may target scoring begin. Compare exhaustive all-legal-action paired terminal ranking (proposed seeds223003/223013, cp2 continuation T.35) against PUCT32/depth12 on the same64 roots, with a3200-cell ceiling frozen before outcomes. Any ceiling/coverage failure requires review, not pruning. No V9 continuation outcome becomes a training label.

Keep the policy head only, frozen trunk/critic, prior-smoothed targets(target weight0.75), anchorKL1 and matched optimizer/batches. Plan4096 updates, doubling the data and update count together to retain V8's expected per-example exposure; do not choose a checkpoint after outcomes. Terminal is the fixed candidate, PUCT the matched control. Freeze both before independent validation; then use the unchanged full development-v4 comparison and fresh-seed confirmation rule. No architecture/engine change is proposed. These downstream stages are not implemented or launched by phase A and must receive an explicit separate implementation/identity freeze before execution.

Run `.venv/bin/python scripts/collect_expanded_sources_v10.py --preflight`, then `--seconds 10800`. Separate namespaces: `runs/expanded-corpus-v10`, `progress/expanded-corpus-v10`. Preserve completed V8/V9r2 and all prior archives. Recovery export must be separate and include exact source NPZ/journals plus source-runtime/dependency pins, with isolated restore verification before publication.
