# V8 completed candidate panel; matched control still running

The fixed terminal candidate completed all 256 development-v4 games with **81 wins, 175 losses, no draws** (31.640625%). The fresh cp2 control has82 wins (32.03125%); the original baseline has83 (32.421875%). Every candidate terminal result was checked against its canonical frozen request. Two deadline-interrupted first attempts remain preserved; their reviewed second attempts completed validly. There are no unresolved candidate errors.

The improvement rule fails against both controls. Against fresh cp2, mean game-score change is **−0.390625 percentage points**, with paired map-bootstrap95% interval **[−2.734375,+1.953125] points**. Against the original baseline it is **−0.78125 points**, interval **[−3.125,+1.953125]**. Aggregate zdb change is zero. No confirmation or promotion is warranted. These results do not establish a meaningful regression either; the uncertainty interval spans zero.

[Completed-panel terminal audit and both comparisons](../progress/terminal-target-head-v8/reviews/completed-panels-20261110T2313.json). The PUCT control must still finish its full256 games. The experiment-wide audit and final recovery export remain pending; this document reports completed panels only.

## Diagnostic interpretation and next review

V8 resolves the earlier underfitting problem: both heads match32/32 training labels after2048 updates, while frozen tensors remain byte-identical. Yet independent first-action validation found zero advantage, and the terminal head fails the broader playing-strength test. More identical SGD is not justified.

Two explanations remain unresolved: the two-seed exhaustive maximum may select noisy terminal targets; alternatively, useful targets may fit the32 roots but transfer poorly to new states. The next proposed bounded mechanism test is to re-evaluate the **already frozen32 training-root selected actions against cp2 greedy actions**, on fresh continuation seeds, with no SGD. This tests target reliability directly before paying for a larger training corpus. It is not fresh model-validation evidence and must never be described as benchmark improvement.

Draft protocol for review after the matched PUCT panel completes:

- Retain all32 roots and their existing fixed selected actions, including unchanged-action roots. Freeze exact input/source hashes and action pairs before collection. No root selection by outcome.
- Four proposed new continuation seeds219003,219013,219029,219041; verify disjointness against all recorded source/continuation/confirmation inventories before freezing. No seed selection based on outcomes.
- Frozen cp2 continuation policy, temperature0.35, identical runtime. Physically share identical-action cells; at most256 distinct terminal games. Score every root equally, retaining all32 in the denominator.
- Report new paired advantage and original selection-seed advantage separately, plus individual roots, maps, roles, sides and source-game clusters. A prospective expansion criterion should require mean paired advantage≥0.10 and a95% source-game-cluster bootstrap lower bound>0, with complete valid evidence. Failure is inconclusive or evidence of target unreliability, not proof that no learning method can help.
- No fitting, benchmark games, candidate selection or confirmation in this diagnostic. Even a pass only motivates a separately preregistered fresh learning experiment with fresh validation. Preserve V8 unchanged.
- Implement in a new run/progress namespace only after full V8 review; three-hour resumable budget, all existing locks/STOPs plus its own, at mostthree one-thread actors and process-tree watchdogs. Freeze implementation/config before any new outcomes.

This is an unlaunched proposal, not an approved result or a completed preregistration. The full PUCT result and independent audit may change the research choice before new data are collected.
