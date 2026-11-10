# Collegejunior crash review and replacement qualification

Original revised baseline: four opponent-runtime failures on wallsofparadis,
both seeds and seats. Two fresh independent reruns (seeds 0 and 2026, learned
seat A) reproduced the same AssertionError in collegejunior.BugSim.simBug:229.
The other seven opponents completed all four games on that map under the same
harness. Source hashes were verified by the evaluator; no bot source was edited.
The failed assertion checks internal pathfinding state. These games remain
unscored, never wins. Original v3 results remain intact.

Before replacement games, select AdamTan12/Icarus, historical Elo1745.9 versus
collegejunior1767.0. It is the nearest-rated available package after excluding
already-retired Finals (documented crashes). Pinned at the same historical
upstream commit as Finals, but a separate package. Qualification requires all
32 games on the exact eight-map/two-seed/both-seat schedule to be valid, with
zero unresolved errors. Selection is based on prior rating and crash validity,
not whether our policy wins. Do not drop wallsofparadis or weaken thresholds.
Any accepted replacement requires a new panel version and baseline qualification.

Raw original errors: ../../progress/development-v3/baseline/evaluation_errors.jsonl.
Independent reproductions and candidate provenance are retained beside this file.
