# Development benchmark v4

This version changes only one opponent from [v3](development-benchmarks-v3.md):
collegejunior is retired after four original and two independent reproducible
BugSim assertion failures on wallsofparadis. Icarus failed qualification with
nine crashes. YokoOno (historical Elo 1734.2, versus collegejunior 1767.0) passed
all 32 qualification games with zero errors. It shares an upstream author with
Xzibit; that limitation is retained rather than choosing replacements by wins.

All eight maps, both seeds, both seats, cp2 checkpoint, score-gain/bootstrap/zdb
nonregression rules and fresh-seed confirmation requirements remain unchanged.
The complete schedule is 256 games. No qualification outcomes are reused in the
fresh baseline. Preserve v3 evidence and never compare overall v3 and v4 rates
as a learned gain. The original training exclusions remain in force.

Use configs/improvement-v4.json with evaluate_development.py and
compare_development.py --config; use configs/benchmarks-development-v4.json with
report_development.py --manifest. Continue using the development-v3 operational
lock/STOP shared by both versions. The 70% tournament-transition priority in
[tournament-transition.md](tournament-transition.md) applies to this current suite.
