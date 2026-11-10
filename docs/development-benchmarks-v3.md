**Next phase:** at **70% overall wins** on the complete benchmark suite, prioritize [tournament compatibility](tournament-transition.md) before further win-rate optimization: Java throughout the deployed bot, standard bytecode, memory, compile and wall-clock limits, and standard API calls only.

# Development benchmark v3: broader maps and measured improvement

The user replaced the all-opponent/all-map completion objective on November 10,
2026. The old twelve-bot gate and raw results remain historical evidence; they
are not the active research success criterion. Never relabel old tiny-only
results as results on this suite.

The eight retained opponents are CS50Crab/rapierV1, spsquared/TSPAARK_QUALS,
HugoIngelsson/Xzibit, abcdezyxwv/bestsofar, IvanGeffner/basic51, vibe26b/zdb,
thewaxmango/collegejunior and uravt/Version41. Krak9n/robot,
seru9/TOTAL_AGGRO, Camleaf/mk1 and the official example player leave the routine
panel based on their low historical strength. This is an explicitly authorized
panel revision, not crash-based retirement; their evidence and source pins remain.

Development maps are tiny, DefaultSmall, DefaultMedium, DefaultLarge, closeup,
knifefight, cheesefarm and wallsofparadis. They deliberately cover more sizes
and layouts; selection is not based on which maps the learner wins. Eight maps
are still a limited development sample, not an all-map performance estimate.
Each checkpoint plays all eight opponents, seeds 0 and 2026, in both seats:
256 scheduled games. Benchmark code and trajectories never enter training.
All four original training-excluded maps remain excluded from gradients.

The fixed baseline is warmstart cp2, pinned in configs/improvement-v3.json.
It needs a NEW complete evaluation on this schedule. Old smoke measurements
and the stopped alphabetical zdb subset cannot supply the baseline.

A development improvement requires complete valid comparable panels, a mean
score gain of at least 0.05 (win=1/draw=.5/loss=0), a paired 95% bootstrap lower
bound above zero, and nonnegative aggregate score gain against zdb. Bootstrap
whole maps, keeping both seeds, both seats and all opponents together (10,000
resamples, fixed seed 602113). Equal schedule counts give equal map/opponent
weights. With only eight map clusters the interval remains a development
heuristic, not strong population-wide statistical assurance. Report every
opponent-map W/L/D count, losses and uncertainty; no required individual win
against bestsofar or basic51, and no 100% target.

Passing is a review milestone, not automatic promotion. Freeze the candidate
before a separately preregistered fresh-seed confirmation against the same
baseline; apply the same improvement rule. Confirmation seeds must be unused
and frozen before outcomes, cannot enter training/tuning, and a failed completed
confirmation consumes them. Preregister that confirmation version before running
it. Periodic broader-map checks remain useful, but winning every map is no longer
the stopping condition. Do not silently change this version's thresholds after
seeing results. Future goals can advance the baseline only after verified gains,
while preserving the original cp2 comparison.

Use scripts/evaluate_development.py for bounded resumable panels and
scripts/compare_development.py for comparisons. Do not resume the user-stopped
all-map zdb run. V8 is an unfinished unlaunched draft: reconcile its evaluation
and review rules with this amendment before freezing or launching it. Preserve
all completed V4–V7 evidence and identities.

The top-level README must show current opponent-by-map win/loss/draw measurements
and completion coverage prominently, refreshing at each published evaluation
boundary. Label partial panels and pending baselines honestly. Tiny-only history
belongs in a historical section, not the headline. Always keep losses visible.
