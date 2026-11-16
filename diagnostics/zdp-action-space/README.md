# zdp action-space investigation (November 16, 2026)

Evidence for [plan sections 16–17](../../docs/rl-es-plan.md#17-what-the-map-and-the-measurements-show-two-tests-set-before-their-games-november-16-2026):
why every optimizer stalled on zdp's decision points, and the inputs to the action-space redesign.

- `design-brief.md`: the brief given to the design panel. It covers the owner's constraints, the measured evidence,
  the requirements R1–R8 and the deliverable format.
- `maps/map-*.json`: seven read-only code maps of the decision points. For each option they record what its executor
  does, when it is offered, why it falls back, its context dependence, its features and its coupling to other points,
  plus a generalization proposal. `map-hidden-strategy.json` lists the strategy hard-coded outside any point.
  `maps/summary-{1,2,3}.txt` are plain-text renderings.
- `maps/verify-*.json`: adversarial verifications of the maps (claims checked, corrections, missed facts, concerns about
  the proposals). Where a verification and a map disagree, the verification wins.
- `analysis/`: read-only analysis scripts, run on the Windows box from the sandbox repo root, with their outputs.
  - `devfb.py` → `devfb-cpi-x0.json`: deviation and fallback rates per point, from the cpi-x0 probe records.
  - `fisher.py` → `fisher-cpi-x0.json`: the per-record policy-gradient signal (Fisher trace in logit space) of the
    floor-mixed policy that rl_train differentiates, at floor 0.005, at cpi-x0's floors and with logits ÷ 3.
  - `zero_const.py`: zeroes the first-layer weights of input columns that were constant in the imitation data, folding
    their contribution into the bias. It built the zdpz test package.
  - `constq`, `throwq`, `decomp`, `teachq`: untrained-input diagnosis at BABY_ACT_PRE. A column that was constant in the
    imitation data (the team rat-trap count) makes the net throw while carrying nothing.
  - `kpts*`, `probe3`, `cpi-x0-summary.json`, `rl-m2-last.txt`: king-point and record statistics used by the maps.
