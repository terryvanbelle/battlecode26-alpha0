# V6 training-only head-fit diagnosis

The terminal head learned substantial probability shifts, but its fixed 128-update fit did not transfer any of the nine changed selected actions into a greedy decision. The evidence supports both an unfinished fit and substantial objective conflict; it does not justify assuming that extra steps alone will solve the problem.

This diagnosis reads only the 32 training roots, 512 training anchors, cp2, the two saved heads, and their training manifests. It uses no validation feature arrays, actions, outcomes, or benchmark evidence. Training metadata contains validation hash-guard strings, which are not used for the fit/capacity calculations. No optimizer step, parameter mutation, engine run, or candidate was produced. Gradients and one hypothetical AdamW direction were evaluated algebraically.

## Probability movement

All nine changed labels gained probability. Their initial logit deficits closed by 44–89%, but the selected action remained below cp2's action. The unchanged 23 training roots also retain their cp2 greedy actions. The full JSON records both actions' probabilities, all three heads' predictions, per-root target cross-entropy, and feature/gradient diagnostics.

| Training root | cp2 action → terminal label | Selected-action probability | Selected minus cp2 logit margin | Initial deficit closed |
|---|---|---|---|---|
| 001 | 27 → 24 | 0.83% → 18.80% | -4.56 → -1.02 | 77.7% |
| 004 | 21 → 32 | 1.34% → 19.46% | -4.07 → -1.15 | 71.7% |
| 007 | 27 → 29 | 1.07% → 19.90% | -4.24 → -0.97 | 77.2% |
| 008 | 21 → 4 | 1.45% → 10.13% | -3.95 → -1.73 | 56.3% |
| 010 | 21 → 6 | 1.85% → 6.78% | -3.57 → -2.00 | 44.0% |
| 014 | 21 → 31 | 0.64% → 11.40% | -4.82 → -1.67 | 65.3% |
| 016 | 21 → 28 | 0.90% → 7.23% | -4.52 → -2.31 | 49.0% |
| 026 | 20 → 6 | 0.95% → 24.66% | -4.56 → -0.99 | 78.3% |
| 027 | 26 → 12 | 0.19% → 30.35% | -6.11 → -0.67 | 89.0% |

The mixed-target cross-entropy falls from **1.49836 to 0.91543**. On the changed nine roots it falls from **3.82439 to 1.77418**; on the other 23 it moves from **0.58817 to 0.57940**. The average mixed-target entropy is 0.54450, so final cross-entropy still exceeds its unconstrained optimum by 0.37093. These are training-set measures, not evidence of better play or of the two-continuation labels being correct.

## Objective conflict and representation

The expected sampler objective weights roots uniformly: each of the 16 map/role/team groups has two roots. For anchors it weights each source game equally, then each team equally, then each anchor within that group equally. The report also evaluates the actual frozen 128-batch sampling frequencies. For comparability, gradients at all three models use the same terminal-selected target distribution; the PUCT-head diagnostic gradient is not its own fitted objective.

| Final terminal-head quantity | Value |
|---|---:|
| Anchor KL, plain storage mean | 0.050669 |
| Anchor KL, expected game/team weighting | 0.050726 |
| Target-CE gradient norm | 2.10687 |
| Balanced anchor-KL gradient norm | 1.86065 |
| CE/KL gradient cosine | −0.64464 |
| Combined gradient norm | 1.68724 |
| Changed-nine/unchanged-23 CE gradient cosine | −0.91172 |

At cp2, anchor KL is approximately zero and its gradient norm is 2.0×10⁻⁷, as expected. At the fitted head, the KL gradient has 88% of the CE gradient's norm and opposes it. The combined norm is 57.5% below the sum of the two norms. The changed-nine and unchanged-23 CE contributions also oppose one another strongly. This is shared-feature interference as well as the anchor constraint.

The negative balanced-KL gradient lowers every changed root's selected-versus-old margin. The negative total Euclidean gradient raises five of nine margins and lowers four; the CE component itself lowers two king-root margins. A hypothetical next AdamW displacement computed from the saved moments and the expected full gradient also raises five and lowers four, with different membership. It is not applied and is not a prediction of the next stochastic minibatch update. Neither gradient calculation establishes a stationary optimum or a simple steps-to-flip estimate.

The augmented 32×129 root feature matrix has full row rank 32 at relative tolerance 10⁻⁶, condition number **235.42**, and no exact duplicate hidden rows. Thus arbitrary logits on these 32 roots are representable by a linear head in exact arithmetic; there is no hard interpolation barrier here. This says nothing about generalization or the weight movement/anchor cost needed to interpolate. The closest conflicting-label roots have cosine similarity about 0.96–1.00, and nearby anchors often strongly prefer the old action. The JSON includes mutual-legality flags so similarity is not mistaken for an incompatible masked classification problem.

The current evidence cannot separate optimization budget from the regularized optimum. It does show why blindly repeating the same sparse training cycle is less informative than a controlled cached-data test.

## Proposed precommitted mechanism test: cached-head-constraint-v7

This is a proposed research diagnostic, not an activated experiment or a promotion plan. After the completed v6 cycle is reviewed, freeze a separate config and source identity for two policy-head-only fits from the original cp2. Use the exact v6 training dataset, selected-action targets, `.75 one-hot + .25 legal-normalized cp2 prior`, AdamW settings, seed 203101, and balanced sampler. Extend the same deterministic batch stream to 2,048 updates. The only between-arm difference is anchor-KL coefficient **1 versus 0**. Preserve cp2, v6 candidates, and all prior evidence.

Record both arms at the fixed 128, 512, and 2,048-update checkpoints. The KL=1 128-update result must reproduce the existing v6 terminal-head weights and optimizer state exactly before interpreting later results. Use one Torch thread, no Java, a 15-minute CPU ceiling, immutable checkpoint/optimizer/RNG records, and the existing global/experiment STOP conventions. Interrupted fits cannot become completed evidence.

At every fixed checkpoint report selected-action agreement on all 32 roots, separately on the changed nine and retained 23, mixed-target CE, correctly weighted anchor KL, anchor greedy changes, parameter displacement, and CE/KL gradient alignment. Do not inspect validation or benchmark examples/outcomes to choose steps, coefficients, labels, or a preferred checkpoint.

Precommit the mechanism classification: if KL=1 reaches all 32 selected greedy actions only after 128 updates, there is direct budget evidence for this training set; if KL=0 reaches all 32 by 2,048 while KL=1 does not, there is evidence that this anchoring recipe impedes transfer within the tested budget; if neither does, investigate optimization/conditioning before changing the feature trunk. Report partial agreement and all checkpoint curves even when neither classification triggers. None of these results establishes strength, and no diagnostic head is automatically promoted or sent back to the existing validation set. Any learning follow-up requires a recorded research review and a separately precommitted fresh validation plan.

## Reproduction

Run the archived standalone script from a checkout with the verified v6 training dataset and checkpoint artifacts restored:

```bash
.venv/bin/python progress/terminal-target-head-v6/source/diagnostics/diagnose_terminal_v6_training.py \
  --output /tmp/v6-training-underfit.json
```

[Machine-readable training diagnosis](../progress/terminal-target-head-v6/training-underfit.json) · [Archived diagnostic source](../progress/terminal-target-head-v6/source/diagnostics/diagnose_terminal_v6_training.py) · [Independent fit audit](../progress/terminal-target-head-v6/independent-fit-audit.json)
