> Live benchmark inspection. Training and first-action validation are complete; benchmark results remain incomplete. Not a final experiment audit or recovery export.

# V8 development transfer

Valid benchmark games: 190/768. No automatic promotion.

![Opponent-map W/L/D](comparison.png)

![Coverage](coverage.png)

The terminal head is the preregistered candidate; the PUCT head is a matched control. Complete-panel improvement uses ≥5 percentage points score gain, positive paired-map bootstrap lower bound and aggregate zdb nonregression. Fresh-seed confirmation is required for promotion. First-action validation is a separately reported mechanism diagnostic, not a substitute for benchmark improvement.

## Comparisons

```json
{}
```

## Independent first-action validation

```json
{
  "complete": true,
  "mean_delta_vs_puct": 0.0,
  "mean_delta_vs_cp2": 0.0,
  "criteria_pass": false
}
```

## Research decision

```json
{
  "complete": false,
  "criteria_pass": false,
  "integrity_verified": false,
  "review_for_expansion": false,
  "automatic_promotion": false,
  "all_map_victory": false,
  "scope": "Precommitted research review rule; no model promotion or all-map victory claim."
}
```

## Fitting diagnostics

![Training histories](fitting.png)

Both heads match all 32 training labels. This is fitting evidence, not playing-strength improvement.
