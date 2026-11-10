# Initial CPU experiment artifacts

Four checkpoints cover the untrained baseline (`0000`) and training iterations
`0001`–`0003`. Each stores the network, configuration, optimizer and RNG metadata.
The final model has 49,242 parameters. `manifest.json` records source provenance,
engine and dependency versions, file hashes and six game summaries.

The six `games/` files contain exact seeds, macro-action histories and terminal
engine states. The final game's 9,177-decision transcript was replay-verified.
Results and limitations are in [the experiment report](../../docs/initial-results.md).

After repository setup, evaluate without modifying published measurement logs:

```bash
.venv/bin/python -m alpha0.train evaluate artifacts/initial/checkpoint-0003.pt \
  --suite smoke --progress-dir runs/recheck/progress
.venv/bin/python scripts/replay_game.py artifacts/initial/games/iter-0003-game-001.json
```

Training continuation also needs the matching replay buffer from the original
local `runs/initial/` directory. Replay buffers are not included in this export.
