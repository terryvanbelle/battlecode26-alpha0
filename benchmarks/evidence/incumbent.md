# Accepted incumbent selection

Source repository: https://github.com/terryvanbelle/battlecode26b-vibe

Checked immutable commit: `fc044c24c37d89d2581562e23439a2571bb2b300`.

Selected package: `bots/zdb` (`zdb.RobotPlayer`), manifest ID `vibe26b_zdb`.
Java source-tree SHA-256: `0c69409ceaad4e69e9f4763c547e2c5f83d9d8c05cfd98245a6a20240fcf485a`.

## Latest handoff state

Verbatim from `HANDOFF.md`, lines 99–108 at the checked commit:

```text
## State (2026-11-08 17:54 PST; TRAINING_LOG.md has the history, older states are in this file's git history)
- **Now (18:02 11-09):**
  - Incumbent still **zdb** (resubmitted #332). zgj rejected at replica block 2 (pooled t 0.83: +8 set 2, -3 set 3,
    Finals where_are_you A and B as zgh). zgo, zgp dropped; zgq/zgr/zgs withdrawn with zgj.
  - **zgt** (zdb + BREAKOUT_TTL 150, pocket escape): screen zd478 (s7) running, chain cq181 (seed 89, 5 pocket maps;
    `chsc.py ~/diag26b/cq181 zdb zgt`; clause on pickupAmt 1.05x and wins).
  - **zgu** (zdb + ATK_STUCK alone, no trail): screen zd479 queued, chain cq182 after cq181 (seed 90, cq177 cells;
    transfer 1.1x and wins).
  - Next for the replica: whichever passes; set 3 first (`paired --dir /dev/shm/rp 21016-21023,21036-21043 IDS`).
  - Next fresh seed: 91. Next screen id: zd480. Next chain: cq183. Next free name: zgv.
```

## Recorded acceptance

Verbatim `progress/attempts.csv` outcome for `zdb`:

ACCEPTED 2026-11-08 09:05 PST over z3x: replica #293 blocks 1-2 pooled 96 cells, 92 alike, D 4 (+4 -0), net +4, t 2.00 (block 1 set 4, 20817-20824, 20842-20845, 20848-20851: D 2 +AdamTan12_Finals averyfineline A and B, t 1.41; block 2 set 2, 20984-20999: D 2 +HugoIngelsson_Xzibit canyoudig A and B, t 1.41); transfer per round 0.999x and 1.002x the twin (floor 0.95x); won 38 to 36 and 35 to 33. Earlier: PENDING 08:21: falsifiers 1 and 2 passed (cq98; s7 zd343 wins 31 to 30, xfer/round 1.011x; s6 zd345 D 0, t 0.00, xfer/round 1.012x; 0 overruns); replica #293 binds. Earlier: PENDING 07:20: falsifier 1 passed (cq98: wins 15 vs 15, 0 overruns/exceptions, matched-round picks 0.928x z -0.88); screens zd343 (s7) and zd345 (s6) and replica #293 still bind. Earlier: SUBMITTED 06:46 to the replica as #293 (block 1 on set 4: 16 rows queued); PENDING chain cq98 (seed 25; small maps vs abrk/araid; wins >= z3x, matched-round picks z > -2) and seed-1 screens zd343 (s7, control zd342) and zd345 (s6, control zd344)

## Working source comparison

Both packages contain 22 Java files. Package-normalized SHA-256 trees differ:

- `bots/zdb`: `39f7c225576fab33a01fbbe7bd6faebe6b47d56301923337acb951d764badc6d`
- `bots/ratbot`: `19fc3bcf8b692476cc0b4228d319fba9b383eaea1bead4daca87a64f00700c87`

The frozen accepted `zdb` snapshot is selected. Its Java bytes also match the
`zdb` snapshot in historical reference f4d6ec971b13a20f7f38fb56d249f62c0251ed05.
No candidate is selected by maximum fitted Elo, and pending or rejected
experiments do not replace the accepted incumbent.
