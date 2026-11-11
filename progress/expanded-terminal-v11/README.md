# V11 expanded terminal-head learning experiment

Active bounded research cycle. Frozen identity `94d03e8b4bfa3582dd2129aa1723c541e8b48fbe5c5c0112417d5e732284e200`. All64 training roots/1024 anchors and32 disjoint validation roots were audited; all96 physical roots reconstructed exactly before launch.

![Collection coverage](coverage.png)

[Derived progress](derived-progress.json) · [Frozen protocol](../../docs/expanded-terminal-v11.md) · [Identity](identity.json) · [Input audit](../validation-v11/reviews/independent-input-audit.json)

Sequence:64 PUCT comparators,2974 selection continuations, two4096-update head fits, independently shared validation continuations, then768 fresh development-v4 benchmark games. The terminal head is the fixed candidate; no post-hoc arm choice. No incomplete data fitting. Every job has immutable output hashes; failed/unclosed jobs require review.

The chart measures work completed, not strength. Latest completed benchmark remains V8 terminal81W175L/256; original cp2 baseline83W173L. No V11 playing-strength result yet. At complete valid180/256 wins, prioritize tournament compatibility.

Review snapshot (2026-11-11 08:45 UTC):524/2974 target continuations valid; selection remains active. This timestamped coverage is not a final integrity certification. The separate dashboard adapter `scripts/publish_v11_development_views.py` is ready for eventual canonical benchmark cells; it does not launch games or change the pinned runtime. At benchmark review boundaries, run it, then `scripts/report_development.py progress/development-v4/v11-terminal_head --manifest configs/benchmarks-development-v4.json` once candidate cells exist.
