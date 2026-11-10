# Rules and map provenance

The rule text (`specs.txt`) and engine-checked digest (`engine-checked-digest.md`)
were downloaded through `terryvanbelle/battlecode26b-vibe`, immutable source
commit `f4d6ec971b13a20f7f38fb56d249f62c0251ed05`. The digest retains its original
engine source citations; its `reference/bc26-engine-src` corresponds to this
project's downloaded `vendor/engine/src`.

- Official game repository: https://github.com/battlecode/battlecode26
- Official release infrastructure: https://releases.battlecode.org/maven/org/battlecode/battlecode26-java/1.2.6/
- Pinned engine: Battlecode 2026 Java 1.2.6. The setup script downloads binary,
  sources and API documentation and validates all three SHA-256 hashes.
- Local API documentation: `vendor/engine/battlecode26-java-1.2.6-javadoc.jar`.
- The 74 `.map26` maps are copied unchanged from the same immutable source
  repository commit; `maps/SHA256SUMS` pins every map. The 43 maps embedded in
  the official engine JAR are byte-identical to the corresponding copied maps.
  The additional 31 are source-repository map corpus additions, not claimed
  to be bundled official maps.

The Java engine remains the authority for movement, combat, cheese, cats,
spawn timing, round transitions and scoring. Local research deviations and
the initial action abstraction are documented in `docs/engine.md`.
