#!/usr/bin/env bash
# Run alpha0.student.Game with the student classpath. Extra JVM flags via JAVA_FLAGS.
cd "$(dirname "$0")/.."
# glibc malloc opens up to 8 arenas per core as the JVM's ~175 threads contend; capping them at 2 cut peak game RSS
# 1104 -> 653 MB (pipes) and 859 -> 720 MB (jail) at -Xmx1200m, same traj_hash, same CPU (VM, 2026-11-15).
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
exec vendor/jdk/bin/java ${JAVA_FLAGS:--Xmx2g} \
  --add-opens=java.base/jdk.internal.misc=ALL-UNNAMED --add-opens=java.base/jdk.internal.math=ALL-UNNAMED \
  --add-opens=java.base/jdk.internal.util=ALL-UNNAMED --add-opens=java.base/jdk.internal.access=ALL-UNNAMED \
  --add-opens=java.base/sun.security.action=ALL-UNNAMED \
  -cp build/student/classes:build/java:vendor/engine/battlecode26-java-1.2.6.jar alpha0.student.Game "$@"
