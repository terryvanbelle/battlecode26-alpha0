#!/usr/bin/env bash
# Run alpha0.student.Game with the student classpath. Extra JVM flags via JAVA_FLAGS.
cd "$(dirname "$0")/.."
exec vendor/jdk/bin/java ${JAVA_FLAGS:--Xmx2g} \
  --add-opens=java.base/jdk.internal.misc=ALL-UNNAMED --add-opens=java.base/jdk.internal.math=ALL-UNNAMED \
  --add-opens=java.base/jdk.internal.util=ALL-UNNAMED --add-opens=java.base/jdk.internal.access=ALL-UNNAMED \
  --add-opens=java.base/sun.security.action=ALL-UNNAMED \
  -cp build/student/classes:build/java:vendor/engine/battlecode26-java-1.2.6.jar alpha0.student.Game "$@"
