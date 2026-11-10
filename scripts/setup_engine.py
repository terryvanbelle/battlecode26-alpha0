#!/usr/bin/env python3
"""Fetch hash-pinned official engine/JDK, apply the reviewed local patch, compile bridge."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ENGINE_VERSION = "1.2.6"
ENGINE_BASE = f"https://releases.battlecode.org/maven/org/battlecode/battlecode26-java/{ENGINE_VERSION}"
ENGINE_HASHES = {
    "battlecode26-java-1.2.6.jar": "ea722532f6d523241ff4f708078563de2a010a39d7d524c66ae1890ab85c04ec",
    "battlecode26-java-1.2.6-sources.jar": "b97b5f10525aaac9565eab0967bce520b48add1e90eae8c47b4628d7274fe4e9",
    "battlecode26-java-1.2.6-javadoc.jar": "afc964b56996948fde0b7b80867ceb431850611b9e1e339f5b7a54f2c3f4bf25",
}
JDK_URL = "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.13%2B9/OpenJDK21U-jdk_x64_linux_hotspot_21.0.13_9.tar.gz"
JDK_HASH = "d7f5d5cf865a2902b70ff549b97586a7d3f234e016f041ab447ea740c7d66d1c"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(url, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or digest(target) != expected:
        tmp = target.with_suffix(target.suffix + ".download")
        with urllib.request.urlopen(url, timeout=120) as source, tmp.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        if digest(tmp) != expected:
            tmp.unlink()
            raise RuntimeError(f"Checksum mismatch downloading {url}")
        tmp.replace(target)
    print(f"verified {target.relative_to(ROOT)}", flush=True)


def main():
    engine = ROOT / "vendor/engine"
    for name, sha in ENGINE_HASHES.items():
        fetch(f"{ENGINE_BASE}/{name}", engine / name, sha)
    source = engine / "src"
    source.mkdir(exist_ok=True)
    with zipfile.ZipFile(engine / "battlecode26-java-1.2.6-sources.jar") as archive:
        archive.extractall(source)
    # Pin a user-local JDK. JAVA_HOME is supported for other architectures/platforms.
    jdk = Path(os.environ["JAVA_HOME"]) if os.environ.get("JAVA_HOME") else ROOT / "vendor/jdk"
    if not (jdk / "bin/javac").exists():
        if os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME must contain a Java 21 JDK with bin/javac")
        archive = ROOT / "vendor/downloads/jdk21.tar.gz"
        fetch(JDK_URL, archive, JDK_HASH)
        unpack = ROOT / "vendor/downloads/jdk-unpack"
        unpack.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(unpack, filter="data")
        extracted = next(p for p in unpack.iterdir() if (p / "bin/javac").exists())
        if jdk.exists():
            shutil.rmtree(jdk)
        shutil.move(str(extracted), str(jdk))
    patched = ROOT / "build/patched"
    patched.mkdir(parents=True, exist_ok=True)
    patched_names = ["battlecode/world/GameWorld.java", "battlecode/world/InternalRobot.java", "battlecode/world/control/PlayerControlProvider.java"]
    for name in patched_names:
        dest = patched / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(ROOT / "patches/local-research.patch")], cwd=patched, check=True)
    output = ROOT / "build/java"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(jdk / "bin/javac"), "--release", "21", "-cp", str(engine / "battlecode26-java-1.2.6.jar"), "-d", str(output), *[str(patched / name) for name in patched_names], *map(str, sorted((ROOT / "java").rglob("*.java")))], check=True)
    manifest = {"version": ENGINE_VERSION, "engine_sha256": ENGINE_HASHES, "patch_sha256": digest(ROOT / "patches/local-research.patch"), "jdk_url": JDK_URL, "jdk_sha256": JDK_HASH, "jdk_used": str(jdk), "java_sources": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "java").rglob("*.java"))}}
    (ROOT / "build/engine-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Engine ready. Runtime classpath: build/java:vendor/engine/battlecode26-java-1.2.6.jar")


if __name__ == "__main__":
    main()
