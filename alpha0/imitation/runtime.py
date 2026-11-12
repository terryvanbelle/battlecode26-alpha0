"""Audit-only JVM launch pins; not an official-limit or teacher approval."""
import os
from pathlib import Path
from .provenance import file_hash, tree_hashes
from .trace import digest

# Reject ambient JVM/native loader injection rather than silently ignoring it.
INJECTION = ('JAVA_TOOL_OPTIONS', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS',
             'LD_PRELOAD', 'LD_LIBRARY_PATH', 'JAVA_OPTIONS')

def clean_environment(environment=None):
    env = dict(os.environ if environment is None else environment)
    if any(env.get(key) for key in INJECTION):
        raise ValueError('ambient JVM/native injection is not permitted')
    return env

def runtime_spec(jdk, flags):
    jdk = Path(jdk).resolve()
    if not isinstance(flags, (list, tuple)) or any(not isinstance(x, str) or not x.startswith('--add-opens=java.base/') or not x.endswith('=ALL-UNNAMED') for x in flags):
        raise ValueError('unsupported diagnostic JVM flags')
    core = {'schema': 'diagnostic-jvm-runtime-v1', 'java': str(jdk/'bin/java'),
            'flags': list(flags), 'release_sha256': file_hash(jdk/'release'),
            'java_sha256': file_hash(jdk/'bin/java'),
            'lib': tree_hashes(jdk/'lib'), 'conf': tree_hashes(jdk/'conf'),
            'scope': 'java executable, release, complete lib and conf trees; system libraries excluded',
            'training_approved': False}
    return {**core, 'runtime_id': digest(core)}

def validate_runtime(spec, jdk, flags):
    if runtime_spec(jdk, flags) != spec:
        raise ValueError('JVM runtime identity changed')

def launch_command(spec, *, classpath, arguments):
    core = {k: v for k, v in spec.items() if k != 'runtime_id'}
    if digest(core) != spec.get('runtime_id'):
        raise ValueError('invalid JVM runtime identity')
    if not classpath or any(not isinstance(x, str) for x in arguments):
        raise ValueError('invalid launch arguments')
    return [spec['java'], *spec['flags'], '-cp', classpath, 'WorldJournalTest', *arguments]
