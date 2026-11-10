#!/usr/bin/env python3
"""Download the twelve immutable benchmark packages and verify every file.

Third-party source stays a local dependency under benchmarks/bots (gitignored).
Only manifest-listed bytes are copied; archives are never extracted wholesale.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid(bot: dict, root: Path = ROOT) -> bool:
    return all((root / f['path']).is_file() and digest((root / f['path']).read_bytes()) == f['sha256'] for f in bot['files'])


def fetch(bot: dict, root: Path = ROOT) -> None:
    if valid(bot, root):
        print(f"verified {bot['id']} ({bot['source_commit'][:12]})", flush=True)
        return
    request = urllib.request.Request(bot['archive_url'], headers={'User-Agent': 'battlecode26-alpha0-benchmark-fetch/1'})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    outputs = []
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        prefixes = {PurePosixPath(m.name).parts[0] for m in members if PurePosixPath(m.name).parts}
        if len(prefixes) != 1:
            raise ValueError('Expected one repository root in the upstream archive')
        prefix = prefixes.pop()
        for file in bot['files']:
            member = archive.getmember(prefix + '/' + file['upstream_path'])
            if not member.isfile():
                raise ValueError(f"Not a regular upstream file: {member.name}")
            content = archive.extractfile(member).read()
            if digest(content) != file['sha256']:
                raise ValueError(f"SHA256 mismatch: {bot['id']} {file['upstream_path']}")
            target = (root / file['path']).resolve()
            target.relative_to((root / 'benchmarks/bots').resolve())
            outputs.append((target, content))
    # Nothing is published until every required upstream file passes its hash.
    for target, content in outputs:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, target)
    if not valid(bot, root):
        raise RuntimeError(f"Post-download verification failed for {bot['id']}")
    print(f"downloaded and verified {bot['id']} ({bot['source_commit'][:12]})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/benchmarks.json')
    parser.add_argument('--bot', action='append', help='Fetch only this exact manifest ID; may repeat')
    parser.add_argument('--root', type=Path, default=ROOT, help='Destination root (useful for download verification)')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    bots = manifest['bots']
    if args.bot:
        unknown = set(args.bot) - {bot['id'] for bot in bots}
        if unknown:
            parser.error('Unknown bot IDs: ' + ', '.join(sorted(unknown)))
        bots = [bot for bot in bots if bot['id'] in args.bot]
    for bot in bots:
        fetch(bot, args.root.resolve())


if __name__ == '__main__':
    main()
