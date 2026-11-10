"""Verify pinned benchmark acquisition without depending on network availability."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('benchmark_fetch', Path(__file__).resolve().parents[1] / 'scripts/fetch_benchmarks.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def fixture(content=b'package demo;'):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        info = tarfile.TarInfo('repository-commit/src/demo/RobotPlayer.java')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    bot = {'id': 'demo', 'source_commit': 'abc123', 'archive_url': 'https://example.test/pinned.tar.gz',
           'files': [{'path': 'benchmarks/bots/demo/RobotPlayer.java',
                      'upstream_path': 'src/demo/RobotPlayer.java',
                      'sha256': hashlib.sha256(content).hexdigest()}]}
    return bot, stream.getvalue()


class FetchBenchmarkTest(unittest.TestCase):
    def test_download_verifies_and_preserves_bytes(self):
        bot, archive = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(fetcher.urllib.request, 'urlopen', return_value=io.BytesIO(archive)):
                fetcher.fetch(bot, root)
            self.assertEqual((root / bot['files'][0]['path']).read_bytes(), b'package demo;')
            self.assertTrue(fetcher.valid(bot, root))

    def test_hash_mismatch_does_not_publish(self):
        bot, archive = fixture()
        bot['files'][0]['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(fetcher.urllib.request, 'urlopen', return_value=io.BytesIO(archive)):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    fetcher.fetch(bot, root)
            self.assertFalse((root / bot['files'][0]['path']).exists())

    def test_verified_download_needs_no_network(self):
        bot, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / bot['files'][0]['path']
            target.parent.mkdir(parents=True)
            target.write_bytes(b'package demo;')
            with mock.patch.object(fetcher.urllib.request, 'urlopen', side_effect=AssertionError('unexpected network')):
                fetcher.fetch(bot, root)


if __name__ == '__main__':
    unittest.main()
