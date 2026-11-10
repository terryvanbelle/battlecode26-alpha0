"""Independent v7 protocol/recovery checks; no real-data SGD or engine runs."""
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from alpha0 import constraint_v7_common as C
from alpha0 import constraint_v7 as S
from alpha0 import constraint_v7_fit as F


class FixedProtocolProof(unittest.TestCase):
    def test_actual_config_is_accepted_and_old_runtime_bytes_unchanged(self):
        config = C.load_config()
        self.assertEqual(config['snapshots'], [128, 512, 2048])
        self.assertEqual(config['total_wall_budget_seconds'], 900)
        self.assertEqual(config['arms'], [{'id': 'kl_1', 'anchor_kl_weight': 1.0},
                                          {'id': 'kl_0', 'anchor_kl_weight': 0.0}])
        # These identity files contain only configuration and source hashes;
        # no validation observations, actions or outcomes are opened.
        for version in (5, 6):
            identity = json.loads((C.ROOT / f'progress/terminal-target-head-v{version}/identity.json').read_text())
            for relative, expected in identity['sources'].items():
                self.assertEqual(hashlib.sha256((C.ROOT / relative).read_bytes()).hexdigest(),
                                 expected, (version, relative))

    def test_extended_stream_reproduces_pinned_training_only_prefix(self):
        config = C.load_config()
        dataset = C.ROOT / config['dataset']['path']
        reference = C.ROOT / config['reference_batches']['path']
        if not dataset.exists() or not reference.exists():
            self.skipTest('Restore the pinned v6 training artifacts for this integration proof')
        self.assertEqual(C.sha256(dataset), config['dataset']['sha256'])
        self.assertEqual(C.sha256(reference), config['reference_batches']['sha256'])
        with np.load(dataset, allow_pickle=False) as data:
            settings = {**config['fit'], 'updates': 2048, 'anchor_kl_weight': 1}
            extended = F.make_batch_schedule(data['root_strata'], data['anchor_game'],
                                            data['anchor_player'], settings)
            prefix = F.make_batch_schedule(data['root_strata'], data['anchor_game'],
                                          data['anchor_player'], {**settings, 'updates': 128})
        with np.load(reference, allow_pickle=False) as saved:
            for key in ('root', 'anchor'):
                self.assertEqual(extended[key].shape, (2048, 32))
                np.testing.assert_array_equal(extended[key][:128], saved[key])
                np.testing.assert_array_equal(prefix[key], saved[key])
        self.assertNotEqual(extended['rng_state'], prefix['rng_state'])

    def test_anchor_metrics_equalize_games_and_teams_despite_storage_quotas(self):
        games = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        teams = np.array([0, 0, 1, 0, 1, 1, 1, 1])
        weight = F.anchor_weights({'anchor_game': games, 'anchor_player': teams}).numpy()
        for game in (0, 1):
            for team in (0, 1):
                self.assertAlmostEqual(float(weight[(games == game) & (teams == team)].sum()), .25)
        self.assertAlmostEqual(float(weight.sum()), 1.)
        with self.assertRaises(ValueError):
            F.anchor_weights({'anchor_game': np.array([0, 0]), 'anchor_player': np.array([0, 0])})


class ClassificationProof(unittest.TestCase):
    @staticmethod
    def rows(kl1=(23, 31, 31), kl0=(23, 31, 31)):
        return {arm: {str(step): {'root_count': 32, 'root_selected_action_matches': count}
                      for step, count in zip((128, 512, 2048), counts)}
                for arm, counts in (('kl_1', kl1), ('kl_0', kl0))}

    def test_budget_evidence_survives_later_regression_without_overlap(self):
        result = F.classify_results(self.rows((23, 32, 31), (23, 32, 32)))
        self.assertEqual(result['classification'], 'budget')
        self.assertTrue(result['budget_evidence'])
        self.assertFalse(result['anchoring_evidence'])
        self.assertEqual(result['first_full_match_step']['kl_1'], 512)
        self.assertFalse(result['automatic_promotion'])

    def test_anchoring_evidence_uses_any_fixed_snapshot_not_selected_endpoint(self):
        result = F.classify_results(self.rows((23, 31, 31), (32, 31, 30)))
        self.assertEqual(result['classification'], 'anchoring')
        self.assertEqual(result['first_full_match_step']['kl_0'], 128)
        self.assertFalse(result['budget_evidence'])
        self.assertEqual(F.classify_results(self.rows())['classification'], 'neither')

    def test_missing_extra_malformed_or_impossible_control_cannot_classify(self):
        invalid = []
        missing = self.rows(); del missing['kl_0']['2048']; invalid.append(missing)
        extra = self.rows(); extra['kl_1']['1024'] = extra['kl_1']['512']; invalid.append(extra)
        for count in (True, 32.0, -1, 33):
            row = self.rows(); row['kl_0']['512']['root_selected_action_matches'] = count; invalid.append(row)
        invalid.append(self.rows((32, 32, 32)))
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                F.classify_results(row)


class ExclusionAndRecoveryProof(unittest.TestCase):
    def test_all_five_locks_and_stop_files_apply(self):
        expected = ('campaign', 'warmstart-v3', 'terminal-target-head-v5',
                    'terminal-target-head-v6', 'cached-head-constraint-v7')
        self.assertEqual(tuple(p.parent.name for p in C.LOCKS), expected)
        self.assertEqual(tuple(p.parent.name for p in C.STOPS), expected)
        with tempfile.TemporaryDirectory() as tmp:
            locks = tuple(Path(tmp) / name / 'lock' for name in expected)
            stops = tuple(Path(tmp) / name / 'STOP' for name in expected)
            with patch.object(C, 'LOCKS', locks), patch.object(C, 'STOPS', stops):
                for held in locks:
                    held.parent.mkdir(parents=True, exist_ok=True)
                    with held.open('a+') as stream:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        with self.assertRaises(BlockingIOError), S.hold_locks():
                            self.fail('held lock was ignored')
                with S.hold_locks():
                    for held in locks:
                        with held.open('a+') as stream, self.assertRaises(BlockingIOError):
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                C.check_stop(float('inf'))
                for stop in stops:
                    stop.touch()
                    with self.assertRaises(C.Interrupted): C.check_stop(float('inf'))
                    stop.unlink()
                with self.assertRaises(C.Interrupted): C.check_stop(time.monotonic()-1)

    def test_matching_actor_with_unreadable_cwd_blocks_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp); actor = proc / '999999'; actor.mkdir()
            (actor / 'cmdline').write_bytes(b'/venv/bin/python\0-m\0alpha0.constraint_v7\0')
            original = Path.resolve
            def denied(path, *args, **kwargs):
                if path == actor / 'cwd': raise PermissionError('namespace boundary')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'resolve', denied):
                found = S.active_actors(proc)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]['reason'], 'unreadable_cwd')
            for module in ('alpha0.evaluate', 'alpha0.victory'):
                (actor / 'cmdline').write_bytes(f'/venv/bin/python\0-m\0{module}\0'.encode())
                with patch.object(Path, 'resolve', denied):
                    self.assertEqual(len(S.active_actors(proc)), 1)

    @staticmethod
    def journal(directory, number, reserved, elapsed=None):
        start_time = datetime(2026, 11, 10, tzinfo=timezone.utc) + timedelta(hours=number)
        start = {'identity_sha256': 'identity', 'invocation': number,
                 'started_at': start_time.isoformat(), 'reserved_wall_seconds': reserved}
        (directory / f'{number:03d}-start.json').write_text(json.dumps(start))
        if elapsed is not None:
            final = {**start, 'ended_at': (start_time+timedelta(seconds=elapsed)).isoformat(),
                     'charged_wall_seconds': elapsed}
            (directory / f'{number:03d}.json').write_text(json.dumps(final))
        return start

    def test_budget_carries_closed_pauses_and_charges_unclosed_reservations(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(C, 'PROGRESS', Path(tmp)):
            directory = Path(tmp) / 'invocations'; directory.mkdir()
            self.journal(directory, 1, 900, elapsed=120)
            self.assertEqual(S.consumed_budget('identity'), 120)
            self.journal(directory, 2, 780)
            self.assertEqual(S.consumed_budget('identity'), 900)
            with self.assertRaises(ValueError): S.consumed_budget('another identity')

    def test_incomplete_or_falsified_budget_ledger_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(C, 'PROGRESS', Path(tmp)):
            directory = Path(tmp) / 'invocations'; directory.mkdir()
            self.journal(directory, 1, 900, elapsed=120)
            start = directory / '001-start.json'; saved = start.read_text(); start.unlink()
            with self.assertRaises(ValueError): S.consumed_budget('identity')
            start.write_text(saved)
            final = directory / '001.json'; row = json.loads(final.read_text()); row['charged_wall_seconds'] = 0
            final.write_text(json.dumps(row))
            with self.assertRaises(ValueError): S.consumed_budget('identity')

    def test_parent_and_worker_thread_configuration_is_idempotent(self):
        import torch
        C.configure_threads()
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)
        C.configure_threads()  # Parent verification repeats identity after worker completion.
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)

    def test_thread_environment_is_explicit_and_single_threaded(self):
        environment = S.worker_environment()
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(environment[name], '1')


if __name__ == '__main__':
    unittest.main()
