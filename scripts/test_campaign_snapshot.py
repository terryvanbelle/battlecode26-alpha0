"""Offline snapshot safety checks. Never contacts GitHub or launches training."""
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
import torch
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("campaign_snapshot", Path(__file__).with_name("campaign_snapshot.py"))
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


def fixture(root):
    run = root/snapshot.RUN
    (run/"replay").mkdir(parents=True)
    torch.save({"model_version": "dual_scale_v2", "iteration": 0}, run/"latest.pt")
    (run/"checkpoint-0000.pt").write_bytes((run/"latest.pt").read_bytes())
    (run/"replay/index.json").write_text('{"games": []}\n')
    (run/"config.json").write_text('{"seed": 42}\n')
    (run/"plan-0001.json").write_text(json.dumps({"jobs": [{"checkpoint": str(run/"latest.pt"),
                                                           "output": str(run/"games/a.json")}]}))
    (run/"worker.log").write_text("excluded log")
    (run/".env").write_text("excluded configuration")
    (run/"api_token.json").write_text('{"fake": "excluded"}')
    progress = root/snapshot.PROGRESS
    progress.mkdir(parents=True)
    (progress/"training.jsonl").write_text('{"iteration": 0}\n')
    (progress/"snapshot.json").write_text('{"old_pointer": true}\n')
    (root/"runs/gate-ledgers").mkdir(parents=True)
    (root/"runs/gate-ledgers/gate.json").write_text('{"consumed": true}\n')
    (root/snapshot.STATUS).write_text('{"state": "checkpoint_boundary"}\n')


def create(root):
    with patch.object(snapshot, "git_identity", return_value={"commit": "a"*40, "worktree_dirty": False}):
        return snapshot.create_snapshot(root)


def malicious_archive(directory, member):
    temporary = directory/"evil.tar.gz"
    manifest = {"format_version": 1, "source_root": "/old", "git": {"commit": "a"*40},
                "files": [{"path": member.name, "size": member.size, "sha256": hashlib.sha256(b"x").hexdigest()}]}
    with tarfile.open(temporary,"w:gz") as archive:
        archive.addfile(member, io.BytesIO(b"x") if member.isfile() else None)
        encoded=json.dumps(manifest).encode()
        info=tarfile.TarInfo(snapshot.MANIFEST); info.size=len(encoded)
        archive.addfile(info,io.BytesIO(encoded))
    digest=snapshot.sha256(temporary)
    target=directory/f"campaign-state-20260101T000000000000Z-{digest[:12]}.tar.gz"
    temporary.rename(target)
    snapshot.atomic_json(Path(str(target)+".manifest.json"),{"format_version":1,"archive":target.name,
                         "sha256":digest,"size":target.stat().st_size})
    return target


class SnapshotTests(unittest.TestCase):
    def test_roundtrip_includes_gates_and_rebases_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); source=base/"source"; target=base/"target"
            fixture(source); target.mkdir()
            made=create(source)
            archive=Path(made["archive"])
            self.assertTrue(archive.is_absolute() and archive.exists())
            with tarfile.open(archive) as tar:
                names=tar.getnames()
            self.assertIn("runs/gate-ledgers/gate.json",names)
            self.assertNotIn(snapshot.POINTER,names)
            self.assertFalse(any(name.endswith((".log",".env","api_token.json")) for name in names))
            report=snapshot.restore_snapshot(target,archive)
            self.assertEqual(report["status"],"restored")
            self.assertEqual((target/snapshot.RUN/"latest.pt").read_bytes(),(source/snapshot.RUN/"latest.pt").read_bytes())
            plan=json.loads((target/snapshot.RUN/"plan-0001.json").read_text())
            self.assertEqual(plan["jobs"][0]["checkpoint"],str(target/snapshot.RUN/"latest.pt"))
            self.assertTrue(json.loads((target/"runs/gate-ledgers/gate.json").read_text())["consumed"])
            with self.assertRaisesRegex(ValueError,"already contains work"):
                snapshot.restore_snapshot(target,archive)

    def test_generation_checkpoint_restores_real_learner_and_identity_safe_plan(self):
        from alpha0.learn_v2 import Learner, _job_identity
        config_path=Path(__file__).resolve().parents[1]/"configs/campaign-v2.json"
        cfg=json.loads(config_path.read_text())
        cfg.update(games_per_iteration=2, workers=1, train_steps=1, batch_size=1,
                   replay_capacity=8, replay_max_games=4, game_position_capacity=2)
        cfg["model"].update(local_width=8, global_width=8, blocks=0, hidden_dim=16)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); source=base/"source"; target=base/"target"
            target.mkdir()
            learner=Learner(source/snapshot.RUN,cfg,source/snapshot.PROGRESS)
            policy=np.zeros(40,np.float32); policy[0]=1
            learner.buffer.add_game("complete", [{"local":np.zeros((33,17,17),np.float16),
                "global_board":np.zeros((33,60,60),np.float16),"scalars":np.zeros(64,np.float32),
                "policy":policy,"legal_mask":np.ones(40,bool),"player":0}],1,{"source":"selfplay"})
            learner.iteration=1; learner.checkpoint()
            plan=learner._plan()
            self.assertFalse((source/snapshot.RUN/"replay/index.json").exists())
            report=create(source)
            snapshot.restore_snapshot(target,report["archive"])
            resumed=Learner(target/snapshot.RUN,cfg,target/snapshot.PROGRESS)
            self.assertEqual(resumed.iteration,1)
            self.assertEqual(resumed.buffer.game_ids,("complete",))
            restored_plan=resumed._plan()
            for original,restored in zip(plan["jobs"],restored_plan["jobs"]):
                self.assertEqual(original["identity"],restored["identity"])
                self.assertEqual(restored["identity"],_job_identity(restored))
                self.assertTrue(restored["checkpoint"].startswith(str(target)))
                self.assertTrue(restored["output"].startswith(str(target)))
            bound={"identity":{"checkpoint_path":str(source/"bound")},
                   "config":{"output":str(source/"bound")},"checkpoint_path":str(source/"movable")}
            updated=snapshot.rebase_json(bound,str(source),str(target))
            self.assertEqual(updated["identity"],bound["identity"])
            self.assertEqual(updated["config"],bound["config"])
            self.assertEqual(updated["checkpoint_path"],str(target/"movable"))

    def test_newer_gate_ledger_prevents_any_run_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); source=base/"source"; target=base/"target"
            fixture(source); (target/"runs/gate-ledgers").mkdir(parents=True)
            ledger=target/"runs/gate-ledgers/gate.json"
            ledger.write_text('{"consumed": true, "newer": true}\n')
            archive=create(source)["archive"]
            with self.assertRaisesRegex(ValueError,"existing artifact differs"):
                snapshot.restore_snapshot(target,archive)
            self.assertTrue(json.loads(ledger.read_text())["newer"])
            self.assertFalse((target/snapshot.RUN).exists())

    def test_active_campaign_lock_blocks_manual_create(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture(root)
            with (root/"runs/campaign/campaign.lock").open("w") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,"campaign is active"):
                    create(root)

    def test_traversal_and_symlink_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            for index,name in enumerate(("../escape",snapshot.RUN+"/evil.json")):
                malicious=tarfile.TarInfo(name)
                if index:
                    malicious.type=tarfile.SYMTYPE; malicious.linkname="/tmp/escape"
                else:
                    malicious.size=1
                target=base/f"target{index}"; target.mkdir()
                archive=malicious_archive(base,malicious)
                with self.assertRaisesRegex(ValueError,"unsafe or unapproved"):
                    snapshot.restore_snapshot(target,archive)
                self.assertFalse((base/"escape").exists())

    def test_corrupt_archive_rejected_before_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); source=base/"source"; target=base/"target"
            fixture(source); target.mkdir(); archive=Path(create(source)["archive"])
            with archive.open("ab") as stream: stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError,"checksum/size/identity"):
                snapshot.restore_snapshot(target,archive)
            self.assertFalse((target/snapshot.RUN).exists())

    def test_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture(root)
            (root/snapshot.RUN/"linked.json").symlink_to(root/snapshot.RUN/"config.json")
            with self.assertRaisesRegex(ValueError,"symlink"):
                create(root)


if __name__ == "__main__":
    unittest.main()
