"""Offline snapshot safety checks. Never contacts GitHub or launches training."""
import fcntl
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
from pathlib import Path
import tarfile
import tempfile
import time
import unittest

import numpy as np
import torch
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("campaign_snapshot", Path(__file__).resolve().parents[1] / "scripts/campaign_snapshot.py")
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


def evaluation_row(number, attempt=1, status="ok"):
    identity={"artifact_version":2,"map":"tiny","seed":number,"side":"A","opponent":"bot",
              "evaluation_config":{"policy_schema":{"action_version":"macro_v2"}}}
    cell=hashlib.sha256(snapshot.canonical(identity).encode()).hexdigest()
    row={"artifact_version":2,"evaluation_cell_id":cell,"identity":identity,"attempt":attempt,
         "status":status,"score":1 if status=="ok" else None,"iteration":0,"checkpoint":"checkpoint-0000.pt",
         "opponent":"bot","map":"tiny","seed":number,"side":"A","rounds":9,"winner":0,
         "terminal":{"terminal":True,"valid_game":True,"benchmark_failures":[],"round":9,
                     "map":"tiny","seed":number,"action_version":"macro_v2","winner":0}}
    if status=="error": row.update(error="original full error\nwith details",error_kind="worker_process_error")
    return row


def write_evidence(progress, rows, training, completed):
    progress.mkdir(parents=True,exist_ok=True)
    all_cells=[evaluation_row(i)["evaluation_cell_id"] for i in range(2)]
    for row in rows:
        cell=progress/"evaluation_cells"/row["evaluation_cell_id"]
        cell.mkdir(parents=True,exist_ok=True)
        (cell/"request.json").write_text(json.dumps({"evaluation_cell_id":row["evaluation_cell_id"],"identity":row["identity"]}))
        (cell/f"attempt-{row['attempt']:04d}.json").write_text(json.dumps(row))
    snapshot.write_jsonl(progress/"evaluations.jsonl",[r for r in rows if r["status"]=="ok"])
    snapshot.write_jsonl(progress/"evaluation_errors.jsonl",[r for r in rows if r["status"]=="error"])
    snapshot.write_jsonl(progress/"training.jsonl",training)
    report={"checkpoint_sha256":"checkpoint", "evaluation_config_sha256":"config", "benchmark_suite_id":"panel",
            "suite":"smoke","stage":"smoke","iteration":0,"expected":2,"expected_cell_ids":all_cells,
            "completed":completed,"pending_cell_ids":all_cells[completed:],"complete":completed==2,
            "full_panel_complete":completed==2,
            "opponents":{"bot":{"completed":completed,"wins":completed,"draws":0,"losses":0}}}
    (progress/"evaluation_reports").mkdir(exist_ok=True)
    (progress/"evaluation_reports/report.json").write_text(json.dumps(report))
    (progress/"status.json").write_text(json.dumps({"historical_completed":completed}))
    return report


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

    def test_compact_snapshot_preserves_trained_state_and_partial_next_iteration(self):
        from alpha0.learn_v2 import Learner
        cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/campaign-v2.json").read_text())
        cfg.update(games_per_iteration=2,workers=1,train_steps=1,batch_size=1,
                   replay_capacity=8,replay_max_games=4,game_position_capacity=2)
        cfg["model"].update(local_width=8,global_width=8,blocks=0,hidden_dim=16)
        def complete_job(job):
            output=Path(job["output"]); output.parent.mkdir(parents=True,exist_ok=True)
            policy=np.zeros((1,40),np.float32); policy[0,0]=1
            np.savez_compressed(output.with_suffix(".npz"),local=np.zeros((1,33,17,17),np.float16),
                global_board=np.zeros((1,33,60,60),np.float16),scalars=np.zeros((1,64),np.float32),
                policy=policy,legal_mask=np.ones((1,40),bool),player=np.array([0],np.int8))
            row={"game_id":job["game_id"],"identity":job["identity"],"status":"ok","attempt":1,
                 "failure_count":0,"terminal_value_a":1,"retained_positions":1,"searched_positions":1,
                 "decisions":1,"data_sha256":snapshot.sha256(output.with_suffix(".npz"))}
            output.write_text(json.dumps(row))
            output.with_name(output.stem+".attempt-0001.json").write_text(json.dumps(row))
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); source=base/"source"; target=base/"target"; target.mkdir()
            learner=Learner(source/snapshot.RUN,cfg,source/snapshot.PROGRESS)
            first=learner._plan()
            for job in first["jobs"]: complete_job(job)
            learner.train_iteration(time.monotonic()+60,source/"STOP")
            self.assertEqual(learner.iteration,1)
            next_plan=learner._plan()
            complete_job(next_plan["jobs"][0])  # One finished game; the next still needs collection.
            old_shards=[Path(job["output"]).with_suffix(".npz") for job in first["jobs"]]
            shard_hashes={path:snapshot.sha256(path) for path in old_shards}
            archive=Path(create(source)["archive"])
            with tarfile.open(archive) as tar:
                names=set(tar.getnames())
                manifest=json.load(tar.extractfile(snapshot.MANIFEST))
            omitted={entry["path"] for entry in manifest["omitted_files"]}
            for path in old_shards:
                self.assertNotIn(path.relative_to(source).as_posix(),names)
                self.assertIn(path.relative_to(source).as_posix(),omitted)
                self.assertEqual(snapshot.sha256(path),shard_hashes[path])  # No local pruning.
                self.assertIn(path.with_suffix(".json").relative_to(source).as_posix(),names)
            self.assertNotIn(snapshot.RUN+"/replay/iteration-0000/index.json",names)
            self.assertIn(snapshot.RUN+"/replay/iteration-0001/index.json",names)
            self.assertIn(snapshot.RUN+"/checkpoint-0000.pt",names)
            self.assertIn(snapshot.RUN+"/checkpoint-0001.pt",names)
            pending_npz=Path(next_plan["jobs"][0]["output"]).with_suffix(".npz")
            self.assertIn(pending_npz.relative_to(source).as_posix(),names)
            snapshot.restore_snapshot(target,archive)
            resumed=Learner(target/snapshot.RUN,cfg,target/snapshot.PROGRESS)
            self.assertEqual(resumed.iteration,1)
            self.assertEqual(resumed.buffer.num_games,2)
            restored_plan=resumed._plan()
            cached=Path(restored_plan["jobs"][0]["output"]).with_suffix(".npz")
            self.assertEqual(snapshot.sha256(cached),snapshot.sha256(pending_npz))
            self.assertFalse(Path(restored_plan["jobs"][1]["output"]).exists())
            self.assertIsNone(resumed.train_iteration(time.monotonic()-1,target/"STOP"))
            complete_job(restored_plan["jobs"][1])
            with patch("alpha0.learn_v2.ProcessPoolExecutor") as executor:
                executor.return_value.__enter__.return_value.submit.side_effect=AssertionError("cached games must not rerun")
                resumed.train_iteration(time.monotonic()+60,target/"STOP")
            self.assertEqual(resumed.iteration,2)
            self.assertEqual(resumed.buffer.num_games,4)

    def test_missing_completed_next_iteration_shard_blocks_compact_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory); fixture(source)
            output=source/snapshot.RUN/"games/a.json"
            output.parent.mkdir(parents=True)
            output.write_text('{"status":"ok","data_sha256":"missing"}')
            with self.assertRaisesRegex(ValueError,"missing or corrupt resume shard"):
                create(source)

    def test_git_and_release_progress_union_in_both_age_orders(self):
        for archive_newer in (True,False):
            with self.subTest(archive_newer=archive_newer), tempfile.TemporaryDirectory() as directory:
                base=Path(directory); source=base/"source"; target=base/"target"
                fixture(source); target.mkdir()
                old_rows=[evaluation_row(0)]
                new_rows=[evaluation_row(0),evaluation_row(1,attempt=1,status="error"),evaluation_row(1,attempt=2)]
                old_train=[{"iteration":0,"loss":0.9}]
                new_train=[*old_train,{"iteration":1,"loss":0.8}]
                write_evidence(source/snapshot.PROGRESS,new_rows if archive_newer else old_rows,
                               new_train if archive_newer else old_train,2 if archive_newer else 1)
                write_evidence(target/snapshot.PROGRESS,old_rows if archive_newer else new_rows,
                               old_train if archive_newer else new_train,1 if archive_newer else 2)
                (target/"runs/campaign").mkdir(parents=True)
                (target/snapshot.STATUS).write_text('{"phase":"different historical process"}')
                archive=create(source)["archive"]
                restored=snapshot.restore_snapshot(target,archive)
                progress=target/snapshot.PROGRESS
                self.assertEqual(snapshot.read_jsonl(progress/"training.jsonl"),new_train)
                evaluations=snapshot.read_jsonl(progress/"evaluations.jsonl")
                self.assertEqual(len(evaluations),2)
                self.assertTrue(all(row["status"]=="ok" for row in evaluations))
                errors=snapshot.read_jsonl(progress/"evaluation_errors.jsonl")
                self.assertEqual(len(errors),1)
                self.assertEqual(errors[0]["error"],"original full error\nwith details")
                self.assertEqual(len(list((progress/"evaluation_cells").glob("*/attempt-*.json"))),3)
                report=json.loads((progress/"evaluation_reports/report.json").read_text())
                self.assertEqual(report["completed"],2)
                self.assertTrue(report["complete"])
                self.assertTrue(restored["progress_ahead_of_checkpoint"])
                self.assertTrue(restored["review_required"])
                self.assertTrue((target/"runs/campaign/STOP").is_file())
                self.assertEqual(json.loads((target/snapshot.STATUS).read_text())["state"],"checkpoint_boundary")

    def test_conflicting_training_or_raw_attempt_never_overwrites(self):
        for conflict in ("training","attempt"):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as directory:
                base=Path(directory); source=base/"source"; target=base/"target"
                fixture(source); target.mkdir()
                rows=[evaluation_row(0)]
                training=[{"iteration":0,"loss":0.9}]
                write_evidence(source/snapshot.PROGRESS,rows,training,1)
                write_evidence(target/snapshot.PROGRESS,rows,training,1)
                if conflict=="training":
                    path=target/snapshot.PROGRESS/"training.jsonl"
                    snapshot.write_jsonl(path,[{"iteration":0,"loss":0.1}])
                else:
                    path=target/snapshot.PROGRESS/"evaluation_cells"/rows[0]["evaluation_cell_id"]/"attempt-0001.json"
                    altered=dict(rows[0]); altered["score"]=0
                    path.write_text(json.dumps(altered))
                original=path.read_bytes()
                with self.assertRaisesRegex(ValueError,"conflicting"):
                    snapshot.restore_snapshot(target,create(source)["archive"])
                self.assertEqual(path.read_bytes(),original)
                self.assertFalse((target/snapshot.RUN).exists())

    def test_git_fallback_exact_commit_roundtrip_and_append_only_history(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); source=base/"source"; remote=base/"remote.git"
            fixture(source)
            subprocess.run(["git","init","--bare","--quiet",str(remote)],check=True)
            original_checkpoint=(source/snapshot.RUN/"latest.pt").read_bytes()
            archive=Path(create(source)["archive"])
            sidecar=json.loads(Path(str(archive)+".manifest.json").read_text())
            with patch.object(snapshot,"remote_url",return_value=str(remote)):
                first=snapshot.publish_git_snapshot(source,archive,sidecar,"owner/repo",snapshot.TAG,"test auth failure")
                (source/snapshot.STATUS).write_text('{"state":"later boundary"}')
                second_archive=Path(create(source)["archive"])
                second_sidecar=json.loads(Path(str(second_archive)+".manifest.json").read_text())
                second=snapshot.publish_git_snapshot(source,second_archive,second_sidecar,"owner/repo",snapshot.TAG,"test auth failure")
                subprocess.run(["git","--git-dir",str(remote),"merge-base","--is-ancestor",first["commit"],second["commit"]],check=True)
                downloads=base/"downloads"; downloads.mkdir()
                downloaded=snapshot.download_git_snapshot(downloads,first)
                self.assertEqual(snapshot.sha256(downloaded["archive"]),sidecar["sha256"])
                latest=snapshot.git_latest_pointer("owner/repo")
                self.assertEqual(latest["archive"],second_archive.name)
            self.assertFalse((source/".git").exists())
            self.assertEqual((source/snapshot.RUN/"latest.pt").read_bytes(),original_checkpoint)
            self.assertEqual(json.loads((source/snapshot.POINTER).read_text())["storage"],"git")

    def test_only_upload_auth_failure_activates_git_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory); fixture(source); archive=create(source)["archive"]
            def fake_gh(*args,**kwargs):
                if args[:2]==("release","upload"):
                    raise subprocess.CalledProcessError(1,["gh"],stderr="HTTP 401: Bad credentials (https://uploads.github.com/...)" )
                raise AssertionError(args)
            with patch.object(snapshot,"release_info",return_value={"assets":[]}), patch.object(snapshot,"gh",side_effect=fake_gh), \
                 patch.object(snapshot,"publish_git_snapshot",return_value={"storage":"git"}) as fallback:
                self.assertEqual(snapshot.publish_snapshot(source,archive),{"storage":"git"})
                self.assertEqual(fallback.call_count,1)
            with patch.object(snapshot,"release_info",return_value={"assets":[]}), \
                 patch.object(snapshot,"gh",side_effect=subprocess.CalledProcessError(1,["gh"],stderr="connection reset")), \
                 patch.object(snapshot,"publish_git_snapshot") as fallback:
                with self.assertRaises(subprocess.CalledProcessError): snapshot.publish_snapshot(source,archive)
                fallback.assert_not_called()

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
