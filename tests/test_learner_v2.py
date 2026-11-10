"""Recovery contracts for persisted v2 self-play plans and optimization."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from alpha0.learn_v2 import Learner, rollout
from alpha0.replay_v2 import GameReplayBuffer


def config():
    value = json.loads((Path(__file__).resolve().parents[1] / "configs/campaign-v2.json").read_text())
    value.update(games_per_iteration=2, workers=1, train_steps=2, batch_size=2,
                 replay_capacity=8, replay_max_games=4, game_position_capacity=2)
    value["model"].update(local_width=8, global_width=8, blocks=0, hidden_dim=16)
    return value


def position(player=0):
    local = np.zeros((33, 17, 17), np.float16)
    global_board = np.zeros((33, 60, 60), np.float16)
    local[0] = 1
    local[9, 8, 8] = 1
    global_board[0, :20, :20] = 1
    global_board[9, 10, 10] = 1
    policy = np.zeros(40, np.float32)
    policy[0] = 1
    return {"local": local, "global_board": global_board, "scalars": np.zeros(64, np.float32),
            "legal_mask": np.ones(40, bool), "policy": policy, "player": player}


def complete_plan(learner):
    plan = learner._plan()
    for index, job in enumerate(plan["jobs"]):
        path = Path(job["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        p = position(index % 2)
        arrays = {key: np.asarray([value]) for key, value in p.items()}
        data = path.with_suffix(".npz")
        np.savez_compressed(data, **arrays)
        row = {"status": "ok", "game_id": job["game_id"], "identity": job["identity"],
               "attempt": 1, "failure_count": 0, "terminal_value_a": 1,
               "retained_positions": 1, "decisions": 1, "searched_positions": 1,
               "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        path.write_text(json.dumps(row))
    return plan


def test_cycle_interruption_produces_no_terminal_labels_or_engine(tmp_path, monkeypatch):
    learner = Learner(tmp_path / "run", config(), tmp_path / "progress")
    job = learner._plan()["jobs"][0]
    job.update(deadline_monotonic=time.monotonic() - 1, attempt=1)
    monkeypatch.setattr("alpha0.learn_v2.EngineEnv", lambda **kw: pytest.fail("expired job must not start Java"))
    result = rollout(job)
    assert result["status"] == "interrupted"
    assert result["failure_count"] == 0
    assert "terminal_value_a" not in result
    assert not Path(job["output"]).with_suffix(".npz").exists()
    assert Path(job["output"]).with_name("iter-0001-game-000.attempt-0001.json").exists()


def test_engine_initialization_failure_is_persisted(tmp_path, monkeypatch):
    learner = Learner(tmp_path / "run", config(), tmp_path / "progress")
    job = learner._plan()["jobs"][0]
    def unavailable(**kwargs):
        raise RuntimeError("engine unavailable")
    monkeypatch.setattr("alpha0.learn_v2.EngineEnv", unavailable)
    result = rollout(job)
    assert result["status"] == "error" and result["failure_count"] == 1
    assert "engine unavailable" in result["error"]
    assert json.loads(Path(job["output"]).read_text())["identity"] == job["identity"]


def test_changed_checkpoint_or_result_cannot_reuse_a_plan(tmp_path):
    learner = Learner(tmp_path / "run", config(), tmp_path / "progress")
    plan = complete_plan(learner)
    output = Path(plan["jobs"][0]["output"])
    row = json.loads(output.read_text())
    row["identity"]["seed"] += 1
    output.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="different training inputs"):
        learner.train_iteration(time.monotonic() + 60, tmp_path / "STOP")
    with learner.checkpoint_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="saved game plan"):
        learner._plan()


def test_plan_alternates_seats_on_each_repeated_map(tmp_path):
    cfg = config()
    cfg["games_per_iteration"] = 2 * len(cfg["maps"])
    learner = Learner(tmp_path / "run", cfg, tmp_path / "progress")
    jobs = learner._plan()["jobs"]
    for map_name in cfg["maps"]:
        assert {job["current_team"] for job in jobs if job["map"] == map_name} == {0, 1}


def test_changed_training_runtime_rejects_saved_plan_and_rollout(tmp_path, monkeypatch):
    learner = Learner(tmp_path / "run", config(), tmp_path / "progress")
    plan = learner._plan()
    assert plan["identity_version"] == 2
    assert "java/alpha0/MacroV2.java" in plan["training_runtime"]["files"]
    assert not any("evaluate.py" in name for name in plan["training_runtime"]["files"])
    changed = dict(plan["training_runtime"], torch_version="changed-runtime")
    monkeypatch.setattr("alpha0.learn_v2._training_runtime_identity", lambda: changed)
    with pytest.raises(ValueError, match="runtime version"):
        learner._plan()
    monkeypatch.setattr("alpha0.learn_v2.EngineEnv", lambda **kw: pytest.fail("stale runtime must not start Java"))
    row = rollout(plan["jobs"][0])
    assert row["status"] == "error" and "runtime changed" in row["error"]
    assert row["training_runtime"] == plan["training_runtime"]
    assert not Path(plan["jobs"][0]["output"]).with_suffix(".npz").exists()


def test_replay_before_failed_checkpoint_does_not_replace_published_replay(tmp_path, monkeypatch):
    learner = Learner(tmp_path / "run", config(), tmp_path / "progress")
    learner.buffer.add_game("new", [position()], 1, {"source": "selfplay"})
    learner.iteration = 1
    monkeypatch.setattr("alpha0.learn_v2.save_checkpoint", lambda *a, **kw: (_ for _ in ()).throw(OSError("interrupted publication")))
    with pytest.raises(OSError, match="interrupted publication"):
        learner.checkpoint()
    restored = Learner(tmp_path / "run", config(), tmp_path / "progress")
    assert restored.iteration == 0 and len(restored.buffer) == 0
    assert GameReplayBuffer.load(tmp_path / "run/replay/iteration-0001").game_ids == ("new",)


def test_resumed_completed_episode_plan_has_identical_next_update(tmp_path):
    direct = Learner(tmp_path / "direct", config(), tmp_path / "progress-direct")
    complete_plan(direct)
    paused = Learner(tmp_path / "resumed", config(), tmp_path / "progress-resumed")
    complete_plan(paused)
    resumed = Learner(tmp_path / "resumed", config(), tmp_path / "progress-resumed")
    direct.train_iteration(time.monotonic() + 60, tmp_path / "STOP")
    resumed.train_iteration(time.monotonic() + 60, tmp_path / "STOP")
    for expected, actual in zip(direct.model.parameters(), resumed.model.parameters()):
        assert torch.equal(expected, actual)
    assert direct.rng.bit_generator.state == resumed.rng.bit_generator.state
    assert resumed.iteration == 1 and resumed.games_total == 2
    assert Learner(tmp_path / "resumed", config(), tmp_path / "progress-resumed").buffer.num_games == 2


def test_stop_during_optimizer_rolls_back_without_losing_completed_games(tmp_path, monkeypatch):
    learner = Learner(tmp_path / "run", config(), tmp_path / "progress")
    plan = complete_plan(learner)
    before = [p.detach().clone() for p in learner.model.parameters()]
    stop = tmp_path / "STOP"
    step = learner.optimizer.step
    def stop_after_step(*args, **kwargs):
        value = step(*args, **kwargs)
        stop.touch()
        return value
    monkeypatch.setattr(learner.optimizer, "step", stop_after_step)
    assert learner.train_iteration(time.monotonic() + 60, stop) is None
    assert learner.iteration == 0 and learner.games_total == 0 and len(learner.buffer) == 0
    assert all(torch.equal(a, b) for a, b in zip(before, learner.model.parameters()))
    assert all(json.loads(Path(job["output"]).read_text())["status"] == "ok" for job in plan["jobs"])


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "build/java/alpha0/EngineBridge.class").exists(),
                    reason="requires the compiled official engine")
def test_real_v2_rollout_cutoff_never_publishes_training_labels(tmp_path):
    cfg = config()
    cfg.update(max_decisions=1, simulations=1, max_search_depth=1)
    learner = Learner(tmp_path / "run", cfg, tmp_path / "progress")
    job = learner._plan()["jobs"][0]
    result = rollout(job)
    assert result["status"] == "error", result
    assert result["decisions"] == 1, result
    assert "incomplete self-play" in result["error"]
    assert "terminal_value_a" not in result
    assert not Path(job["output"]).with_suffix(".npz").exists()
