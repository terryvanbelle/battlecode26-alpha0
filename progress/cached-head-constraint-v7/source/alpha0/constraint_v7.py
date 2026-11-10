"""One-worker, training-only supervisor for cached-head-constraint-v7.

The 900-second wall budget is shared across resumed invocations. No engine,
validation or benchmark command exists in this supervisor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import ctypes
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import time

from . import constraint_v7_common as C

ACTOR_MODULES = frozenset(("alpha0.constraint_v7", "alpha0.terminal_v5", "alpha0.terminal_v6",
                          "alpha0.root_ranking_v4", "alpha0.campaign", "alpha0.learn_v2", "alpha0.train",
                          "alpha0.evaluate", "alpha0.victory"))
ACTOR_SCRIPTS = frozenset(("run_warmstart_experiment.py", "run_campaign.py"))


class ReviewRequired(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def active_actors(proc_root=Path("/proc")):
    """Unknown cwd for an identified actor blocks launch instead of hiding it."""
    found = []
    for process in Path(proc_root).iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            command = [part for part in process.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0") if part]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            found.append({"pid": int(process.name), "command": [], "reason": "unreadable_command"})
            continue
        if not command:
            continue
        executable = Path(command[0]).name
        python_actor = "python" in executable and any(value in ACTOR_MODULES or Path(value).name in ACTOR_SCRIPTS for value in command[1:])
        java_actor = executable == "java" and "alpha0.EngineBridge" in command
        if not (python_actor or java_actor):
            continue
        try:
            cwd = process.joinpath("cwd").resolve(strict=True)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as error:
            try:
                process.stat()
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                pass
            found.append({"pid": int(process.name), "command": command, "reason": "unreadable_cwd", "error": type(error).__name__})
            continue
        if cwd == C.ROOT.resolve():
            found.append({"pid": int(process.name), "command": command, "reason": "active_actor"})
    return found


@contextmanager
def hold_locks():
    with ExitStack() as stack:
        for path in C.LOCKS:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def kill_group(process, immediate=False):
    signals = ((signal.SIGKILL, 3),) if immediate else ((signal.SIGTERM, 1), (signal.SIGKILL, 3))
    for signum, timeout in signals:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        if signum == signal.SIGKILL:
            return
    if process.poll() is None:
        raise RuntimeError("worker process group did not terminate")


def consumed_budget(identity_hash):
    """An unfinalized invocation is charged its full reservation conservatively."""
    total = 0.
    directory = C.PROGRESS / "invocations"
    files = list(directory.glob("*.json"))
    for path in files:
        if not re.fullmatch(r"\d{3,}(?:-start)?\.json", path.name):
            raise ValueError("invalid invocation journal filename")
        if not path.name.endswith("-start.json") and not path.with_name(path.stem + "-start.json").exists():
            raise ValueError("final invocation is missing its start reservation")
    for path in sorted(directory.glob("*-start.json")):
        start = read(path)
        if start.get("identity_sha256") != identity_hash:
            raise ValueError("invocation identity changed")
        number = int(path.name.split("-")[0])
        if number < 1 or path.name != f"{number:03d}-start.json" or type(start.get("invocation")) is not int or start["invocation"] != number:
            raise ValueError("invocation number differs from journal filename")
        try:
            started = datetime.fromisoformat(start["started_at"])
            if started.tzinfo is None:
                raise ValueError("missing timezone")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid invocation start timestamp") from error
        reserved = start.get("reserved_wall_seconds")
        if type(reserved) not in (int, float) or not math.isfinite(reserved) or not 0 < reserved <= 900:
            raise ValueError("invalid wall budget reservation")
        final_path = path.with_name(path.name.replace("-start.json", ".json"))
        if final_path.exists():
            final = read(final_path)
            if (final.get("identity_sha256") != identity_hash or final.get("started_at") != start.get("started_at")
                    or type(final.get("invocation")) is not int or final.get("invocation") != number
                    or final.get("reserved_wall_seconds") != reserved):
                raise ValueError("final invocation does not match its reservation")
            charge = final["charged_wall_seconds"]
            try:
                ended = datetime.fromisoformat(final["ended_at"])
                duration = (ended-started).total_seconds()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid final invocation timestamp") from error
            if type(charge) not in (int, float) or not math.isfinite(charge) or duration < 0 or abs(duration-charge) > 2:
                raise ValueError("wall budget accounting differs from recorded duration")
        else:
            charge = start["reserved_wall_seconds"]
        if type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0 or charge > start["reserved_wall_seconds"] + 10:
            raise ValueError("invalid wall budget accounting")
        total += charge
    return total


def worker_environment():
    result = os.environ.copy()
    result.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    return result


def run_worker(descriptor_path, deadline, status):
    """Own the worker process group until it exits or is forcibly reaped."""
    process = None
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    log_path = descriptor_path.with_suffix(".log")
    try:
        C.check_stop(deadline)
        with log_path.open("ab") as log:
            blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGALRM}
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
            try:
                process = subprocess.Popen([sys.executable, "-u", "-m", "alpha0.constraint_v7", "--worker", str(descriptor_path)],
                    cwd=C.ROOT, stdout=log, stderr=subprocess.STDOUT, env=worker_environment(), start_new_session=True)
                status(worker_pid=process.pid, phase="fitting")
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            last_status = time.monotonic()
            while process.poll() is None:
                C.check_stop(deadline)
                if time.monotonic() - last_status >= 5:
                    status(worker_pid=process.pid, phase="fitting")
                    last_status = time.monotonic()
                time.sleep(min(.1, max(.001, deadline-time.monotonic())))
            if process.returncode:
                raise ReviewRequired(f"worker exit {process.returncode}; inspect {log_path.relative_to(C.ROOT)}")
    finally:
        if process is not None:
            kill_group(process, immediate=time.monotonic() >= deadline)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        status(worker_pid=None, phase="worker_quiescent",
               worker_tree_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)


def worker(descriptor_path):
    descriptor_path = descriptor_path.resolve()
    if not descriptor_path.is_relative_to((C.RUN / "workers").resolve()):
        raise ValueError("worker descriptor is outside this experiment")
    descriptor = read(descriptor_path)
    parent = descriptor["parent_pid"]
    if os.getppid() != parent or os.getpgrp() != os.getpid():
        raise ValueError("worker must be launched in its supervisor's isolated process group")
    # Parent death cannot leave a standalone fitter running.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot configure worker parent-death signal")
    if os.getppid() != parent:
        raise C.Interrupted("supervisor disappeared")
    for path in C.LOCKS:
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            raise ValueError("supervisor does not hold all five locks")
    deadline = descriptor["deadline_monotonic"]
    C.check_stop(deadline)
    remaining = max(1, math.ceil(deadline-time.monotonic()))
    resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining))
    C.configure_threads()
    config = C.load_config()
    expected = read(C.PROGRESS / "identity.json")
    if descriptor["identity_sha256"] != C.fingerprint(expected) or C.identity(config) != expected:
        raise ValueError("worker runtime/input identity changed")
    from .constraint_v7_fit import run_fit
    run_fit(config, expected, deadline)
    C.check_stop(deadline)
    if C.identity(config) != expected:
        raise ValueError("worker runtime/input identity changed during fitting")


def execute(seconds=None):
    config = C.load_config()
    limit = config["total_wall_budget_seconds"]
    if seconds is not None and (not math.isfinite(seconds) or not 0 < seconds <= limit):
        raise ValueError("requested budget must be positive and at most 900 seconds")
    with hold_locks():
        C.check_stop(float("inf"))
        actors = active_actors()
        if actors:
            raise ReviewRequired(f"existing or unreadable actors block launch: {actors}")
        expected = C.identity(config)
        identity_hash = C.fingerprint(expected)
        C.immutable_json(C.PROGRESS / "identity.json", expected)
        C.immutable_json(C.PROGRESS / "config.json", config)
        for name in expected["sources"]:
            C.immutable_bytes(C.PROGRESS / "source" / name, (C.ROOT / name).read_bytes())
        remaining = limit - consumed_budget(identity_hash)
        if remaining <= 0:
            raise ReviewRequired("total 900-second wall budget exhausted")
        budget = min(remaining, seconds if seconds is not None else remaining)
        directory = C.PROGRESS / "invocations"
        number = len(list(directory.glob("*-start.json"))) + 1
        started = time.monotonic()
        deadline = started + budget
        start = {"experiment_id": config["experiment_id"], "identity_sha256": identity_hash,
                 "started_at": C.now(), "pid": os.getpid(), "reserved_wall_seconds": budget,
                 "wall_budget_previously_consumed": limit-remaining, "invocation": number}
        C.immutable_json(directory / f"{number:03d}-start.json", start)
        descriptor = {"identity_sha256": identity_hash, "parent_pid": os.getpid(), "deadline_monotonic": deadline,
                      "invocation": number, "started_at": start["started_at"]}
        descriptor_path = C.RUN / "workers" / f"invocation-{number:03d}.json"
        C.immutable_json(descriptor_path, descriptor)
        live = {**start, "phase": "starting", "worker_pid": None}
        def status(**extra):
            live.update(extra, updated_at=C.now(), elapsed_seconds=time.monotonic()-started)
            C.atomic_json(C.RUN / "status.json", live)
        outcome, reason, verified = "complete_review_required", None, False
        supervisor_cpu = time.process_time()
        signal.setitimer(signal.ITIMER_REAL, budget)
        try:
            run_worker(descriptor_path, deadline, status)
            C.check_stop(deadline)
            if C.identity(config) != expected:
                raise ValueError("final identity changed")
            if active_actors():
                raise ReviewRequired("actor appeared before final verification")
            from .constraint_v7_fit import verify_completed
            report = verify_completed(config, expected)
            verified = True
        except C.Interrupted as error:
            outcome, reason = "paused", str(error)
        except ReviewRequired as error:
            outcome, reason = "review_required", str(error)
        except BaseException as error:
            outcome, reason = "invalidated", f"{type(error).__name__}: {error}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                unchanged = C.identity(config) == expected
            except Exception:
                unchanged = False
            if not unchanged:
                outcome, reason, verified = "invalidated", "runtime or frozen training inputs changed", False
            if verified and unchanged:
                C.immutable_json(C.PROGRESS / "final-verification.json", {"identity_sha256": identity_hash,
                    "complete": True, "integrity_verified": True, "training_only": True,
                    "promotion_allowed": False, "fit_report_sha256": C.fingerprint(report)})
            report_error = None
            try:
                from .constraint_v7_report import report as build_report, write_report
                summary = build_report(C.RUN, C.PROGRESS, config, expected, integrity_verified=verified and unchanged)
                C.atomic_json(C.PROGRESS / "summary.json", summary)
                write_report(summary, C.PROGRESS)
            except Exception as error:
                report_error = f"{type(error).__name__}: {error}"
            result = {**start, "ended_at": C.now(), "phase": outcome, "reason": reason,
                      "charged_wall_seconds": time.monotonic()-started,
                      "worker_tree_cpu_seconds": live.get("worker_tree_cpu_seconds", 0),
                      "supervisor_cpu_seconds": time.process_time()-supervisor_cpu,
                      "integrity_verified": verified and unchanged, "worker_pid": None,
                      "training_only": True, "engine_games": 0, "promotion_allowed": False,
                      "report_error": report_error}
            C.immutable_json(directory / f"{number:03d}.json", result)
            C.atomic_json(C.RUN / "status.json", result)
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(signum, lambda number, _: (_ for _ in ()).throw(C.Interrupted(signal.Signals(number).name)))
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM})
    if args.worker:
        worker(args.worker)
    elif args.preflight:
        with hold_locks():
            C.check_stop(float("inf"))
            config = C.load_config()
            expected = C.identity(config)
            actors = active_actors()
            if actors:
                raise ReviewRequired(f"actors block launch: {actors}")
            print(json.dumps({"identity_sha256": C.fingerprint(expected), "training_only": True,
                              "remaining_wall_seconds": config["total_wall_budget_seconds"]-consumed_budget(C.fingerprint(expected)),
                              "actors": [], "sgd_launched": False}))
    else:
        print(json.dumps(execute(args.seconds)), flush=True)


if __name__ == "__main__":
    main()
