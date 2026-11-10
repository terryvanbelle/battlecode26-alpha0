"""Bounded counterfactual search probe on archived OWN self-play, with no SGD."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from alpha0.env import EngineEnv
from alpha0.features_v2 import make_evaluator
from alpha0.learn_v2 import load_checkpoint, _training_runtime_identity
from alpha0.mcts import SearchConfig, search

OUT = ROOT / "diagnostics/warmstart-plateau/counterfactual-v1"
RUN = ROOT / "runs/warmstart-v3"
CONFIGS = ((32, 12), (128, 12), (128, 96))
CONTINUATION_SEEDS = (102031, 102047)


class Deadline(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def retain(env, *states):
    """Keep a counterfactual root plus the live branch, using the frozen adapter."""
    keep = {state.snapshot for state in states if state is not None and state.snapshot is not None}
    release = list(env._snapshots - keep)
    if release:
        env._request({"cmd": "release", "snapshots": release})
        env._snapshots.difference_update(release)


def main():
    torch.set_num_threads(1)
    stops = (ROOT / "runs/campaign/STOP", RUN / "STOP")
    if any(path.exists() for path in stops):
        raise ValueError("STOP exists; no diagnostic started")
    control = ROOT / "runs/warmstart-probe"
    control.mkdir(parents=True, exist_ok=True)
    with (control / "probe.lock").open("a") as probe_lock, (RUN / "experiment.lock").open("a") as training_lock:
        for lock in (probe_lock, training_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if OUT.exists():
            raise ValueError("probe output already exists; never overwrite diagnostic evidence")
        OUT.mkdir(parents=True)
        started = time.monotonic()
        deadline = started + 15*60
        checkpoint = RUN / "checkpoint-0002.pt"
        model, payload = load_checkpoint(checkpoint)
        model.eval()
        infer = make_evaluator(model)
        gate = json.loads((ROOT / "configs/victory_gate.json").read_text())
        reserved = set(gate["stages"]["confirmation"]["seeds"])
        assert not reserved.intersection(CONTINUATION_SEEDS)
        runtime = _training_runtime_identity()
        paths = [RUN / "games" / f"iter-0001-game-{i:03d}.json" for i in range(4)]
        config = {"experiment_id": "warmstart-cp2-counterfactual-v1", "checkpoint_sha256": digest(checkpoint),
                  "source_games": {str(p.relative_to(ROOT)): digest(p) for p in paths},
                  "search_configs": [{"simulations": s, "max_depth": d} for s, d in CONFIGS],
                  "root_noise": True, "root_noise_rng_seed": "20261114 + root ordinal, identical across conditions",
                  "continuation_policy": "frozen cp2 for both teams, temperature 0.35; no subsequent MCTS",
                  "continuation_seeds": CONTINUATION_SEEDS, "max_first_actions_per_root": 3,
                  "selection": "first baby/king turn per source game with powered-bite macro37 legal; one of each role",
                  "first_actions": "greedy policy, deepest-search argmax, powered bite; fill duplicate slots with retreat/trap/hunt/guard",
                  "max_roots": 8, "wall_budget_seconds": 900, "threads": 1,
                  "benchmark_data_used": False, "gradient_updates": 0, "runtime": runtime,
                  "probe_source_sha256": digest(__file__)}
        publish(OUT / "config.json", config)
        results = []

        def budget(env):
            if any(path.exists() for path in stops):
                raise Deadline("STOP requested; unfinished continuations have no terminal label")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Deadline("15-minute probe deadline")
            env.response_timeout = min(10, max(.01, remaining))

        def run_root(env, root, source, decision):
            actor = next(r for r in root.raw["robots"] if r["id"] == root.raw["current_id"])
            policy, value = infer(root)
            greedy = max(root.legal_actions, key=lambda a: float(policy[a]))
            record = {"root_ordinal": len(results), "game_id": source["game_id"], "map": source["map"],
                      "seed": source["seed"], "decision_index": decision, "round": root.raw["round"],
                      "actor": actor, "player": root.player, "legal_actions": root.legal_actions,
                      "policy": policy.tolist(), "network_value": float(value), "greedy_action": greedy,
                      "root_state_sha256": hashlib.sha256(json.dumps({k:v for k,v in root.raw.items() if k!='snapshot'},sort_keys=True).encode()).hexdigest(),
                      "searches": [], "continuations": []}
            results.append(record)
            for simulations, depth in CONFIGS:
                budget(env)
                terminal_nodes, max_round_delta, leaf_values = 0, 0, []
                def transition(state, action):
                    nonlocal terminal_nodes, max_round_delta
                    budget(env)
                    child = env.transition(state, action)
                    terminal_nodes += child.terminal
                    max_round_delta = max(max_round_delta, child.raw['round'] - root.raw['round'])
                    return child
                def evaluate(state):
                    budget(env)
                    p, v = infer(state)
                    leaf_values.append(float(v))
                    return p, v
                began = time.monotonic()
                found = search(root, evaluate, transition, 40,
                               config=SearchConfig(simulations=simulations, max_depth=depth),
                               rng=np.random.default_rng(20261114 + record['root_ordinal']), add_root_noise=True)
                record['searches'].append({'simulations':simulations,'depth_cap':depth,
                    'selected_action':max(found.visits,key=found.visits.get),'policy':found.policy.tolist(),
                    'root_value':found.value,'nodes':found.nodes,'max_robot_turn_depth':found.max_depth,
                    'max_round_delta':max_round_delta,'terminal_nodes_created':terminal_nodes,
                    'leaf_value_mean':float(np.mean(leaf_values)),'leaf_value_std':float(np.std(leaf_values)),
                    'seconds':time.monotonic()-began})
                retain(env, root)
                publish(OUT / f"root-{record['root_ordinal']:02d}.json", record)
            candidates=[]
            for action in (greedy, record['searches'][-1]['selected_action'],37,24,28,22,20):
                if action in root.legal_actions and action not in candidates:
                    candidates.append(action)
                if len(candidates)==3:
                    break
            for action in candidates:
                for seed in CONTINUATION_SEEDS:
                    row={'first_action':action,'policy_rng_seed':seed,'status':'incomplete','terminal_value_root':None}
                    record['continuations'].append(row)
                    rng=np.random.default_rng(seed)
                    began=time.monotonic()
                    state=root
                    try:
                        budget(env)
                        state=env.transition(root,action)
                        decisions=1
                        retain(env,root,state)
                        while not state.terminal:
                            budget(env)
                            if decisions>=10000 or time.monotonic()-began>120:
                                raise Deadline('per-continuation cap; no terminal label')
                            p,_=infer(state)
                            legal=np.asarray(state.legal_actions,dtype=int)
                            logits=np.log(np.maximum(p[legal],1e-12))/.35
                            weights=np.exp(logits-logits.max());weights/=weights.sum()
                            state=env.transition(state,int(rng.choice(legal,p=weights)))
                            decisions+=1
                            retain(env,root,state)
                        row.update(status='ok',terminal_value_root=state.terminal_value_a*(1 if root.player==0 else -1),
                                   winner=state.raw['winner'],terminal_round=state.raw['round'],decisions=decisions)
                    except Deadline as error:
                        row.update(status='interrupted',error=str(error))
                        if time.monotonic()>=deadline:
                            raise
                    finally:
                        row['seconds']=time.monotonic()-began
                        retain(env,root)
                        publish(OUT / f"root-{record['root_ordinal']:02d}.json",record)
            print(json.dumps({'root':record['root_ordinal'],'game':source['game_id'],'role':actor['type'],
                              'round':record['round'],'search_actions':[r['selected_action'] for r in record['searches']],
                              'continuations':[(r['first_action'],r['policy_rng_seed'],r['terminal_value_root']) for r in record['continuations']]}),flush=True)

        stopped=None
        try:
            for path in paths:
                source=json.loads(path.read_text())
                assert source['source']=='selfplay' and source['opponent_checkpoint'] is None and source['status']=='ok'
                selected=set()
                with EngineEnv(action_version='macro_v2',response_timeout=10) as env:
                    state=env.reset(source['map'],source['seed'])
                    for decision, action in enumerate(source['actions']):
                        budget(env)
                        actor=next(r for r in state.raw['robots'] if r['id']==state.raw['current_id'])
                        role=actor['type']
                        if role not in selected and role in ('RAT_KING','BABY_RAT') and 37 in state.legal_actions:
                            run_root(env,state,source,decision)
                            selected.add(role)
                        if len(selected)==2:
                            break
                        state=env.transition(state,action)
                        retain(env,state)
        except Deadline as error:
            stopped=str(error)
        finally:
            unchanged=runtime==_training_runtime_identity() and digest(checkpoint)==config['checkpoint_sha256']
            publish(OUT / 'summary.json',{'experiment_id':config['experiment_id'],'roots':results,
                    'elapsed_seconds':time.monotonic()-started,'stopped':stopped,'runtime_and_checkpoint_unchanged':unchanged,
                    'completed_continuations':sum(r['status']=='ok' for x in results for r in x['continuations']),
                    'total_continuations':sum(len(x['continuations']) for x in results),
                    'benchmark_data_used':False,'gradient_updates':0})
            if not unchanged:
                raise ValueError('runtime/checkpoint changed during diagnostic')


if __name__=='__main__':
    main()
