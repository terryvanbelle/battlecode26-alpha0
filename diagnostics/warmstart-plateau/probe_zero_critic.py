"""Paired zero-critic search control; uses only roots of counterfactual-v1."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from alpha0.env import EngineEnv
from alpha0.features_v2 import make_evaluator
from alpha0.learn_v2 import load_checkpoint, _training_runtime_identity
from alpha0.mcts import SearchConfig, search

spec=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('probe.py'))
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def main():
    torch.set_num_threads(1)
    stops=(ROOT/'runs/campaign/STOP',P.RUN/'STOP')
    if any(path.exists() for path in stops):
        raise ValueError('STOP exists; no diagnostic started')
    with (ROOT/'runs/warmstart-probe/probe.lock').open('a') as lock, (P.RUN/'experiment.lock').open('a') as training:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(training,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=json.loads((P.OUT/'summary.json').read_text())
        prior_config=json.loads((P.OUT/'config.json').read_text())
        output=P.OUT/'zero-critic.json'
        if output.exists():raise ValueError('immutable diagnostic already exists')
        remaining=min(180,900-original['elapsed_seconds'])
        if remaining<=5:raise ValueError('no remaining total probe budget')
        started=time.monotonic();deadline=started+remaining
        model,_=load_checkpoint(P.RUN/'checkpoint-0002.pt');model.eval()
        infer=make_evaluator(model)
        runtime=_training_runtime_identity()
        assert P.digest(P.RUN/'checkpoint-0002.pt')==prior_config['checkpoint_sha256']
        roots=original['roots'];results=[]
        def budget(env):
            if any(path.exists() for path in stops):
                raise P.Deadline('STOP requested; incomplete diagnostic excluded from terminal evidence')
            seconds=deadline-time.monotonic()
            if seconds<=0:raise P.Deadline('remaining paired-probe deadline')
            env.response_timeout=min(10,max(.01,seconds))
        stopped=None
        try:
            for game_id in dict.fromkeys(r['game_id'] for r in roots):
                source=json.loads((P.RUN/'games'/f'{game_id}.json').read_text())
                desired={r['decision_index']:r for r in roots if r['game_id']==game_id}
                with EngineEnv(action_version='macro_v2',response_timeout=10) as env:
                    state=env.reset(source['map'],source['seed'])
                    for decision,action in enumerate(source['actions']):
                        budget(env)
                        if decision in desired:
                            prior=desired[decision]
                            hashed=hashlib.sha256(json.dumps({k:v for k,v in state.raw.items() if k!='snapshot'},sort_keys=True).encode()).hexdigest()
                            assert hashed==prior['root_state_sha256']
                            for sims,depth in P.CONFIGS:
                                began=time.monotonic();max_round_delta=0;terminals=0
                                def transition(position,selected):
                                    nonlocal max_round_delta,terminals
                                    budget(env);child=env.transition(position,selected)
                                    max_round_delta=max(max_round_delta,child.raw['round']-state.raw['round'])
                                    terminals+=child.terminal
                                    return child
                                def evaluate(position):
                                    budget(env);policy,_=infer(position);return policy,0.0
                                found=search(state,evaluate,transition,40,
                                    config=SearchConfig(simulations=sims,max_depth=depth),
                                    rng=np.random.default_rng(20261114+prior['root_ordinal']),add_root_noise=True)
                                results.append({'root_ordinal':prior['root_ordinal'],'simulations':sims,'depth_cap':depth,
                                    'policy':found.policy.tolist(),'selected_action':max(found.visits,key=found.visits.get),
                                    'max_robot_turn_depth':found.max_depth,'max_round_delta':max_round_delta,
                                    'terminal_nodes_created':terminals,'seconds':time.monotonic()-began})
                                P.retain(env,state)
                        if decision>=max(desired):break
                        state=env.transition(state,action);P.retain(env,state)
        except P.Deadline as error:stopped=str(error)
        finally:
            assert runtime==_training_runtime_identity()
            P.publish(output,{'experiment_id':'warmstart-cp2-counterfactual-zero-critic-v1',
                'checkpoint_sha256':prior_config['checkpoint_sha256'],'probe_source_sha256':P.digest(__file__),
                'policy_and_roots_identical':True,'critic_output':0.0,'root_noise_identical':True,
                'results':results,'elapsed_seconds':time.monotonic()-started,'stopped':stopped,
                'total_with_terminal_probe_seconds':original['elapsed_seconds']+time.monotonic()-started,
                'benchmark_data_used':False,'gradient_updates':0,'runtime_unchanged':True})
            print(json.dumps({'zero_critic_searches':len(results),'elapsed_seconds':time.monotonic()-started,'stopped':stopped}))


if __name__=='__main__':main()
