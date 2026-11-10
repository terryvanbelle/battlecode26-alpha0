#!/usr/bin/env python3
"""Training-only cached-head diagnosis: no validation/benchmark reads or updates."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import torch
from torch.nn import functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize(prior,mask):
    p=np.where(mask,prior,0).astype(np.float64)
    return (p/p.sum(-1,keepdims=True)).astype(np.float32)

def flat(parts):
    return torch.cat([p.reshape(-1) for p in parts])

def cosine(a,b):
    den=float(a.norm()*b.norm())
    return float(torch.dot(a,b))/den if den else None

def scalar(x):
    return float(x.detach())

def masked_logp(hidden,weight,bias,mask):
    logits=F.linear(hidden,weight,bias)
    return logits,logits.masked_fill(~mask,torch.finfo(torch.float32).min).log_softmax(-1)

def outputs(state,rh,ah,rm,am):
    with torch.no_grad():
        logits,rlog=masked_logp(rh,state['policy.weight'],state['policy.bias'],rm)
        _,alog=masked_logp(ah,state['policy.weight'],state['policy.bias'],am)
    return logits,rlog,alog

def parameter_grad(loss,params,**kwargs):
    return flat(torch.autograd.grad(loss,params,retain_graph=True,**kwargs)).detach()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace/battlecode26-alpha0'))
    p.add_argument('--output',type=Path,default=Path('/tmp/v6-training-underfit.json'))
    args=p.parse_args(); root=args.root.resolve(); run=root/'runs/terminal-target-head-v6'
    inputs={'dataset':run/'dataset/training.npz','training_metadata':run/'dataset/training.json',
        'fit_manifest':run/'fit/manifest.json','cp2':root/'runs/warmstart-v3/checkpoint-0002.pt',
        'terminal_head':run/'fit/terminal_head.pt','puct_head':run/'fit/puct_head.pt'}
    hashes={k:sha(v) for k,v in inputs.items()}
    meta=json.loads(inputs['training_metadata'].read_text()); manifest=json.loads(inputs['fit_manifest'].read_text())
    assert manifest['complete'] and meta['complete'] and meta['source']=='own_selfplay'
    assert not meta['benchmark_data_used'] and not meta['validation_data_used']
    assert hashes['dataset']==meta['npz_sha256']==manifest['dataset']['sha256']
    assert hashes['training_metadata']==manifest['dataset']['metadata_sha256']
    assert hashes['cp2']=='20d9e813349010630940e1cdfbaee5206490c7d323997688d26f47508bbeee7c'
    payloads={k:torch.load(inputs[k],map_location='cpu',weights_only=True) for k in ('cp2','terminal_head','puct_head')}
    for arm in ('terminal_head','puct_head'):
        assert hashes[arm]==manifest['checkpoints'][arm]['sha256']
    with np.load(inputs['dataset'],allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    assert data['root_hidden'].shape==(32,128) and data['anchor_hidden'].shape==(512,128)
    # Match the loader normalization and then the head_loss normalization.
    for pref in ('root','anchor'):
        data[pref+'_prior']=normalize(data[pref+'_prior'],data[pref+'_mask'])
    rp=torch.as_tensor(normalize(data['root_prior'],data['root_mask']))
    ap=torch.as_tensor(normalize(data['anchor_prior'],data['anchor_mask']))
    rh=torch.as_tensor(data['root_hidden']); ah=torch.as_tensor(data['anchor_hidden'])
    rm=torch.as_tensor(data['root_mask'],dtype=torch.bool); am=torch.as_tensor(data['anchor_mask'],dtype=torch.bool)
    terminal=torch.as_tensor(data['terminal_actions']); puct=torch.as_tensor(data['puct_actions'])
    q=.25*rp+.75*F.one_hot(terminal,40).float()
    qentropy=-torch.xlogy(q,q).sum(-1)
    states={k:v['model_state_dict'] for k,v in payloads.items()}
    pred={k:outputs(s,rh,ah,rm,am) for k,s in states.items()}
    old=pred['cp2'][1].argmax(-1)
    assert torch.equal(old,puct)
    discord=np.flatnonzero(data['terminal_actions']!=data['puct_actions'])
    assert len(discord)==9
    anchor_game=data['anchor_game']; anchor_team=data['anchor_player']
    games=sorted(set(anchor_game.tolist())); weights=np.zeros(512,dtype=np.float64)
    anchor_counts=[]
    for game in games:
        for team in (0,1):
            idx=np.flatnonzero((anchor_game==game)&(anchor_team==team)); assert len(idx)>0
            weights[idx]=1/(len(games)*2*len(idx))
            anchor_counts.append({'game_index':game,'team':team,'anchors':len(idx)})
    assert abs(weights.sum()-1)<1e-12
    aw=torch.as_tensor(weights,dtype=torch.float32)
    root_frequency=torch.bincount(payloads['terminal_head']['batch_indices']['root'].flatten(),minlength=32).float()/4096
    anchor_frequency=torch.bincount(payloads['terminal_head']['batch_indices']['anchor'].flatten(),minlength=512).float()/4096
    gradient_results={}; grad_vectors={}; root_grads={}
    for label in ('cp2','terminal_head','puct_head'):
        w=states[label]['policy.weight'].clone().requires_grad_(True)
        b=states[label]['policy.bias'].clone().requires_grad_(True); params=(w,b)
        _,rlog=masked_logp(rh,w,b,rm); _,alog=masked_logp(ah,w,b,am)
        root_ce=-(q*rlog).sum(-1)
        per_anchor_kl=(torch.xlogy(ap,ap)-ap*alog).sum(-1)
        ce=root_ce.mean(); kl_uniform=per_anchor_kl.mean(); kl_weighted=(per_anchor_kl*aw).sum()
        ce_g=parameter_grad(ce,params); kl_g=parameter_grad(kl_weighted,params); total_g=ce_g+kl_g
        changed_g=parameter_grad(root_ce[discord].sum()/32,params)
        same_idx=np.flatnonzero(data['terminal_actions']==data['puct_actions'])
        unchanged_g=parameter_grad(root_ce[same_idx].sum()/32,params)
        assert torch.allclose(changed_g+unchanged_g,ce_g,atol=2e-6,rtol=2e-6)
        empirical_ce=(root_ce*root_frequency).sum()
        empirical_kl=(per_anchor_kl*anchor_frequency).sum()
        empirical_g=parameter_grad(empirical_ce+empirical_kl,params)
        roots_g=[parameter_grad(root_ce[i],params) for i in range(32)]
        root_grads[label]=roots_g
        grad_vectors[label]={'ce':ce_g,'anchor_kl':kl_g,'total':total_g}
        gradient_results[label]={
            'terminal_target_ce':scalar(ce),'changed9_target_ce_mean':scalar(root_ce[discord].mean()),
            'unchanged23_target_ce_mean':scalar(root_ce[same_idx].mean()),
            'changed9_ce_contribution_gradient_norm':float(changed_g.norm()),
            'unchanged23_ce_contribution_gradient_norm':float(unchanged_g.norm()),
            'changed_vs_unchanged_ce_gradient_cosine':cosine(changed_g,unchanged_g),
            'anchor_kl_uniform_storage':scalar(kl_uniform),
            'anchor_kl_game_team_balanced':scalar(kl_weighted),'expected_objective':scalar(ce+kl_weighted),
            'ce_gradient_norm':float(ce_g.norm()),'balanced_kl_gradient_norm':float(kl_g.norm()),
            'total_gradient_norm':float(total_g.norm()),'ce_kl_gradient_cosine':cosine(ce_g,kl_g),
            'ce_total_gradient_cosine':cosine(ce_g,total_g),
            'relative_kl_norm':float(kl_g.norm()/ce_g.norm()),
            'cancellation_fraction':1-float(total_g.norm()/(ce_g.norm()+kl_g.norm())),
            'mean_target_entropy':float(qentropy.mean()),
            'terminal_target_excess_ce':scalar(ce)-float(qentropy.mean()),
            'fixed128batch_frequency_ce':scalar(empirical_ce),'fixed128batch_frequency_kl':scalar(empirical_kl),
            'fixed128batch_frequency_total_gradient_norm':float(empirical_g.norm()),
            'expected_vs_fixed128batch_gradient_cosine':cosine(total_g,empirical_g)}
        if label=='terminal_head':
            stored=payloads[label]['fit_metrics']
            assert abs(scalar(ce)-stored['target_ce'])<2e-6 and abs(scalar(kl_uniform)-stored['anchor_kl'])<2e-6
    x=data['root_hidden'].astype(np.float64); a=data['anchor_hidden'].astype(np.float64)
    xn=np.linalg.norm(x,axis=1); an=np.linalg.norm(a,axis=1)
    assert (xn>0).all() and (an>0).all()
    cosroots=x@x.T/np.outer(xn,xn); cosanchors=x@a.T/np.outer(xn,an)
    augmented=np.column_stack([x,np.ones(32)])
    singular=np.linalg.svd(augmented,compute_uv=False)
    normalized_singular=np.linalg.svd(augmented/np.linalg.norm(augmented,axis=1,keepdims=True),compute_uv=False)
    nearest_rows=[]
    delta=flat([states['terminal_head']['policy.weight']-states['cp2']['policy.weight'],
                states['terminal_head']['policy.bias']-states['cp2']['policy.bias']])
    # A hypothetical 129th expected-gradient AdamW displacement, evaluated
    # algebraically from saved moments; it is never applied to a model.
    opt=payloads['terminal_head']['optimizer_state_dict']
    moments=opt['state']; group=opt['param_groups'][0]
    moment1=flat([moments[i]['exp_avg'] for i in (0,1)])
    moment2=flat([moments[i]['exp_avg_sq'] for i in (0,1)])
    theta=flat([states['terminal_head']['policy.weight'],states['terminal_head']['policy.bias']])
    eg=grad_vectors['terminal_head']['total']; beta1,beta2=group['betas']; step=129
    next_m=beta1*moment1+(1-beta1)*eg; next_v=beta2*moment2+(1-beta2)*eg.square()
    hypothetical_delta=-group['lr']*(next_m/(1-beta1**step)/(torch.sqrt(next_v/(1-beta2**step))+group['eps'])
        +group['weight_decay']*theta)
    def root_ref(j):
        return {'root_index':int(j),'root_id':meta['root_ids'][j],
                'stratum':data['root_strata'][j].tolist(),'terminal_action':int(terminal[j]),'cp2_action':int(old[j])}
    for i in discord:
        target=int(terminal[i]); previous=int(old[i]); row={'root_index':int(i),**root_ref(i),'selected_action':target}
        for label in ('cp2','terminal_head','puct_head'):
            logits,logp,_=pred[label]
            ce=-(q[i]*logp[i]).sum()
            row[label]={'selected_probability':float(logp[i,target].exp()),
                'cp2_action_probability':float(logp[i,previous].exp()),
                'selected_minus_cp2_logit_margin':float(logits[i,target]-logits[i,previous]),
                'selected_minus_best_other_logit_margin':float(logits[i,target]-logits[i].masked_fill(~rm[i],-float('inf')).masked_fill(F.one_hot(torch.tensor(target),40).bool(),-float('inf')).max()),
                'mixed_target_ce':float(ce),'mixed_target_excess_ce':float(ce-qentropy[i]),
                'hard_selected_nll':float(-logp[i,target]),'greedy_action':int(logp[i].argmax())}
        row['probability_gain_factor']=row['terminal_head']['selected_probability']/row['cp2']['selected_probability']
        row['margin_improvement']=row['terminal_head']['selected_minus_cp2_logit_margin']-row['cp2']['selected_minus_cp2_logit_margin']
        row['initial_logit_deficit_fraction_closed']=row['margin_improvement']/(-row['cp2']['selected_minus_cp2_logit_margin'])
        same=[j for j in range(32) if j!=i and terminal[j]==terminal[i]]
        conflict=[j for j in range(32) if j!=i and terminal[j]!=terminal[i]]
        mutual_conflict=[j for j in conflict if bool(rm[j,target]) and bool(rm[i,int(terminal[j])])]
        def nearest(group):
            if not group:return None
            j=max(group,key=lambda j:(cosroots[i,j],-j))
            return {**root_ref(j),'cosine':float(cosroots[i,j]),'euclidean_distance':float(np.linalg.norm(x[i]-x[j])),
                'this_selected_action_legal_there':bool(rm[j,target]),'neighbor_selected_action_legal_here':bool(rm[i,int(terminal[j])])}
        top=np.argsort(-cosanchors[i],kind='stable')[:5]
        row['features']={'norm':float(xn[i]),'nearest_same_target_root':nearest(same),'nearest_conflicting_target_root':nearest(conflict),
            'nearest_mutually_legal_conflicting_target_root':nearest(mutual_conflict),
            'nearest_anchors':[{'anchor_index':int(j),'input_hash':meta['anchor_input_hashes'][j],
                'source_id':meta['anchor_refs'][j].get('source_id',meta['anchor_refs'][j].get('game_id')),
                'player':int(anchor_team[j]),'cosine':float(cosanchors[i,j]),
                'euclidean_distance':float(np.linalg.norm(x[i]-a[j])),
                'cp2_greedy_action':int(ap[j].argmax()),'cp2_selected_action_probability':float(ap[j,target]),
                'selected_action_legal':bool(am[j,target]),'old_action_legal':bool(am[j,previous])} for j in top]}
        # Margin is linear in the 5,160 head parameters. Evaluate directions only;
        # never mutate parameters, construct an optimizer, or take a step.
        mw=torch.zeros_like(states['cp2']['policy.weight']); mb=torch.zeros_like(states['cp2']['policy.bias'])
        mw[target]=rh[i]; mw[previous]=-rh[i]; mb[target]=1; mb[previous]=-1
        margin_g=flat([mw,mb])
        assert abs(float(torch.dot(margin_g,delta))-row['margin_improvement'])<2e-5
        directions={}
        for name,g in grad_vectors['terminal_head'].items():
            n=float(g.norm())
            directions[name]={'raw_negative_gradient_margin_derivative':-float(torch.dot(margin_g,g)),
                'per_unit_parameter_norm_margin_derivative':-float(torch.dot(margin_g,g))/n if n else None,
                'own_root_ce_gradient_cosine':cosine(root_grads['terminal_head'][i],g)}
        own=root_grads['terminal_head'][i]
        others=[(cosine(own,g),j) for j,g in enumerate(root_grads['terminal_head']) if j!=i]
        worst_cos,worst_j=min(others)
        row['final_gradient']={'own_root_ce_norm':float(own.norm()),'directions':directions,
            'hypothetical_expected_gradient_adam_step129_margin_change':float(torch.dot(margin_g,hypothetical_delta)),
            'most_opposing_other_root':{**root_ref(worst_j),'gradient_cosine':worst_cos},
            'minimum_parameter_l2_for_pairwise_tie_with_cp2_action_root_alone':max(0,-row['terminal_head']['selected_minus_cp2_logit_margin'])/float(margin_g.norm())}
        nearest_rows.append(row)
    changed={k:{'roots':int((out[1].argmax(-1)!=old).sum()),
        'anchors':int((out[2].argmax(-1)!=pred['cp2'][2].argmax(-1)).sum())} for k,out in pred.items()}
    report={'format_version':1,'experiment_id':'terminal-target-head-v6',
        'scope':'training roots and training anchors only; gradient directions evaluated without updates',
        'source_sha256':sha(__file__),'input_files':{k:{'path':str(v.relative_to(root)),'sha256':hashes[k]} for k,v in inputs.items()},
        'no_parameter_updates':True,'no_engine_runs':True,'no_validation_inputs_or_outcomes_read':True,
        'no_benchmark_inputs_or_outcomes_read':True,'torch_threads':torch.get_num_threads(),
        'root_count':32,'anchor_count':512,'discordant_selected_targets':len(discord),'changed_greedy_actions':changed,
        'anchor_weighting':{'source_games':len(games),'game_team_strata':len(anchor_counts),
            'method':'uniform game, uniform team, uniform anchor within game/team','storage_counts':anchor_counts,
            'min_weight':float(weights.min()),'max_weight':float(weights.max())},
        'gradient_objective_definition':'All three models, including puct_head, are diagnosed against q=.75 terminal-selected one-hot + .25 legal-normalized cp2 prior; this is not the PUCT arm own fitted objective.',
        'expected_objective_gradients':gradient_results,'discordant_roots':nearest_rows,
        'hypothetical_expected_gradient_adam_step129':{'not_applied':True,'parameter_delta_l2':float(hypothetical_delta.norm()),
            'caveat':'Algebraic projection from actual saved moments using full expected gradient instead of a stochastic minibatch; not a forecast of the actual next step.'},
        'feature_geometry':{'root_hidden_norm_min':float(xn.min()),'root_hidden_norm_median':float(np.median(xn)),
            'root_hidden_norm_max':float(xn.max()),'anchor_hidden_norm_min':float(an.min()),
            'anchor_hidden_norm_median':float(np.median(an)),'anchor_hidden_norm_max':float(an.max()),
            'augmented_root_singular_values':singular.tolist(),'augmented_root_relative_singular_values':(singular/singular[0]).tolist(),
            'augmented_root_rank_rtol_1e_6':int((singular>singular[0]*1e-6).sum()),
            'augmented_root_rank_rtol_1e_9':int((singular>singular[0]*1e-9).sum()),
            'augmented_root_condition_number':float(singular[0]/singular[-1]),
            'row_normalized_augmented_root_singular_values':normalized_singular.tolist(),
            'row_normalized_augmented_root_condition_number':float(normalized_singular[0]/normalized_singular[-1]),
            'exact_duplicate_hidden_pairs':[[i,j] for i in range(32) for j in range(i+1,32) if np.array_equal(x[i],x[j])]},
        'terminal_head_parameter_delta_l2':float(delta.norm()),
        'limitations':['Selected target probabilities, CE and gradient diagnoses are in-sample, not generalization measurements.',
            'A gradient direction is infinitesimal and uses Euclidean parameter geometry; AdamW preconditioning and finite updates can differ.',
            'Hidden cosine similarity alone does not establish incompatible labels or a linear-capacity limit.',
            'The root feature rank does not measure the regularized optimum or costs imposed by the anchor distribution.',
            'Training metadata includes validation hash guard strings; no validation feature arrays, actions, or outcomes were accessed.']}
    assert {k:sha(v) for k,v in inputs.items()}==hashes,'input files changed during diagnosis'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if args.output.exists():assert args.output.read_bytes()==raw,'refusing to replace different report'
    else:
        with args.output.open('xb') as f:f.write(raw)
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output),'gradients':gradient_results,
        'feature_rank':report['feature_geometry']['augmented_root_rank_rtol_1e_6'],
        'condition':report['feature_geometry']['augmented_root_condition_number'],
        'rows':[{'root_id':r['root_id'],'p0':r['cp2']['selected_probability'],'p1':r['terminal_head']['selected_probability'],
          'margin0':r['cp2']['selected_minus_cp2_logit_margin'],'margin1':r['terminal_head']['selected_minus_cp2_logit_margin'],
          'ce_descent':r['final_gradient']['directions']['ce']['per_unit_parameter_norm_margin_derivative'],
          'kl_descent':r['final_gradient']['directions']['anchor_kl']['per_unit_parameter_norm_margin_derivative'],
          'net_descent':r['final_gradient']['directions']['total']['per_unit_parameter_norm_margin_derivative']} for r in nearest_rows]},indent=2))

if __name__=='__main__':main()
