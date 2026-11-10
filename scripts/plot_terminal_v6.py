#!/usr/bin/env python3
"""Plot frozen v6 evidence; incomplete observations are coverage, never losses."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("cp2", "terminal_head", "puct_head")
LABELS = {"cp2": "Fresh cp2", "terminal_head": "Terminal-ranking head", "puct_head": "Matched PUCT head"}
COLORS = {"cp2": "#64748b", "terminal_head": "#0f766e", "puct_head": "#b45309"}


def numeric(value, name, low=0, high=1):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return value


def same(value, expected, name):
    if value is None and expected is None:
        return
    if expected is None or type(value) not in (int,float) or not math.isfinite(value) or abs(value-expected)>1e-9:
        raise ValueError(f"Recorded {name} disagrees with measured scores")


def report_view(summary):
    if summary.get("experiment_id") != "terminal-target-head-v6" or summary.get("schema_version") != 1:
        raise ValueError("Expected a v6 report")
    sampling = summary["sampling"]
    if (sampling.get("source_temperature") != 1.0 or sampling.get("continuation_temperature") != .35 or
        sampling.get("source_policy") != "frozen_cp2_both_teams" or sampling.get("v5_data_reused") is not False or
        sampling.get("seed_inventory") != "paired_v5_same_seeds_fresh_generation"):
        raise ValueError("V6 source/continuation protocol differs from the coverage amendment")
    decision=summary["decision"]
    if decision.get("automatic_promotion") is not False or decision.get("all_map_victory") is not False:
        raise ValueError("This pilot cannot claim model promotion or all-map victory")
    validation=summary["validation"]; roots=validation["roots"]
    if len({root['root_id'] for root in roots}) != len(roots):
        raise ValueError("Repeated validation roots")
    for root in roots:
        if not root['complete']:
            if any(root['means'][arm] is not None for arm in ARMS) or root['delta_vs_puct'] is not None or root['delta_vs_cp2'] is not None:
                raise ValueError("Incomplete roots cannot carry final scores")
            continue
        for arm in ARMS:
            scores=root['scores'][arm]
            if len(scores)!=4 or any(type(score) not in (int,float) or score not in (0,.5,1) for score in scores):
                raise ValueError("Root lacks four genuine paired outcomes")
            same(root['means'][arm],sum(scores)/4,'root mean')
        for a in ARMS:
            for b in ARMS:
                if root['actions'][a]==root['actions'][b] and root['scores'][a]!=root['scores'][b]:
                    raise ValueError("Identical first actions must share physical outcomes")
        same(root['delta_vs_puct'],root['means']['terminal_head']-root['means']['puct_head'],'paired PUCT delta')
        same(root['delta_vs_cp2'],root['means']['terminal_head']-root['means']['cp2'],'paired cp2 delta')
    validation_complete=(len(roots)==16 and all(root['complete'] for root in roots) and validation['unresolved_errors']==0)
    if validation['complete'] is not validation_complete:
        raise ValueError("Validation completion omits planned roots/errors")
    delta=sum(root['delta_vs_puct'] for root in roots)/16 if validation_complete else None
    cp2_delta=sum(root['delta_vs_cp2'] for root in roots)/16 if validation_complete else None
    same(validation['mean_delta_vs_puct'],delta,'equal-root validation mean')
    same(validation['mean_delta_vs_cp2'],cp2_delta,'equal-root cp2 mean')
    panels=summary['benchmarks']['panels']
    if panels and set(panels)!=set(ARMS):
        raise ValueError("Expected all three benchmark arm labels")
    for panel in panels.values():
        if panel['expected']!=24 or len(panel['opponents'])!=12:
            raise ValueError("Benchmark panel lacks twelve opponent pairs")
        scores=[]
        for opponent in panel['opponents'].values():
            sides=opponent['sides']
            for side in ('A','B'):
                if sides[side] is not None:
                    score=numeric(sides[side]['score'],'benchmark outcome')
                    if score not in (0,.5,1): raise ValueError('Invalid terminal outcome')
                    scores.append(score)
            complete=all(sides[side] is not None for side in ('A','B')) and opponent['unresolved_errors']==0
            if opponent['complete'] is not complete: raise ValueError('Incorrect opponent coverage')
            same(opponent['mean_score'],sum(sides[side]['score'] for side in ('A','B'))/2 if complete else None,'opponent mean')
        if len(scores)!=panel['completed']: raise ValueError('Benchmark score/count mismatch')
        complete=len(scores)==24 and panel['unresolved_errors']==0
        if panel['complete'] is not complete: raise ValueError('Incorrect full-panel coverage')
        same(panel['mean_score'],sum(scores)/24 if complete else None,'panel score')
    benchmark_complete=bool(panels) and all(panel['complete'] for panel in panels.values())
    if summary['benchmarks']['complete'] is not benchmark_complete:
        raise ValueError('Benchmark completion differs from all three panels')
    if decision['review_for_expansion']:
        if not (decision['criteria_pass'] and decision['integrity_verified'] and decision['complete'] and validation_complete and benchmark_complete
                and summary['benchmarks']['identity_verified'] and summary['benchmarks']['per_opponent_nonregression']
                and delta>=.10 and cp2_delta>=0 and sum(root['delta_vs_puct']>0 for root in roots)>=2
                and min(root['delta_vs_puct'] for root in roots)>=-.25):
            raise ValueError('Expansion claim lacks complete qualifying evidence')
    state=('READY FOR RESEARCH REVIEW' if decision['review_for_expansion'] else
           'AWAITING FINAL INTEGRITY AUDIT' if decision['criteria_pass'] else
           'DID NOT MEET RESEARCH RULE' if decision['complete'] else 'INCOMPLETE')
    return {'state':state,'roots':roots,'panels':panels,'validation_complete':validation_complete,'benchmark_complete':benchmark_complete}


def atomic_text(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name); stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try: os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def save_figure(fig,path):
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.png',delete=False) as stream: temporary=Path(stream.name)
    try:
        fig.savefig(temporary,dpi=150,facecolor='white'); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def generate(summary,output_dir):
    view=report_view(summary); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    digest=hashlib.sha256(encoded).hexdigest()
    snapshot=output_dir/'report-snapshots'/f'{digest}.json'
    if not snapshot.exists(): atomic_text(snapshot,json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'bc26-alpha0-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    roots=view['roots']; ax=axes[0]
    for index,root in enumerate(roots):
        if root['complete']:
            ax.scatter(index-.12,root['delta_vs_puct'],color='#0f766e',s=38,label='Versus matched PUCT' if index==0 else None)
            ax.scatter(index+.12,root['delta_vs_cp2'],color='#7c3aed',s=35,marker='D',label='Versus cp2' if index==0 else None)
        else: ax.axvspan(index-.45,index+.45,color='#e2e8f0',alpha=.8)
    ax.axhline(0,color='#64748b',linewidth=1); ax.axhline(.10,color='#0f766e',linestyle=':',linewidth=1)
    ax.set_ylim(-1.08,1.08); ax.set_xlim(-.6,max(15,len(roots)-1)+.6)
    ax.set_xticks(range(len(roots)),[root['root_id'].replace('validation-','') for root in roots],rotation=90)
    ax.set_ylabel('Terminal-head mean score advantage'); ax.set_xlabel('Frozen validation root')
    ax.set_title('First action only · cp2 plays the remainder',fontsize=11)
    if any(root['complete'] for root in roots):
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([],[],color='#0f766e',marker='o',linestyle='',label='Versus matched PUCT'),
                           Line2D([],[],color='#7c3aed',marker='D',linestyle='',label='Versus cp2')],fontsize=8)
    if not any(root['complete'] for root in roots): ax.text(.5,.75,'No complete paired root scores',transform=ax.transAxes,ha='center',color='#64748b')
    ax=axes[1]
    for index,arm in enumerate(ARMS):
        panel=view['panels'].get(arm)
        if panel and panel['complete']:
            ax.bar(index,panel['score_sum'],color=COLORS[arm],width=.6)
            ax.text(index,panel['score_sum']+.45,f"{panel['score_sum']:g}/24",ha='center',fontsize=10)
        else:
            ax.text(index,.35,f"{panel['completed'] if panel else 0}/24 valid\npanel incomplete",ha='center',va='bottom',fontsize=9,color='#64748b')
    ax.axhline(20,color='#475569',linestyle='--',label='Historical cp2: 20/24 (reference only)')
    ax.set_ylim(0,26); ax.set_xlim(-.6,2.6); ax.set_xticks(range(3),[LABELS[a] for a in ARMS],fontsize=9)
    ax.set_ylabel('Score points from 24 games'); ax.set_title('Whole-policy deployment · fresh 12-bot panels',fontsize=11)
    ax.legend(loc='lower left',fontsize=8)
    fig.suptitle(f"Terminal-target-head-v6 · {view['state']}",fontweight='bold')
    fig.text(.5,.015,'Gray/blank means unresolved, not a loss. Different panels measure different behaviors; neither is the all-map victory gate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93)); save_figure(fig,output_dir/'comparison.png'); plt.close(fig)
    selection=summary['selection']; validation=summary['validation']; bench=summary['benchmarks']; anchors=summary['anchors']
    stages=[('Training roots',selection['frozen_roots'],selection['planned_roots']),('KL anchors',anchors['completed'],512),
            ('Training terminal cells',selection['completed_cells'],selection['expected_cells']),
            ('Validation roots',validation['completed_roots'],16),
            ('Validation terminal cells',validation['completed_unique_cells'],validation['expected_unique_cells']),
            ('Fresh benchmark cells',bench['completed'],72)]
    fig,ax=plt.subplots(figsize=(10,4.7))
    for index,(name,done,total) in enumerate(stages):
        ax.barh(index,1,color='#e2e8f0')
        if total: ax.barh(index,min(1,done/total),color='#0f766e')
        ax.text(1.02,index,f'{done:,}/{total:,}' if total else f'{done:,}; plan not frozen',va='center',fontsize=10)
    ax.set_yticks(range(len(stages)),[row[0] for row in stages]); ax.invert_yaxis(); ax.set_xlim(0,1.35)
    ax.set_xticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']); ax.set_title('Coverage of preregistered allocation')
    fig.tight_layout(); save_figure(fig,output_dir/'coverage.png'); plt.close(fig)
    lines=['# Terminal-target-head-v6 progress','',f"**{view['state']}. No automatic promotion or all-map victory claim.**",'',
           f"Frozen experiment identity: `{summary['identity_sha256']}`. Exact plotted [report snapshot](report-snapshots/{digest}.json).",'',
           '![Coverage](coverage.png)','','![Separate transfer and deployment checks](comparison.png)','',
           '## What these panels measure','',
           '- **Preregistered source sampling:** frozen cp2 on both teams at temperature **1.0** on all four development maps. Continuations remain at **0.35**. The complete source/reserve and continuation seed inventory is paired with v5; every v6 source game and cache is regenerated. No v5 trajectory is reused.',
           '- **First-action validation:** a fitted head chooses one action; unchanged cp2 controls both teams afterward. Four paired continuation seeds per root; roots, not cells, receive equal weight.',
           '- **Whole-policy benchmark:** fresh cp2 and both fixed candidate policies each play all twelve bots on tiny/seed0/both seats. Every action uses that deployed policy. Historical 20/24 is a labeled reference and supplies no missing fresh results.',
           '- Incomplete/error games have no score. Earlier failed attempts remain recorded even when a valid retry resolves the cell. No benchmark data enters targets or gradients.','',
           '## Coverage','', '| Stage | Observed | Planned |','|---|---:|---:|']
    lines.extend(f'| {name} | {done:,} | {total if total is not None else "Not yet frozen"} |' for name,done,total in stages)
    lines += ['',f"Unresolved errors: training cells **{selection['unresolved_errors']}**, validation cells **{validation['unresolved_errors']}**, benchmark cells **{bench['unresolved_errors']}**.",'',
              '| Source split | Games attempted | Games completed | Games represented by roots | Skipped exact gradient-input duplicates | Start-only attempts |',
              '|---|---:|---:|---:|---:|---:|']
    for split,counts in summary['collection'].items():
        lines.append(f"| {split} | {counts['attempted_source_games']} | {counts['completed_source_games']} | {counts['source_games_represented_by_roots']} | {counts['skipped_exact_input_duplicates']} | {counts.get('start_only_attempts',0)} |")
    lines += ['',f"Validation has **{validation['distinct_source_games_represented']}** distinct represented source games and **{validation['distinct_validation_inputs']}** distinct model inputs. Continuation count is not an independent training-state or scenario count.",'',
              '## Paired first-action validation','', '| Root | Map / role / team | cp2 | Terminal head | PUCT head | Terminal − PUCT | Terminal − cp2 |','|---|---|---:|---:|---:|---:|---:|']
    fmt=lambda value:'—' if value is None else f'{value:.3f}'
    for root in roots:
        lines.append(f"| {root['root_id']} | {root['map']} / {root['role']} / {root['player']} | {fmt(root['means']['cp2'])} | {fmt(root['means']['terminal_head'])} | {fmt(root['means']['puct_head'])} | {fmt(root['delta_vs_puct'])} | {fmt(root['delta_vs_cp2'])} |")
    lines += ['',f"Complete equal-root mean advantages: versus PUCT **{fmt(validation['mean_delta_vs_puct'])}**, versus cp2 **{fmt(validation['mean_delta_vs_cp2'])}**. Missing roots are not dropped from these means.",'',
              '## Every whole-policy benchmark comparison','', '| Opponent | Fresh cp2 A/B | Terminal head A/B | PUCT head A/B | Terminal vs cp2 |','|---|---|---|---|---|']
    def seats(arm,name):
        opponent=view['panels'].get(arm,{}).get('opponents',{}).get(name)
        if not opponent: return '— / —'
        return ' / '.join('—' if opponent['sides'][seat] is None else {0:'L',.5:'D',1:'W'}[opponent['sides'][seat]['score']] for seat in ('A','B'))
    for name,comparison in bench['opponent_comparisons'].items():
        result='Pending' if not comparison['complete'] else 'No regression' if comparison['nonregression'] else 'Regressed'
        lines.append(f"| `{name}` | {seats('cp2',name)} | {seats('terminal_head',name)} | {seats('puct_head',name)} | {result} |")
    lines += ['', 'The research rule requires complete independent validation, mean terminal-head advantage ≥0.10 over PUCT, at least two improved roots, no root worse than −0.25, nonnegative mean versus cp2, all 72 fresh benchmark games without unresolved errors, and no regression against **any** individual benchmark opponent. Final input/runtime integrity must also be verified. Passing only permits a recorded review of expansion.','']
    atomic_text(output_dir/'README.md','\n'.join(lines))
    return {'report_sha256':digest,'comparison_png':str(output_dir/'comparison.png'),'coverage_png':str(output_dir/'coverage.png'),'readme':str(output_dir/'README.md')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'progress/terminal-target-head-v6/summary.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'progress/terminal-target-head-v6')
    args=parser.parse_args(); print(json.dumps(generate(json.loads(args.report.read_text()),args.output_dir),indent=2))


if __name__=='__main__': main()
