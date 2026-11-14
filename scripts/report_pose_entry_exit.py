"""Summarize a user-confirmed frozen-model pose comparison, without live arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.pronoun_diagnostics import CURRENT_SHA,BASELINE_SHA,recover_competitor_margin


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(live,weights,bias):
    """No label inference: require explicit per-trial confirmation and condition."""
    rows=[];counts={}
    for t in live['trials']:
        if t.get('user_confirmation') is not True or t.get('trial_status')!='confirmed_valid':continue
        label=t['user_confirmed_label'];key=label+'/'+t['condition']
        count=counts.setdefault(key,{'confirmed_captures':0,'scored_full':0,'correct_full':0,'unavailable_full':0})
        count['confirmed_captures']+=1
        full=t['windows']['full']
        count['unavailable_full']+=full['status']!='scored'
        if full['status']=='scored':
            count['scored_full']+=1
            count['correct_full']+=full['variants']['original']['current']['top5'][0]['label']==label
        for name,window in t['windows'].items():
            row={'capture_id':t['capture_id'],'label':label,'condition':t['condition'],
                 'repetition':t['repetition'],'window':name,'status':window['status']}
            if window['status']=='scored':
                entry=window['variants']['original'];target=live['classes'].index(label)
                first=entry['current']['top5'][0]['label']
                row.update(top5=entry['current']['top5'],expected_score=entry['current']['scores'][target],
                    baseline_top5=entry['original_temporal_model']['top5'],
                    head_intervention_top1={k:v['top5'][0]['label'] for k,v in entry['head_interventions'].items()},
                    destructive_variant_top1={k:v['current']['top5'][0]['label'] for k,v in window['variants'].items()},
                    competitor_margin=recover_competitor_margin(entry['margin_explanation'],entry['current']['scores'],
                        weights,bias,live['classes'],label,first) if first!=label else None)
            else:row.update(reason=window.get('reason'),quality=window.get('quality'))
            rows.append(row)
    return {'counts':counts,'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',type=Path,required=True)
    parser.add_argument('--confirm-all',action='store_true',help='Only after user explicitly confirms gestures, conditions and normal capture')
    parser.add_argument('--exclude',nargs='*',type=int,default=[])
    args=parser.parse_args();live=read(args.live)
    if live.get('status')!='complete' or not live.get('pose_entry_exit_mode'):raise ValueError('Requires completed pose comparison')
    if args.confirm_all:
        backup=args.live.with_name(args.live.stem+'.before-condition-confirmation.json')
        if not backup.exists():backup.write_bytes(args.live.read_bytes())
        for t in live['trials']:
            if t['capture_id'] in args.exclude or t.get('user_confirmation') is False or t.get('trial_status') in ('processing_error','interrupted'):
                t.update(user_confirmation=False)
                if t['capture_id'] in args.exclude:t.update(trial_status='user_reported_invalid_condition')
            else:t.update(user_confirmation=True,trial_status='confirmed_valid',user_confirmed_label=t['prompted_label'])
        live['user_completion_confirmed']=all(t['user_confirmation'] is True for t in live['trials'])
        args.live.write_text(json.dumps(live,indent=2)+'\n',encoding='utf-8')
    checkpoint=ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt'
    baseline=ROOT/'models/asl-gru-fifteen-signs-temporal-v1/temporal/best.pt'
    before={str(p.relative_to(ROOT)):sha(p) for p in (checkpoint,baseline)}
    assert sha(checkpoint)==CURRENT_SHA and sha(baseline)==BASELINE_SHA
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for name,h in state['config']['source_sha256'].items():
        path=ROOT/'src/sign_language'/name
        assert sha(path)==h
        before[str(path.relative_to(ROOT))]=h
    result=summarize(live,state['model']['classifier.weight'].double().numpy(),state['model']['classifier.bias'].double().numpy())
    result.update(status='complete' if live['user_completion_confirmed'] else 'complete_with_exclusions_or_pending',
        date_local='2026-11-14',session_id=live['session_id'],live_report=live,
        protected_sha256=before,protected_unchanged=all(sha(ROOT/p)==h for p,h in before.items()),
        training_performed=False,media_saved=False,coordinates_saved=False,embeddings_saved=False,
        limits=['Two performances per label/condition, not a general accuracy estimate.',
                'Unscorable captures are reported separately, not as predicted wrong signs.',
                'Timed windows are proxies; fixed normalization uses retrospective full-capture context.',
                'Aggregate presence/motion does not prove accurate joints. Conditions are distinct performances.'])
    if not result['protected_unchanged']:raise AssertionError('Frozen source changed')
    output=ROOT/'docs/assets/pose-entry-exit-results-2026-11-14.json'
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    confirmed=[t for t in live['trials'] if t.get('user_confirmation') is True and t.get('trial_status')=='confirmed_valid']
    if confirmed:
        fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
        phases=['first_second','middle_second','last_second']
        for ax,t in zip(axes.flat,confirmed):
            selected={r['window']:r for r in result['rows'] if r['capture_id']==t['capture_id']}
            for suffix,title,color in [('', 'Refit','#2563eb'),('_fixed_normalization','Full-context fixed','#0d9488')]:
                values=[selected[p+suffix].get('expected_score',np.nan) for p in phases]
                ax.plot(range(3),values,'o-',label=title,color=color)
            full=selected['full'];first=full['top5'][0]['label'] if full['status']=='scored' else 'UNAVAILABLE'
            ax.set(title=f'#{t["capture_id"]} {t["prompted_label"]} {t["condition"]}\nFull: {first}',
                   xticks=range(3),xticklabels=['0–1s','1–2s','2–3s'],ylim=(-.03,1.03))
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(confirmed):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.supylabel('Intended-label softmax score; gaps mean unscorable')
        fig.suptitle('Frozen pronoun controls: actual timed phases; two repetitions per condition')
        fig.savefig(output.with_suffix('.png'),dpi=150);plt.close(fig)
    print(json.dumps(result['counts'],indent=2))
    for row in result['rows']:
        if row['window'] in ('full','middle_second','middle_second_fixed_normalization'):
            print(row['capture_id'],row['label'],row['condition'],row['window'],row.get('top5',row.get('reason')))


if __name__=='__main__':main()
