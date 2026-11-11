"""Report the fixed pooling comparison from saved metrics; no training or rescoring."""
import argparse
import json
from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import write_json


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--live-report',type=Path);a=p.parse_args()
    old=ROOT/'models/asl-gru-five-signs-early-v2/regularized';new=ROOT/'models/asl-gru-five-signs-mean-pool-v1'
    report=read(new/'report.json');original=report['historical_baseline_test'];pooled=report['test']
    before=read(old/'test-predictions.json');after=read(new/'test-predictions.json')
    if [x['video_file'] for x in before]!=[x['video_file'] for x in after]:raise ValueError('Test rows differ')
    report['paired_test_changes']={'improved':sum(not x['correct'] and y['correct'] for x,y in zip(before,after)),
                                 'regressed':sum(x['correct'] and not y['correct'] for x,y in zip(before,after))}
    report['runtime_consistency']=read(new/'runtime-consistency.json')
    old_epochs=[read(f) for f in sorted(old.glob('epoch-*.json'))];new_epochs=[read(f) for f in sorted(new.glob('epoch-*.json'))]
    report['epoch_metrics']={'original':old_epochs,'mean_pool':new_epochs}
    live=None
    if a.live_report is not None:
        live=read(a.live_report)
        if live['status']!='complete' or live['user_completion_confirmed'] is not True:
            raise ValueError('Only completed, user-confirmed live comparisons may be reported')
        if live['checkpoint_sha256']!=original['checkpoint_sha256'] or live['candidate']['checkpoint_sha256']!=pooled['checkpoint_sha256']:
            raise ValueError('Live checkpoints do not match report')
        report['live']=live
        report['live_summary']={name:{'correct':sum(t['variants'][key]['correct'] for t in live['trials']),
                   'trials':len(live['trials'])} for name,key in [('baseline','original'),('mean_pool','mean_pool')]}
    output=ROOT/'docs/assets/gru-mean-pooling-2026-11-11.json';write_json(output,report)
    fig,axes=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Five-sign GRU: final-state versus masked mean readout',fontsize=15)
    ax=axes[0,0]
    for name,rows,colors in [('Original',old_epochs,('#64748b','#2563eb')),('Mean pool',new_epochs,('#16a34a','#f59e0b'))]:
        for split,color,style in [('train_clean',colors[0],'--'),('val',colors[1],'-')]:
            ax.plot([r['epoch'] for r in rows],[r[split]['top1']*100 for r in rows],style,color=color,label=name+' '+('train' if split=='train_clean' else 'val'))
    ax.axvline(18,color='#2563eb',alpha=.4);ax.axvline(4,color='#f59e0b',alpha=.4)
    ax.set_xlabel('Epoch');ax.set_ylabel('Correct (%)');ax.set_ylim(0,105);ax.set_title('Same selection policy; selected epochs 18 and 4');ax.legend(fontsize=8)
    ax=axes[0,1]
    for rows,color,label in [(old_epochs,'#2563eb','Original'),(new_epochs,'#f59e0b','Mean pool')]:
        ax.plot([r['epoch'] for r in rows],[r['val']['loss'] for r in rows],color=color,label=label)
    ax.set_xlabel('Epoch');ax.set_ylabel('Cross-entropy');ax.set_title('Validation loss (19 clips)');ax.legend()
    ax=axes[1,0];labels=original['labels'];pos=np.arange(len(labels))
    for offset,test,label,color in [(-.18,original,'Original','#2563eb'),(.18,pooled,'Mean pool','#f59e0b')]:
        vals=[test['per_class'][g]['recall']*100 for g in labels]
        bars=ax.bar(pos+offset,vals,.35,label=label,color=color)
        for bar,g in zip(bars,labels):
            item=test['per_class'][g];ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{item["correct"]}/{item["support"]}',ha='center',fontsize=8)
    ax.set_xticks(pos,labels,fontsize=9);ax.set_ylim(0,115);ax.set_ylabel('Recall (%)');ax.set_title('Fixed test set: 42/60 → 47/60; NO remains weak');ax.legend(fontsize=8)
    ax=axes[1,1]
    if live is not None:
        pos=np.arange(len(live['trials']))
        for offset,key,label,color in [(-.18,'original','Original','#2563eb'),(.18,'mean_pool','Mean pool','#f59e0b')]:
            vals=[t['variants'][key]['expected_score']*100 for t in live['trials']]
            bars=ax.bar(pos+offset,vals,.35,label=label,color=color)
            for bar,t in zip(bars,live['trials']):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,'OK' if t['variants'][key]['correct'] else 'X',ha='center',fontsize=8)
        ax.set_xticks(pos,[f'{t["capture_id"]}\n{t["expected"]}' for t in live['trials']],fontsize=8)
        ax.set_ylim(0,110);ax.set_ylabel('Expected-sign score (%)');ax.set_title('Identical live inputs; OK/X indicates top-1 outcome');ax.legend(fontsize=8)
    else:ax.text(.5,.5,'Live paired comparison pending',ha='center',va='center');ax.axis('off')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output.with_suffix('.png'),dpi=160);plt.close(fig)
    print('Saved',output)
    if live is not None:print(json.dumps(report['live_summary']))


if __name__=='__main__':main()
