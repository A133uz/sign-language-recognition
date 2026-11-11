"""Publish reproducible aggregate diagnostic evidence; exclude banks and webcam media."""
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
from sign_language.five_sign_pilot import write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();base=a.source
    raw={name:json.loads((base/file).read_text()) for name,file in {
        'offline':'offline-report.json','phase_controls':'phase-report.json','live':'live-report.json',
        'camera':'camera-report.json','scale_isolation':'scale-isolation-detailed.json'}.items()}
    live=raw['live'];offline=raw['offline']
    if live['status']!='complete' or live['user_completion_confirmed'] is not True:raise ValueError('Unconfirmed/incomplete live trials')
    for section in ('offline','camera'):
        if raw[section]['status']!='complete':raise ValueError('Incomplete diagnostic')
    live_counts={name:sum(t['variants'][name]['correct'] for t in live['trials']) for name in live['trials'][0]['variants']}
    heads={name:sum(t['diagnostic_heads'][name]['correct'] for t in live['trials']) for name in live['trials'][0]['diagnostic_heads']}
    cache=ROOT/'data/training/asl-five-signs-128-v1';manifest=json.loads((cache/'manifest.json').read_text())
    unchanged=all(sha256(cache/f)==digest for f,digest in manifest['files_sha256'].items())
    checkpoint=ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt'
    if not unchanged or sha256(checkpoint)!=live['checkpoint_sha256']:raise AssertionError('Sources changed')
    summary={'live_variant_correct_out_of_6':live_counts,'live_probe_correct_out_of_6':heads,
             'source_checkpoint_unchanged':True,'source_cache_unchanged':True,
             'test_predictions_not_rescored':True,'production_model_trained':False,
             'scratch_heads_fitted_on_training_only':True,'new_live_media_or_coordinates_saved':False,
             'interpretation':'Multiple demonstrated sensitivities, not proof of one universal cause. Small selected diagnostic samples are not independent webcam benchmarks.'}
    output=ROOT/'docs/assets/recognition-forensics-2026-11-11.json'
    write_json(output,{'summary':summary,**raw})
    fig,axes=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Frozen five-sign GRU: diagnostic evidence (small samples)',fontsize=15)
    methods=['Original GRU','New linear head','New MLP head','Hand summaries']
    train=[66,75,75,75];val=[15,15,15,17];webcam=[live_counts['original'],heads['hidden_linear'],heads['hidden_mlp'],heads['hands_linear']]
    positions=np.arange(4);ax=axes[0,0]
    for offset,counts,total,label,color in [(-.24,train,75,'Train (75)','#9ca3af'),(0,val,19,'Validation (19)','#2563eb'),(.24,webcam,6,'Guided live (6)','#f59e0b')]:
        bars=ax.bar(positions+offset,np.asarray(counts)/total*100,.23,label=label,color=color)
        for b,count in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{count}/{total}',ha='center',fontsize=7)
    ax.set_xticks(positions,methods,rotation=13);ax.set_ylim(0,115);ax.set_ylabel('Correct (%)');ax.set_title('Replacing the head did not solve live recognition');ax.legend(fontsize=8)
    case=live['trials'][4];prefix=case['prefix_progression'];ax=axes[0,1]
    for label,color in [('THANKYOU','#16a34a'),('NO','#dc2626')]:
        values=[next(x['score'] for x in v['top5'] if x['label']==label)*100 for v in prefix]
        ax.plot([v['fraction']*100 for v in prefix],values,'o-',label=label,color=color)
    ax.set_xlabel('Captured sequence included (%)');ax.set_ylabel('Uncalibrated score (%)');ax.set_ylim(0,100)
    ax.set_title('THANK YOU trial #5: later input overturns recognition');ax.legend()
    isolation=raw['scale_isolation'];ax=axes[1,0]
    keys=['original','scaled_fresh_mediapipe','scaled_exact_landmarks','restore_pose']
    vals=[isolation['variants'][k]['expected_score']*100 for k in keys]
    ax.bar(np.arange(4),vals,color=['#2563eb','#dc2626','#16a34a','#8b5cf6'])
    ax.set_xticks(np.arange(4),['Original','Smaller person\nnew tracking','Exact landmark\nscale transform','Restore original\npose features'],fontsize=8)
    ax.set_ylim(0,100);ax.set_ylabel('HELLO score (%)');ax.set_title('Same public HELLO video: scale sensitivity isolated')
    for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.1f}%',ha='center')
    ax=axes[1,1];training=[x['hand_frame_fraction']*100 for x in offline['splits']['train']['features']]
    observed=[t['feature_summary']['hand_frame_fraction']*100 for t in live['trials']]
    ax.boxplot([training,observed],tick_labels=['Train clips (75)','Guided live (6)'],showfliers=True)
    ax.scatter(np.full(6,2.),observed,color='#f59e0b',zorder=3)
    ax.set_ylabel('Frames with detected hands (%)');ax.set_ylim(0,105)
    ax.set_title('Visibility context differs; this is not detector accuracy')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2));print('Evidence:',output)


if __name__=='__main__':main()
