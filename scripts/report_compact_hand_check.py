"""Report confirmed scalar articulation comparisons; no training or model rescoring."""
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    live=json.loads(a.report.read_text());reference=json.loads(a.reference.read_text())
    if live['status']!='complete' or live['user_completion_confirmed'] is not True:
        raise ValueError('Require a complete, user-confirmed six-trial session')
    if len(live['trials'])!=6 or any('error' in t for t in live['trials']):raise ValueError('Trial count/errors need review')
    metrics=['thumb_index_xy_range','index_tip_wrist_median','palm_axis_xyz_excursion_deg','wrist_body_xy_excursion']
    compact=[]
    for t in live['trials']:
        target=t['expected'];values=t['articulation']['metrics'];expected_ref=t['training_reference_comparison'][target]
        compact.append({'capture_id':t['capture_id'],'expected':target,
            'baseline':t['variants']['original']['top5'][0], 'mean_pool':t['variants']['mean_pool']['top5'][0],
            'usable_frames':t['articulation']['usable_frames'],'slot_change_flags':t['articulation']['single_hand_slot_changes'],
            'metrics':{k:{'live':values.get(k),'training':expected_ref.get(k)} for k in metrics}})
    counts={key:sum(t['variants'][key]['correct'] for t in live['trials']) for key in ('original','mean_pool')}
    output=ROOT/'docs/assets/compact-hand-live-2026-11-12.json'
    result={'summary':{'correct_out_of_6':counts,'trials':compact,
              'limits':'Dominant slot includes detected rest. Distances/angles are estimates; percentile ranks are not recognition confidence or proof of tracking accuracy. No coordinates/media saved.'},
            'live':live,'reference':reference}
    write_json(output,result)
    fig,axes=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Compact-hand diagnostic: source features versus matched webcam inputs',fontsize=14)
    for ax,key,title,units in zip(axes.flat[:3],metrics[:3],
        ['Finger opening range (thumb/index)','Index fingertip distance from wrist','Estimated palm-axis directional excursion'],
        ['Palm-normalized distance','Palm-normalized distance','Degrees; estimated xyz proxy']):
        labels=['NO','YES','HELLO'];groups=[]
        for gloss in labels:
            values=[r['articulation']['metrics'].get(key) for r in reference['records'] if r['split']=='train' and r['gloss']==gloss]
            groups.append([v for v in values if v is not None])
        ax.boxplot(groups,tick_labels=labels,showfliers=True)
        for t in live['trials']:
            v=t['articulation']['metrics'].get(key)
            if v is None:continue
            x=labels.index(t['expected'])+1
            ax.scatter(x,v,color='#ea580c',marker='D',zorder=3)
            ax.annotate('#'+str(t['capture_id']),(x,v),xytext=(7,2),textcoords='offset points',fontsize=8)
        ax.set_title(title,fontsize=11);ax.set_ylabel(units);ax.text(.02,.98,'Boxes: training clips | diamonds: live trials',transform=ax.transAxes,va='top',fontsize=8)
    ax=axes[1,1];pos=np.arange(6)
    for offset,key,label,color in [(-.18,'original','Original GRU','#2563eb'),(.18,'mean_pool','Mean-pool GRU','#f59e0b')]:
        bars=ax.bar(pos+offset,[t['variants'][key]['expected_score']*100 for t in live['trials']],.35,label=label,color=color)
        for b,t in zip(bars,live['trials']):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,'OK' if t['variants'][key]['correct'] else 'X',ha='center',fontsize=8)
    ax.set_xticks(pos,[f'#{t["capture_id"]}\n{t["expected"]}' for t in live['trials']],fontsize=8)
    ax.set_ylim(0,115);ax.set_ylabel('Expected-sign score (%)');ax.set_title('Same observations; scores are uncalibrated',fontsize=11);ax.legend(fontsize=8)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps({'correct_out_of_6':counts,'trials':compact},indent=2));print('Saved',output)


if __name__=='__main__':main()
