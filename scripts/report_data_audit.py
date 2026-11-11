"""Publish compact aggregate audit evidence after the full read-only scan finishes."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def quantiles(values):
    return dict(zip(('min','p05','p25','median','p75','p95','p99','max'),
                    np.quantile(values,[0,.05,.25,.5,.75,.95,.99,1]).tolist(),strict=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);args=parser.parse_args()
    source=args.audit
    summary=json.loads((source/'summary.json').read_text())
    status=json.loads((source/'status.json').read_text())
    if summary['status']!='complete' or summary['clips']!=83399 or status['completed_shards']!=836:
        raise ValueError('Complete all 836 audit shards before publishing a final report')
    rows=[]
    for path in sorted((source/'parts').glob('*.json')):
        part=json.loads(path.read_text())
        if part['identity']!=summary['identity']:raise ValueError('Audit identity mismatch')
        rows.extend(part['clips'])
    slots=json.loads((source/'slot-change-summary.json').read_text())
    focused=json.loads((source/'visual-review/focused-summary.json').read_text())
    fresh=json.loads((source/'fresh-extraction-check.json').read_text())
    metadata=json.loads((source/'metadata-statistics.json').read_text())
    valid=[r for r in rows if 'frames' in r]
    extended={k:v for k,v in summary.items() if k not in ('exact_duplicate_groups',)}
    extended['duplicate_groups_count']=len(summary['exact_duplicate_groups'])
    extended['duplicate_group_examples']=summary['exact_duplicate_groups'][:5]
    extended['duration_seconds_quantiles']=quantiles([r['frames']/r['fps'] for r in valid])
    extended['frame_count_quantiles']=quantiles([r['frames'] for r in valid])
    extended['video_dimensions']=dict(Counter(f"{r['frame_shape'][1]}x{r['frame_shape'][0]}" for r in valid))
    extended['clips_over_128_frames']=sum(r['frames']>128 for r in valid)
    extended['slot_change_screen']=slots
    extended['focused_train_val']=focused
    extended['official_metadata_statistics']=metadata
    extended['fresh_extraction_check']=fresh
    extended['decoded_sample_all_black_clips']=sum(r.get('decode',{}).get('black_frame_fraction',0)==1 for r in rows)
    extended['decoded_sample_high_duplicate_thumbnail_clips']=sum(r.get('decode',{}).get('identical_thumbnail_fraction',0)>.9 for r in rows)
    extended['interpretation']={
        'slot_changes':'Heuristic candidates; visually confirmed identity error in one YES clip. Not a corpus-wide ground-truth error rate.',
        'hand_coverage':'Fraction of full video frames, including real rest periods. Not detector accuracy.',
        'source_crc':'Compared extracted bytes with retained archive central-directory CRC32; new SHA256 baseline, not historical per-video SHA identity.',
        'video_decoding':'298 selected full-video decodes, not every video.',
        'semantic_labels':'No semantic mislabel confirmed; ASL-competent review required.',
    }
    destination=ROOT/'docs/assets/video-feature-audit-2026-11-11.json'
    destination.write_text(json.dumps(extended,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#2563eb','#d97706','#15803d']
    for s,color in zip(('train','val','test'),colors,strict=True):
        selected=[r['hand_coverage'] for r in valid if r['split']==s]
        axes[0,0].hist(selected,bins=np.linspace(0,1,21),density=True,histtype='step',linewidth=2,label=s,color=color)
    axes[0,0].set(title='Whole-clip hand detection coverage',xlabel='Fraction of frames with any hand',ylabel='Density');axes[0,0].legend()
    durations=np.array([r['frames']/r['fps'] for r in valid])
    axes[0,1].hist(durations,bins=np.geomspace(durations.min(),durations.max(),45),color='#475569')
    axes[0,1].set_xscale('log');axes[0,1].set(title='Source clip durations',xlabel='Seconds (log scale)',ylabel='Clips')
    rates=[100*slots[s]['clips_with_nearby_slot_changes']/slots[s]['clips'] for s in slots]
    bars=axes[1,0].bar(list(slots),rates,color=colors)
    axes[1,0].bar_label(bars,fmt='%.1f%%');axes[1,0].set(title='Nearby hand-slot changes: candidates',ylabel='Clips flagged (%)',ylim=(0,max(rates)*1.25))
    bars=axes[1,1].bar(list(focused),[100*v['median_hand_fraction'] for v in focused.values()],color='#7c3aed')
    axes[1,1].bar_label(bars,fmt='%.1f%%');axes[1,1].set(title='Five focus signs: train + validation only',ylabel='Median whole-clip hand coverage (%)',ylim=(0,60))
    fig.suptitle('ASL Citizen audit: integrity and representation quality',fontsize=15)
    fig.savefig(ROOT/'docs/assets/video-feature-audit-2026-11-11.png',dpi=150)
    plt.close(fig)
    print(json.dumps({k:extended[k] for k in ('clips','clips_with_errors','duration_seconds_quantiles','clips_over_128_frames','duplicate_groups_count','checkpoint_unchanged')},indent=2))


if __name__=='__main__':main()
