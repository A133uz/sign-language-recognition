"""Verify and plot actual five-sign example counts; no inference or data changes."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json


def main():
    cache=ROOT/'data/training/asl-five-signs-128-v1';manifest=json.loads((cache/'manifest.json').read_text())
    names=[g for g,i in sorted(manifest['labels'].items(),key=lambda x:x[1])];splits={}
    for split in ('train','val','test'):
        index_path=cache/f'{split}-index.json';target_path=cache/f'{split}-targets.npy'
        for path in (index_path,target_path):
            if sha256(path)!=manifest['files_sha256'][path.name]:raise ValueError('Payload differs from manifest')
        rows=json.loads(index_path.read_text());targets=np.load(target_path,allow_pickle=False)
        features=np.load(cache/f'{split}-features.npy',mmap_mode='r');lengths=np.load(cache/f'{split}-lengths.npy')
        if not len(rows)==len(targets)==len(features)==len(lengths):raise ValueError('Row count mismatch')
        if len({r['video_file'] for r in rows})!=len(rows):raise ValueError('Repeated source videos')
        if not np.array_equal(targets,np.asarray([manifest['labels'][r['gloss']] for r in rows])):raise ValueError('Target/label mismatch')
        counts=Counter(r['gloss'] for r in rows)
        if dict(counts)!=manifest['counts'][split]:raise ValueError('Manifest count mismatch')
        splits[split]={'clips':len(rows),'unique_signers':len({r['participant_id'] for r in rows}),
            'counts':{g:counts[g] for g in names},'percent':{g:counts[g]/len(rows)*100 for g in names},
            'unique_signers_by_class':{g:len({r['participant_id'] for r in rows if r['gloss']==g}) for g in names}}
    totals={g:sum(splits[s]['counts'][g] for s in splits) for g in names};total=sum(totals.values())
    train_counts=list(splits['train']['counts'].values())
    report={'cache_id':manifest['cache_id'],'unit':'cached source video clips, not frames or augmentation variants',
        'verification':'Target arrays, label/index rows, feature/length row counts and manifest counts agree; index/target checksums verified.',
        'splits':splits,'total_clips':total,'total_by_class':totals,
        'total_percent_by_class':{g:totals[g]/total*100 for g in names},
        'split_percent':{s:splits[s]['clips']/total*100 for s in splits},
        'training_max_min_ratio':max(train_counts)/min(train_counts),
        'models_using_same_subset':['five-sign last-state GRU','five-sign mean-pool GRU','five-sign ST-GCN'],
        'limits':'Clip balance does not establish coverage of signing variations, detection quality, or per-class learnability. Official signer-disjoint splits unchanged.'}
    output=ROOT/'docs/assets/five-sign-class-distribution-2026-11-12.json';write_json(output,report)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));pos=np.arange(5)
    for offset,split,label,color in [(-.24,'train','Train (75)','#2563eb'),(0,'val','Validation (19)','#f59e0b'),(.24,'test','Test (60)','#16a34a')]:
        values=[splits[split]['counts'][g] for g in names]
        bars=axes[0].bar(pos+offset,values,.23,label=label,color=color)
        for b,v in zip(bars,values):axes[0].text(b.get_x()+b.get_width()/2,v+.15,str(v),ha='center',fontsize=9)
    axes[0].set_xticks(pos,names);axes[0].set_ylim(0,19);axes[0].set_yticks(np.arange(0,19,2));axes[0].set_ylabel('Video clips');axes[0].set_title('Five-class subset: 154 clips across official splits');axes[0].legend(fontsize=8)
    values=[splits['train']['percent'][g] for g in names];bars=axes[1].bar(pos,values,color='#2563eb',width=.65)
    for b,v,g in zip(bars,values,names):axes[1].text(b.get_x()+b.get_width()/2,v+.4,f'{v:.1f}%\n({splits["train"]["counts"][g]} clips)',ha='center',fontsize=9,zorder=4,bbox={'facecolor':'white','edgecolor':'none','pad':.5})
    axes[1].axhline(20,color='#475569',linestyle='--',label='Equal five-class share: 20%')
    axes[1].set_xticks(pos,names);axes[1].set_ylim(0,30);axes[1].set_ylabel('Share of training examples (%)');axes[1].set_title('Training is nearly balanced: 14–16 clips per class');axes[1].legend(fontsize=8,loc='upper right')
    fig.tight_layout();fig.savefig(output.with_suffix('.png'),dpi=160);plt.close(fig)
    lines=['# Current five-sign class distribution — 2026-11-12','',
        'Verified against the actual cached target arrays and index rows. Counts are video clips, not frames or augmented variants.','',
        '| Sign | Train | Validation | Test | Total | Training share |','|---|---:|---:|---:|---:|---:|']
    for g in names:lines.append(f'| {g} | {splits["train"]["counts"][g]} | {splits["val"]["counts"][g]} | {splits["test"]["counts"][g]} | {totals[g]} | {splits["train"]["percent"][g]:.1f}% |')
    lines+=['| **Total** | **75** | **19** | **60** | **154** | **100%** |','',
        'The official signer-disjoint split is preserved: 24 training, 4 validation and 11 test signers. Split shares are 48.7%, 12.3% and 39.0%. This is not a random 80/20 split.','',
        'Training counts are nearly balanced (largest/smallest = 16/14 = 1.14). Strong clip-count imbalance does not explain the NO/YES failures on its own. The much clearer limitation is only 14–16 source training clips per class; balance does not guarantee tracking quality or coverage of signers, viewpoints and signing variations. Online augmentation does not add new source clips.','',
        'The five-sign GRU, mean-pooling GRU and ST-GCN experiments use this same subset. No data, splits, model weights or training settings were changed. No inference or new test-set scoring occurred.','',
        '[Graph](assets/five-sign-class-distribution-2026-11-12.png) · [Verified counts and signer details](assets/five-sign-class-distribution-2026-11-12.json)','',
        'Reproduce: `.\\.venv\\Scripts\\python.exe scripts/report_five_sign_distribution.py`','']
    (ROOT/'docs/five-sign-class-distribution-2026-11-12.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
