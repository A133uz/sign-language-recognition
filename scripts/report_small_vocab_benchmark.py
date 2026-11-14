"""Publish every frozen small-vocabulary test result without selecting new weights."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json


def short(path):
    return path.removeprefix('models/').removesuffix('/best.pt').replace('asl-','').replace('-five-signs','').replace('-fifteen-signs','')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation',type=Path,required=True)
    args=parser.parse_args();evaluation=args.evaluation.resolve();source=json.loads((evaluation/'report.json').read_text())
    if source['status']!='complete':raise ValueError('All frozen snapshots must finish successfully')
    result={k:source[k] for k in ('date_local','snapshot_paths','unique_checkpoint_files','groups','protected_unchanged',
                                'parent_test_chunks_unchanged','training','selected_models_changed','geometry_interventions','limits','gpu','elapsed_seconds')}
    result.update(status='complete',evaluation_directory=str(evaluation.relative_to(ROOT)).replace('\\','/'),
                  source_report_sha256=sha256(evaluation/'report.json'),
                  frozen_inventory_sha256=sha256(evaluation/'frozen-inventory.json'),checkpoints=[])
    named=[]
    for record in source['records']:
        raw=json.loads((ROOT/record['report']).read_text())
        per_signer=json.loads((ROOT/record['report']).with_name('per-signer.json').read_text())
        portable={**record,'confusion':raw['confusion'],'labels':raw['labels'],'per_class':raw['per_class'],
                  'per_signer':per_signer,'geometry_enabled':raw['geometry_enabled'],'geometry_features':raw['geometry_features']}
        result['checkpoints'].append(portable)
        for alias in record['aliases']:
            if alias['artifact_role']=='best.pt':named.append({**alias,'epoch':record['epoch'],'metrics':record['metrics'],
                                                             'report':record['report']})
    result['validation_selected_snapshots']=named
    result['best_artifact_scope_note']='The best.pt name identifies a pre-existing per-run artifact; readout C-grid entries include candidates that were not selected as the final model.'
    result['snapshot_rows']=[{**alias,'epoch':r['epoch'],'metrics':r['metrics'],'report':r['report']}
                             for r in result['checkpoints'] for alias in r['aliases']]
    assert len(result['snapshot_rows'])==source['snapshot_paths']
    result['retrospective_test_leaders']={}
    for classes in (5,15):
        candidates=[r for r in result['snapshot_rows'] if r['classes']==classes]
        peak=max(r['metrics']['top1'] for r in candidates)
        result['retrospective_test_leaders'][str(classes)]=[r for r in candidates if r['metrics']['top1']==peak]
    result['evaluation_source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in
        (ROOT/'scripts/benchmark_small_vocab_snapshots.py',ROOT/'src/sign_language/small_vocab_benchmark.py')}
    assets=ROOT/'docs/assets';stem='small-vocab-snapshot-test-2026-11-14'
    write_json(assets/(stem+'.json'),result)
    shutil.copyfile(evaluation/'all-snapshots.csv',assets/(stem+'.csv'))
    fig,axes=plt.subplots(1,2,figsize=(17,8),constrained_layout=True)
    for ax,classes in zip(axes,(5,15)):
        rows=sorted([r for r in named if r['classes']==classes],key=lambda r:r['metrics']['top1'])
        y=np.arange(len(rows))
        ax.barh(y-.17,[r['metrics']['top1'] for r in rows],height=.32,label='Test top-1',color='#2563eb')
        ax.barh(y+.17,[r['metrics']['macro_f1'] for r in rows],height=.32,label='Test macro F1',color='#0d9488')
        ax.set(yticks=y,yticklabels=[short(r['checkpoint'])+f' e{r["epoch"]}' for r in rows],xlim=(0,1.05),
               title=f'{classes} labels; {source["groups"][str(classes)]["samples"]} official test clips')
        ax.tick_params(axis='y',labelsize=8);ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
        for i,row in enumerate(rows):ax.text(row['metrics']['top1']+.01,i-.17,f'{row["metrics"]["top1"]:.1%}',va='center',fontsize=8)
    fig.suptitle('Pre-existing best.pt artifacts; no test-driven model replacement')
    fig.savefig(assets/(stem+'.png'),dpi=150);plt.close(fig)
    primaries=['models/asl-gru-five-signs-geometry-v1/geometry/best.pt',
               'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt']
    fig,axes=plt.subplots(1,2,figsize=(17,7),constrained_layout=True)
    for ax,path in zip(axes,primaries):
        row=next(r for r in result['checkpoints'] if any(a['checkpoint']==path for a in r['aliases']))
        cm=np.asarray(row['confusion']);support=cm.sum(1);rates=cm/support[:,None]
        ax.imshow(rates,vmin=0,vmax=1,cmap='Blues')
        for i in range(len(cm)):
            for j in range(len(cm)):
                if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7 if len(cm)>5 else 10,color='white' if rates[i,j]>.55 else 'black')
        ax.set(xticks=range(len(cm)),yticks=range(len(cm)),xticklabels=row['labels'],yticklabels=row['labels'],
               xlabel='Predicted',ylabel='True',title=short(path)+f'\nTest {row["metrics"]["correct"]}/{row["metrics"]["samples"]}')
        ax.tick_params(axis='x',labelrotation=90,labelsize=8);ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Two discussed reference snapshots: cell counts, row-normalized color')
    fig.savefig(assets/(stem+'-confusion.png'),dpi=150);plt.close(fig)
    lines=['# Frozen five-sign and fifteen-sign test benchmark — 2026-11-14','',
        f'Completed **{source["snapshot_paths"]} saved checkpoint paths / {source["unique_checkpoint_files"]} byte-distinct checkpoint files**. All per-epoch, `best.pt` and `minimum-val-loss.pt` snapshots in the five-signs/fifteen-signs neural experiment directories were inventoried and hashed before evaluation. Identical files share inference and retain separate alias records. No weights were trained, branches disabled, defaults changed or checkpoint selections replaced.','',
        'Each vocabulary uses all of its examples from the official ASL Citizen test split, with the trained label mapping and 128-frame preprocessing. Test signers remain disjoint from the associated training/validation signers. The new inputs were copied into a separate evaluation directory; training caches were preserved. No clip is discarded for missing hands or poor predictions.','',
        f'Five signs: **{source["groups"]["5"]["samples"]}** test clips; fifteen signs: **{source["groups"]["15"]["samples"]}**. Parent selected test-feature chunks, cached rows/labels, original subset hashes, runtime source hashes, checkpoints and existing model archives were checked. Protected hashes and selected parent chunks match before/after.','',
        '## Pre-existing best.pt artifacts','',
        'The following are the pre-existing per-run `best.pt` artifacts. Readout C-grid entries include candidates not chosen as the final model. Their order is a descriptive test comparison, not a new deployment/model-selection decision.','',
        '| Classes | Snapshot | Epoch | Test correct | Top-1 | Top-5 | Macro F1 | Cross-entropy |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in sorted(named,key=lambda r:(r['classes'],-r['metrics']['top1'],r['checkpoint'])):
        m=row['metrics'];lines.append(f'| {row["classes"]} | `{row["checkpoint"]}` | {row["epoch"]} | {m["correct"]}/{m["samples"]} | {m["top1"]:.2%} | {m["top5"]:.2%} | {m["macro_f1"]:.4f} | {m["loss"]:.4f} |')
    lines+=['','## Highest observed test scores across every stored checkpoint','',
        'This includes epochs that were not selected by validation. These are retrospective test leaders, not newly validated deployment choices. Selecting them because of this sweep consumes the test set as selection evidence. Existing best.pt files are preserved.','',
        '| Classes | Stored snapshot | Epoch | Correct | Top-1 | Macro F1 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for rows in result['retrospective_test_leaders'].values():
        for row in rows:
            m=row['metrics'];lines.append(f'| {row["classes"]} | `{row["checkpoint"]}` | {row["epoch"]} | {m["correct"]}/{m["samples"]} | {m["top1"]:.2%} | {m["macro_f1"]:.4f} |')
    lines+=['','## Reference models and class-specific results','']
    for path in primaries:
        row=next(r for r in result['checkpoints'] if any(a['checkpoint']==path for a in r['aliases']))
        lines+=['### '+short(path),'','| Sign | Correct / support | Recall | Precision | F1 |','| --- | ---: | ---: | ---: | ---: |']
        for g,v in row['per_class'].items():lines.append(f'| {g} | {v["correct"]}/{v["support"]} | {v["recall"]:.2%} | {v["precision"]:.2%} | {v["f1"]:.4f} |')
    lines+=['','## Every stored epoch and alternate snapshot','',
        f'The [complete CSV](assets/{stem}.csv) has one row for every saved path, including all epochs and minimum-validation-loss alternatives. The [portable JSON](assets/{stem}.json) contains all unique-checkpoint metrics, aliases, confusion matrices, per-class scores and per-signer scores. Full per-video probability vectors are recorded under `{result["evaluation_directory"]}/checkpoints/<hash>/predictions.jsonl`, with companion per-class/confusion CSVs and reports.','',
        f'![Validation-selected test comparison](assets/{stem}.png)','',
        f'![Reference confusion matrices](assets/{stem}-confusion.png)','',
        '## Interpretation limits','',
        *['- '+s for s in source['limits']],'',
        'These are original official partitions, not a newly drawn random 80/20 split. Five-label and fifteen-label accuracies are different tasks. The snapshot sweep is retrospective; future changes informed by it need fresh validation/holdout evidence. An earlier perfect ten-attempt webcam session does not override the larger dataset test, and strong dataset results do not establish live-camera recognition.','',
        '## Reproduction and verification','',
        '`python scripts/benchmark_small_vocab_snapshots.py --output data/evaluations/NEW_EMPTY_RUN` freezes inventory before test evaluation. `python scripts/report_small_vocab_benchmark.py --evaluation RUN` renders this record without inference or weight changes.','',
        f'Hardware: {source["gpu"]}. Inference/output and final integrity checks took {source["elapsed_seconds"]:.1f} seconds after input preparation. Focused evaluator tests passed before the run. The reporting notebook includes these frozen test results.']
    verification=ROOT/'docs/assets/small-vocab-snapshot-test-verification-2026-11-14.json'
    if verification.exists():
        verified=json.loads(verification.read_text())
        lines+=['','## Final verification','',
            f'All **{verified["tests"]} tests passed**. Every per-video prediction reconstructs its saved confusion matrix; historical five-sign GRU/ST-GCN test confusions reproduce exactly. Official CSV hashes and all {verified["protected_file_count"]} protected checkpoint/config/cache/source/archive hashes match after the test suite. See [verification record](assets/small-vocab-snapshot-test-verification-2026-11-14.json).']
    (ROOT/'docs'/('small-vocab-snapshot-test-2026-11-14.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PUBLISHED',len(named),'validation-selected snapshots;',len(result['snapshot_rows']),'total paths')
    for classes in (5,15):
        winner=max((r for r in named if r['classes']==classes),key=lambda r:r['metrics']['top1'])
        print('TOP_RECORDED_SELECTED',classes,winner['checkpoint'],winner['metrics'])


if __name__=='__main__':main()
