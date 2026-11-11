"""Report the repeated ST-GCN mean-pooling run without inference or training."""
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


def read(path):return json.loads(path.read_text())


def main():
    old=ROOT/'models/asl-stgcn27-five-signs-early-v1';new=ROOT/'models/asl-stgcn27-five-signs-mean-pool-repeat-v1'
    report=read(new/'report.json')
    if report['status']!='complete':raise ValueError('Training is incomplete')
    a=torch.load(old/'best.pt',map_location='cpu',weights_only=True)['model']
    b=torch.load(new/'best.pt',map_location='cpu',weights_only=True)['model']
    if a.keys()!=b.keys():raise ValueError('Model parameter schemas differ')
    differences=[name for name in a if not torch.equal(a[name],b[name])]
    before=read(old/'test-predictions.json');after=read(new/'test-predictions.json')
    if [(x['video_file'],x['target']) for x in before]!=[(x['video_file'],x['target']) for x in after]:raise ValueError('Test rows differ')
    report['repeat_comparison']={'all_model_tensors_bitwise_equal':not differences,
        'unequal_tensor_count':len(differences),
        'maximum_absolute_tensor_difference':max(float((a[n].float()-b[n].float()).abs().max()) for n in a),
        'test_top1_predictions_equal':all(x['prediction']==y['prediction'] for x,y in zip(before,after)),
        'improved_test_clips':sum(not x['correct'] and y['correct'] for x,y in zip(before,after)),
        'regressed_test_clips':sum(x['correct'] and not y['correct'] for x,y in zip(before,after)),
        'note':'The old and repeated model both already use mean pooling. This is not an architectural ablation.'}
    report['runtime_consistency']=read(new/'runtime-consistency.json')
    report['epoch_metrics']=[read(p) for p in sorted(new.glob('epoch-*.json'))]
    spec=report['specification']
    if any(sha256(ROOT/name)!=digest for name,digest in spec['protected_weights_before'].items()):raise AssertionError('Preserved weights changed')
    output=ROOT/'docs/assets/stgcn-mean-pool-repeat-2026-11-12.json';write_json(output,report)
    rows=report['epoch_metrics'];selected=report['test']['epoch']
    fig,axes=plt.subplots(1,3,figsize=(15,4.7));x=[r['epoch'] for r in rows]
    for split,label,color in [('train_clean','Clean train','#2563eb'),('val','Validation','#f59e0b')]:
        axes[0].plot(x,[r[split]['top1']*100 for r in rows],marker='o',label=label,color=color)
        axes[1].plot(x,[r[split]['loss'] for r in rows],marker='o',label=label,color=color)
    for ax in axes[:2]:ax.axvline(selected,linestyle='--',color='#16a34a',label=f'Selected epoch {selected}');ax.set_xlabel('Epoch');ax.legend(fontsize=8)
    axes[0].set(title='Existing mean-pool ST-GCN: fresh run',ylabel='Correct (%)',ylim=(0,105))
    axes[1].set(title='Cross-entropy',ylabel='Loss')
    result=report['test'];matrix=np.asarray(result['confusion']);axes[2].imshow(matrix,cmap='Blues',vmin=0,vmax=14)
    axes[2].set(xticks=range(5),yticks=range(5),xticklabels=result['labels'],yticklabels=result['labels'],
        title=f'Test: {matrix.trace()}/60 ({result["top1"]:.1%})',xlabel='Predicted',ylabel='True')
    axes[2].tick_params(axis='x',rotation=45)
    for i in range(5):
        for j in range(5):axes[2].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>7 else 'black')
    fig.tight_layout();fig.savefig(output.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps({'test':result,'comparison':report['repeat_comparison']},indent=2));print(output)


if __name__=='__main__':main()
