"""Record the completed five-sign ST-GCN/GRU comparison and exact learning curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'models/asl-stgcn27-five-signs-early-v1'
    report=json.loads((folder/'report.json').read_text())
    gru=json.loads((ROOT/'models/asl-gru-five-signs-early-v2/report.json').read_text())['models']['regularized']
    if report['status']!='complete':raise ValueError('ST-GCN run is not complete')
    old=ROOT/'models/asl-stgcn27-v1'
    old_epochs={str(e):json.loads((old/f'epoch-{e:03d}.json').read_text()) for e in (32,33,39,49)}
    gru_predictions=json.loads((ROOT/'models/asl-gru-five-signs-early-v2/regularized/test-predictions.json').read_text())
    stgcn_predictions=json.loads((folder/'test-predictions.json').read_text())
    identity=lambda rows:[(r['video_file'],r['participant_id'],r['target']) for r in rows]
    if identity(gru_predictions)!=identity(stgcn_predictions):raise ValueError('Comparison test rows differ')
    payload={'stgcn':report,'gru_regularized_reference':gru,'same_test_rows':True,
             'old_full_vocabulary_epoch_records':old_epochs,
             'runtime_consistency':json.loads((folder/'runtime-consistency.json').read_text()),
             'config':json.loads((folder/'config.json').read_text()),
             'note':'A fixed five-sign task. Old full-vocabulary train-mode gaps are not clean train/eval comparisons.'}
    (ROOT/'docs/assets/five-sign-stgcn-2026-11-11.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for col,(title,path,result) in enumerate([
            ('Regularized GRU',ROOT/'models/asl-gru-five-signs-early-v2/regularized',gru['training']),
            ('ST-GCN27',folder,report['training'])]):
        rows=sorted((json.loads(p.read_text()) for p in path.glob('epoch-*.json')),key=lambda r:r['epoch'])
        epoch=result['selection']['selected_epoch'];x=[r['epoch'] for r in rows]
        for split,label,color in [('train_clean','Clean training','#2563eb'),('val','Validation','#d97706')]:
            axes[0,col].plot(x,[100*r[split]['top1'] for r in rows],label=label,color=color)
            axes[1,col].plot(x,[r[split]['loss'] for r in rows],color=color)
        axes[2,col].plot(x,[100*r['accuracy_gap'] for r in rows],color='#7c3aed')
        axes[2,col].axhline(0,color='gray',linestyle=':',linewidth=1)
        for row in range(3):
            axes[row,col].axvline(epoch,color='#15803d',linestyle='--',label=f'Selected epoch {epoch}')
            axes[row,col].set_xlabel('Epoch');axes[row,col].grid(alpha=.2)
        axes[0,col].set(title=title,ylabel='Top-1 accuracy (%)',ylim=(0,105));axes[0,col].legend(fontsize=8)
        axes[1,col].set_ylabel('Cross-entropy')
        axes[2,col].set_ylabel('Clean train − validation (pp)')
    fig.suptitle('Same five signs and splits: GRU / ST-GCN early selection',fontsize=15)
    fig.savefig(ROOT/'docs/assets/five-sign-stgcn-2026-11-11.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,title,result in [(axes[0],'Regularized GRU',gru['test']),(axes[1],'ST-GCN27',report['test'])]:
        cm=np.array(result['confusion']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=14)
        ax.set(xticks=range(5),yticks=range(5),xticklabels=result['labels'],yticklabels=result['labels'],
               xlabel='Predicted label',ylabel='True label',title=f"{title}: {result['top1']:.1%} ({int(cm.trace())}/60)")
        ax.tick_params(axis='x',rotation=45)
        for i in range(5):
            for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>7 else 'black')
    fig.savefig(ROOT/'docs/assets/five-sign-stgcn-confusion-2026-11-11.png',dpi=150);plt.close(fig)
    print(json.dumps({k:report[k] for k in ('previous_weights_unchanged','cache_unchanged')},indent=2))
    print(json.dumps({'selected_epoch':report['training']['selection']['selected_epoch'],
                      'selected_metrics':report['training']['selected_metrics'],'test':report['test']},indent=2))


if __name__=='__main__':main()
