"""Summarize completed five-sign trials and plot comparable clean learning curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports={}
    for version in ('v1','v2'):
        directory=ROOT/f'models/asl-gru-five-signs-early-{version}'
        report=json.loads((directory/'report.json').read_text())
        if report['status']!='complete':raise ValueError('Complete both trials before reporting')
        reports[version]=report
    reports['v2']['runtime_consistency']=json.loads((ROOT/'models/asl-gru-five-signs-early-v2/regularized/runtime-consistency.json').read_text())
    asset=ROOT/'docs/assets/five-sign-early-pilot-2026-11-11.json'
    asset.write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for column,name in enumerate(('baseline','regularized')):
        directory=ROOT/'models/asl-gru-five-signs-early-v2'/name
        rows=sorted((json.loads(p.read_text()) for p in directory.glob('epoch-*.json')),key=lambda r:r['epoch'])
        selected=reports['v2']['models'][name]['training']['selection']['selected_epoch']
        x=[r['epoch'] for r in rows]
        for split,label,color in [('train_clean','Clean training','#2563eb'),('val','Validation','#d97706')]:
            axes[0,column].plot(x,[100*r[split]['top1'] for r in rows],label=label,color=color)
            axes[1,column].plot(x,[r[split]['loss'] for r in rows],label=label,color=color)
        axes[2,column].plot(x,[100*r['accuracy_gap'] for r in rows],color='#7c3aed')
        axes[2,column].axhline(0,color='#64748b',linestyle=':',linewidth=1)
        for row in range(3):
            axes[row,column].axvline(selected,color='#15803d',linestyle='--',label=f'Selected epoch {selected}')
            axes[row,column].set_xlabel('Epoch');axes[row,column].grid(alpha=.2)
        axes[0,column].set(title=name.capitalize()+' GRU — five labels',ylabel='Top-1 accuracy (%)',ylim=(0,105))
        axes[0,column].legend(fontsize=8)
        axes[1,column].set_ylabel('Cross-entropy')
        axes[2,column].set_ylabel('Clean train − validation (pp)')
    fig.suptitle('Five-sign pilot v2: earlier selection and generalization gap',fontsize=15)
    fig.savefig(ROOT/'docs/assets/five-sign-early-pilot-2026-11-11.png',dpi=150)
    plt.close(fig)
    for name,result in reports['v2']['models'].items():
        train=result['training'];chosen=train['selected_metrics'];test=result['test']
        print(json.dumps({'model':name,'trained_epochs':train['epochs_completed'],'selected_epoch':chosen['epoch'],
                          'clean_train_top1':chosen['train_clean']['top1'],'val_top1':chosen['val']['top1'],
                          'accuracy_gap':chosen['accuracy_gap'],'test_top1':test['top1'],'test_macro_f1':test['macro_f1'],
                          'stop_reason':train['stop_reason'],'learning_status':train['learning_status']},indent=2))


if __name__=='__main__':main()
