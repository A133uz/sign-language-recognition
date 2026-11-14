"""Record the official pretrained baseline alongside frozen project results."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import write_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',type=Path)
    parser.add_argument('--confirm-all',action='store_true',help='Only after explicit user confirmation of all four prompted captures')
    args=parser.parse_args()
    path=ROOT/'docs/assets/official-citizen-i3d-baseline-2026-11-14.json';baseline=json.loads(path.read_text())
    benchmark=json.loads((ROOT/'docs/assets/small-vocab-snapshot-test-2026-11-14.json').read_text())
    if baseline['status']!='complete':raise ValueError('Baseline incomplete')
    selected=['models/asl-gru-five-signs-geometry-v1/geometry/best.pt',
              'models/asl-gru-five-signs-geometry-v1/geometry/epoch-005.pt',
              'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt']
    refs=[]
    for checkpoint in selected:
        record=next(r for r in benchmark['checkpoints'] if any(a['checkpoint']==checkpoint for a in r['aliases']))
        report_path=ROOT/record['report'];data=json.loads(report_path.read_text());classes=str(record['classes'])
        data['predictions']=[json.loads(line) for line in report_path.with_name('predictions.jsonl').read_text().splitlines()]
        official={r['video_file']:r for r in baseline['restricted_vocabulary'][classes]['predictions']}
        if set(official)!={r['video_file'] for r in data['predictions']}:raise ValueError('Comparison clips differ')
        improved=sum(official[r['video_file']]['correct'] and not r['correct'] for r in data['predictions'])
        regressed=sum(r['correct'] and not official[r['video_file']]['correct'] for r in data['predictions'])
        refs.append({'checkpoint':checkpoint,'epoch':record['epoch'],'classes':record['classes'],'metrics':record['metrics'],
                     'official_correct_project_wrong':improved,'project_correct_official_wrong':regressed})
    wanted={r['video_file'] for r in baseline['full_vocabulary']['predictions']};full_refs=[]
    for name in ('asl-gru-regularized-test-v1','asl-stgcn27-test-v1'):
        folder=ROOT/'data/evaluations'/name
        old=json.loads((folder/'report.json').read_text());predictions=[]
        if sha256(Path(old['checkpoint']))!=old['checkpoint_sha256']:raise ValueError('Historical full-vocabulary weights differ')
        for line in (folder/'predictions.jsonl').open():
            row=json.loads(line)
            if row['video_file'] in wanted:predictions.append(row)
        if len(predictions)!=len(wanted):raise ValueError('Missing historical test rows')
        full_refs.append({'evaluation':name,'classes':2731,'samples':len(predictions),
                          'correct':sum(r['top1_correct'] for r in predictions),
                          'top1':sum(r['top1_correct'] for r in predictions)/len(predictions),
                          'top5':sum(r['top5_correct'] for r in predictions)/len(predictions),
                          'source_report_sha256':sha256(folder/'report.json')})
    result={'status':'complete','date_local':'2026-11-14','checkpoint_sha256':baseline['checkpoint_sha256'],
            'official_full':{k:v for k,v in baseline['full_vocabulary'].items() if k!='predictions'},
            'official_restricted':{k:{q:v for q,v in value.items() if q!='predictions'} for k,value in baseline['restricted_vocabulary'].items()},
            'project_small_vocab':refs,'project_full_vocab_same_videos':full_refs,
            'baseline_report_sha256':sha256(path),'training':False,'existing_models_changed':False,
            'preprocessing_matches_upstream':True,
            'limits':['Restricted scores renormalize the released 2731 outputs; no weights or head were retrained.',
                      'Released I3D trained on the full vocabulary; small project models trained on selected-class subsets. This is not an architecture-only ablation.',
                      '182 chosen official test videos are not the entire 32941-video official test set; no published-score reproduction claim.',
                      'The epoch-5 GRU was identified retrospectively by its test score.',
                      'Order-dependent upstream endpoint padding is seeded and disclosed; webcam performance is unmeasured in this run.']}
    destination=ROOT/'docs/assets/official-citizen-i3d-comparison-2026-11-14.json'
    if args.live:
        live=json.loads(args.live.read_text())
        if live['status']!='complete' or len(live['trials'])!=4 or live['checkpoint_sha256']!=baseline['checkpoint_sha256']:
            raise ValueError('Requires completed, matching four-trial baseline')
        if args.confirm_all:
            backup=args.live.with_name(args.live.stem+'.before-user-confirmation.json')
            if not backup.exists():backup.write_bytes(args.live.read_bytes())
            live['user_completion_confirmed']=True
            for trial in live['trials']:
                if trial.get('user_confirmation') is False:raise ValueError('Do not override an explicitly invalid trial')
                trial.update(user_confirmation=True,user_confirmed_label=trial['prompted_label'])
                if trial['trial_status']=='captured_pending_confirmation':trial['trial_status']='confirmed_valid'
            write_json(args.live,live)
        confirmed=[t for t in live['trials'] if t.get('user_confirmation') is True and t.get('trial_status')=='confirmed_valid']
        result['webcam']={'session':live,'confirmed_trials':len(confirmed),
            'full_correct':sum(t['full_top5'][0]['label']==t['user_confirmed_label'] for t in confirmed),
            'pilot15_correct':sum(t['pilot15_top5'][0]['label']==t['user_confirmed_label'] for t in confirmed),
            'source_report_sha256':sha256(args.live),
            'limits':'Four trials, one signer/session; descriptive diagnostic only. Full 2731-way versus restricted 15-way are different output tasks. No personal media or coordinates saved.'}
        result['limits'][-1]='Order-dependent upstream endpoint padding is seeded/disclosed. A separate confirmed four-trial webcam check is reported descriptively.'
    elif destination.exists():
        previous=json.loads(destination.read_text())
        if previous.get('checkpoint_sha256')==result['checkpoint_sha256'] and 'webcam' in previous:result['webcam']=previous['webcam']
    if 'webcam' in result:
        result['limits'][-1]='Order-dependent upstream endpoint padding is seeded/disclosed. A separate confirmed four-trial webcam check is reported descriptively.'
    write_json(destination,result)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,classes in zip(axes,(5,15)):
        rows=[r for r in refs if r['classes']==classes]
        labels=['Official I3D\nrestricted outputs']+['GRU geometry e'+str(r['epoch'])+('\nretrospective' if 'epoch-005' in r['checkpoint'] else '\npreselected') for r in rows]
        accuracy=[result['official_restricted'][str(classes)]['metrics']['top1']]+[r['metrics']['top1'] for r in rows]
        f1=[result['official_restricted'][str(classes)]['metrics']['macro_f1']]+[r['metrics']['macro_f1'] for r in rows]
        x=np.arange(len(labels));a=ax.bar(x-.18,accuracy,.35,label='Top-1',color='#2563eb');ax.bar(x+.18,f1,.35,label='Macro F1',color='#0d9488')
        ax.bar_label(a,labels=[f'{v:.2%}' for v in accuracy],padding=3,fontsize=9)
        ax.set(xticks=x,xticklabels=labels,ylim=(0,1.12),title=f'{classes} labels / {result["official_restricted"][str(classes)]["metrics"]["samples"]} identical test videos')
        ax.legend(fontsize=8)
    fig.suptitle('Official pretrained pixel baseline versus project landmark snapshots')
    fig.savefig(ROOT/'docs/assets/official-citizen-i3d-comparison-2026-11-14.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k in ('official_full','project_small_vocab','project_full_vocab_same_videos')},indent=2))


if __name__=='__main__':main()
