"""Freeze all saved five/fifteen-sign neural checkpoints, then evaluate official test."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json
from sign_language.expanded_pilot import digest_chunk
from sign_language.extraction import load_rows,shard_plan
from sign_language.recognition import SignRecognizer
from sign_language.small_vocab_benchmark import select_test_subset,evaluate_model

PARENT=ROOT/'data/training/asl-gru-128-v1'
CACHES={5:ROOT/'data/training/asl-five-signs-128-v1',15:ROOT/'data/training/asl-fifteen-signs-128-v1'}


def read(p):return json.loads(p.read_text())
def rel(p):return str(Path(p).relative_to(ROOT)).replace('\\','/')


def freeze_inventory():
    records=[];protected={}
    for directory in sorted((ROOT/'models').iterdir()):
        if not directory.is_dir() or not any(n in directory.name for n in ('five-signs','fifteen-signs')):continue
        for path in sorted(directory.rglob('*.pt')):
            config=read(path.with_name('config.json'));labels=read(path.with_name('labels.json'));classes=len(labels)
            if classes not in CACHES:raise ValueError('Unexpected snapshot vocabulary')
            manifest=read(CACHES[classes]/'manifest.json')
            if config['cache_id']!=manifest['cache_id'] or labels!=manifest['labels']:raise ValueError('Training-cache mismatch')
            h=sha256(path);protected[rel(path)]=h
            for p in (path.with_name('config.json'),path.with_name('labels.json')):protected[rel(p)]=sha256(p)
            records.append({'checkpoint':rel(path),'checkpoint_sha256':h,'classes':classes,
                'architecture':config.get('architecture','gru'),'artifact_role':path.name,
                'validation_selected_artifact':path.name=='best.pt'})
    for cache in CACHES.values():
        manifest=read(cache/'manifest.json')
        for name,h in manifest['files_sha256'].items():
            p=cache/name
            if sha256(p)!=h:raise ValueError('Existing subset changed: '+str(p))
            protected[rel(p)]=h
        for name in ('manifest.json','status.json'):protected[rel(cache/name)]=sha256(cache/name)
    # Preserve existing archives and serialized diagnostic classifiers too.
    for p in (ROOT/'models/archives').glob('*.zip'):protected[rel(p)]=sha256(p)
    for p in (ROOT/'models').rglob('*.pkl'):protected[rel(p)]=sha256(p)
    if not records:raise ValueError('No saved snapshots')
    return records,protected


def prepare_inputs(output,protected):
    manifest=read(PARENT/'manifest.json');status=read(PARENT/'status.json')
    if status['status']!='complete' or status['cache_id']!=manifest['cache_id']:raise ValueError('Parent cache incomplete')
    source_rows,source_labels,csv_hashes=load_rows(ROOT/'data/asl_citizen/ASL_Citizen')
    expected=[r for _,group in shard_plan(source_rows,('test',),100,None) for r in group]
    index=read(PARENT/'test-index.json')
    if expected!=index or manifest['labels']!=source_labels:raise ValueError('Official test index mismatch')
    arrays=[np.load(PARENT/f'test-{n}.npy',mmap_mode='r') for n in ('features','lengths','targets')]
    if any(len(a)!=len(index) for a in arrays):raise ValueError('Parent test arrays mismatch')
    for name in ('test-index.json','test-lengths.npy','test-targets.npy','manifest.json','status.json'):
        protected[rel(PARENT/name)]=sha256(PARENT/name)
    protected[rel(ROOT/'src/sign_language/preprocessing.py')]=manifest['identity']['preprocessing_sha256']
    needed={i for i,r in enumerate(index) if r['gloss'] in read(CACHES[15]/'manifest.json')['labels']}
    checked={}
    for marker in sorted(PARENT.glob('test-*.prepared.json')):
        record=read(marker);start,count=record['start'],record['count'];covered=needed.intersection(range(start,start+count))
        if not covered:continue
        if record['cache_id']!=manifest['cache_id'] or digest_chunk(arrays,start,count)!=record['sha256']:
            raise ValueError('Parent test feature checksum mismatch: '+marker.name)
        checked[rel(marker)]={k:record[k] for k in ('start','count','sha256')}
        protected[rel(marker)]=sha256(marker);needed.difference_update(covered)
    if needed:raise ValueError('Unverified selected rows')
    groups={};descriptions={}
    for classes,cache in CACHES.items():
        subset=read(cache/'manifest.json');labels=subset['labels']
        if subset['identity']['parent_cache_id']!=manifest['cache_id']:raise ValueError('Subset has another parent')
        rows,x,n,y=select_test_subset(index,*arrays,labels,manifest['labels'])
        people=sorted({r['participant_id'] for r in rows})
        if set(people)&set(subset['participants']['train']+subset['participants']['val']):raise ValueError('Signer leakage')
        folder=output/'inputs'/str(classes);folder.mkdir(parents=True,exist_ok=True)
        for name,value in [('features',x.astype(np.float16)),('lengths',n),('targets',y)]:
            np.save(folder/f'test-{name}.npy',value,allow_pickle=False)
        write_json(folder/'test-index.json',rows)
        # Existing five-class test is an independent byte/value consistency control.
        five_match=None
        if classes==5:
            if rows!=read(cache/'test-index.json'):raise ValueError('Existing five-sign test rows differ')
            for name,value in [('features',x),('lengths',n),('targets',y)]:
                if not np.array_equal(value,np.load(cache/f'test-{name}.npy')):raise ValueError('Existing five-sign test data differ')
            five_match=True
        names=[g for g,i in sorted(labels.items(),key=lambda item:item[1])]
        descriptions[str(classes)]={'samples':len(rows),'labels':labels,'participants':people,
            'counts':{g:sum(r['gloss']==g for r in rows) for g in names},
            'training_cache_id':subset['cache_id'],'existing_five_test_matches':five_match,
            'files_sha256':{rel(p):sha256(p) for p in folder.iterdir()}}
        groups[classes]=(rows,x,n,y,names)
    for p,h in descriptions.items():write_json(output/'inputs'/p/'manifest.json',h)
    return groups,{'groups':descriptions,'verified_parent_test_chunks':checked,'official_csv_sha256':csv_hashes}


def write_tables(folder,result,rows):
    with (folder/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row,p in zip(rows,result['predictions'],strict=True):stream.write(json.dumps({**row,**p})+'\n')
    with (folder/'per-class.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['gloss','support','correct','precision','recall','f1'])
        for g,v in result['per_class'].items():writer.writerow([g,*[v[k] for k in ('support','correct','precision','recall','f1')]])
    with (folder/'confusion.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['true / predicted',*result['labels']])
        for name,row in zip(result['labels'],result['confusion'],strict=True):writer.writerow([name,*row])
    people={}
    for row,p in zip(rows,result['predictions'],strict=True):
        person=people.setdefault(row['participant_id'],{'samples':0,'correct':0});person['samples']+=1;person['correct']+=int(p['correct'])
    for value in people.values():value['top1']=value['correct']/value['samples']
    write_json(folder/'per-signer.json',people)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a new benchmark output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    inventory,protected=freeze_inventory()
    frozen={'date_local':'2026-11-14','frozen_utc':datetime.now(timezone.utc).isoformat(),'snapshots':inventory,
        'protected_sha256':protected.copy(),'scope':'Every saved .pt, including best/minimum-loss/epoch snapshots, under five-signs/fifteen-signs experiment directories; identical files share inference.',
        'training':False,'model_selection_changed':False,'geometry_interventions':False,
        'test_use':'Retrospective frozen sweep. Historical five-sign test already seen. Do not use these results as a fresh holdout for later tuning.'}
    write_json(output/'frozen-inventory.json',frozen)
    print('FROZEN',len(inventory),'checkpoint paths;',len({r['checkpoint_sha256'] for r in inventory}),'unique files',flush=True)
    groups,input_provenance=prepare_inputs(output,protected);write_json(output/'input-provenance.json',input_provenance)
    print('TEST_INPUTS',{k:len(v[0]) for k,v in groups.items()},flush=True)
    unique={}
    for row in inventory:unique.setdefault(row['checkpoint_sha256'],[]).append(row)
    records=[];began=time.monotonic()
    write_json(output/'status.json',{'status':'evaluating','completed':0,'total':len(unique)})
    for h,aliases in unique.items():
        item=aliases[0];path=ROOT/item['checkpoint'];folder=output/'checkpoints'/h[:20];folder.mkdir(parents=True)
        started=time.monotonic()
        try:
            recognizer=SignRecognizer(path,ROOT/'models')
            rows,x,n,y,names=groups[item['classes']]
            if recognizer.names!=names:raise ValueError('Model/input label order mismatch')
            for name,expected in recognizer.config['source_sha256'].items():
                source=ROOT/'src/sign_language'/name
                # Training-only sources may legitimately evolve; runtime sources are checked by SignRecognizer.
                if source.exists() and sha256(source)==expected:protected[rel(source)]=expected
            result=evaluate_model(recognizer.model,x,n,y,names,recognizer.device,amp=recognizer.config.get('amp',True))
            result.update(status='complete',checkpoint_sha256=h,checkpoint_epoch=recognizer.epoch,
                architecture=item['architecture'],classes=item['classes'],aliases=aliases,
                geometry_enabled=bool(recognizer.config.get('geometry_features')),
                geometry_features=len(recognizer.config.get('geometry_features',[])),
                no_geometry_interventions=True,training=False,elapsed_seconds=time.monotonic()-started)
            write_tables(folder,result,rows);write_json(folder/'report.json',result)
            records.append({'checkpoint_sha256':h,'report':rel(folder/'report.json'),'epoch':recognizer.epoch,
                'classes':item['classes'],'architecture':item['architecture'],'metrics':result['metrics'],'aliases':aliases,'status':'complete'})
            print(f'{len(records)}/{len(unique)} {item["checkpoint"]}: {result["metrics"]["correct"]}/{len(rows)}; macro F1={result["metrics"]["macro_f1"]:.4f}',flush=True)
            del recognizer
        except Exception as error:
            failure={'status':'failed','checkpoint_sha256':h,'aliases':aliases,'error':f'{type(error).__name__}: {error}'}
            write_json(folder/'failure.json',failure);records.append(failure)
            print('CHECKPOINT_FAILED',item['checkpoint'],failure['error'],flush=True)
        write_json(output/'status.json',{'status':'evaluating','completed':len(records),'total':len(unique),'records':records})
    unchanged=all(sha256(ROOT/p)==h for p,h in protected.items())
    arrays=[np.load(PARENT/f'test-{n}.npy',mmap_mode='r') for n in ('features','lengths','targets')]
    chunks_unchanged=all(digest_chunk(arrays,r['start'],r['count'])==r['sha256'] for r in input_provenance['verified_parent_test_chunks'].values())
    if not unchanged or not chunks_unchanged:raise AssertionError('Protected checkpoint/source/cache bytes changed')
    summary={'status':'complete' if all(r['status']=='complete' for r in records) else 'complete_with_failures',
        'date_local':'2026-11-14','snapshot_paths':len(inventory),'unique_checkpoint_files':len(unique),
        'groups':input_provenance['groups'],'records':records,'protected_sha256':protected,
        'protected_unchanged':unchanged,'parent_test_chunks_unchanged':chunks_unchanged,
        'training':False,'selected_models_changed':False,'geometry_interventions':False,
        'elapsed_seconds':time.monotonic()-began,'gpu':torch.cuda.get_device_name(0),
        'limits':['Five- and fifteen-label tasks have different chance levels and samples; rank within vocabulary.',
                  'Five-class top-five is automatically 100%, not discriminative.',
                  'All saved epochs are retrospective test comparisons; existing validation selections stay unchanged.',
                  'Historical full-vocabulary experiments already used these official test signers/videos; no claim of a newly sealed holdout.',
                  'Dataset test scores do not establish webcam accuracy.']}
    write_json(output/'report.json',summary);write_json(output/'status.json',summary)
    fields=['classes','checkpoint','artifact_role','epoch','top1','top5','macro_precision','macro_recall','macro_f1','weighted_f1','loss','samples','checkpoint_sha256']
    with (output/'all-snapshots.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for record in records:
            if record['status']!='complete':continue
            for alias in record['aliases']:
                writer.writerow({'classes':record['classes'],'checkpoint':alias['checkpoint'],'artifact_role':alias['artifact_role'],
                    'epoch':record['epoch'],'checkpoint_sha256':record['checkpoint_sha256'],
                    **{k:record['metrics'][k] for k in fields if k in record['metrics']}})
    print('BENCHMARK_FINISHED',summary['status'],len(records),'unique checkpoints;',summary['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__':main()
