"""Released official I3D on the same test videos, full and restricted vocabularies."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.official_citizen_i3d import OfficialCitizenI3D,sha
from sign_language.five_sign_pilot import write_json
from sign_language.evaluation import classification_metrics


def softmax(z):
    p=np.exp(np.asarray(z,dtype=np.float64)-np.max(z));return p/p.sum()


def restricted_report(logits,rows,names,official_names):
    ids=[official_names.index(g) for g in names];matrix=np.zeros((len(names),len(names)),np.int64)
    predictions=[];loss=0.;hit5=0
    for z,row in zip(logits,rows,strict=True):
        if row['gloss'] not in names:continue
        p=softmax(z[ids]);target=names.index(row['gloss']);order=np.argsort(-p,kind='stable');pred=int(order[0])
        matrix[target,pred]+=1;loss-=float(np.log(max(p[target],np.finfo(float).tiny)));hit5+=int(target in order[:5])
        predictions.append({**row,'expected':row['gloss'],'prediction':names[pred],'correct':target==pred,
            'scores':p.tolist(),'top5':[{'label':names[int(i)],'score':float(p[i])} for i in order[:5]]})
    metrics,precision,recall,f1=classification_metrics(matrix)
    metrics.update(samples=len(predictions),correct=int(matrix.trace()),loss=loss/len(predictions),top5=hit5/len(predictions))
    return {'classes':len(names),'labels':names,'metrics':metrics,'confusion':matrix.tolist(),
            'per_class':{g:{'support':int(matrix[i].sum()),'correct':int(matrix[i,i]),'precision':float(precision[i]),
                            'recall':float(recall[i]),'f1':float(f1[i])} for i,g in enumerate(names)},'predictions':predictions,
            'interpretation':'Output logits restricted and renormalized to the same pilot labels; original 2731-output weights/head unchanged, no training.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=1);args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Preserve previous baseline evidence')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    baseline=OfficialCitizenI3D(ROOT)
    reference=ROOT/'data/evaluations/small-vocab-snapshot-test-2026-11-14-v1'
    rows=json.loads((reference/'inputs/15/test-index.json').read_text())
    requested={r['video_file'] for r in rows}
    source=ROOT/'data/asl_citizen/ASL_Citizen/splits/test.csv'
    with source.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream);header=next(reader);mapping={r[1]:r for r in reader if r[1] in requested}
    if set(mapping)!=requested:raise ValueError('Missing official test records')
    with (out/'official-test-subset.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(header)
        for row in rows:
            original=mapping[row['video_file']]
            if original[0]!=row['participant_id'] or original[2].strip()!=row['gloss']:raise ValueError('Test identity differs')
            writer.writerow(original)
    protected={str(baseline.checkpoint.relative_to(ROOT)):sha(baseline.checkpoint),
               str(source.relative_to(ROOT)):sha(source)}
    catalog=ROOT/'models/model-catalog.json'
    if catalog.exists():
        # Curation intentionally removed earlier checkpoints; protect the current retained set.
        for record in json.loads(catalog.read_text())['retained_models']:
            protected[record['checkpoint']]=record['checkpoint_sha256']
    else:
        old=json.loads((reference/'report.json').read_text())['protected_sha256']
        for name,h in old.items():
            if '/models/' in '/'+name or name.startswith('models/'):protected[name]=h
    for p in baseline.folder.glob('*.json'):protected[str(p.relative_to(ROOT))]=sha(p)
    for name,h in baseline.config['source_sha256'].items():protected[str((ROOT/baseline.config['upstream_directory']/name).relative_to(ROOT))]=h
    if not all(sha(ROOT/p)==h for p,h in protected.items()):raise ValueError('Protected baseline/reference differs')
    frozen={'status':'frozen','checkpoint_sha256':baseline.config['checkpoint_sha256'],'commit':baseline.config['commit'],
        'examples':len(rows),'test_subset_csv_sha256':sha(out/'official-test-subset.csv'),'seed':0,'batch_size':args.batch_size,
        'upstream_loader_unchanged':True,'float32_no_amp':True,'training':False,'landmark_extraction':False,
        'channel_order':'OpenCV BGR, matching upstream despite RGB loader name',
        'padding':'Unmodified random first/last append padding; numpy seed 0 and fixed input order, workers=0.',
        'reproduction_limit':'Official script shuffles data with workers=2; order-dependent endpoint padding can differ. This is not a full-test reproduction of the paper.',
        'protected_sha256':protected}
    write_json(out/'protocol.json',frozen)
    dataset=baseline.dataset(out/'official-test-subset.csv')
    all_logits=[];began=time.monotonic();video_hashes={};torch.cuda.reset_peak_memory_stats()
    for start in range(0,len(rows),args.batch_size):
        batch=[]
        for i in range(start,min(start+args.batch_size,len(rows))):
            row=rows[i];video=ROOT/'data/asl_citizen/ASL_Citizen/videos'/row['video_file'];video_hashes[str(video.relative_to(ROOT))]=sha(video)
            frames,info,_=dataset[i]
            if info['filename']!=row['video_file']:raise ValueError('Loader changed sample identity')
            batch.append(frames)
        z=baseline.logits(torch.stack(batch));all_logits.extend(z)
        write_json(out/'status.json',{'status':'evaluating','processed':len(all_logits),'total':len(rows)})
        if len(all_logits)%10==0 or len(all_logits)==len(rows):print('OFFICIAL_I3D',len(all_logits),'/',len(rows),flush=True)
    logits=np.asarray(all_logits,np.float32);np.save(out/'full-vocabulary-logits.npy',logits,allow_pickle=False)
    full_predictions=[];hit1=hit5=0;mrr=dcg=0.
    for row,z in zip(rows,logits,strict=True):
        p=softmax(z);order=np.argsort(-p,kind='stable');target=baseline.labels[row['gloss']];rank=int(np.flatnonzero(order==target)[0])+1
        hit1+=rank==1;hit5+=rank<=5;mrr+=1/rank;dcg+=1/np.log2(rank+1)
        full_predictions.append({**row,'rank':rank,'prediction':baseline.names[int(order[0])],
            'top5':[{'label':baseline.names[int(i)],'score':float(p[i])} for i in order[:5]],'true_label_score':float(p[target])})
    reports={}
    for classes in (5,15):
        labels=json.loads((reference/f'inputs/{classes}/manifest.json').read_text())['labels']
        names=[g for g,i in sorted(labels.items(),key=lambda item:item[1])]
        reports[str(classes)]=restricted_report(logits,rows,names,baseline.names)
    protected.update(video_hashes)
    unchanged=all(sha(ROOT/p)==h for p,h in protected.items())
    if not unchanged:raise ValueError('Protected video/checkpoint/reference changed')
    report={'status':'complete','date_local':'2026-11-14','checkpoint_sha256':baseline.config['checkpoint_sha256'],
        'source_config':baseline.config,'protocol':frozen,'protected_sha256':protected,'protected_unchanged':True,
        'full_vocabulary':{'classes':2731,'samples':len(rows),'correct':int(hit1),'top1':hit1/len(rows),'top5':hit5/len(rows),
                           'mrr':mrr/len(rows),'dcg':dcg/len(rows),'predictions':full_predictions},
        'restricted_vocabulary':reports,'training':False,'existing_models_changed':False,
        'inference_seconds':time.monotonic()-began,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}
    write_json(out/'report.json',report);write_json(out/'status.json',{'status':'complete','samples':len(rows)})
    write_json(ROOT/'docs/assets/official-citizen-i3d-baseline-2026-11-14.json',report)
    print('OFFICIAL_I3D_COMPLETE','full',hit1,len(rows),'restricted',{k:r['metrics'] for k,r in reports.items()},flush=True)


if __name__=='__main__':main()
