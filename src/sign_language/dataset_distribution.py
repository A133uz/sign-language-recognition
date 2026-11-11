"""Read-only class/signer counts from official ASL Citizen CSV metadata."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS=('train','val','test')


def file_sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def balance_statistics(counts):
    values=np.asarray(counts,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Expected finite nonnegative class counts')
    n=len(values);total=float(values.sum());ordered=np.sort(values)
    p=values[values>0]/total if total else np.asarray([])
    entropy=float(-(p*np.log(p)).sum()) if len(p) else 0.
    gini=float(2*(np.arange(1,n+1)*ordered).sum()/(n*total)-(n+1)/n) if total else None
    return {'classes':n,'clips':int(total),'zero_count_classes':int((values==0).sum()),
        'min':int(values.min()),'p10':float(np.quantile(values,.1)), 'q25':float(np.quantile(values,.25)),
        'median':float(np.median(values)),'q75':float(np.quantile(values,.75)),
        'p90':float(np.quantile(values,.9)),'max':int(values.max()),'mean':float(values.mean()),
        'std_population':float(values.std()),'cv':float(values.std()/values.mean()) if total else None,
        'max_min_ratio':float(values.max()/values.min()) if values.min()>0 else None,
        'gini':max(0.,min(1.,gini)) if gini is not None else None,
        'normalized_entropy':entropy/np.log(n) if total and n>1 else (1. if total else None),
        'largest_class_share':float(values.max()/total) if total else None,
        'top_10pct_classes_rounded_up':int(np.ceil(n*.1)),
        'top_10pct_clip_share':float(ordered[-int(np.ceil(n*.1)):].sum()/total) if total else None}


def gather_distribution(root):
    root=Path(root);folder=root/'data/asl_citizen/ASL_Citizen/splits'
    class_counts={};class_people={};people={};signer_rows=[];seen=set();sources={}
    for split in SPLITS:
        path=folder/f'{split}.csv';sources[path.relative_to(root).as_posix()]=file_sha256(path)
        counts=Counter();groups=defaultdict(set);signer_clips=Counter();signer_classes=defaultdict(set)
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if not {'Participant ID','Video file','Gloss'}<=set(reader.fieldnames or []):raise ValueError('Unexpected CSV schema')
            for row in reader:
                person,video,gloss=[row[k].strip() for k in ('Participant ID','Video file','Gloss')]
                if not all((person,video,gloss)):raise ValueError('Blank metadata field')
                if video in seen:raise ValueError('Duplicate source video across metadata splits')
                seen.add(video);counts[gloss]+=1;groups[gloss].add(person)
                signer_clips[person]+=1;signer_classes[person].add(gloss)
        class_counts[split]=counts;class_people[split]=groups;people[split]=set(signer_clips)
        signer_rows.extend({'participant_id':p,'split':split,'clips':signer_clips[p],
                            'classes':len(signer_classes[p])} for p in sorted(signer_clips))
    for a,b in (('train','val'),('train','test'),('val','test')):
        if people[a]&people[b]:raise ValueError('Participant leakage between official splits')
    names=sorted(set().union(*(set(x) for x in class_counts.values())))
    rows=[]
    for gloss in names:
        counts={s:int(class_counts[s][gloss]) for s in SPLITS}
        signers={s:len(class_people[s].get(gloss,set())) for s in SPLITS}
        rows.append({'gloss':gloss,**counts,'total':sum(counts.values()),
                     **{s+'_signers':signers[s] for s in SPLITS},'total_signers':sum(signers.values())})
    stats={s:balance_statistics([r[s] for r in rows]) for s in (*SPLITS,'total')}
    cache=root/'data/training/asl-gru-128-v1';cache_check={'status':'not_available','scope':'Counts only; no feature values or predictions inspected'}
    if (cache/'manifest.json').exists():
        manifest=json.loads((cache/'manifest.json').read_text());mapping=manifest['labels']
        if set(mapping)!=set(names) or set(mapping.values())!=set(range(len(names))):raise ValueError('Prepared cache vocabulary differs')
        results={}
        for split in SPLITS:
            target=cache/f'{split}-targets.npy';features=cache/f'{split}-features.npy'
            labels=np.load(target,mmap_mode='r',allow_pickle=False)
            if labels.ndim!=1 or not np.issubdtype(labels.dtype,np.integer) or labels.min()<0 or labels.max()>=len(names):raise ValueError('Invalid cached targets')
            actual=np.bincount(labels,minlength=len(names))
            expected=np.zeros(len(names),np.int64)
            for gloss,class_id in mapping.items():expected[class_id]=class_counts[split][gloss]
            if not np.array_equal(actual,expected):raise ValueError('Cached class counts differ from official metadata')
            shape=np.load(features,mmap_mode='r',allow_pickle=False).shape
            if shape[0]!=len(labels):raise ValueError('Feature/target row count differs')
            results[split]={'clips':len(labels),'feature_shape':list(shape),'counts_match':True}
            sources[target.relative_to(root).as_posix()]=file_sha256(target)
        cache_check={'status':'verified','cache_id':manifest['cache_id'],'splits':results,
                     'scope':'All target-array counts and feature row counts only; no model inference or feature quality audit'}
    focus_names=('HELLO','NO','PLEASE','THANKYOU','YES');focus=[r for r in rows if r['gloss'] in focus_names]
    subset=root/'data/training/asl-five-signs-128-v1/manifest.json';focus_check={'status':'not_available'}
    if subset.exists():
        m=json.loads(subset.read_text())
        okay=all(m['counts'][s][r['gloss']]==r[s] for r in focus for s in SPLITS)
        if not okay:raise ValueError('Current five-sign subset differs from source counts')
        focus_check={'status':'verified','cache_id':m['cache_id'],'all_source_examples_retained':True}
    return {'schema':'asl-citizen-class-distribution-v1','report_date_local':'2026-11-12','timezone':'Asia/Tashkent',
        'dataset':'Microsoft ASL Citizen','source':'Official local train.csv / val.csv / test.csv; gloss strings preserved',
        'source_sha256':sources,'class_rows':rows,'signer_rows':signer_rows,'statistics':stats,
        'split_summary':{s:{'clips':int(sum(class_counts[s].values())),'classes':len(class_counts[s]),'signers':len(people[s])} for s in SPLITS},
        'integrity':{'unique_video_count':len(seen),'duplicate_videos':0,'participant_disjoint':True,
                     'video_existence_or_decoding_checked':False},
        'prepared_cache_check':cache_check,'five_sign_rows':focus,'five_sign_cache_check':focus_check,
        'limits':'Class balance describes source-clip frequency, not landmark correctness, sign quality, semantic uniqueness or recognition accuracy. No videos decoded, models scored, data edited or splits changed.'}
