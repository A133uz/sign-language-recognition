"""Separate five-class GRU/ST-GCN experiments; old inputs and weights are read-only."""
import json
import random
import shutil
import time
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from sign_language.five_sign_pilot import POLICY, early_stop_reason, prepare_subset, select_epoch, sha256, write_json
from sign_language.gru_model import SignGRU
from sign_language.preprocessing import CONFIG
from sign_language.regularization import PROFILES
from sign_language.training import LandmarkDataset, atomic_checkpoint, cuda_check, epoch_pass


def visible_metrics(report):
    # Top-5 is necessarily 100% with five classes and must not be evidence of quality.
    return {k:v for k,v in report.items() if k!='top5'}


def classifier(config):
    architecture=config.get('architecture','gru')
    if architecture=='gru':
        return SignGRU(**config['model'])
    if architecture=='gru-mean-pool-v1':
        from sign_language.gru_mean_pool_model import SignMeanPoolGRU
        return SignMeanPoolGRU(**config['model'])
    if architecture=='stgcn27-v1':
        from sign_language.stgcn_graph import CONFIG as GRAPH
        from sign_language.stgcn_model import SignSTGCN
        if config.get('graph')!=GRAPH:raise ValueError('ST-GCN graph differs')
        return SignSTGCN(**config['model'])
    raise ValueError('Unknown pilot architecture')


def train_variant(cache, output, profile, epochs, gpu, batch_size=4, architecture='gru'):
    if architecture not in ('gru','gru-mean-pool-v1','stgcn27-v1'):raise ValueError('Unknown pilot architecture')
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((cache/'manifest.json').read_text());regularization=PROFILES[profile]
    seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cudnn.enabled=architecture=='stgcn27-v1'
    torch.backends.cudnn.benchmark=False
    config={'architecture':architecture,'model':{'input_size':254,'hidden_size':256,'layers':2,
             'classes':5,'dropout':regularization['dropout']},'preprocessing':CONFIG,
            'cache_id':manifest['cache_id'],'batch_size':batch_size,'epochs':epochs,'patience':3,
            'early_stopping_warmup_epochs':5,'early_stopping_learning_gate':.5,
            'seed':seed,'optimizer':'AdamW','learning_rate':.001,
            'weight_decay':regularization['weight_decay'],'profile':profile,
            'regularization':regularization,'cudnn_enabled':architecture=='stgcn27-v1','amp':True,'loader_workers':2,
            'selection':POLICY,'partition':{'mode':'official_signer_disjoint_five_sign_subset','test_used_for_selection':False},
            'source_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                ('five_sign_training.py','five_sign_pilot.py','gru_model.py','preprocessing.py','training.py','regularization.py')}}
    if architecture=='stgcn27-v1':
        from sign_language.stgcn_graph import CONFIG as GRAPH
        config['graph']=GRAPH
        config['model']={'classes':5,'dropout':regularization['dropout']}
        config['source_sha256'].update({name:sha256(Path(__file__).with_name(name)) for name in ('stgcn_model.py','stgcn_graph.py')})
    if architecture=='gru-mean-pool-v1':
        config['source_sha256']['gru_mean_pool_model.py']=sha256(Path(__file__).with_name('gru_mean_pool_model.py'))
        config['readout']='masked arithmetic mean of top-layer GRU outputs over true sequence length; dropout then unchanged linear head'
    write_json(output/'config.json',config);write_json(output/'labels.json',manifest['labels'])
    model=classifier(config).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=regularization['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');criterion=nn.CrossEntropyLoss()
    generator=torch.Generator().manual_seed(seed)
    loaders={s:DataLoader(LandmarkDataset(cache,s),batch_size=batch_size if s=='train' else 16,shuffle=s=='train',num_workers=2,
                          persistent_workers=True,pin_memory=True,generator=generator if s=='train' else None)
             for s in ('train','val')}
    clean_loader=DataLoader(LandmarkDataset(cache,'train'),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    records=[];best_loss=float('inf');best_accuracy=-1;loss_stale=accuracy_stale=0;reason='epoch_cap'
    began=time.monotonic()
    try:
        for epoch in range(1,epochs+1):
            write_json(output/'status.json',{'status':'training','epoch':epoch,'epochs':epochs,'gpu':gpu})
            training=epoch_pass(model,loaders['train'],criterion,'cuda',optimizer,scaler,
                                augment=regularization['augmentation'] is not None)
            clean=epoch_pass(model,clean_loader,criterion,'cuda',scaler=scaler)
            validation=epoch_pass(model,loaders['val'],criterion,'cuda',scaler=scaler)
            record={'epoch':epoch,'train':visible_metrics(training),'train_clean':visible_metrics(clean),
                    'val':visible_metrics(validation),'accuracy_gap':clean['top1']-validation['top1'],
                    'loss_gap':validation['loss']-clean['loss']}
            records.append(record)
            state={'config':config,'model':model.state_dict(),'labels':manifest['labels'],'epoch':epoch,'metrics':record}
            atomic_checkpoint(output/f'epoch-{epoch:03d}.pt',state)
            write_json(output/f'epoch-{epoch:03d}.json',record)
            if validation['loss']<best_loss-POLICY['loss_min_delta']:
                best_loss=validation['loss'];loss_stale=0
            else:loss_stale+=1
            if validation['top1']>best_accuracy+1e-12:
                best_accuracy=validation['top1'];accuracy_stale=0
            else:accuracy_stale+=1
            print(f'{profile} {epoch:02d}/{epochs}: clean train {clean["top1"]:.1%}, val {validation["top1"]:.1%}, gap {record["accuracy_gap"]:+.1%}, val CE {validation["loss"]:.4f}',flush=True)
            stop=early_stop_reason(records,loss_stale,accuracy_stale)
            if stop is not None:
                reason=stop;break
        chosen,selection=select_epoch(records)
        shutil.copyfile(output/f'epoch-{chosen["epoch"]:03d}.pt',output/'best.pt')
        minimum=min(records,key=lambda r:r['val']['loss'])
        shutil.copyfile(output/f'epoch-{minimum["epoch"]:03d}.pt',output/'minimum-val-loss.pt')
        result={'status':'selected','profile':profile,'epochs_completed':len(records),'epoch_cap':epochs,
                'stop_reason':reason,'selection':selection,'selected_metrics':chosen,
                'minimum_val_loss_epoch':minimum['epoch'],'selected_checkpoint_sha256':sha256(output/'best.pt'),
                'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-began,
                'test_used_for_selection':False,
                'learning_status':'underfit' if max(r['train_clean']['top1'] for r in records)<.5 else 'learned_above_gate'}
        write_json(output/'selection.json',result);write_json(output/'status.json',result)
        return result
    finally:
        # Release persistent loaders before the next variant starts its workers.
        for loader in loaders.values():
            iterator=getattr(loader,'_iterator',None)
            if iterator is not None:iterator._shutdown_workers()
        del model;torch.cuda.empty_cache()


def test_selected(cache, output):
    """Only called after the validation-selected checkpoint is frozen."""
    state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    torch.backends.cudnn.enabled=state['config'].get('cudnn_enabled',False)
    model=classifier(state['config']).cuda().eval();model.load_state_dict(state['model'])
    names=[g for g,i in sorted(state['labels'].items(),key=lambda item:item[1])]
    dataset=LandmarkDataset(cache,'test');loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
    confusion=np.zeros((5,5),np.int64);loss=0.;predictions=[];offset=0
    index=json.loads((cache/'test-index.json').read_text())
    with torch.inference_mode():
        for frames,lengths,targets in loader:
            frames=frames.cuda();targets=targets.cuda()
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(frames,lengths);batch_loss=nn.functional.cross_entropy(logits,targets,reduction='sum')
            predicted=logits.argmax(1).cpu().numpy();actual=targets.cpu().numpy();loss+=float(batch_loss)
            for y,p in zip(actual,predicted,strict=True):confusion[y,p]+=1
            for row,y,p in zip(index[offset:offset+len(actual)],actual,predicted,strict=True):
                predictions.append({**row,'target':names[int(y)],'prediction':names[int(p)],'correct':bool(y==p)})
            offset+=len(actual)
    tp=confusion.diagonal().astype(float);support=confusion.sum(1);predicted=confusion.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(5),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(5),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(5),where=precision+recall>0)
    result={'samples':len(dataset),'top1':float(tp.sum()/len(dataset)),'loss':loss/len(dataset),
            'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),
            'labels':names,'confusion':confusion.tolist(),'epoch':state['epoch'],
            'checkpoint_sha256':sha256(output/'best.pt'),'test_used_for_selection':False,
            'per_class':{g:{'support':int(support[i]),'correct':int(tp[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,g in enumerate(names)}}
    write_json(output/'test-report.json',result)
    write_json(output/'test-predictions.json',predictions)
    del model;torch.cuda.empty_cache()
    return result


def run(root):
    root=Path(root);cache=root/'data/training/asl-five-signs-128-v1'
    output=root/'models/asl-gru-five-signs-early-v2'
    if output.exists():raise FileExistsError('Pilot output already exists; preserved unchanged')
    protected=[root/'models/asl-gru-128-baseline/best.pt',
               root/'models/asl-gru-128-cv-regularized-v1/official-refit/best.pt',
               root/'models/asl-stgcn27-v1/best.pt']
    before={str(p.relative_to(root)):sha256(p) for p in protected}
    gpu=cuda_check();print(json.dumps(gpu),flush=True)
    if cache.exists():
        subset=json.loads((cache/'manifest.json').read_text())
        if subset['identity']['parent_cache_id']!=json.loads((root/'data/training/asl-gru-128-v1/manifest.json').read_text())['cache_id']:
            raise ValueError('Subset parent differs')
        for name,wanted in subset['files_sha256'].items():
            if sha256(cache/name)!=wanted:raise ValueError('Subset checksum differs: '+name)
    else:
        subset=prepare_subset(root/'data/training/asl-gru-128-v1',cache)
    output.mkdir(parents=True)
    write_json(output/'experiment.json',{'gpu':gpu,'subset':subset,'protected_weights_before':before,'policy':POLICY})
    results={}
    for name,profile,cap in [('baseline','baseline',10),('regularized','regularized-v1',28)]:
        results[name]={'training':train_variant(cache,output/name,profile,cap,gpu)}
    for name in results:
        results[name]['test']=test_selected(cache,output/name)
        status=json.loads((output/name/'status.json').read_text());status['status']='completed';status['test_evaluated']=True
        write_json(output/name/'status.json',status)
    after={str(p.relative_to(root)):sha256(p) for p in protected}
    report={'status':'complete','classes':list(subset['labels']),'counts':subset['counts'],
            'participants':subset['participants'],'models':results,'policy':POLICY,
            'protected_weights_before':before,'protected_weights_after':after,'previous_weights_unchanged':before==after,
            'note':'New five-output models trained from scratch. Closed set; top-5 is uninformative. No cleaning or normalization change.'}
    write_json(output/'report.json',report)
    print(json.dumps({name:{'epoch':r['training']['selection']['selected_epoch'],'test_top1':r['test']['top1']} for name,r in results.items()},indent=2),flush=True)
    return report
