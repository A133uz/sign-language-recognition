"""Inference-only metrics and official small-vocabulary test selection."""
import numpy as np
import torch
from sign_language.evaluation import classification_metrics


def select_test_subset(index,features,lengths,targets,labels,parent_labels):
    if sorted(labels.values())!=list(range(len(labels))):raise ValueError('Noncontiguous model labels')
    if any(row['split']!='test' for row in index):raise ValueError('Expected official test rows only')
    ids=[i for i,row in enumerate(index) if row['gloss'] in labels]
    if not ids or len({index[i]['video_file'] for i in ids})!=len(ids):raise ValueError('Empty/duplicate test selection')
    if not np.array_equal(targets[ids],[parent_labels[index[i]['gloss']] for i in ids]):raise ValueError('Parent target mismatch')
    x=np.array(features[ids],dtype=np.float32);n=np.array(lengths[ids],dtype=np.int64)
    y=np.array([labels[index[i]['gloss']] for i in ids],dtype=np.int64)
    if x.shape!=(len(ids),128,254) or not np.isfinite(x).all() or np.any((n<1)|(n>128)):
        raise ValueError('Invalid test tensor/lengths')
    if set(y)!=set(labels.values()):raise ValueError('A class is absent from test')
    rows=[{**index[i],'parent_index':i,'parent_class_id':int(targets[i]),'class_id':int(y[j])} for j,i in enumerate(ids)]
    return rows,x,n,y


def evaluate_model(model,features,lengths,targets,names,device,*,amp=True,batch_size=16):
    """All examples, original 128-step grid, no augmentation or branch overrides."""
    model.eval();matrix=np.zeros((len(names),len(names)),np.int64)
    losses=0.;hits5=0;predictions=[]
    with torch.inference_mode():
        for start in range(0,len(targets),batch_size):
            end=min(start+batch_size,len(targets))
            x=torch.from_numpy(features[start:end]).to(device)
            n=torch.from_numpy(lengths[start:end]).long();y=torch.from_numpy(targets[start:end]).to(device,dtype=torch.long)
            with torch.autocast(device_type=torch.device(device).type,enabled=amp and torch.device(device).type=='cuda'):
                logits=model(x,n).float()
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite model logits')
            losses+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum'))
            probabilities=logits.softmax(1).cpu().numpy()
            for offset,p in enumerate(probabilities):
                target=int(targets[start+offset]);order=np.argsort(-p,kind='stable');pred=int(order[0])
                matrix[target,pred]+=1;hits5+=int(target in order[:min(5,len(names))])
                predictions.append({'expected':names[target],'prediction':names[pred],'correct':pred==target,
                    'scores':[float(v) for v in p],
                    'top5':[{'label':names[int(i)],'score':float(p[i])} for i in order[:5]]})
    metrics,precision,recall,f1=classification_metrics(matrix)
    metrics.update(samples=len(targets),correct=int(matrix.trace()),top5=hits5/len(targets),loss=losses/len(targets))
    per_class={g:{'support':int(matrix[i].sum()),'correct':int(matrix[i,i]),'precision':float(precision[i]),
                  'recall':float(recall[i]),'f1':float(f1[i])} for i,g in enumerate(names)}
    return {'metrics':metrics,'labels':names,'confusion':matrix.tolist(),'per_class':per_class,'predictions':predictions}
