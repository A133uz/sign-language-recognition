"""Fixed binary diagnostic using explicit scalar geometry, never a five-sign router."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS=('NO','YES')
FEATURES=(
    'thumb_index_xy_median','thumb_index_xy_range','thumb_index_xyz_median','thumb_index_xyz_range',
    'thumb_middle_xy_median','thumb_middle_xy_range','thumb_middle_xyz_median','thumb_middle_xyz_range',
    'index_tip_wrist_median','middle_tip_wrist_median',
    'index_pip_angle_median_deg','index_pip_angle_range_deg',
    'middle_pip_angle_median_deg','middle_pip_angle_range_deg',
    'palm_axis_xy_excursion_deg','palm_axis_xyz_excursion_deg','wrist_body_xy_excursion',
)


def vectorize(articulation):
    if articulation.get('insufficient'):raise ValueError('Insufficient tracked hand geometry')
    metrics=articulation['metrics']
    if any(name not in metrics for name in FEATURES):raise ValueError('Geometry feature schema differs')
    values=np.asarray([np.nan if metrics[name] is None else metrics[name] for name in FEATURES],np.float64)
    if np.isinf(values).any():raise ValueError('Infinite geometry feature')
    return values


def make_probe():
    return Pipeline([
        ('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('scale',StandardScaler()),
        ('classifier',LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=42)),
    ])


def evaluate(targets,probabilities):
    targets=np.asarray(targets,dtype=np.int64);probabilities=np.asarray(probabilities)
    if probabilities.shape!=(len(targets),2) or not np.isfinite(probabilities).all():raise ValueError('Invalid binary probabilities')
    predicted=probabilities.argmax(1);matrix=np.zeros((2,2),np.int64)
    np.add.at(matrix,(targets,predicted),1)
    tp=matrix.diagonal().astype(float);support=matrix.sum(1);counts=matrix.sum(0)
    precision=np.divide(tp,counts,out=np.zeros(2),where=counts>0)
    recall=np.divide(tp,support,out=np.zeros(2),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(2),where=(precision+recall)>0)
    return {'samples':len(targets),'correct':int(tp.sum()),'top1':float(tp.sum()/len(targets)),
        'balanced_accuracy':float(recall.mean()),'macro_f1':float(f1.mean()),'confusion':matrix.tolist(),
        'labels':list(LABELS),'per_class':{name:{'support':int(support[i]),'correct':int(tp[i]),
                    'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,name in enumerate(LABELS)}}


def grouped_cross_validation(x,y,groups):
    """Fixed settings, train-only signer-held-out folds; fit preprocessing afresh."""
    x=np.asarray(x);y=np.asarray(y);groups=np.asarray(groups)
    predictions=np.full((len(y),2),np.nan);assigned=np.zeros(len(y),np.int64);folds=[]
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
    for fold,(train,heldout) in enumerate(splitter.split(x,y,groups),1):
        if set(groups[train])&set(groups[heldout]):raise AssertionError('Signer leakage')
        if len(np.unique(y[train]))!=2:raise ValueError('Fold training lacks a class')
        model=make_probe();model.fit(x[train],y[train]);prob=model.predict_proba(x[heldout])
        if not np.array_equal(model.classes_,[0,1]):raise AssertionError('Label order changed')
        predictions[heldout]=prob;assigned[heldout]+=1
        folds.append({'fold':fold,'train_indices':train.tolist(),'heldout_indices':heldout.tolist(),
            'train_signers':sorted(set(groups[train])),'heldout_signers':sorted(set(groups[heldout])),
            'metrics':evaluate(y[heldout],prob)})
    if not np.all(assigned==1):raise AssertionError('Each training row must have one held-out prediction')
    return predictions,folds
