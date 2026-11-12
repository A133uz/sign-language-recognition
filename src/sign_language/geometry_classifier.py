"""Fixed five-label geometry-only diagnostic; no GRU or label-dependent routing."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from sign_language.yes_no_geometry import FEATURES, make_probe

LABELS = ('HELLO', 'NO', 'PLEASE', 'THANKYOU', 'YES')


def geometry_vector(articulation):
    if articulation.get('insufficient'):
        return np.full(len(FEATURES), np.nan)
    metrics = articulation['metrics']
    if any(name not in metrics for name in FEATURES):
        raise ValueError('Geometry schema differs')
    vector = np.asarray([np.nan if metrics[name] is None else metrics[name] for name in FEATURES], dtype=float)
    if np.isinf(vector).any():
        raise ValueError('Infinite geometry value')
    return vector


def evaluate(targets, probabilities):
    y = np.asarray(targets)
    p = np.asarray(probabilities)
    if y.ndim != 1 or not len(y) or not np.isin(y, np.arange(len(LABELS))).all():
        raise ValueError('Expected nonempty integer five-class targets')
    if p.shape != (len(y), len(LABELS)) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid five-class probabilities')
    if not np.allclose(p.sum(1), 1):
        raise ValueError('Probabilities must sum to one')
    y = y.astype(np.int64)
    predicted = p.argmax(1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(1)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(len(LABELS)), where=matrix.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros(len(LABELS)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(LABELS)), where=(precision + recall) > 0)
    return {'samples': len(y), 'correct': int(tp.sum()), 'top1': float(tp.sum()/len(y)),
            'macro_f1': float(f1.mean()), 'macro_f1_present_classes': float(f1[support > 0].mean()),
            'labels': list(LABELS), 'confusion': matrix.tolist(),
            'per_class': {name: {'support': int(support[i]), 'correct': int(tp[i]),
                                 'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
                          for i, name in enumerate(LABELS)}}


def grouped_oof(x, targets, groups):
    x, y, groups = np.asarray(x), np.asarray(targets), np.asarray(groups)
    probabilities = np.full((len(y), len(LABELS)), np.nan)
    assigned = np.zeros(len(y), dtype=int)
    folds = []
    for number, (train, heldout) in enumerate(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(x, y, groups), 1):
        if set(groups[train]) & set(groups[heldout]):
            raise AssertionError('Signer leakage')
        if not np.array_equal(np.unique(y[train]), np.arange(len(LABELS))):
            raise ValueError('Fold training lacks a class')
        model = make_probe().fit(x[train], y[train])
        if not np.array_equal(model.classes_, np.arange(len(LABELS))):
            raise AssertionError('Label order differs')
        probabilities[heldout] = model.predict_proba(x[heldout])
        assigned[heldout] += 1
        folds.append({'fold': number, 'train_indices': train.tolist(), 'heldout_indices': heldout.tolist(),
                      'train_signers': sorted(str(g) for g in set(groups[train])),
                      'heldout_signers': sorted(str(g) for g in set(groups[heldout])),
                      'metrics': evaluate(y[heldout], probabilities[heldout])})
    if not np.all(assigned == 1):
        raise AssertionError('Each row needs one out-of-fold prediction')
    return probabilities, folds
