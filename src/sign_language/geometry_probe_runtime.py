"""Restore the trusted local five-class probe with its fitted preprocessing."""
import json
from pathlib import Path
import pickle

import numpy as np

from sign_language.five_sign_pilot import sha256
from sign_language.geometry_classifier import FEATURES, LABELS, geometry_vector


class GeometryProbe:
    def __init__(self, path, models_root=None):
        self.path = Path(path).resolve()
        root = Path(models_root).resolve() if models_root is not None else Path(__file__).resolve().parents[2] / 'models'
        if not self.path.is_relative_to(root):
            raise ValueError('Probe must be a local project model artifact')
        self.config = json.loads(self.path.with_name('config.json').read_text())
        report = json.loads(self.path.with_name('report.json').read_text())
        if self.config != report['config']:
            raise ValueError('Probe configuration differs from fitted report')
        if self.config['features'] != list(FEATURES) or self.config['labels'] != list(LABELS):
            raise ValueError('Probe feature or label schema differs')
        for name, expected in self.config['source_sha256'].items():
            if Path(name).name != name or sha256(Path(__file__).with_name(name)) != expected:
                raise ValueError('Probe source differs: ' + name)
        self.sha256 = sha256(self.path)
        if self.sha256 != report['model_sha256']:
            raise ValueError('Probe model checksum differs from fitted report')
        # Only load a trusted locally generated artifact after integrity checks.
        with self.path.open('rb') as stream:
            self.model = pickle.load(stream)
        if not np.array_equal(self.model.classes_, np.arange(len(LABELS))) or self.model.n_features_in_ != len(FEATURES):
            raise ValueError('Fitted probe schema differs')
        self.names = list(LABELS)

    def score(self, articulation, expected):
        target = self.names.index(expected)
        probabilities = self.model.predict_proba(geometry_vector(articulation)[None])[0]
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1):
            raise ValueError('Invalid probe scores')
        order = np.argsort(-probabilities)
        return {'top5': [{'label': self.names[int(i)], 'score': float(probabilities[i])} for i in order],
                'expected_score': float(probabilities[target]),
                'expected_rank': int(np.flatnonzero(order == target)[0] + 1),
                'correct': bool(order[0] == target)}
