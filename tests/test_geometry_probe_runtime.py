import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest

HAS_DEPENDENCIES = (importlib.util.find_spec('torch') is not None
                    and importlib.util.find_spec('sklearn') is not None)
if HAS_DEPENDENCIES:
    import torch
    import numpy as np
    from sign_language.geometry_probe_runtime import GeometryProbe
    from sign_language.geometry_classifier import FEATURES, LABELS, make_probe
    from sign_language.five_sign_pilot import sha256


@unittest.skipUnless(HAS_DEPENDENCIES,'Optional training dependencies unavailable')
class GeometryProbeRuntimeTests(unittest.TestCase):
    def test_fitted_pipeline_restores_without_refitting_and_checks_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(__file__).resolve().parents[1] / 'src/sign_language'
            x = np.random.default_rng(42).normal(size=(30, 17))
            y = np.tile(np.arange(5), 6)
            model = make_probe().fit(x, y)
            path = root / 'model.pkl'
            with path.open('wb') as stream:
                pickle.dump(model, stream)
            config = {'features': list(FEATURES), 'labels': list(LABELS), 'cache_id': 'fixture',
                      'source_sha256': {name: sha256(source / name) for name in
                          ('geometry_classifier.py', 'yes_no_geometry.py', 'hand_articulation.py')}}
            (root / 'config.json').write_text(json.dumps(config))
            (root / 'report.json').write_text(json.dumps({'config': config, 'model_sha256': sha256(path)}))
            runtime = GeometryProbe(path, models_root=root)
            record = {'metrics': dict(zip(FEATURES, x[0], strict=True))}
            before = runtime.model.named_steps['scale'].mean_.copy()
            first = runtime.score(record, 'NO')
            second = runtime.score(record, 'HELLO')
            self.assertEqual(first['top5'], second['top5'])
            predicted = dict((v['label'], v['score']) for v in first['top5'])
            np.testing.assert_allclose([predicted[name] for name in LABELS], model.predict_proba(x[:1])[0])
            np.testing.assert_array_equal(runtime.model.named_steps['scale'].mean_, before)
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                GeometryProbe(path, models_root=root)
            with self.assertRaisesRegex(ValueError, 'local project'):
                GeometryProbe(path, models_root=root / 'other')
