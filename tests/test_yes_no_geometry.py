import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

HAS_SKLEARN = (importlib.util.find_spec('sklearn') is not None
               and importlib.util.find_spec('torch') is not None)
if HAS_SKLEARN:
    import torch  # Load Windows native libraries in the project's supported order.
    from sign_language.yes_no_geometry import (
        FEATURES, evaluate, grouped_cross_validation, make_probe, vectorize,
    )


@unittest.skipUnless(HAS_SKLEARN, 'Optional scikit-learn extra unavailable')
class GeometryDiagnosticTests(unittest.TestCase):
    def test_only_fixed_geometry_schema_enters_model(self):
        record = {'metrics': {name: i for i, name in enumerate(FEATURES)},
                  'participant_id': 'held-out', 'expected': 'YES'}
        record['metrics']['expected'] = 999
        np.testing.assert_array_equal(vectorize(record), np.arange(len(FEATURES)))
        record['metrics'][FEATURES[0]] = None
        self.assertTrue(np.isnan(vectorize(record)[0]))
        record['insufficient'] = True
        with self.assertRaises(ValueError):
            vectorize(record)
        record['insufficient'] = False
        record['metrics'][FEATURES[0]] = np.inf
        with self.assertRaises(ValueError):
            vectorize(record)
        del record['metrics'][FEATURES[0]]
        with self.assertRaises(ValueError):
            vectorize(record)

    def test_validation_cannot_change_imputation_or_scaling(self):
        x = np.array([[0., np.nan], [2., 10.], [4., 20.], [6., 30.]])
        model = make_probe().fit(x, [0, 0, 1, 1])
        np.testing.assert_array_equal(model.named_steps['impute'].statistics_, [3., 20.])
        mean = model.named_steps['scale'].mean_.copy()
        model.predict_proba([[1.e6, np.nan], [-1.e6, 1.e9]])
        np.testing.assert_array_equal(model.named_steps['scale'].mean_, mean)

    def test_grouped_oof_fits_each_pipeline_on_training_signers_only(self):
        groups = np.repeat(np.arange(9), 2)
        y = np.tile([0, 1], 9)
        x = np.column_stack([y + groups * .01, groups.astype(float)])
        fitted = []

        def factory():
            model = make_probe()
            fitted.append(model)
            return model

        with patch('sign_language.yes_no_geometry.make_probe', side_effect=factory):
            probabilities, folds = grouped_cross_validation(x, y, groups)
        heldout = []
        for model, fold in zip(fitted, folds, strict=True):
            self.assertFalse(set(fold['train_signers']) & set(fold['heldout_signers']))
            train = fold['train_indices']
            np.testing.assert_allclose(model.named_steps['scale'].mean_, x[train].mean(0))
            heldout.extend(fold['heldout_indices'])
        self.assertEqual(sorted(heldout), list(range(len(y))))
        self.assertEqual(probabilities.shape, (18, 2))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_metrics_use_no_yes_label_order_and_truth_rows(self):
        metrics = evaluate([0, 0, 1, 1], [[.9, .1], [.2, .8], [.1, .9], [.7, .3]])
        self.assertEqual(metrics['confusion'], [[1, 1], [1, 1]])
        self.assertEqual(metrics['labels'], ['NO', 'YES'])
        self.assertEqual(metrics['correct'], 2)
        self.assertEqual(metrics['macro_f1'], .5)
        with self.assertRaises(ValueError):
            evaluate([0], [[.5, np.nan]])
