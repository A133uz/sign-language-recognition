import importlib.util
import unittest
from unittest.mock import patch

HAS_DEPENDENCIES = (importlib.util.find_spec('torch') is not None
                    and importlib.util.find_spec('sklearn') is not None)
if HAS_DEPENDENCIES:
    import torch
    import numpy as np
    from sign_language.geometry_classifier import (
        FEATURES, LABELS, geometry_vector, evaluate, grouped_oof, make_probe,
    )


@unittest.skipUnless(HAS_DEPENDENCIES, 'Optional training dependencies unavailable')
class FiveClassGeometryTests(unittest.TestCase):
    def test_geometry_schema_retains_missing_clips_and_ignores_metadata(self):
        record = {'insufficient': False, 'metrics': {n: i for i, n in enumerate(FEATURES)},
                  'expected': 'NO', 'participant_id': 'P99'}
        np.testing.assert_array_equal(geometry_vector(record), np.arange(17))
        record['metrics'][FEATURES[0]] = None
        self.assertTrue(np.isnan(geometry_vector(record)[0]))
        record['insufficient'] = True
        self.assertTrue(np.isnan(geometry_vector(record)).all())
        record['insufficient'] = False
        del record['metrics'][FEATURES[0]]
        with self.assertRaises(ValueError):
            geometry_vector(record)

    def test_five_label_metrics_preserve_confusion_order_and_reject_bad_scores(self):
        y = np.arange(5)
        result = evaluate(y, np.eye(5))
        self.assertEqual(result['labels'], list(LABELS))
        self.assertEqual(result['correct'], 5)
        self.assertEqual(result['macro_f1'], 1.)
        self.assertEqual(result['confusion'], np.eye(5, dtype=int).tolist())
        result = evaluate([0, 1], np.eye(5)[:2])
        self.assertAlmostEqual(result['macro_f1'], .4)
        self.assertEqual(result['macro_f1_present_classes'], 1.)
        for targets, scores in [([-1], [[1, 0, 0, 0, 0]]), ([0], [[2, 0, 0, 0, 0]]),
                                 ([0], [[np.nan, 0, 0, 0, 0]])]:
            with self.assertRaises(ValueError):
                evaluate(targets, scores)

    def test_five_class_grouped_oof_keeps_signers_and_preprocessing_separate(self):
        groups = np.repeat(np.arange(9), 5)
        y = np.tile(np.arange(5), 9)
        x = np.column_stack([np.eye(5)[y], groups.astype(float)])
        fitted = []

        def factory():
            model = make_probe()
            fitted.append(model)
            return model

        with patch('sign_language.geometry_classifier.make_probe', side_effect=factory):
            p, folds = grouped_oof(x, y, groups)
        assigned = []
        for model, fold in zip(fitted, folds, strict=True):
            self.assertFalse(set(fold['train_signers']) & set(fold['heldout_signers']))
            np.testing.assert_allclose(model.named_steps['scale'].mean_, x[fold['train_indices']].mean(0))
            np.testing.assert_array_equal(model.classes_, np.arange(5))
            assigned.extend(fold['heldout_indices'])
        self.assertEqual(sorted(assigned), list(range(len(y))))
        self.assertEqual(p.shape, (45, 5))
        self.assertTrue(np.isfinite(p).all())
