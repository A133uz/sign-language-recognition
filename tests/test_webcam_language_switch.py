"""Controls and output selection for the two-language webcam launcher."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from webcam_language_switch import LanguageSelector, load_recognizer, overlay, top_prediction


def test_unavailable_rsl_preserves_asl_and_retries():
    calls = []
    def loader(language):
        calls.append(language)
        if language == 'RSL':
            raise FileNotFoundError('missing checkpoint')
        return SimpleNamespace(name=language)
    selector = LanguageSelector(loader)
    changed, status = selector.select('RSL', 'ready')
    assert not changed and 'unavailable' in status
    assert selector.language == 'ASL' and selector.recognizer.name == 'ASL'
    selector.select('RSL', 'ready')
    assert calls == ['ASL', 'RSL', 'RSL']


def test_switch_only_between_captures_and_reuses_models():
    calls = []
    def loader(language):
        calls.append(language)
        return SimpleNamespace(name=language)
    selector = LanguageSelector(loader)
    for state in ('countdown', 'recording', 'processing'):
        assert not selector.select('RSL', state)[0]
        assert selector.language == 'ASL'
    assert selector.select('RSL', 'ready')[0]
    assert selector.recognizer.name == 'RSL'
    assert selector.select('ASL', 'ready')[0]
    assert selector.select('RSL', 'ready')[0]
    assert calls == ['ASL', 'RSL']


def test_rsl_missing_checkpoint_never_loads_asl():
    with TemporaryDirectory() as directory:
        try:
            load_recognizer('RSL', 'cpu', rsl_checkpoint=Path(directory) / 'missing.pt')
        except FileNotFoundError as exc:
            assert 'No trained SLOVO checkpoint' in str(exc)
        else:
            raise AssertionError('A missing RSL checkpoint must fail')


def test_each_model_uses_its_own_best_label():
    assert top_prediction({'top1': {'label': 'HELLO'}})['label'] == 'HELLO'
    assert top_prediction({'top5': [{'label': 'ПРИВЕТ'}, {'label': 'ПОКА'}]})['label'] == 'ПРИВЕТ'


def test_unicode_overlay_keeps_frame_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = overlay(frame, 'RSL', 'Ready', 'ПРИВЕТ')
    assert result.shape == frame.shape and result.dtype == frame.dtype
    assert np.any(result != frame)


class TestLanguageSwitch(unittest.TestCase):
    def test_unavailable(self): test_unavailable_rsl_preserves_asl_and_retries()
    def test_between_captures(self): test_switch_only_between_captures_and_reuses_models()
    def test_missing_checkpoint(self): test_rsl_missing_checkpoint_never_loads_asl()
    def test_output_mapping(self): test_each_model_uses_its_own_best_label()
    def test_unicode(self): test_unicode_overlay_keeps_frame_size()
