import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sign_language.hands import draw_hands
from sign_language.webcam import run_webcam


class PreviewTests(unittest.TestCase):
    def test_skeleton_draws_without_modifying_the_source_pixels(self):
        source = np.zeros((240, 320, 3), np.uint8)
        points = [SimpleNamespace(x=.15 + i * .025, y=.2 + i * .02) for i in range(21)]
        view = draw_hands(source, SimpleNamespace(hand_landmarks=[points]))
        self.assertTrue(np.any(view))
        self.assertFalse(np.any(source))

    def test_camera_read_failure_releases_the_camera(self):
        camera = SimpleNamespace(isOpened=lambda: True, set=lambda *a: True,
                                 read=lambda: (False, None), release=lambda: None)
        with patch("sign_language.webcam.cv2.VideoCapture", return_value=camera), \
             patch.object(camera, "release") as release, \
             patch("sign_language.webcam.HandTracker") as tracker, \
             patch("sign_language.webcam.cv2.destroyAllWindows"):
            report = run_webcam()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["media_saved"])
        self.assertFalse(report["training_performed"])
        release.assert_called_once()
        tracker.return_value.__exit__.assert_called_once()
