"""Check that view diagnostics measure crop geometry and never invent missing detections."""

import numpy as np
import pytest

from sign_language.view_quality import analyze_capture_view, crop_geometry, summarize_model_view


def test_crop_geometry_matches_backend_sizes_and_rounding():
    assert crop_geometry(640, 480, "wlasl") == (341, 256, 58, 16)
    assert crop_geometry(640, 480, "citizen") == (256, 256, 16, 16)
    assert crop_geometry(200, 100, "citizen") == (452, 226, 114, 1)
    with pytest.raises(ValueError):
        crop_geometry(0, 480, "wlasl")


def test_hand_size_and_visibility_use_only_detected_hands():
    features = np.zeros((3, 160), dtype=np.float32)
    points = np.stack(
        [np.linspace(0.45, 0.55, 21), np.linspace(0.4, 0.6, 21), np.zeros(21)], axis=1
    )
    features[0, :63] = points.ravel()
    points[:, 0] += 1
    features[1, :63] = points.ravel()
    features[:2, 63] = 1
    result = summarize_model_view(features, 640, 480)
    assert result["sampled_frames"] == 3
    assert result["frames_with_hands"] == result["detected_hand_instances"] == 2
    assert result["all_hand_points_inside_fraction"] == 0.5
    assert result["hand_points_inside_fraction"] == 0.5
    assert result["hand_bbox_short_side_px"]["p50"] == pytest.approx(34.1, abs=0.0001)
    assert result["hand_bbox_long_side_px"]["p50"] == pytest.approx(51.2, abs=0.0001)
    assert result["shoulder_span_px"] is None
    empty = summarize_model_view(np.zeros((2, 128)), 640, 480)
    assert empty["frames_with_hands"] == 0
    assert empty["hand_points_inside_fraction"] is None
    assert empty["hand_bbox_short_side_px"] is None


def test_pose_reference_requires_visible_nose_and_shoulders():
    features = np.zeros((2, 160), dtype=np.float32)
    pose = features[:, 128:156].reshape(2, 7, 4)
    pose[:, :3, 3] = 1
    pose[:, 0, 1] = 0.2
    pose[:, 1, :2], pose[:, 2, :2] = [0.35, 0.5], [0.65, 0.5]
    pose[1, 0, 3] = 0.1
    result = summarize_model_view(features, 640, 480)
    assert result["pose_context_frames"] == 1
    assert result["shoulder_span_px"]["p50"] == pytest.approx(102.3, abs=0.0001)
    assert result["nose_y_px"]["p50"] == pytest.approx(35.2, abs=0.0001)


def test_capture_analysis_keeps_only_aggregate_results():
    pytest.importorskip("cv2")

    class Extractor:
        calls = 0

        def extract(self, frame):
            assert frame.shape == (360, 480, 3)
            self.calls += 1
            return np.zeros(160, dtype=np.float32)

    extractor = Extractor()
    result = analyze_capture_view([np.zeros((480, 640, 3), dtype=np.uint8)], extractor, "wlasl")
    assert extractor.calls == 32
    assert result["frames_with_hands"] == 0
    assert all(not isinstance(value, np.ndarray) for value in result.values())
    with pytest.raises(ValueError, match="nonempty"):
        analyze_capture_view([], extractor, "wlasl")
