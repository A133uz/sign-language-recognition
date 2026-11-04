"""Exercise WLASL's inclusive frame bounds on generated video."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sign_language.clips import extract_interval  # noqa: E402


def test_extract_interval_preserves_inclusive_bounds_and_existing_output(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((48, 64, 3), i * 25, dtype=np.uint8))
    writer.release()
    output = tmp_path / "clip.mp4"
    assert extract_interval(source, output, 3, 5) == 3
    cap = cv2.VideoCapture(str(output))
    means = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        means.append(frame.mean())
    cap.release()
    assert len(means) == 3
    assert means[0] == pytest.approx(50, abs=8)
    assert means[-1] == pytest.approx(100, abs=8)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        extract_interval(source, output, 1, 2)
    assert output.read_bytes() == original
    missing = tmp_path / "short.mp4"
    with pytest.raises(ValueError, match="ended before"):
        extract_interval(source, missing, 7, 10)
    assert not missing.exists()


def test_native_fps_is_resampled_before_applying_annotations(tmp_path):
    source = tmp_path / "fifty_fps.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 48))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 15, dtype=np.uint8))
    writer.release()
    output = tmp_path / "resampled.mp4"
    assert extract_interval(source, output, 2, 4, fps=25) == 3
    cap = cv2.VideoCapture(str(output))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25)
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        values.append(frame.mean())
    cap.release()
    assert values == pytest.approx([30, 60, 90], abs=8)
