"""Check invariance and retained finger/body information, independent of classifier weights."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sign_language.feature_geometry import body_geometry_inputs  # noqa: E402


def example():
    rng = np.random.default_rng(17)
    x = torch.from_numpy(rng.normal(0, 0.1, (1, 32, 288)).astype(np.float32))
    x[:, :, 128:130] = torch.tensor([0.5, 0.3])
    x[:, :, 132:134] = torch.tensor([0.3, 0.6])
    x[:, :, 136:138] = torch.tensor([0.7, 0.6])
    x[:, :, [131, 135, 139, 63, 127, 223, 287]] = 1
    return x


def test_geometry_is_finite_with_missing_or_degenerate_hands_and_body():
    x = torch.zeros((2, 32, 288))
    encoded = body_geometry_inputs(x)
    assert encoded.shape == (2, 745) and torch.isfinite(encoded).all()
    assert torch.count_nonzero(encoded) == 0
    x[:, :, [63, 223]] = 1
    assert torch.isfinite(body_geometry_inputs(x)).all()
    with pytest.raises(ValueError, match="32-frame world-hand"):
        body_geometry_inputs(x[:, :16])


def test_image_translation_and_uniform_scale_preserve_body_relative_geometry():
    x = example()
    transformed = x.clone()
    for start in (0, 64):
        points = transformed[:, :, start : start + 63].reshape(1, 32, 21, 3)
        points[..., :2] = points[..., :2] * 0.8 + 0.15
    for start in (128, 132, 136):
        transformed[:, :, start : start + 2] = transformed[:, :, start : start + 2] * 0.8 + 0.15
    assert torch.allclose(body_geometry_inputs(x), body_geometry_inputs(transformed), atol=1e-5)


def test_finger_shape_is_rotation_invariant_but_finger_location_is_retained():
    x = example()
    rotated = x.clone()
    rotation = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for start in (160, 224):
        points = rotated[:, :, start : start + 63].reshape(1, 32, 21, 3)
        points[:] = points @ rotation.T
    assert torch.allclose(body_geometry_inputs(x), body_geometry_inputs(rotated), atol=1e-5)
    moved = x.clone()
    moved[:, :, 8 * 3 + 1] += 0.1  # Index fingertip moves; wrist is unchanged.
    assert not torch.allclose(body_geometry_inputs(x), body_geometry_inputs(moved))
