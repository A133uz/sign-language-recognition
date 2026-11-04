"""Test the representation shared by offline and live inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from sign_language.landmarks import (
    FEATURE_SIZE,
    POSE_FEATURE_SIZE,
    WORLD_FEATURE_SIZE,
    encode_body_context,
    encode_hands,
    encode_world_hands,
    sample_sequence,
)


def test_hand_order_presence_and_image_coordinates():
    def hand(x):
        return [SimpleNamespace(x=x, y=0.25, z=-0.1) for _ in range(21)]

    frame = encode_hands([hand(0.8), hand(0.2)])
    assert frame.shape == (FEATURE_SIZE,)
    assert frame[0] == pytest.approx(0.2)
    assert frame[64] == pytest.approx(0.8)
    assert frame[63] == frame[127] == 1
    assert encode_hands([]).sum() == 0
    assert encode_hands([hand(0.4)])[127] == 0


def test_temporal_sampling_and_invalid_inputs():
    frames = np.repeat(np.arange(3, dtype=np.float32)[:, None], FEATURE_SIZE, axis=1)
    assert sample_sequence(frames, 5)[:, 0].tolist() == [0, 0, 1, 2, 2]
    with pytest.raises(ValueError):
        sample_sequence(np.empty((0, FEATURE_SIZE)))
    with pytest.raises(ValueError):
        sample_sequence(np.full((1, FEATURE_SIZE), np.nan))


def test_body_context_tracks_pose_and_normalized_wrist_offset():
    hands = encode_hands([[SimpleNamespace(x=0.6, y=0.4, z=0) for _ in range(21)]])
    pose = [SimpleNamespace(x=0.5, y=0.3, z=0, visibility=1.0) for _ in range(33)]
    pose[11].x, pose[12].x = 0.3, 0.7
    encoded = encode_body_context(hands, [pose])
    assert encoded.shape == (POSE_FEATURE_SIZE,)
    assert encoded[128] == pytest.approx(0.5)
    assert encoded[156] == pytest.approx(0.25)
    assert encoded[157] == pytest.approx(0.25)
    assert sample_sequence(encoded[None]).shape == (32, POSE_FEATURE_SIZE)
    assert not encode_body_context(hands, [])[128:].any()


def test_world_shape_is_wrist_centered_scale_invariant_and_keeps_image_slot_pairing():
    image_hands = [[SimpleNamespace(x=x)] for x in (0.8, 0.2)]
    points = np.stack([np.arange(21), np.arange(21) * 0.2, np.arange(21) * 0.1], axis=1)

    def world(array):
        return [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in array]

    encoded = encode_world_hands(image_hands, [world(points), world(points * 3 + [4, 5, 6])])
    assert encoded.shape == (128,)
    assert encoded[:63] == pytest.approx(encoded[64:127], abs=0.00001)
    assert encoded[:3].tolist() == [0, 0, 0]
    assert encoded[63] == encoded[127] == 1
    assert sample_sequence(np.zeros((2, WORLD_FEATURE_SIZE))).shape == (32, WORLD_FEATURE_SIZE)
    assert not encode_world_hands([], []).any()
    with pytest.raises(ValueError, match="match"):
        encode_world_hands(image_hands, [])


def test_world_slots_follow_image_wrist_order_instead_of_world_coordinate_order():
    image = [[SimpleNamespace(x=x)] for x in (0.8, 0.2)]
    world = [
        [SimpleNamespace(x=float(i), y=0, z=0) for i in range(21)],
        [SimpleNamespace(x=0, y=float(i), z=0) for i in range(21)],
    ]
    encoded = encode_world_hands(image, world)
    assert encoded[27:30].tolist() == [0, 1, 0]
    assert encoded[91:94].tolist() == [1, 0, 0]
