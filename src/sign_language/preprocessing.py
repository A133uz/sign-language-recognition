"""Versioned ASL hand/body preprocessing shared by training and future inference."""

from itertools import permutations

import numpy as np

VERSION = 'asl-body-local-hands-v1'
POSE_POINTS = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
INPUT_SIZE = 254
CONFIG = {
    'version': VERSION, 'max_frames': 128, 'input_size': INPUT_SIZE,
    'pose_points': POSE_POINTS, 'max_anchor_gap': 5, 'clip_features': 8.0,
    'sampling': 'uniform ordered indices including endpoints; no upsampling',
    'padding': 'right-zero-padding; true lengths for packed GRU',
    'coordinates': 'width units: x, y*height/width; hand z remains a local channel',
    'body': 'per-frame shoulder midpoint, median valid shoulder width per sign',
    'hand_shape': 'wrist-relative xyz divided by median palm size per hand',
    'pose_depth': 'omitted; not fused with hand depth',
    'augmentation': 'none in initial baseline',
}


def canonical_hands(arrays, xy_pose, pose_mask, aspect):
    """Assign anatomical slots using pose wrists, labels and short-term continuity."""
    raw = np.asarray(arrays['hands_xyz'], dtype=np.float32).copy()
    raw[..., 1] *= aspect
    present = np.asarray(arrays['hand_present'], dtype=bool)
    labels = np.asarray(arrays['handedness'])
    scores = np.asarray(arrays['handedness_score'])
    result = np.zeros_like(raw)
    valid = np.zeros_like(present)
    previous = [None, None]
    age = [100, 100]
    for t in range(len(raw)):
        detected = np.flatnonzero(present[t])
        age = [a+1 for a in age]
        if not len(detected):
            continue
        costs = np.zeros((len(detected), 2))
        shoulder = np.linalg.norm(xy_pose[t, 11]-xy_pose[t, 12])
        scale = max(float(shoulder), .05)
        for i, source in enumerate(detected):
            wrist = raw[t, source, 0, :2]
            for slot in (0, 1):
                cost = .35*float(scores[t, source])*(int(labels[t, source]) != slot)
                joint = (15, 16)[slot]
                if pose_mask[t, joint]:
                    cost += np.linalg.norm(wrist-xy_pose[t, joint])/scale
                if age[slot] <= 5 and previous[slot] is not None:
                    cost += .2*np.linalg.norm(wrist-previous[slot])/scale
                costs[i, slot] = cost
        assignments = list(permutations((0, 1), len(detected)))
        assignment = min(assignments, key=lambda a: sum(costs[i, slot] for i, slot in enumerate(a)))
        for source, slot in zip(detected, assignment):
            result[t, slot] = raw[t, source]
            valid[t, slot] = True
            previous[slot] = result[t, slot, 0, :2]
            age[slot] = 0
    return result, valid


def normalize_sequence(arrays, frame_shape, max_frames=128):
    """Return unpadded [<=128,254] features, retaining explicit missingness."""
    if not 1 <= max_frames <= 128:
        raise ValueError('Use a maximum of 1–128 frames')
    pose = np.asarray(arrays['pose_xyz'], dtype=np.float32)[..., :2].copy()
    pose_mask = np.asarray(arrays['pose_valid_mask'], dtype=bool).copy()
    if not len(pose) or len(frame_shape) != 3 or min(frame_shape[:2]) <= 0:
        raise ValueError('Empty sequence or invalid image dimensions')
    aspect = frame_shape[0]/frame_shape[1]
    pose[..., 1] *= aspect
    if not np.isfinite(pose).all() or not np.isfinite(arrays['hands_xyz']).all():
        raise ValueError('Nonfinite raw coordinates')
    hands, present = canonical_hands(arrays, pose, pose_mask, aspect)
    center = (pose[:, 11]+pose[:, 12])/2
    widths = np.linalg.norm(pose[:, 11]-pose[:, 12], axis=1)
    anchors = pose_mask[:, 11] & pose_mask[:, 12] & (widths >= .02)
    scale = float(np.median(widths[anchors])) if anchors.any() else 1.0
    anchor_valid = anchors.copy()
    # Interpolate only short internal shoulder gaps, never extrapolate long absences.
    good = np.flatnonzero(anchors)
    for left, right in zip(good[:-1], good[1:]):
        if 1 < right-left <= CONFIG['max_anchor_gap']+1:
            alpha = np.arange(1, right-left, dtype=np.float32)/(right-left)
            center[left+1:right] = center[left]*(1-alpha[:, None])+center[right]*alpha[:, None]
            anchor_valid[left+1:right] = True
    body_pose = (pose[:, POSE_POINTS]-center[:, None])/scale
    body_pose_mask = pose_mask[:, POSE_POINTS] & anchor_valid[:, None]
    body_pose[~body_pose_mask] = 0
    body_hands = (hands[..., :2]-center[:, None, None])/scale
    body_hands[~(present & anchor_valid[:, None])] = 0
    local = hands-hands[:, :, :1]
    palm = np.linalg.norm(local[:, :, [5, 9, 13, 17], :2], axis=-1).mean(axis=-1)
    local_valid = np.zeros_like(present)
    for slot in (0, 1):
        usable = present[:, slot] & (palm[:, slot] >= .005)
        if usable.any():
            local[:, slot] /= float(np.median(palm[usable, slot]))
            local_valid[:, slot] = usable
        local[~usable, slot] = 0
    features = np.concatenate([
        body_pose.reshape(len(pose), -1), body_pose_mask.astype(np.float32),
        body_hands.reshape(len(pose), -1), local.reshape(len(pose), -1),
        present.astype(np.float32), local_valid.astype(np.float32),
        anchor_valid[:, None].astype(np.float32),
    ], axis=1)
    assert features.shape[1] == INPUT_SIZE
    features = np.clip(features, -CONFIG['clip_features'], CONFIG['clip_features'])
    if len(features) > max_frames:
        features = features[np.linspace(0, len(features)-1, max_frames).round().astype(int)]
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite normalized features')
    return features.astype(np.float32), {
        'frames': len(pose), 'frames_with_hands': int(present.any(axis=1).sum()),
        'frames_with_body_anchor': int(anchor_valid.sum()),
    }
