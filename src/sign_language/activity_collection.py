"""Local-only, consented storage for coarse-labeled hand/pose observations."""

import hashlib
import io
import json
import os
from pathlib import Path
import uuid

import numpy as np

from sign_language.dataset import write_json

SCHEMA = 'activity-hand-pose-v1'
SHAPES = {'hands_xyz': (2, 21, 3), 'hand_present': (2,), 'handedness': (2,),
          'handedness_score': (2,), 'pose_xyz': (33, 3), 'pose_visibility': (33,),
          'pose_presence': (33,), 'pose_detected': (), 'pose_valid_mask': (33,)}


def save_trial(directory, number, prompt, category, observations, timestamps, frame_shape):
    """Write one immutable NPZ plus metadata; no face, images or embeddings."""
    directory = Path(directory)
    if number < 1 or not observations or len(observations) != len(timestamps):
        raise ValueError('Invalid trial data')
    if category not in {'rest', 'unrelated_movement', 'known_sign', 'unknown_sign', 'held_pose'}:
        raise ValueError('Unsupported prompt category')
    if len(frame_shape) != 3 or min(frame_shape[:2]) <= 0:
        raise ValueError('Invalid source dimensions')
    times = np.asarray(timestamps, dtype=np.int64)
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase')
    arrays = {}
    for key, shape in SHAPES.items():
        if any(set(item) != set(SHAPES) for item in observations):
            raise ValueError('Unexpected observation schema')
        value = np.stack([item[key] for item in observations])
        if value.shape != (len(times), *shape) or value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError(f'Invalid observation array: {key}')
        arrays[key] = value
    arrays['timestamp_ms'] = times
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'trial-{number:03d}.npz'
    meta = directory / f'trial-{number:03d}.json'
    if target.exists() or meta.exists():
        raise FileExistsError('Trial already exists')
    content = io.BytesIO()
    np.savez_compressed(content, **arrays)
    payload = content.getvalue()
    temporary = directory / f'.{uuid.uuid4().hex}.tmp'
    try:
        # Exclusive creation protects already-recorded trial files.
        with temporary.open('xb') as handle:
            handle.write(payload)
        os.replace(temporary, target)
    finally:
        if temporary.exists(): temporary.unlink()
    metadata = {'schema': SCHEMA, 'trial': number, 'prompt': prompt, 'category': category,
                'prompt_confirmation': 'pending', 'frame_count': len(times),
                'frame_shape': list(frame_shape), 'duration_ms': int(times[-1]-times[0]),
                'sha256': hashlib.sha256(payload).hexdigest(), 'video_saved': False,
                'face_mesh_saved': False, 'file': target.name}
    write_json(meta, metadata)
    return metadata


def validate_trial(directory, metadata):
    """Verify bytes and numeric shapes without loading pickle data."""
    path = Path(directory) / metadata['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Trial hash mismatch')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(SHAPES) | {'timestamp_ms'}:
            raise ValueError('Unexpected archive fields')
        count = metadata['frame_count']
        for key, shape in SHAPES.items():
            value = arrays[key]
            if value.shape != (count, *shape) or value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
                raise ValueError(f'Invalid observation array: {key}')
        timestamps = arrays['timestamp_ms']
        if timestamps.shape != (count,) or timestamps.dtype.kind not in 'iu' or np.any(np.diff(timestamps) <= 0):
            raise ValueError('Invalid timestamps')
    return True
