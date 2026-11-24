"""Aggregate, experimental motion measurements from one raw observation window."""

import numpy as np

from sign_language.preprocessing import normalize_sequence


def motion_summary(snapshot, max_gap_seconds=.25):
    """Return percentile speeds and coverage; never return or persist coordinates.

    Wrist displacement is relative to shoulder width. Finger displacement is
    wrist-relative and palm-scaled. These are noisy proxies, not a sign detector.
    """
    timestamps, body, _face, _face_present, _blend, shape = snapshot
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Need increasing timestamps')
    features, _ = normalize_sequence(body, shape)
    if len(timestamps) > 128:
        timestamps = timestamps[np.linspace(0, len(timestamps) - 1, 128).round().astype(int)]
    dt = np.diff(timestamps) / 1000.0
    usable_gap = (dt > 0) & (dt <= max_gap_seconds)
    wrists = features[:, 39:123].reshape(-1, 2, 21, 2)[:, :, 0]
    local = features[:, 123:249].reshape(-1, 2, 21, 3)[..., :2]
    present = features[:, 249:251] > .5
    local_valid = features[:, 251:253] > .5
    anchor = features[:, 253] > .5
    wrist_valid = present[1:] & present[:-1] & anchor[1:, None] & anchor[:-1, None] & usable_gap[:, None]
    finger_valid = local_valid[1:] & local_valid[:-1] & usable_gap[:, None]
    wrist_speed = np.linalg.norm(np.diff(wrists, axis=0), axis=-1) / dt[:, None]
    finger_speed = np.linalg.norm(np.diff(local, axis=0), axis=-1).mean(axis=-1) / dt[:, None]

    def percentile(values, mask):
        selected = values[mask]
        return float(np.percentile(selected, 75)) if selected.size else None

    return {'wrist_speed_p75_shoulder_widths_per_second': percentile(wrist_speed, wrist_valid),
            'handshape_speed_p75_palm_widths_per_second': percentile(finger_speed, finger_valid),
            'valid_wrist_pairs': int(wrist_valid.sum()), 'valid_handshape_pairs': int(finger_valid.sum()),
            'eligible_time_pairs': int(usable_gap.sum()), 'sampled_frames': len(timestamps)}
