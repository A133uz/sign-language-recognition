"""Privacy-preserving, aggregate detector-coverage diagnostics."""

import numpy as np

from sign_language.face_gru_recognition import normalize_joined


def window_health(snapshot):
    """Explain model quality checks without retaining any landmark positions."""
    timestamps, body, face, present, blend, shape = snapshot
    raw_hands = np.asarray(body['hand_present'], dtype=bool).any(axis=1)
    raw_shoulders = np.asarray(body['pose_valid_mask'], dtype=bool)[:, [11, 12]].all(axis=1)
    result = {'observed_frames': int(len(timestamps)), 'raw_hand_frames': int(raw_hands.sum()),
              'raw_shoulder_frames': int(raw_shoulders.sum()),
              'raw_face_frames': int(np.asarray(present, dtype=bool).sum())}
    try:
        _, quality = normalize_joined(body, face, present, blend, shape)
    except (ValueError, AssertionError):
        result.update(normalization_status='failed', model_hand_frames=None,
                      model_body_anchor_frames=None, model_face_frames=None)
        return result
    result.update(normalization_status='ok', model_hand_frames=quality['frames_with_hands'],
                  model_body_anchor_frames=quality['frames_with_body_anchor'],
                  model_face_frames=quality['frames_with_face'])
    return result


def failure_category(error, health):
    """Classify only known fixed messages; never serialize exception text."""
    text = str(error)
    if text == 'Keep hands and shoulders visible':
        missing_hands = health['model_hand_frames'] == 0
        missing_body = health['model_body_anchor_frames'] == 0
        if missing_hands and missing_body:
            return 'no_hands_and_no_body_anchor'
        if missing_hands:
            return 'no_hands'
        if missing_body:
            return 'no_body_anchor'
        return 'hand_body_check_unresolved'
    if text == 'Keep face visible':
        return 'no_valid_face'
    if text == 'Window needs eight frames and increasing timestamps':
        return 'invalid_window_timestamps_or_length'
    if text == 'Nonfinite prediction':
        return 'nonfinite_prediction'
    return 'other_value_error'
