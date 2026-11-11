"""Aggregate-only input diagnostics; no landmark arrays are written to disk."""

import numpy as np
import torch

from sign_language.preprocessing import canonical_hands, normalize_sequence


def distribution(values):
    values = np.asarray(values)
    return {'p05': float(np.quantile(values, .05)), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'max': float(np.max(values))} if values.size else None


def hand_visibility_summary(hand_present):
    """Scalar/count-only capture-ending measurements, never joint coordinates."""
    present=np.asarray(hand_present,dtype=bool)
    if present.ndim!=2 or present.shape[1]!=2 or not len(present):
        raise ValueError('Expected nonempty [frames,2] hand masks')
    visible=present.any(axis=1);tail=visible[-min(10,len(visible)):]
    return {'hand_presence_by_quarter':[float(part.mean()) if len(part) else None for part in np.array_split(visible,4)],
            'final_frame_has_hand':bool(visible[-1]),'final_window_frames':len(tail),
            'final_window_hand_frames':int(tail.sum()),'hand_frame_fraction':float(visible.mean())}


def observation_statistics(arrays, shape):
    pose = arrays['pose_xyz'][..., :2].copy()
    pose[..., 1] *= shape[0]/shape[1]
    mask = arrays['pose_valid_mask']
    valid = mask[:, 11] & mask[:, 12]
    width = np.linalg.norm(pose[:, 11]-pose[:, 12], axis=-1)
    width = width[valid]
    hands, present = canonical_hands(arrays, pose, mask, shape[0]/shape[1])
    palms = np.linalg.norm(hands[:, :, [5, 9, 13, 17], :2]-hands[:, :, :1, :2], axis=-1).mean(-1)
    features, quality = normalize_sequence(arrays, shape)
    body = features[:, 39:123].reshape(-1, 2, 21, 2)
    local = features[:, 123:249].reshape(-1, 2, 21, 3)
    raw_present = arrays['hand_present']
    outside = (arrays['hands_xyz'][..., :2] < 0) | (arrays['hands_xyz'][..., :2] > 1)
    quarter_counts = []
    for group in np.array_split(np.arange(len(pose)), 4):
        quarter_counts.append(float(present[group].any(1).mean()) if len(group) else None)
    return {'frame_shape': list(shape), 'quality': quality,
            'hand_frame_fraction': float(present.any(1).mean()),
            'two_hand_frame_fraction': float(present.all(1).mean()),
            'anatomical_hand_frame_fractions': present.mean(0).tolist(),
            'hand_presence_by_quarter': quarter_counts,
            'shoulder_width_image_fraction': distribution(width),
            'palm_size_pixels': distribution(palms[present]*shape[1]),
            'out_of_image_hand_xy_fraction': float(outside[raw_present].mean()) if raw_present.any() else None,
            'body_hand_abs': distribution(np.abs(body)),
            'local_hand_xy_abs': distribution(np.abs(local[..., :2])),
            'local_hand_depth_abs': distribution(np.abs(local[..., 2])),
            'feature_clipped_fraction': float((np.abs(features) >= 7.999).mean())}


def score_observations(recognizer, arrays, shape, expected):
    features, quality = normalize_sequence(arrays, shape)
    features = features.astype(np.float16).astype(np.float32)
    collected = {}

    def observe_hidden(module, inputs, output):
        hidden = output[1].detach().float()
        collected.update(hidden_abs_mean=float(hidden.abs().mean()),
                         hidden_saturated_fraction=float((hidden.abs() > .99).float().mean()))

    hook = recognizer.model.gru.register_forward_hook(observe_hidden)
    try:
        with torch.inference_mode(), torch.autocast('cuda', enabled=recognizer.device.type == 'cuda'):
            logits = recognizer.model(torch.from_numpy(features)[None].to(recognizer.device),
                                      torch.tensor([len(features)])).float()[0]
        scores = logits.softmax(-1)
        values, ids = scores.topk(5)
        target = recognizer.labels[expected]
        names = [recognizer.names[i] for i in ids.tolist()]
        return {'top5': [{'label': name, 'score': score} for name, score in zip(names, values.tolist())],
                'expected_score': float(scores[target]),
                'expected_rank_min': int((logits > logits[target]).sum())+1,
                'expected_in_top5': expected in names, 'top1_correct': names[0] == expected,
                'quality': quality, 'input_frames': len(features), **collected}
    finally:
        hook.remove()


def compare_capture(recognizer, frames, timestamps, expected):
    arrays, shape, metadata = recognizer.observe_frames(frames, timestamps)
    n = len(frames)
    variants = {'original': score_observations(recognizer, arrays, shape, expected)}
    for name, first, last in (('first_80pct', 0, .8), ('last_80pct', .2, 1), ('middle_60pct', .2, .8)):
        start, end = round(n*first), round(n*last)
        subset = {key: value[start:end] for key, value in arrays.items()}
        variants[name] = score_observations(recognizer, subset, shape, expected)
    mirrored = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    reflected, reflected_shape, _ = recognizer.observe_frames(mirrored, timestamps)
    variants['mirrored_pixels'] = score_observations(recognizer, reflected, reflected_shape, expected)
    return {'expected': expected, **metadata, 'input_statistics': observation_statistics(arrays, shape),
            'mirrored_input_statistics': observation_statistics(reflected, reflected_shape), 'variants': variants}
