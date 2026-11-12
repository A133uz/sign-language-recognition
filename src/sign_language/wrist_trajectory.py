"""Exploratory trajectory measurements; not sign labels or tracking ground truth."""
import numpy as np


def contiguous_spans(valid):
    ids = np.flatnonzero(valid)
    return [group for group in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(group)]


def path_metrics(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('Expected at least two finite XY points')
    steps = np.diff(p, axis=0)
    distances = np.linalg.norm(steps, axis=1)
    length = float(distances.sum())
    gap = float(np.linalg.norm(p[-1] - p[0]))
    area = float(.5 * np.sum(p[:, 0]*np.roll(p[:, 1], -1) - p[:, 1]*np.roll(p[:, 0], -1)))
    perimeter = length + gap
    circularity = float(np.clip(4*np.pi*abs(area)/perimeter**2, 0, 1)) if perimeter > 1.e-8 else 0.
    sv = np.linalg.svd(p - p.mean(0), compute_uv=False)
    ratio = float(sv[1]/sv[0]) if sv[0] > 1.e-8 else 0.
    good = distances > 1.e-5
    pairs = good[:-1] & good[1:]
    cross = steps[:-1, 0]*steps[1:, 1] - steps[:-1, 1]*steps[1:, 0]
    dot = (steps[:-1]*steps[1:]).sum(1)
    turns = np.arctan2(cross[pairs], dot[pairs])
    closure = min(1., gap/length) if length > 1.e-8 else 1.
    return {'points': len(p), 'path_length': length, 'endpoint_distance': gap,
            'endpoint_over_path': closure, 'axis_ratio': ratio,
            'x_extent': float(np.ptp(p[:, 0])), 'y_extent': float(np.ptp(p[:, 1])),
            'signed_turns': float(turns.sum()/(2*np.pi)),
            'turn_coherence': float(abs(turns.sum())/np.abs(turns).sum()) if np.abs(turns).sum() > 1.e-8 else 0.,
            'closed_polygon_circularity': circularity,
            'loop_probe_score': circularity*(1-closure)}


def trajectory_summary(points, valid, fps):
    p, mask = np.asarray(points), np.asarray(valid, dtype=bool)
    if p.ndim != 2 or p.shape != (len(mask), 2) or not np.isfinite(p).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid aligned trajectory/mask/FPS')
    spans = contiguous_spans(mask)
    usable = [span for span in spans if len(span) >= 2]
    result = {'total_frames': len(p), 'valid_frames': int(mask.sum()), 'contiguous_spans': len(spans),
              'longest_span_frames': max((len(s) for s in spans), default=0), 'longest_span': None,
              'best_loop_probe': None,
              'limits': 'Projected path metrics, not confirmed circles or ground-truth joints; entry/exit motion and tracking jitter can dominate.'}
    if not usable:
        return result
    longest = max(usable, key=len)
    result['longest_span'] = {'start': int(longest[0]), 'end': int(longest[-1]), **path_metrics(p[longest])}
    best = None
    for span in usable:
        # Prespecified 0.4/0.8/1.2-second windows, no gap bridging or smoothing.
        for seconds in (.4, .8, 1.2):
            count = max(6, round(seconds*fps))
            for offset in range(max(0, len(span)-count+1)):
                ids = span[offset:offset+count]
                metrics = path_metrics(p[ids])
                candidate = {'start': int(ids[0]), 'end': int(ids[-1]), 'window_seconds': seconds, **metrics}
                if best is None or candidate['loop_probe_score'] > best['loop_probe_score']:
                    best = candidate
    result['best_loop_probe'] = best
    return result
