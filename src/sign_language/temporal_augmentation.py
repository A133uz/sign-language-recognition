"""Ordered nearest-frame time augmentation; original geometry and masks stay intact."""
from dataclasses import dataclass

import torch
from torch.utils.data import default_collate

CONFIG = {'version': 'ordered-speed-edge-spans-v1', 'speed_length_factor': [.85, 1.15],
          'edge_span_probability': .5, 'edge_span_length_factor': [.5, 1.5],
          'max_frames': 128, 'interpolation': 'nearest original frame; ordered endpoints retained per span',
          'edges': 'existing leading/trailing hand-absent spans only; not ground-truth rest',
          'whole_sequence_identity_probability': .2}


def _uniform(bounds):
    return bounds[0] + (bounds[1] - bounds[0]) * float(torch.rand(()))


def temporal_clip(clip, *, speed_factor=None, edge_factors=None):
    """Modify sample counts without inventing poses or interpolating binary masks.

    Speed factor above one gives more time steps. Up/downsampling changes
    apparent speed, not camera FPS metadata. All nonempty spans keep endpoints.
    Tracking absence is a proxy, not an action-boundary detector.
    """
    if clip.ndim != 2 or clip.shape[1] != 254 or not 1 <= len(clip) <= 128:
        raise ValueError('Expected 1-128 real frames with 254 channels')
    if speed_factor is None and edge_factors is None and float(torch.rand(())) < .2:
        return clip.clone()
    speed_factor = _uniform(CONFIG['speed_length_factor']) if speed_factor is None else speed_factor
    if not .5 <= speed_factor <= 1.5:
        raise ValueError('Invalid temporal speed factor')
    present = (clip[:, 249:251] > 0).any(1)
    ids = torch.nonzero(present, as_tuple=False).flatten()
    if not len(ids):
        return clip.clone()  # No evidence of active hands: do not invent a motion span.
    first, last = int(ids[0]), int(ids[-1]) + 1
    parts = (clip[:first], clip[first:last], clip[last:])
    if edge_factors is None:
        edge_factors = tuple(_uniform(CONFIG['edge_span_length_factor']) if float(torch.rand(())) < .5 else 1. for _ in range(2))
    if len(edge_factors) != 2 or any(not .5 <= value <= 1.5 for value in edge_factors):
        raise ValueError('Invalid edge-span factors')
    factors = (edge_factors[0], speed_factor, edge_factors[1])
    minimum = [min(len(part), 2) for part in parts]
    counts = [max(low, round(len(part) * factor)) for part, factor, low in zip(parts, factors, minimum, strict=True)]
    # Keep the total bounded while retaining the endpoints of every existing span.
    if sum(counts) > 128:
        ratio = 128 / sum(counts)
        counts = [max(low, round(n * ratio)) for n, low in zip(counts, minimum, strict=True)]
        while sum(counts) > 128:
            i = max(range(3), key=lambda j: counts[j] - minimum[j])
            counts[i] -= 1
    selected = []
    for part, count in zip(parts, counts, strict=True):
        if count:
            indices = torch.linspace(0, len(part) - 1, count, device=clip.device).round().long()
            selected.append(part[indices])
    return torch.cat(selected).clone()


@dataclass
class TemporalCollator:
    enabled: bool = False

    def __call__(self, samples):
        if not self.enabled:
            return default_collate(samples)
        frames, lengths, targets = [], [], []
        for padded, length, target in samples:
            if padded.shape != (128, 254):
                raise ValueError('Expected padded [128,254] input')
            if not 1 <= int(length) <= len(padded):
                raise ValueError('Invalid real sequence length')
            sequence = temporal_clip(padded[:int(length)])
            result = torch.zeros_like(padded)
            result[:len(sequence)] = sequence
            frames.append(result)
            lengths.append(len(sequence))
            targets.append(target)
        return torch.stack(frames), torch.tensor(lengths, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
