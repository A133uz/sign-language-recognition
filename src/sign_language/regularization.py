"""Conservative training-only perturbations of prepared landmark geometry."""

import torch

PROFILES = {
    'baseline': {'dropout': .3, 'weight_decay': .0001, 'augmentation': None},
    'regularized-v1': {
        'dropout': .5, 'weight_decay': .001,
        'augmentation': {'body_scale': [.9, 1.1], 'roll_degrees': [-5., 5.]},
    },
}


def augment_batch(frames):
    """Use one roll/scale per sign; preserve masks, padding, order and local z."""
    result = frames.clone()
    batch = len(frames)
    angle = (torch.rand(batch, 1, 1, device=frames.device)*10-5)*(torch.pi/180)
    scale = torch.rand(batch, 1, 1, device=frames.device)*.2+.9
    cosine, sine = angle.cos(), angle.sin()

    def transform(xy, size):
        original = xy.clone()
        xy[..., 0] = size*(original[..., 0]*cosine-original[..., 1]*sine)
        xy[..., 1] = size*(original[..., 0]*sine+original[..., 1]*cosine)

    transform(result[..., :26].view(batch, frames.shape[1], 13, 2), scale)
    transform(result[..., 39:123].view(batch, frames.shape[1], 42, 2), scale)
    transform(result[..., 123:249].view(batch, frames.shape[1], 42, 3)[..., :2], 1.)
    return result.clamp(-8, 8)
