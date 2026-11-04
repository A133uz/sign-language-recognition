"""Geometry derived from the existing world-hand archive, shared by training and capture."""

import torch


def body_geometry_inputs(features: torch.Tensor) -> torch.Tensor:
    """Summarize finger/body location and finger shape in 745 numeric values.

    Image points are nose-centered and divided by shoulder width. World-hand
    points are already wrist-centered and size-normalized by the raw schema.
    Finger joint cosines and distances reduce dependence on camera rotation;
    this is an experimental representation, not a sign-language rule engine.
    """
    if features.ndim != 3 or features.shape[1:] != (32, 288):
        raise ValueError("body_geometry requires 32-frame world-hand sequences")
    n, t, _ = features.shape
    nose = features[:, :, 128:130]
    shoulder_delta = features[:, :, 132:134] - features[:, :, 136:138]
    width = torch.linalg.vector_norm(shoulder_delta, dim=-1, keepdim=True)
    visible = features[:, :, [131, 135, 139]].amin(dim=-1)
    body = (visible >= 0.5) & (width[:, :, 0] > 0.03)
    hands = features[:, :, :128].reshape(n, t, 2, 64)
    image = hands[:, :, :, :63].reshape(n, t, 2, 21, 3)[..., :2]
    image_valid = hands[:, :, :, 63]
    offsets = (image - nose[:, :, None, None]) / width[:, :, None, None].clamp_min(0.03)
    offsets = offsets * (image_valid * body[:, :, None])[:, :, :, None, None]
    world = features[:, :, 160:288].reshape(n, t, 2, 64)
    points = world[:, :, :, :63].reshape(n, t, 2, 21, 3)
    pieces = []
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for chain in chains:
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:], strict=True):
            left, right = (
                points[:, :, :, a] - points[:, :, :, b],
                points[:, :, :, c] - points[:, :, :, b],
            )
            denom = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
            cosine = (left * right).sum(-1) / denom.clamp_min(1e-6)
            pieces.append(cosine[..., None])
    tips = (4, 8, 12, 16, 20)
    for index, tip in enumerate(tips):
        pieces.append(torch.linalg.vector_norm(points[:, :, :, tip], dim=-1)[..., None])
        for other in tips[index + 1 :]:
            distance = points[:, :, :, tip] - points[:, :, :, other]
            pieces.append(torch.linalg.vector_norm(distance, dim=-1)[..., None])
    shape = torch.cat(pieces, dim=-1) * world[:, :, :, 63:64]
    frames = torch.cat(
        (
            offsets.reshape(n, t, -1),
            shape.reshape(n, t, -1),
            image_valid,
            world[:, :, :, 63],
            body[:, :, None].to(features.dtype),
        ),
        dim=-1,
    )
    return torch.cat(
        [frames[:, start : start + 8].mean(1) for start in (0, 8, 16, 24)]
        + [frames.std(1, correction=0)],
        dim=1,
    )
