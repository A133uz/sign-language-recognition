"""27-joint XY graph adapter from the existing, verified landmark cache.

Joint subset/topology follows Microsoft's ASL Citizen ST-GCN configuration.
Normalization and anatomical hand slots retain this project's preprocessing.
"""

import torch

BODY = (0, 2, 5, 11, 12, 13, 14)
BODY_FEATURE_INDICES = (0, 1, 2, 7, 8, 9, 10)
HAND = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
EDGES = ((2, 0), (1, 0), (0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (6, 17),
         (7, 8), (7, 9), (9, 10), (7, 11), (11, 12), (7, 13), (13, 14),
         (7, 15), (15, 16), (17, 18), (17, 19), (19, 20), (17, 21), (21, 22),
         (17, 23), (23, 24), (17, 25), (25, 26))
CONFIG = {'version': 'asl-stgcn27-xy-v1', 'nodes': 27, 'channels': 2,
          'body_joints': list(BODY), 'hand_joints': list(HAND),
          'hand_order': ['anatomical_left', 'anatomical_right'],
          'edges': [list(edge) for edge in EDGES], 'partitions': 'self/inward/outward, column-normalized',
          'normalization': 'existing asl-body-local-hands-v1 body-relative XY; missing coordinates zero',
          'temporal_padding': 'masked before/after each block; valid-time/node average pooling'}


def adjacency():
    matrices = []
    for links in ([(i, i) for i in range(27)], EDGES, [(b, a) for a, b in EDGES]):
        matrix = torch.zeros(27, 27)
        for source, target in links:
            matrix[source, target] = 1
        matrix /= matrix.sum(0, keepdim=True).clamp_min(1)
        matrices.append(matrix)
    return torch.stack(matrices)


def graph_input(frames):
    if frames.ndim != 3 or frames.shape[-1] != 254:
        raise ValueError('Expected [batch,time,254] prepared landmark features')
    n, t, _ = frames.shape
    body = frames[..., :26].reshape(n, t, 13, 2)[..., list(BODY_FEATURE_INDICES), :]
    hands = frames[..., 39:123].reshape(n, t, 2, 21, 2)[..., list(HAND), :]
    nodes = torch.cat((body, hands[:, :, 0], hands[:, :, 1]), dim=2)
    return nodes.permute(0, 3, 1, 2).contiguous()
