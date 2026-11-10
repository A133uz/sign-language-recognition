"""ST-GCN classifier: graph + temporal convolutions and valid-time pooling.

Reference: Yan et al., AAAI 2018, arXiv:1801.07455 and Microsoft's
ASL-citizen-code/ST-GCN. This is a project-specific implementation, not the
authors' pretrained model or an exact reproduction of their preprocessing.
"""

import torch
from torch import nn

from sign_language.stgcn_graph import adjacency, graph_input


class GraphTemporalBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1, residual=True):
        super().__init__()
        self.spatial = nn.Conv2d(incoming, outgoing*3, 1)
        self.temporal = nn.Sequential(nn.BatchNorm2d(outgoing), nn.ReLU(),
                                      nn.Conv2d(outgoing, outgoing, (9, 1), (stride, 1), (4, 0)),
                                      nn.BatchNorm2d(outgoing))
        self.residual = (nn.Identity() if incoming == outgoing and stride == 1 else
                         nn.Sequential(nn.Conv2d(incoming, outgoing, 1, (stride, 1)), nn.BatchNorm2d(outgoing))) if residual else None
        self.stride = stride

    def forward(self, x, matrix):
        residual = self.residual(x) if self.residual is not None else 0
        spatial = self.spatial(x)
        n, _, t, v = spatial.shape
        spatial = spatial.reshape(n, 3, -1, t, v)
        aggregated = torch.einsum('nkctv,kvw->nctw', spatial, matrix)
        return (self.temporal(aggregated)+residual).relu()


class SignSTGCN(nn.Module):
    def __init__(self, classes=2731, dropout=.5):
        super().__init__()
        self.register_buffer('adjacency', adjacency())
        self.input_bn = nn.BatchNorm1d(54)
        settings = [(2, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 1),
                    (64, 128, 2), (128, 128, 1), (128, 128, 1),
                    (128, 256, 2), (256, 256, 1), (256, 256, 1)]
        self.blocks = nn.ModuleList(GraphTemporalBlock(a, b, stride, residual=i != 0)
                                    for i, (a, b, stride) in enumerate(settings))
        self.edge_importance = nn.ParameterList(nn.Parameter(torch.ones(3, 27, 27)) for _ in settings)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, classes)

    @staticmethod
    def mask(x, lengths):
        valid = torch.arange(x.shape[2], device=x.device)[None] < lengths[:, None]
        return x.masked_fill(~valid[:, None, :, None], 0)

    def forward(self, frames, lengths):
        lengths = lengths.to(frames.device)
        x = graph_input(frames)
        n, c, t, v = x.shape
        x = self.mask(x, lengths)
        x = self.input_bn(x.permute(0, 3, 1, 2).reshape(n, v*c, t))
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        x = self.mask(x, lengths)
        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.adjacency*importance)
            lengths = (lengths+block.stride-1)//block.stride
            x = self.mask(x, lengths)
        pooled = x.sum((2, 3))/(lengths[:, None].clamp_min(1)*v)
        return self.classifier(self.dropout(pooled))
