"""Small, unidirectional isolated-sign GRU and jointly trained linear head."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SignGRU(nn.Module):
    def __init__(self, input_size=254, hidden_size=256, layers=2, classes=2731, dropout=.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, frames, lengths):
        packed = pack_padded_sequence(frames, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.classifier(self.dropout(hidden[-1]))
