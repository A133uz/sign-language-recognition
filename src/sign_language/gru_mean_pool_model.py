"""Parameter-matched GRU using a masked mean of the top-layer time outputs."""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sign_language.gru_model import SignGRU


def masked_temporal_mean(outputs, lengths):
    """Exclude padding, retain real no-hand/rest frames, and accumulate AMP safely."""
    if outputs.ndim != 3 or lengths.ndim != 1 or len(lengths) != len(outputs):
        raise ValueError('Expected [batch,time,features] and [batch] lengths')
    if torch.any(lengths < 1) or torch.any(lengths > outputs.shape[1]):
        raise ValueError('Lengths must identify nonempty real sequences')
    lengths = lengths.to(outputs.device)
    mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :] < lengths[:, None]
    dtype = torch.float32 if outputs.dtype in (torch.float16, torch.bfloat16) else outputs.dtype
    values = outputs.masked_fill(~mask[..., None], 0).to(dtype)
    return values.sum(dim=1) / lengths[:, None].to(dtype)


class SignMeanPoolGRU(SignGRU):
    """Same GRU/head parameters and initialization as SignGRU; only readout differs."""

    def forward(self, frames, lengths):
        packed = pack_padded_sequence(frames, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.gru(packed)
        outputs, actual_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        pooled = masked_temporal_mean(outputs, actual_lengths)
        return self.classifier(self.dropout(pooled))
