"""Mean-pool GRU with 17 explicit clip-level geometry inputs to its linear head."""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sign_language.gru_mean_pool_model import SignMeanPoolGRU, masked_temporal_mean
from sign_language.hand_articulation import summarize_articulation

FEATURES = (
    'thumb_index_xy_median', 'thumb_index_xy_range',
    'thumb_index_xyz_median', 'thumb_index_xyz_range',
    'thumb_middle_xy_median', 'thumb_middle_xy_range',
    'thumb_middle_xyz_median', 'thumb_middle_xyz_range',
    'index_tip_wrist_median', 'middle_tip_wrist_median',
    'index_pip_angle_median_deg', 'index_pip_angle_range_deg',
    'middle_pip_angle_median_deg', 'middle_pip_angle_range_deg',
    'palm_axis_xy_excursion_deg', 'palm_axis_xyz_excursion_deg',
    'wrist_body_xy_excursion',
)


def geometry_batch(frames, lengths):
    """Deterministic CPU summaries of real frames, including augmented inputs.

    Missing/insufficient tracking becomes NaN and is subsequently imputed using
    training statistics. No target, identity, time stamp or metadata is used.
    This fixed measurement branch is not differentiated; the GRU branch is.
    """
    values = frames.detach().float().cpu().numpy()
    counts = lengths.detach().cpu().numpy()
    if values.ndim != 3 or values.shape[2] != 254 or counts.shape != (len(values),):
        raise ValueError('Expected [batch,time,254] and [batch] lengths')
    if not np.isfinite(counts).all() or np.any(counts != counts.astype(int)):
        raise ValueError('Lengths must be finite integers')
    rows = []
    for sequence, count in zip(values, counts, strict=True):
        count = int(count)
        if count < 1 or count > len(sequence):
            raise ValueError('Invalid real sequence length')
        summary = summarize_articulation(sequence[:count])
        metrics = summary['metrics']
        rows.append([np.nan if metrics.get(name) is None else metrics[name] for name in FEATURES])
    return np.asarray(rows, dtype=np.float32)


def fit_geometry_statistics(training_geometry):
    """Fit once on official training clips only; validation is never an argument."""
    x = np.asarray(training_geometry, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not len(x) or np.isinf(x).any():
        raise ValueError('Invalid training geometry')
    median = np.asarray([np.median(column[np.isfinite(column)])
                         if np.isfinite(column).any() else 0. for column in x.T])
    imputed = np.where(np.isnan(x), median, x)
    mean = imputed.mean(0)
    scale = imputed.std(0)
    scale[scale < 1.e-8] = 1.
    return {key: value.tolist() for key, value in
            [('median', median), ('mean', mean), ('scale', scale)]}


class SignGeometryMeanPoolGRU(SignMeanPoolGRU):
    """Concatenate standardized geometry after pooling; 85 extra weights for 5 labels."""
    def __init__(self, input_size=254, hidden_size=256, layers=2, classes=5, dropout=.5):
        super().__init__(input_size, hidden_size, layers, classes, dropout)
        original = self.classifier
        # Preserve the control's recurrent/head initialization AND RNG position.
        with torch.random.fork_rng(devices=[]):
            self.classifier = nn.Linear(hidden_size + len(FEATURES), classes)
        with torch.no_grad():
            self.classifier.weight[:, :hidden_size].copy_(original.weight)
            self.classifier.weight[:, hidden_size:].zero_()
            self.classifier.bias.copy_(original.bias)
        self.register_buffer('geometry_median', torch.zeros(len(FEATURES)))
        self.register_buffer('geometry_mean', torch.zeros(len(FEATURES)))
        self.register_buffer('geometry_scale', torch.ones(len(FEATURES)))

    def set_geometry_statistics(self, statistics):
        for name in ('median', 'mean', 'scale'):
            values = torch.as_tensor(statistics[name], dtype=torch.float32)
            if values.shape != (len(FEATURES),) or not torch.isfinite(values).all():
                raise ValueError('Invalid geometry statistics')
            if name == 'scale' and torch.any(values <= 0):
                raise ValueError('Geometry scale must be positive')
            getattr(self, 'geometry_' + name).copy_(values)

    def standardize_geometry(self, geometry):
        if geometry.ndim != 2 or geometry.shape[1] != len(FEATURES) or torch.isinf(geometry).any():
            raise ValueError('Invalid geometry matrix')
        geometry = torch.where(torch.isnan(geometry), self.geometry_median, geometry)
        return (geometry - self.geometry_mean) / self.geometry_scale

    def forward(self, frames, lengths, geometry=None, zero_geometry=False):
        packed = pack_padded_sequence(frames, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.gru(packed)
        outputs, actual_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        pooled = self.dropout(masked_temporal_mean(outputs, actual_lengths))
        if zero_geometry:
            explicit = torch.zeros((len(frames), len(FEATURES)), device=frames.device)
        else:
            if geometry is None:
                geometry = geometry_batch(frames, lengths)
            raw = torch.as_tensor(geometry, dtype=torch.float32, device=frames.device)
            if raw.shape[0] != len(frames):
                raise ValueError('Geometry batch differs')
            explicit = self.standardize_geometry(raw)
        return self.classifier(torch.cat([pooled, explicit], dim=1))
