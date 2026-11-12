"""Same-capture geometry intervention; persist scalar explanations, never inputs."""
import numpy as np
import torch

from sign_language.gru_geometry_model import FEATURES
from sign_language.hand_articulation import summarize_articulation
from sign_language.preprocessing import CONFIG, normalize_sequence
from sign_language.recognition_diagnostics import hand_visibility_summary
from sign_language.recognition_forensics import scores


def compare_geometry_capture(control, candidate, frames, timestamps, expected, geometry_probe=None):
    if control.labels != candidate.labels or control.config['cache_id'] != candidate.config['cache_id']:
        raise ValueError('Comparison requires identical labels and training cache')
    if control.config.get('architecture') != 'gru-mean-pool-v1' or candidate.config.get('architecture') != 'gru-mean-pool-geometry-v1':
        raise ValueError('Expected mean-pool control and geometry candidate')
    target = control.labels[expected]  # Used for scoring only; never passed to inference.
    arrays, shape, metadata = control.observe_frames(frames, timestamps)
    features, quality = normalize_sequence(arrays, shape)
    if not quality['frames_with_hands'] or not quality['frames_with_body_anchor']:
        raise ValueError('Capture requires hands and a valid shoulder anchor')
    features = features.astype(np.float16).astype(np.float32)
    padded = np.zeros((1, CONFIG['max_frames'], 254), dtype=np.float32)
    padded[0, :len(features)] = features
    lengths = torch.tensor([len(features)])
    collected = []
    hook = candidate.model.classifier.register_forward_pre_hook(
        lambda module, args: collected.append(args[0].detach().float().cpu().numpy()))
    try:
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=control.device.type == 'cuda'):
                original = control.model(torch.from_numpy(padded).to(control.device), lengths).float()[0].cpu().numpy()
            batch = torch.from_numpy(padded).to(candidate.device)
            with torch.autocast('cuda', enabled=candidate.device.type == 'cuda'):
                full = candidate.model(batch, lengths).float()[0].cpu().numpy()
                off = candidate.model(batch, lengths, zero_geometry=True).float()[0].cpu().numpy()
    finally:
        hook.remove()
    if len(collected) != 2:
        raise AssertionError('Expected full and geometry-off classifier inputs')
    width = candidate.model.gru.hidden_size
    model = candidate.model
    weights = model.classifier.weight.detach().float().cpu().numpy()
    bias = model.classifier.bias.detach().float().cpu().numpy()
    no, hello = candidate.labels['NO'], candidate.labels['HELLO']
    delta = weights[hello] - weights[no]
    joined = collected[0][0]
    if joined.shape != (width + len(FEATURES),):
        raise ValueError('Geometry input schema differs')
    recurrent = float(np.dot(delta[:width], joined[:width]))
    geometry = float(np.dot(delta[width:], joined[width:]))
    bias_margin = float(bias[hello] - bias[no])
    total = float(full[hello] - full[no])
    disabled = float(off[hello] - off[no])
    articulation = summarize_articulation(features)
    per_feature = {name: {
        'measurement': articulation['metrics'].get(name),
        'standardized_value': float(joined[width + i]),
        'hello_minus_no_logit_contribution': float(delta[width + i] * joined[width + i]),
    } for i, name in enumerate(FEATURES)}
    result = {
        'expected': expected, **metadata, 'quality': quality,
        'variants': {'original': scores(original, control.names, target),
                     'mean_pool': scores(full, candidate.names, target),
                     'geometry_off': scores(off, candidate.names, target)},
        'variant_meanings': {'original': 'Separately trained mean-pool control',
                             'mean_pool': 'Geometry candidate with full inputs',
                             'geometry_off': 'Same candidate, standardized geometry set to zero'},
        'hello_minus_no_margin': {
            'positive_favors': 'HELLO', 'negative_favors': 'NO',
            'full_model': total, 'geometry_disabled_model': disabled,
            'actual_geometry_effect': total - disabled,
            'recurrent_component_fp32': recurrent, 'geometry_component_fp32': geometry,
            'classifier_bias_component_fp32': bias_margin,
            'amp_decomposition_residual': total - recurrent - geometry - bias_margin,
        },
        'geometry_measurements': per_feature,
        'tracking_summary': hand_visibility_summary(arrays['hand_present']),
        'articulation': articulation,
        'identical_observations_for_all_variants': True,
        'interpretation': 'Geometry-off is a diagnostic intervention, not a separately trained model; scalar margins explain decisions on this capture only.',
    }
    if geometry_probe is not None:
        if geometry_probe.names != candidate.names or geometry_probe.config['cache_id'] != candidate.config['cache_id']:
            raise ValueError('Probe and GRU must share labels and training cache')
        result['variants']['geometry_only'] = geometry_probe.score(articulation, expected)
        result['variant_meanings']['geometry_only'] = 'Frozen five-class geometry-only logistic regression; CPU'
    return result
