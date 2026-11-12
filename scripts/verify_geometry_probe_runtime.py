"""Restore frozen geometry probe and reproduce previously recorded scores; no webcam."""
import json
from pathlib import Path
import sys

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))
from sign_language.geometry_probe_runtime import GeometryProbe
from sign_language.five_sign_pilot import sha256, write_json


def main():
    probe = GeometryProbe(ROOT / 'models/asl-five-class-geometry-only-v1/model.pkl')
    report = json.loads(probe.path.with_name('report.json').read_text())
    source = ROOT / 'docs/assets/gru-geometry-guided-2026-11-12.json'
    if sha256(source) != report['retrospective_live']['source_sha256']:
        raise ValueError('Retrospective source changed')
    trials = json.loads(source.read_text())['live']['trials']
    max_difference = 0.
    for trial, recorded in zip(trials, report['retrospective_live']['predictions'], strict=True):
        result = probe.score(trial['articulation'], trial['expected'])
        actual = {p['label']: p['score'] for p in result['top5']}
        ordered = np.asarray([actual[name] for name in probe.names])
        expected = np.asarray(recorded['scores'])
        np.testing.assert_allclose(ordered, expected, rtol=1.e-12, atol=1.e-12)
        max_difference = max(max_difference, float(np.abs(ordered - expected).max()))
    verification = {'status': 'verified', 'probe_sha256': probe.sha256, 'device': 'cpu',
                    'labels': probe.names, 'compared_rows': len(trials), 'max_score_difference': max_difference,
                    'webcam_opened': False, 'training_performed': False, 'test_split_read': False}
    write_json(ROOT / '.cache/five-class-geometry-runtime-20261112.json', verification)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
