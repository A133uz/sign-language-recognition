"""Copy into a new Colab cell after the 100-clip comparison; no inference."""

import json
import os
from pathlib import Path
import subprocess

TOP_CLIPS = 5
PREVIEW_CLIPS = 3
if not all(name in globals() for name in ('ENV_PYTHON', 'comparison_dir')):
    raise RuntimeError('Run in the notebook holding the completed comparison and its saved files.')
comparison_audit_dir = Path(comparison_dir)
if not (comparison_audit_dir / 'summary.json').is_file():
    raise FileNotFoundError(comparison_audit_dir / 'summary.json')
if not 1 <= PREVIEW_CLIPS <= TOP_CLIPS <= 20:
    raise ValueError('Choose 1 <= PREVIEW_CLIPS <= TOP_CLIPS <= 20.')

comparison_audit_program = r'''
import itertools, json, sys
from pathlib import Path
import numpy as np


def read_features(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def frame_audit(cpu, gpu, frame):
    left = np.flatnonzero(cpu['hand_present'][frame])
    right = np.flatnonzero(gpu['hand_present'][frame])
    distances, labels_agree = [], []
    if len(left) and len(right):
        choices = ([(left,p) for p in itertools.permutations(right, len(left))]
                   if len(left) <= len(right) else
                   [(p,right) for p in itertools.permutations(left, len(right))])
        match = min(choices, key=lambda pair: sum(
            np.linalg.norm(cpu['hands_xyz'][frame,a,:,:2] - gpu['hands_xyz'][frame,b,:,:2], axis=-1).mean()
            for a,b in zip(*pair)))
        for a,b in zip(*match):
            distances.extend(np.linalg.norm(cpu['hands_xyz'][frame,a,:,:2] - gpu['hands_xyz'][frame,b,:,:2], axis=-1))
            labels_agree.append(cpu['handedness'][frame,a] == gpu['handedness'][frame,b])
    label = {-1:'unknown', 0:'Left', 1:'Right'}
    return {
        'array_frame': frame, 'source_frame': int(cpu['frame_index'][frame]),
        'timestamp_ms': int(cpu['timestamp_ms'][frame]),
        'cpu_hands': len(left), 'gpu_hands': len(right),
        'cpu_reported_labels': [label.get(int(cpu['handedness'][frame,i]), 'unknown') for i in left],
        'gpu_reported_labels': [label.get(int(gpu['handedness'][frame,i]), 'unknown') for i in right],
        'matched_labels_disagree': bool(labels_agree and not all(labels_agree)),
        'xy_mean': float(np.mean(distances)) if distances else None,
        'xy_p95': float(np.percentile(distances,95)) if distances else None,
    }


def audit_clip(item, cpu, gpu):
    if not np.array_equal(cpu['frame_index'], gpu['frame_index']) or not np.array_equal(cpu['timestamp_ms'], gpu['timestamp_ms']):
        raise ValueError('CPU/GPU frame timing differs')
    details = [frame_audit(cpu,gpu,index) for index in range(len(cpu['frame_index']))]
    worst = max(details, key=lambda frame: frame['xy_p95'] if frame['xy_p95'] is not None else -1)
    return {
        'key': item['key'], 'video_file': item['video_file'], 'gloss': item['gloss'],
        'clip_hand_xy': item['comparison']['hand_xy_distance'],
        'frames': len(details),
        'frames_with_different_hand_counts': sum(f['cpu_hands'] != f['gpu_hands'] for f in details),
        'frames_cpu_only_hands': sum(f['cpu_hands'] > 0 and f['gpu_hands'] == 0 for f in details),
        'frames_gpu_only_hands': sum(f['gpu_hands'] > 0 and f['cpu_hands'] == 0 for f in details),
        'frames_both_no_hands': sum(f['cpu_hands'] == 0 and f['gpu_hands'] == 0 for f in details),
        'frames_with_disagreeing_matched_labels': sum(f['matched_labels_disagree'] for f in details),
        'frames_xy_p95_over_0_05': sum(f['xy_p95'] is not None and f['xy_p95'] > .05 for f in details),
        'frames_xy_p95_over_0_10': sum(f['xy_p95'] is not None and f['xy_p95'] > .10 for f in details),
        'worst_frame': worst,
    }


def preview(work, dataset, reports, limit):
    import cv2
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(min(limit,len(reports)), 2, figsize=(12,4*min(limit,len(reports))), squeeze=False)
    edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
             (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
    for row, item in enumerate(reports[:limit]):
        index = item['worst_frame']['array_frame']
        capture = cv2.VideoCapture(str(dataset/'videos'/item['video_file']))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, item['worst_frame']['source_frame'])
            ok, image = capture.read()
        finally:
            capture.release()
        for column, mode in enumerate(('cpu','gpu')):
            data = read_features(work/mode/(item['key']+'.npz'))
            axis = axes[row,column]
            if ok:
                height,width = image.shape[:2]
                axis.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), extent=(0,1,1,0), aspect=height/width)
            for slot in np.flatnonzero(data['hand_present'][index]):
                category = int(data['handedness'][index,slot])
                color = {0:'cyan',1:'orange'}.get(category,'magenta')
                points = data['hands_xyz'][index,slot]
                for a,b in edges:
                    axis.plot(points[[a,b],0], points[[a,b],1], color=color, linewidth=1.4)
                axis.scatter(points[:,0], points[:,1], s=7, color=color)
                axis.text(points[0,0],points[0,1],{0:'Left',1:'Right'}.get(category,'?'),
                          color='white',fontsize=8,bbox={'facecolor':'black','alpha':.6,'pad':1})
            axis.set(xlim=(0,1), ylim=(1,0), title=f'{mode.upper()}: {item["gloss"]}, source frame {item["worst_frame"]["source_frame"]}')
            axis.axis('off')
    figure.tight_layout()
    figure.savefig(work/'worst-hand-frames.png',dpi=110)
    plt.close(figure)


if __name__ == '__main__':
    work, count, image_count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    summary = json.loads((work/'summary.json').read_text())
    manifest = json.loads((work/'manifest.json').read_text())
    ranked = sorted(summary['pairs'], key=lambda p: p['comparison']['hand_xy_distance']['p95'] or 0, reverse=True)[:count]
    reports = [audit_clip(item, read_features(work/'cpu'/(item['key']+'.npz')),
                          read_features(work/'gpu'/(item['key']+'.npz'))) for item in ranked]
    (work/'worst-hand-audit.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
    print(json.dumps(reports,indent=2,allow_nan=False))
    if reports:
        preview(work,Path(manifest['dataset_root']),reports,image_count)
    print('Reported Left/Right labels are model outputs, not ground truth.')
    print('XY thresholds 0.05/0.10 are triage counts, not accuracy or acceptance limits.')
'''

comparison_audit_script = comparison_audit_dir / 'audit_saved_features.py'
comparison_audit_script.write_text(comparison_audit_program, encoding='utf-8')
print('Auditing saved comparison only; no MediaPipe inference, downloads or training.', flush=True)
comparison_audit_process = subprocess.run(
    [str(ENV_PYTHON), '-u', str(comparison_audit_script), str(comparison_audit_dir), str(TOP_CLIPS), str(PREVIEW_CLIPS)],
    env={**os.environ, 'MPLBACKEND':'Agg'}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding='utf-8', errors='replace', timeout=180,
)
print(comparison_audit_process.stdout)
if comparison_audit_process.returncode:
    raise RuntimeError('Saved-feature audit failed; inspect output above.')
comparison_hand_audit = json.loads((comparison_audit_dir/'worst-hand-audit.json').read_text())
from IPython.display import Image, display
if (comparison_audit_dir/'worst-hand-frames.png').exists():
    display(Image(filename=str(comparison_audit_dir/'worst-hand-frames.png')))
print('Keep participant previews private. Share the JSON output and describe which hand(s) each overlay follows.')
