"""Copy the entire file into Colab AFTER the successful NVIDIA EGL retry."""

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import uuid

CLIPS = 10                  # Same complete training clips for both delegates.
CLIP_TIMEOUT_SECONDS = 300   # Includes interpreter and tracker initialization.
if not all(name in globals() for name in ('PROJECT', 'ENV_PYTHON', 'DATASET_ROOT', 'egl_retry_env')):
    raise RuntimeError('Run setup, GPU check and the successful EGL retry first.')
if not 2 <= CLIPS <= 100:
    raise ValueError('Choose 2-100 comparison clips.')

# A benchmark-only copy: never patch the source used by full extraction.
comparison_dir = Path('/content') / f'asl-delegate-compare-{uuid.uuid4().hex[:10]}'
comparison_dir.mkdir()
comparison_gpu_source = comparison_dir / 'gpu-src'
shutil.copytree(Path(PROJECT) / 'src', comparison_gpu_source,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
comparison_source_hashes = {}
for filename in ('hands.py', 'raw_landmarks.py'):
    path = comparison_gpu_source / 'sign_language' / filename
    original = path.read_text(encoding='utf-8')
    if original.count('delegate=mp.tasks.BaseOptions.Delegate.CPU') != 1:
        raise RuntimeError(f'Unexpected {filename}; stop rather than patching unknown source.')
    changed = original.replace('delegate=mp.tasks.BaseOptions.Delegate.CPU',
                               'delegate=mp.tasks.BaseOptions.Delegate.GPU')
    path.write_text(changed, encoding='utf-8')
    comparison_source_hashes[filename] = {
        'cpu': hashlib.sha256(original.encode()).hexdigest(),
        'gpu': hashlib.sha256(changed.encode()).hexdigest(),
    }

with (Path(DATASET_ROOT) / 'splits/train.csv').open(encoding='utf-8-sig', newline='') as stream:
    comparison_rows = list(csv.DictReader(stream))
comparison_rows = random.Random(20261107).sample(comparison_rows, CLIPS)
for row in comparison_rows:
    path = (Path(DATASET_ROOT) / 'videos' / row['Video file']).resolve()
    if not path.is_relative_to((Path(DATASET_ROOT) / 'videos').resolve()) or not path.is_file():
        raise RuntimeError(f'Missing or unsafe video path: {path}')

comparison_worker = r'''
import io, itertools, json, sys, time
from pathlib import Path
import numpy as np


def error_stats(values):
    values = np.asarray(values, dtype=float)
    return {'points': int(values.size), 'mean': float(values.mean()) if values.size else None,
            'p95': float(np.percentile(values, 95)) if values.size else None,
            'max': float(values.max()) if values.size else None}


def compare_arrays(cpu, gpu):
    for data in (cpu, gpu):
        if any(not np.isfinite(value).all() for value in data.values()):
            raise ValueError('Nonfinite feature values')
    frames = len(cpu['frame_index'])
    if not frames or not np.array_equal(cpu['frame_index'], gpu['frame_index']) or not np.array_equal(cpu['timestamp_ms'], gpu['timestamp_ms']):
        raise ValueError('CPU/GPU frame indices or timestamps differ')
    expected = {'hands_xyz': (frames,2,21,3), 'hand_present': (frames,2),
                'handedness': (frames,2), 'pose_xyz': (frames,33,3),
                'pose_valid_mask': (frames,33), 'pose_detected': (frames,)}
    for key, shape in expected.items():
        if cpu[key].shape != shape or gpu[key].shape != shape:
            raise ValueError('Unexpected array shape: ' + key)
    distances, handedness_equal = [], []
    for frame in range(frames):
        left = np.flatnonzero(cpu['hand_present'][frame])
        right = np.flatnonzero(gpu['hand_present'][frame])
        if not len(left) or not len(right):
            continue
        # Detection-order slots may swap. Match hands by minimum XY distance;
        # count missing/extra hands separately instead of comparing zero fills.
        candidates = ([(left, p) for p in itertools.permutations(right, len(left))]
                      if len(left) <= len(right) else
                      [(p, right) for p in itertools.permutations(left, len(right))])
        best = min(candidates, key=lambda pair: sum(
            np.linalg.norm(cpu['hands_xyz'][frame,a,:,:2] - gpu['hands_xyz'][frame,b,:,:2], axis=-1).mean()
            for a,b in zip(*pair)))
        for a,b in zip(*best):
            distances.extend(np.linalg.norm(cpu['hands_xyz'][frame,a,:,:2] - gpu['hands_xyz'][frame,b,:,:2], axis=-1))
            handedness_equal.append(cpu['handedness'][frame,a] == gpu['handedness'][frame,b])
    common = cpu['pose_valid_mask'] & gpu['pose_valid_mask'] & cpu['pose_detected'][:,None] & gpu['pose_detected'][:,None]
    pose_distances = np.linalg.norm(cpu['pose_xyz'][...,:2] - gpu['pose_xyz'][...,:2], axis=-1)[common]
    return {
        'frames': frames,
        'hand_count_agreement': float(np.mean(cpu['hand_present'].sum(axis=1) == gpu['hand_present'].sum(axis=1))),
        'pose_presence_agreement': float(np.mean(cpu['pose_detected'] == gpu['pose_detected'])),
        'pose_valid_mask_agreement': float(np.mean(cpu['pose_valid_mask'] == gpu['pose_valid_mask'])),
        'matched_handedness_agreement': float(np.mean(handedness_equal)) if handedness_equal else None,
        'hand_xy_distance': error_stats(distances), 'pose_xy_distance': error_stats(pose_distances),
    }


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_preview(work, pairs, dataset):
    import cv2
    import matplotlib.pyplot as plt
    # Show up to three clips with largest hand-coordinate differences.
    examples = sorted(pairs, key=lambda p: p['comparison']['hand_xy_distance']['mean'] or 0, reverse=True)[:3]
    figure, axes = plt.subplots(len(examples), 2, figsize=(10, 4 * len(examples)), squeeze=False)
    edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
             (9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
    for row, item in enumerate(examples):
        cpu = load_npz(work / 'cpu' / (item['key'] + '.npz'))
        gpu = load_npz(work / 'gpu' / (item['key'] + '.npz'))
        index = len(cpu['frame_index']) // 2
        capture = cv2.VideoCapture(str(dataset / 'videos' / item['video_file']))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(cpu['frame_index'][index]))
            ok, image = capture.read()
        finally:
            capture.release()
        for column, (name, data) in enumerate((('CPU', cpu), ('GPU', gpu))):
            axis = axes[row,column]
            if ok:
                height, width = image.shape[:2]
                axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), extent=(0,1,1,0), aspect=height/width)
            for slot in np.flatnonzero(data['hand_present'][index]):
                points = data['hands_xyz'][index,slot]
                for a,b in edges:
                    axis.plot(points[[a,b],0], points[[a,b],1], color='yellow', linewidth=1)
                axis.scatter(points[:,0], points[:,1], s=5, color='cyan')
            valid = data['pose_valid_mask'][index] & data['pose_detected'][index]
            axis.scatter(data['pose_xyz'][index,valid,0], data['pose_xyz'][index,valid,1], s=8, color='lime')
            axis.set(xlim=(0,1), ylim=(1,0), title=f'{name}: {item["gloss"]}, frame {index}')
            axis.axis('off')
    figure.tight_layout()
    figure.savefig(work / 'comparison-preview.png', dpi=110)
    plt.close(figure)


def summarize(work):
    manifest = json.loads((work / 'manifest.json').read_text())
    pairs, failed = [], []
    for item in manifest['clips']:
        results = {mode: json.loads((work / mode / (item['key'] + '.json')).read_text()) for mode in ('cpu','gpu')}
        if any(record['status'] != 'ok' for record in results.values()):
            failed.append({**item, 'results': results})
            continue
        try:
            comparison = compare_arrays(load_npz(work/'cpu'/(item['key']+'.npz')), load_npz(work/'gpu'/(item['key']+'.npz')))
        except Exception as error:
            failed.append({**item, 'error': str(error)})
            continue
        pairs.append({**item, 'comparison': comparison, 'results': results})
    summary = {'selected_clips': len(manifest['clips']), 'completed_pairs': len(pairs), 'failed_pairs': failed,
               'pairs': pairs, 'timing_scope': 'One worker each; tracker creation, full decode, inference and NPZ compression. Excludes Drive/shard writes and interpreter startup.'}
    for mode in ('cpu','gpu'):
        total = sum(p['results'][mode]['seconds'] for p in pairs)
        frames = sum(p['comparison']['frames'] for p in pairs)
        logs = [(work/mode/(item['key']+'.log')).read_text(errors='replace') for item in manifest['clips']]
        summary[mode] = {'paired_extraction_seconds': total, 'seconds_per_clip': total/len(pairs) if pairs else None,
                         'frames': frames, 'frames_per_second': frames/total if total else None,
                         'frames_with_hands': sum(p['results'][mode]['frames_with_hands'] for p in pairs),
                         'frames_with_pose': sum(p['results'][mode]['frames_with_pose'] for p in pairs),
                         'clips_logging_tensor_write_error': sum('Tensors are designed for single writes' in log for log in logs)}
    summary['cpu_over_gpu_speedup'] = summary['cpu']['paired_extraction_seconds']/summary['gpu']['paired_extraction_seconds'] if pairs else None
    total_frames = sum(p['comparison']['frames'] for p in pairs)
    summary['agreement'] = {
        key: sum(p['comparison'][key] * p['comparison']['frames'] for p in pairs)/total_frames if pairs else None
        for key in ('hand_count_agreement', 'pose_presence_agreement', 'pose_valid_mask_agreement')
    }
    for key in ('hand_xy_distance', 'pose_xy_distance'):
        observed = [p['comparison'][key] for p in pairs if p['comparison'][key]['points']]
        points = sum(item['points'] for item in observed)
        summary['agreement'][key] = {
            'matched_points': points,
            'point_weighted_mean': sum(item['mean']*item['points'] for item in observed)/points if points else None,
            'largest_clip_p95': max((item['p95'] for item in observed), default=None),
        }
    # Require renderer evidence too; the generic success marker alone is insufficient.
    paired_gpu_logs = [(work/'gpu'/(p['key']+'.log')).read_text(errors='replace') for p in pairs]
    gpu_logs = '\n'.join(paired_gpu_logs)
    summary['nvidia_renderer_observed'] = 'renderer: Tesla' in gpu_logs or 'renderer: NVIDIA' in gpu_logs
    summary['paired_clips_with_nvidia_renderer'] = sum('renderer: Tesla' in log or 'renderer: NVIDIA' in log for log in paired_gpu_logs)
    summary['software_renderer_observed'] = 'llvmpipe' in gpu_logs.lower()
    (work/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    if pairs:
        make_preview(work, pairs, Path(manifest['dataset_root']))
    print(json.dumps({k:v for k,v in summary.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summarize(Path(sys.argv[2]))
    else:
        source, video, models, output = map(Path, sys.argv[2:])
        sys.path.insert(0, str(source))
        from sign_language.raw_landmarks import extract_clip
        began = time.perf_counter()
        data, record = extract_clip(video, models/'hand_landmarker.task', models/'pose_landmarker_lite.task')
        record.update(status='ok', seconds=time.perf_counter()-began)
        output.with_suffix('.npz').write_bytes(data)
        output.with_suffix('.json').write_text(json.dumps(record, allow_nan=False))
'''

comparison_script = comparison_dir / 'worker.py'
comparison_script.write_text(comparison_worker, encoding='utf-8')
comparison_manifest = {
    'dataset_root': str(DATASET_ROOT), 'workers_per_delegate': 1,
    'source_hashes': comparison_source_hashes,
    'clips': [{'key': f'{index:03d}', 'video_file': row['Video file'], 'gloss': row['Gloss']}
              for index, row in enumerate(comparison_rows)],
}
(comparison_dir / 'manifest.json').write_text(json.dumps(comparison_manifest, indent=2))
# Give both modes the same driver environment; only their delegate differs.
comparison_env = {**egl_retry_env, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1',
                  'TF_CPP_MIN_LOG_LEVEL': '0', 'GLOG_minloglevel': '0'}
for mode in ('cpu', 'gpu'):
    (comparison_dir / mode).mkdir()
print('Comparison folder:', comparison_dir, flush=True)
print('One worker each. Both hands and pose; complete videos; no Drive writes or training.', flush=True)
for index, item in enumerate(comparison_manifest['clips']):
    order = ('cpu','gpu') if index % 2 == 0 else ('gpu','cpu')
    for mode in order:
        output = comparison_dir / mode / item['key']
        source = Path(PROJECT)/'src' if mode == 'cpu' else comparison_gpu_source
        command = [str(ENV_PYTHON), '-u', str(comparison_script), 'extract', str(source),
                   str(Path(DATASET_ROOT)/'videos'/item['video_file']), str(Path(PROJECT)/'models'), str(output)]
        with output.with_suffix('.log').open('w', encoding='utf-8') as log:
            try:
                process = subprocess.run(command, env=comparison_env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=CLIP_TIMEOUT_SECONDS)
                if process.returncode or not output.with_suffix('.json').exists():
                    output.with_suffix('.json').write_text(json.dumps({'status': 'failed', 'exit_code': process.returncode}))
            except subprocess.TimeoutExpired:
                output.with_suffix('.json').write_text(json.dumps({'status': 'failed', 'error': 'timeout'}))
        record = json.loads(output.with_suffix('.json').read_text())
        timing = f'{record["seconds"]:.2f}s, {record["frames"]} frames' if record['status'] == 'ok' else str(record)
        print(f'{index+1}/{CLIPS} {mode.upper()} {item["gloss"]}: {timing}', flush=True)

comparison_summary_process = subprocess.run(
    [str(ENV_PYTHON), '-u', str(comparison_script), 'summary', str(comparison_dir)],
    env=comparison_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding='utf-8', errors='replace', timeout=180,
)
print(comparison_summary_process.stdout)
if comparison_summary_process.returncode:
    raise RuntimeError(f'Comparison summary failed; inspect files under {comparison_dir}')
cpu_gpu_comparison = json.loads((comparison_dir / 'summary.json').read_text())
try:
    from IPython.display import Image, display
    if (comparison_dir / 'comparison-preview.png').exists():
        display(Image(filename=str(comparison_dir / 'comparison-preview.png')))
except ImportError:
    print('Preview saved:', comparison_dir / 'comparison-preview.png')
print('Keep participant previews private. XY distances measure CPU/GPU disagreement, not accuracy.')
print('Share the printed summary and note whether either skeleton looks wrong. Full extraction still uses CPU.')
