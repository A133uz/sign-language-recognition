"""Build two portable, executed reporting notebooks from existing artifacts only.

Uses the existing NumPy/Matplotlib environment; no kernel installation, model
inference, training, webcam capture or environment changes occur. Authored
analysis cells execute in a normal Python process and their real PNG/stdout
outputs are stored using the standard notebook v4 format.
"""

import argparse
import base64
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ('gru_baseline', 'GRU 256 baseline', 'models/asl-gru-128-baseline', None),
    ('gru_regularized', 'GRU 256 regularized', 'models/asl-gru-128-cv-regularized-v1/official-refit',
     'data/evaluations/asl-gru-regularized-test-v1'),
    ('gru64', 'GRU 64 baseline', 'models/asl-gru-64-baseline', None),
    ('stgcn', 'ST-GCN27', 'models/asl-stgcn27-v1', 'data/evaluations/asl-stgcn27-test-v1'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gather_snapshot():
    provenance = {}

    def read(relative):
        path = ROOT/relative
        if not path.exists():
            return None
        provenance[relative] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    models = {}
    for key, label, folder, evaluation in MODELS:
        path = ROOT/folder
        config = read(folder+'/config.json')
        if config is None:
            continue
        status = read(folder+'/status.json') or {}
        records = [read(p.relative_to(ROOT).as_posix()) for p in sorted(path.glob('epoch-*.json'))]
        records = [r for r in records if r and not r.get('smoke')]
        chosen = min(records, key=lambda r: r['val']['loss']) if records else None
        test = read(evaluation+'/report.json') if evaluation else None
        if test and test['status'] == 'complete':
            chosen = next(r for r in records if r['epoch'] == test['checkpoint_epoch'])
            assert abs(chosen['val']['loss']-test['validation_at_selection']['loss']) < 1e-9
            assert sha(path/'best.pt') == test['checkpoint_sha256']
        hidden = config['model'].get('hidden_size', 0)
        parameters = status.get('parameters') or (9*hidden**2+3505*hidden+2731 if hidden else None)
        item = {'label': label, 'folder': folder, 'status': status.get('status', 'unknown'),
                'config': config, 'history': records, 'selected': chosen,
                'parameters': parameters, 'test': test}
        if test and test['status'] == 'complete':
            per_class = ROOT/evaluation/'per-class.csv'
            if per_class.exists():
                provenance[per_class.relative_to(ROOT).as_posix()] = sha(per_class)
                with per_class.open(newline='',encoding='utf-8') as stream:
                    item['per_class'] = [{**row, 'support': int(row['support']),
                                          **{name: float(row[name]) for name in ('precision','recall','f1')}}
                                         for row in csv.DictReader(stream)]
            item['per_signer'] = read(evaluation+'/per-signer.json') or {}
            matrix_path = ROOT/evaluation/'confusion.npz'
            provenance[matrix_path.relative_to(ROOT).as_posix()] = sha(matrix_path)
            with np.load(matrix_path, allow_pickle=False) as archive:
                matrix, names = archive['counts'], archive['labels'].tolist()
            focus = ['HELLO','THANKYOU','NO','FROWN1','NONSENSE','THUNDER']
            focus = [name for name in focus if name in names]
            loc = [names.index(name) for name in focus]
            item['focused_confusion'] = {'labels': focus, 'counts': matrix[np.ix_(loc,loc)].tolist(),
                                         'full_row_support': matrix.sum(1)[loc].tolist()}
            np.fill_diagonal(matrix, 0)
            order = np.argpartition(matrix.ravel(), -12)[-12:]
            order = order[np.argsort(matrix.ravel()[order])[::-1]]
            item['top_confusions'] = [{'true': names[int(i)//len(names)], 'predicted': names[int(i)%len(names)],
                                       'count': int(matrix.ravel()[i])} for i in order if matrix.ravel()[i] > 0]
        # Absolute artifact paths and transient process IDs are not needed in a portable report.
        if item['test']:
            item['test'] = {k:v for k,v in item['test'].items() if k != 'checkpoint'}
        models[key] = item
    extraction = read('data/landmarks/asl-citizen-raw-v1-cpu-hands-pose/summary.json') or {}
    extraction = {k:v for k,v in extraction.items() if k not in
                  ('output','session_elapsed_seconds','generated_shards_this_session','newly_processed_clips','newly_decoded_frames')}
    snapshot = {'schema': 'sign-project-report-v1', 'generated_utc': datetime.now(timezone.utc).isoformat(),
                'models': models, 'extraction': extraction,
                'raw_schema': read('data/landmarks/asl-citizen-raw-v1-cpu-hands-pose/feature-schema.json'),
                'cache': read('data/training/asl-gru-128-v1/manifest.json'),
                'cv': read('models/asl-gru-128-cv-regularized-v1/summary.json'),
                'stgcn_engine_manifest': read('models/asl-stgcn27-v1/engine_snapshot/manifest.json'),
                'results_verification': read('.cache/stgcn-training/results-verification.json'),
                'investigation': read('docs/assets/recognition-investigation-2026-11-10.json'),
                'stgcn_batch_probe': read('.cache/stgcn-batch-benchmark.json'),
                'dataset': {'name': 'Microsoft ASL Citizen', 'classes': 2731, 'total_clips': 83399,
                            'splits': {'train':40154,'val':10304,'test':32941},
                            'signers': {'train':35,'val':6,'test':11}},
                'cpu_benchmark': {'source':'docs/local-cpu-benchmark-2026-11-07.md', 'clips':100,'frames':9257,
                                  'workers':[1,2,4], 'seconds':[314.28,161.35,103.89],
                                  'seconds_per_clip':[3.143,1.613,1.039],
                                  'extrapolated_hours':[72.8,37.4,24.1]},
                'environment': {'python':platform.python_version(), **{p:importlib.metadata.version(p)
                                 for p in ('numpy','matplotlib','mediapipe','opencv-contrib-python','torch')}}}
    # The label map is unnecessary for graphs; retain its identity without duplicating 2,731 labels.
    if snapshot['cache']:
        snapshot['cache'].pop('labels',None)
    try:
        snapshot['git_commit'] = subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',
                                                         'rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        snapshot['worktree_has_changes'] = bool(subprocess.check_output(
            ['git','-c',f'safe.directory={ROOT.as_posix()}','status','--porcelain'],cwd=ROOT,text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        snapshot['git_commit'] = None
    for path in ('docs/local-cpu-benchmark-2026-11-07.md','sign_language_project_summary.md',
                 'docs/recognition-investigation-2026-11-10.md','docs/stgcn-training.md',
                 'scripts/build_project_notebooks.py'):
        provenance[path] = sha(ROOT/path)
    snapshot['source_sha256'] = provenance
    latest = read('docs/assets/fifteen-sign-temporal-2026-11-12.json')
    if latest is not None:
        if latest.get('status') != 'complete':
            raise ValueError('Current pilot notebook requires completed paired results')
        snapshot['fifteen_sign_temporal'] = {
            'report': latest,
            'configs': {name: read(f'models/asl-gru-fifteen-signs-temporal-v1/{name}/config.json')
                        for name in ('control','temporal')},
            'runtime_verification': read('.cache/fifteen-sign-runtime-20261112.json'),
        }
    readout = read('docs/assets/please-trajectory-readout-study-2026-11-12.json')
    if readout is not None:
        runtime = read('docs/assets/please-trajectory-runtime-2026-11-12.json')
        if readout.get('status') != 'complete' or not readout.get('selected'):
            raise ValueError('Trajectory report requires a completed, qualified readout study')
        if not runtime or runtime.get('status') != 'complete' or not runtime.get('fresh_acceptance_met'):
            raise ValueError('Trajectory report requires completed fresh-video application checks')
        snapshot['trajectory_readout'] = {
            'study': readout, 'runtime': runtime,
            'config': read(str(Path(readout['selected']['checkpoint']).with_name('config.json'))),
            'ablation': read('docs/assets/please-trajectory-readout-ablation-2026-11-12.json'),
            'joint_attempt': read('docs/assets/please-trajectory-fix-2026-11-12.json'),
        }
    snapshot['stgcn_complete'] = bool(models.get('stgcn',{}).get('status') == 'completed' and
                                     (models.get('stgcn',{}).get('test') or {}).get('status') == 'complete')
    return snapshot


def require_final(snapshot):
    if not snapshot.get('stgcn_complete'):
        raise ValueError('Final notebooks require completed ST-GCN training and frozen test evaluation')
    metrics = snapshot['models']['stgcn']['test']['metrics']
    if metrics['samples'] != 32941:
        raise ValueError('ST-GCN test evaluation did not cover the complete official test split')


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                       *['| '+' | '.join(map(str,row))+' |' for row in rows]])


class Notebook:
    def __init__(self, name):
        self.name, self.cells = name, []

    def markdown(self, source):
        self.cells.append({'id':f'cell-{len(self.cells)+1:03}', 'cell_type':'markdown',
                           'metadata':{}, 'source':textwrap.dedent(source).strip().splitlines(keepends=True)})

    def code(self, source):
        self.cells.append({'id':f'cell-{len(self.cells)+1:03}', 'cell_type':'code', 'metadata':{},
                           'source':textwrap.dedent(source).strip().splitlines(keepends=True),
                           'execution_count':None,'outputs':[]})

    def execute(self, working_directory, figures):
        import os
        namespace = {'__name__':'__main__'}
        previous_cwd = Path.cwd()
        show = plt.show
        figures.mkdir(parents=True,exist_ok=True)
        count, image_count = 0, 0
        try:
            os.chdir(working_directory)
            for cell in self.cells:
                if cell['cell_type'] != 'code':
                    continue
                count += 1
                stream = io.StringIO()

                def flush():
                    value = stream.getvalue()
                    if value:
                        cell['outputs'].append({'output_type':'stream','name':'stdout','text':value})
                        stream.seek(0);stream.truncate(0)

                def capture_show(*args,**kwargs):
                    nonlocal image_count
                    flush()
                    for number in list(plt.get_fignums()):
                        fig = plt.figure(number)
                        buffer = io.BytesIO()
                        fig.savefig(buffer,format='png',dpi=160,bbox_inches='tight',facecolor='white')
                        payload = buffer.getvalue()
                        image_count += 1
                        (figures/f'{self.name}-{image_count:02}.png').write_bytes(payload)
                        cell['outputs'].append({'output_type':'display_data','metadata':{},
                                                'data':{'image/png':base64.b64encode(payload).decode('ascii'),
                                                        'text/plain':f'Figure {image_count}: {self.name}'}})
                        plt.close(fig)
                plt.show = capture_show
                with contextlib.redirect_stdout(stream):
                    exec(compile(''.join(cell['source']),f'{self.name}/{cell["id"]}','exec'),namespace)
                    if plt.get_fignums():
                        capture_show()
                flush()
                cell['execution_count'] = count
        finally:
            plt.show = show
            plt.close('all')
            os.chdir(previous_cwd)
        result = {'nbformat':4,'nbformat_minor':5,'metadata':{
                    'kernelspec':{'display_name':'Python (sign-language project)','language':'python','name':'python3'},
                    'language_info':{'name':'python','version':platform.python_version()},
                    'generated_by':'scripts/build_project_notebooks.py',
                    'execution_method':'Authored read-only cells executed by the Python report builder; not a Jupyter kernel session'},
                  'cells':self.cells}
        validate_notebook(result)
        return result


def validate_notebook(notebook):
    """Structural, source-compilation and actual embedded-image checks without installing Jupyter."""
    assert notebook['nbformat'] == 4 and notebook['nbformat_minor'] == 5
    seen = set()
    for cell in notebook['cells']:
        assert cell['id'] not in seen
        seen.add(cell['id'])
        assert isinstance(cell['source'],list) and isinstance(cell['metadata'],dict)
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']),'<notebook>','exec')
            assert isinstance(cell['execution_count'],int)
            for output in cell['outputs']:
                assert output['output_type'] in ('stream','display_data')
                if output['output_type'] == 'display_data':
                    assert base64.b64decode(output['data']['image/png']).startswith(b'\x89PNG\r\n\x1a\n')


SETUP = r'''
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

def find_snapshot():
    for folder in (Path.cwd(), *Path.cwd().parents):
        for relative in ('artifacts/experiment_snapshot.json', 'notebooks/artifacts/experiment_snapshot.json'):
            candidate = folder/relative
            if candidate.exists():
                return candidate
    raise FileNotFoundError('Keep the notebooks/artifacts folder beside these notebooks.')

SNAPSHOT_PATH = find_snapshot()
S = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
M = S['models']
COLORS = {'gru_baseline':'#2878b5','gru_regularized':'#219653','gru64':'#e49320','stgcn':'#8e44ad'}
plt.rcParams.update({'figure.figsize':(11,5),'font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
                     'figure.constrained_layout.use':True})
def finish(title):
    plt.gcf().suptitle(title,fontsize=14,fontweight='bold')
    plt.show()
print('Snapshot:',S['generated_utc'])
print('ST-GCN final evaluation complete:',S['stgcn_complete'])
print('Read-only report: no training, extraction, downloads, webcam or weight updates.')
'''


def make_workflow(snapshot):
    n = Notebook('01_project_workflow_and_investigation')
    pending = '' if snapshot['stgcn_complete'] else '**DRAFT SNAPSHOT: ST-GCN is still training; final refresh is scheduled after evaluation.**'
    n.markdown(f'''# Sign-language project: workflow, decisions and investigation

    {pending}

    This notebook records implemented work, experimental evidence, unsuccessful
    checks and remaining limitations. It is a read-only report: Run All reads
    the companion snapshot and redraws figures; it never starts training.
    Figures/tables are embedded, so the saved notebook is readable without videos,
    landmark archives, checkpoints or a running kernel. Rerunning needs Python,
    NumPy and Matplotlib (plus an IPython kernel in a notebook editor).

    Training/program code remains in ordinary Python modules. The notebook is a
    research record, not a second implementation. For detailed actions see the
    [investigation log](../docs/recognition-investigation-2026-11-10.md) and
    [project summary](../sign_language_project_summary.md).''')
    n.code(SETUP)
    if snapshot.get('fifteen_sign_temporal'):
        append_current_pilot(n, snapshot, include_results=False)
    n.markdown('''## 1. Scope and implementation history

    - Goal: one isolated ASL sign from a webcam clip → its text/gloss. Continuous
      signing and sentence generation are future work.
    - Earlier WLASL/SVM/I3D experiments were removed at the user's request.
      They are historical experiments, not active fallback models. Their
      unpreserved measurements are not invented or included as comparable scores.
    - ASL now uses Microsoft ASL Citizen. Slovo is selected for a separate RSL
      extension; its landmark/schema audit and RSL training are still pending.
    - Repository setup was simplified to one project environment using uv;
      the initially added CI pipeline was removed at the user's request.
    - Work progressed through dataset auditing, MediaPipe hand preview,
      CPU/GPU extraction diagnostics, resumable full extraction, normalization,
      GRU baselines/CV, held-out testing, webcam diagnosis and the ST-GCN comparison.

    Sources: [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/),
    [ASL Citizen code](https://github.com/microsoft/ASL-citizen-code),
    [Slovo](https://github.com/hukenovs/slovo),
    [decisions](../docs/project-decisions.md).''')
    n.markdown('''## 2. Dataset and split audit

    Keep official signer-disjoint splits; random clip-level 80/20 splitting can
    mix signers. Train selects parameters, validation selects the checkpoint,
    and frozen test evaluation measures performance on additional held-out signers.
    These are ASL classification results, not RSL or webcam accuracy.''')
    n.code('''
    d=S['dataset']
    print(d['name'], '|',d['total_clips'],'videos |',d['classes'],'labels')
    for split,count in d['splits'].items():
        print(f'{split:>5}: {count:>6,} clips; {d["signers"][split]} signers')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axs,('splits','signers'),('Clips by official split','Signers by official split')):
        values=list(d[key].values())
        bars=ax.bar(list(d[key]),values,color=['#2878b5','#e49320','#219653'])
        ax.bar_label(bars,labels=[f'{x:,}' for x in values],padding=3)
        ax.set_title(title);ax.set_ylim(0,max(values)*1.16)
    finish('ASL Citizen: official data partitions')
    ''')
    n.markdown('''## 3. MediaPipe extraction and recovery

    MediaPipe's pretrained hand/pose models detect landmarks; they do not classify
    the word. Local extraction used CPU hand and pose-lite detectors and retained
    decoded frames, timestamps, handedness and missing/confidence masks in
    checksummed NPZ/ZIP shards. Completed shards survive worker-count changes.
    A hidden supervisor, duplicate-writer lock, atomic shard markers and bounded
    recovery were added after worker-pool crashes. The final selected count was
    two workers. Crash mitigation was not claimed as a proven native root-cause fix.

    Colab multiprocessing/GPU/EGL diagnostics were also investigated. GPU inference
    became possible, but tensor-write errors and coordinate disagreements prevented
    that extraction path from being adopted. CUDA GRU/ST-GCN training is separate
    from the MediaPipe GPU-delegate experiment.

    The local benchmark used the same 100 clips/9,257 frames for every setting.
    Its full-corpus hours are **extrapolations**, not measured completion durations.
    The extractor's last-session elapsed time is not a full-corpus wall time.
    [Extraction guide](../docs/landmark-extraction.md),
    [local benchmark](../docs/local-cpu-benchmark-2026-11-07.md),
    [job recovery/controls](../docs/local-extraction.md).''')
    n.code('''
    e=S['extraction'];b=S['cpu_benchmark']
    print('Extraction status:',e.get('status'))
    print('Successful clips:',e.get('successful_clips'),'| failures:',e.get('failed_clips'))
    print('Shards:',e.get('completed_shards'),'| archive GiB:',round(e.get('feature_archive_bytes',0)/2**30,3))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].bar([str(x) for x in b['workers']],b['seconds_per_clip'],color='#2878b5')
    axs[0].set(xlabel='CPU workers',ylabel='Seconds / clip',title='Measured 100-clip benchmark')
    total=e.get('decoded_frames',1)
    values=[100*e.get('frames_with_hands',0)/total,100*e.get('frames_with_pose',0)/total]
    bars=axs[1].bar(['At least one hand','Pose'],values,color=['#e49320','#219653'])
    axs[1].bar_label(bars,labels=[f'{x:.2f}%' for x in values],padding=3)
    axs[1].set(ylabel='Frames with detections (%)',ylim=(0,112),title='Full raw-cache detection coverage')
    finish('Extraction speed and coverage — neither is sign-recognition accuracy')
    ''')
    n.markdown('''## 4. Normalization and sequence construction

    Raw detector hand slots are not stable anatomical identities. Preprocessing
    uses pose wrists, handedness scores and short-term continuity to assign them.
    XY coordinates are aspect-corrected, centered at shoulders and divided by a
    clip-level median shoulder width. Hand-shape XYZ is additionally wrist-relative
    and palm-normalized; hand and pose depth are not fused into metric 3D.
    Missingness is explicit; only short internal anchor gaps are interpolated.

    Long signs are sampled across the **whole** clip to 128 ordered frames;
    shorter signs retain true lengths with right padding. The GRU uses packed
    sequences. ST-GCN masks temporal padding at every block and during pooling.

    | GRU feature group | Dimensions |
    | --- | ---: |
    | 13 pose joints × XY | 26 |
    | Pose-valid masks | 13 |
    | Two hands × 21 joints × body-relative XY | 84 |
    | Two hands × 21 joints × local XYZ | 126 |
    | Hand present / local-valid / body-anchor flags | 2 + 2 + 1 |
    | Total | **254** |

    ST-GCN selects seven pose points and ten points per hand from the existing
    body-relative XY cache. Its graph input is `[batch,2,128,27]`; no new landmark
    extraction was needed. The different node/channel representation means this
    is a **model-plus-input comparison**, not an isolated layer replacement.
    [Preprocessor](../src/sign_language/preprocessing.py),
    [graph adapter](../src/sign_language/stgcn_graph.py).''')
    n.code('''
    assert sum([26,13,84,126,2,2,1])==254
    print('Prepared-cache identity:',S['cache']['cache_id'])
    print('Preprocessing:',S['cache']['identity']['preprocessing']['version'])
    for key,m in M.items():
        print(m['label'],':',m['config']['model'],'|',m['parameters'],'parameters')
    ''')
    n.markdown('''## 5. Model experiments and evidence boundaries

    1. Two-layer 256-unit GRU baseline; checkpoint selected by validation loss.
    2. Three-fold signer-grouped development CV within the official train split,
       then a fresh regularized official-split refit. Dropout, weight decay and
       geometric augmentation changed together; their separate effects were not isolated.
    3. A 64-unit comparison using original baseline settings; its lower training
       and validation scores do not establish better generalization.
    4. Frozen official test evaluation of the regularized GRU.
    5. A separately trained ST-GCN27 with graph/temporal convolutions and pooling.

    Earlier ordinary-PyTorch cuDNN GRU shutdown crashes were isolated; the working
    GRU uses ATen CUDA with cuDNN disabled. ST-GCN convolutional cuDNN passed a
    separate forward/backward/save/reload/exit smoke test. Both train on the GPU.
    All weights/checkpoints remain in separate experiment folders.
    The second notebook plots these experiments from their saved epoch JSONs.

    One training update follows this sequence (illustrative summary, not executed):

    ```python
    optimizer.zero_grad(set_to_none=True)
    logits = model(prepared_sequences, true_lengths)
    loss = cross_entropy(logits, target_class_ids)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    ```

    AMP overflow handling skips unsafe updates. Validation disables gradients,
    dropout and augmentation. A scheduler responds to validation loss. Atomic
    `last.pt` stores optimizer/scaler/scheduler/RNG state for resume; `best.pt`
    stores the selected model, labels and configuration. Test evaluation uses
    `eval()` plus inference mode and verifies that the checkpoint is unchanged.

    Implementations: [GRU](../src/sign_language/gru_model.py),
    [GRU trainer](../src/sign_language/training.py),
    [ST-GCN](../src/sign_language/stgcn_model.py),
    [ST-GCN trainer](../src/sign_language/stgcn_training.py),
    [frozen evaluation](../src/sign_language/evaluation.py).
    Commands are documented in the linked guides; they are not executed by Run All.''')
    if snapshot['stgcn_complete']:
        current=snapshot['models']['stgcn'];baseline=snapshot['models']['gru_regularized']
        metrics=current['test']['metrics'];gain=(metrics['top1']-baseline['test']['metrics']['top1'])*100
        n.markdown(f'''### Completed ST-GCN result

        Training stopped after **{len(current['history'])} epochs**. The selected
        checkpoint is **epoch {current['selected']['epoch']}**, using minimum
        validation cross-entropy. On all **{metrics['samples']:,} test clips** it
        achieved **{metrics['top1']*100:.2f}% top-1**, **{metrics['top5']*100:.2f}%
        top-5**, and **{metrics['macro_f1']*100:.2f}% macro F1**. Top-1 improved
        by **{gain:.2f} percentage points** over the regularized GRU overall.
        Some individual signs regressed; notebook 02 includes paired class
        comparisons. ST-GCN webcam performance has not been evaluated.''')
    n.markdown('''## 6. GRU webcam failure investigation

    These live checks used the regularized GRU; they are not ST-GCN webcam tests.
    The user reported THANKYOU→FROWN1, HELLO→NONSENSE and NO→THUNDER. A subsequent
    user-confirmed six-trial check tested the same captures under predefined
    timing windows and actual image reflection. No videos, landmark arrays or
    feature vectors were saved. Every variant had zero top-one and top-five
    matches on those six trials; this is one signer's diagnostic, not a general
    webcam-accuracy estimate.

    Checks found matching model/label/cache/detector identities and six exact
    fresh-video reproductions. Ordinary AMP/batch differences were negligible.
    The strongest supported weakness was hand-visibility/rest-context sensitivity;
    detector failures and smaller hand images contributed additional evidence.
    Their individual causal contributions were not uniquely separated.''')
    n.code('''
    investigation=S['investigation']
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    quarters=investigation['training_mean_hand_presence_by_quarter']
    for name,values in quarters.items():axs[0].plot(range(1,5),np.array(values)*100,'o-',label=name)
    live=investigation['live_input_comparisons']
    for row in live:
        if row['expected']=='HELLO':axs[0].plot(range(1,5),np.array(row['hand_presence_by_quarter'])*100,'--',alpha=.6,color='#8e44ad')
    axs[0].plot([],[],'--',color='#8e44ad',label='Two live HELLO trials')
    axs[0].set(xlabel='Quarter of isolated-sign clip',ylabel='Frames with hands (%)',xticks=[1,2,3,4],ylim=(0,105))
    axs[0].legend(fontsize=8)
    modes=['original','fill_tail_hand_context','fill_head_hand_context','fill_both_hand_context']
    values=[investigation['validation_sample_perturbations'][x]['top1']*100 for x in modes]
    bars=axs[1].bar(['Original','Fill trailing','Fill leading','Fill both'],values,color=['#219653','#e49320','#2878b5','#c0392b'])
    axs[1].bar_label(bars,fmt='%.1f%%',padding=3)
    axs[1].set(ylabel='Top-1 vs original labels (%)',ylim=(0,75),title='512 validation clips: synthetic context probe')
    finish('Detected-hand context differs; counterfactual probes show sensitivity')
    ''')
    n.code('''
    rows=investigation['live_input_comparisons']
    names=[f"{i+1} {r['expected']}" for i,r in enumerate(rows)]
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    axs[0].bar(names,[r['hand_frame_fraction']*100 for r in rows],color='#2878b5')
    axs[0].set(ylabel='Frames with hand detections (%)',ylim=(0,105),title='Original six prompted captures')
    axs[0].tick_params(axis='x',rotation=40)
    live_sizes=[r['palm_size_pixels_median'] or 0 for r in rows]
    training=[investigation['dataset_training_summaries'][r['expected']]['palm_size_pixels_median']['median'] for r in rows]
    x=np.arange(len(rows))
    axs[1].bar(x-.2,live_sizes,.4,label='Live capture',color='#8e44ad')
    axs[1].bar(x+.2,training,.4,label='Median training clip',color='#219653')
    axs[1].set(xticks=x,xticklabels=names,ylabel='Palm-size proxy (pixels)',title='Zero means no hand detection, not a zero-size hand')
    axs[1].tick_params(axis='x',rotation=40);axs[1].legend(fontsize=8)
    finish('Tracking and hand-image detail: measured input differences')
    ''')
    n.markdown('''The context probe extends detected-hand feature states into previously missing
    endpoint frames. It is synthetic, changes coordinates as well as masks and
    is **not** an approved detector correction or universally label-preserving
    augmentation. The smaller-image video probe also introduced black borders.
    Mirroring did not solve live recognition. ST-GCN has not yet demonstrated
    a fix for any of these live failures.

    ## 7. Reproducibility, limits and next work

    - The companion JSON holds the actual reports/histories behind these graphs.
      Its source hashes identify the snapshots used; live source files may evolve.
    - Checkpoints, raw videos and raw landmarks remain ignored by Git. Embedded
      report outputs are sufficient to read the notebooks without those assets.
    - No RSL training/evaluation, continuous signing, automatic sign boundaries,
      calibrated confidence or validated unknown-sign rejection is claimed.
    - Future controlled work can address input context, camera detail, detection
      quality and generalization, preserving the existing models as baselines.
    - See [ST-GCN training controls](../docs/stgcn-training.md),
      [GRU controls](../docs/local-training.md),
      [test evaluation](../docs/test-evaluation-2026-11-10.md), and
      [complete investigation](../docs/recognition-investigation-2026-11-10.md).

    Dataset terms and upstream licenses continue to apply. No personal webcam
    media is included in this notebook.''')
    return n


def append_trajectory_readout(n, snapshot, include_results=True):
    """Latest targeted development result, including the unsuccessful controls."""
    current=snapshot['trajectory_readout']
    selected=current['study']['selected']
    n.markdown('''## Latest development model: frozen GRU + wrist-path readout

    The four known validation PLEASE clips now rank PLEASE first, while MY stays
    4/4. **These cases informed feature design and validation tuning. They are
    development results, not independent evidence of webcam/generalization
    accuracy.** The 219/60 official train/validation subset and all 15 labels
    are preserved. No new official test data was read or webcam opened.

    Keep the temporal baseline's epoch-13 recurrent weights fixed. Add six
    label-independent wrist-path descriptors to the original 17 summaries:
    loop strength, turn coherence, absolute turning, axis ratio, endpoint/path
    ratio and path length. Windows of 12/24/36 cached steps never bridge absent
    tracking. Only the readout is refitted on original training examples, using
    L2-regularized multinomial cross-entropy (C=0.1, lbfgs). No further GRU epochs
    run for this candidate. Epoch 13 identifies the inherited backbone.

    New input to the head: 256 pooled + 23 standardized summaries = 279 values.
    Linear(279,15) has 4,200 parameters; total neural parameters are 792,168.
    Training-only medians/means/scales are frozen. Runtime is Torch, including
    a float32 head; sklearn is only the fitting tool. MediaPipe stays frozen.
    Source videos/caches and previous models/defaults are unchanged.''')
    n.code(r'''
    TR = S['trajectory_readout']
    TS = TR['study']['selected']
    print('Checkpoint:', TS['checkpoint'])
    print('SHA-256:', TS['checkpoint_sha256'])
    print('C:', TS['C'], '| frozen backbone epoch:', TR['config']['backbone_epoch'])
    print('Application verification: all 60 cached score vectors match; max difference:',
          TR['runtime']['cached_max_score_difference'])
    print('Fresh original-video PLEASE/MY:', TR['runtime']['fresh_per_class'])
    print('Features:', ', '.join(TR['config']['geometry_features']))
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set(xlim=(0,12),ylim=(0,10)); ax.axis('off')
    def tr_box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=color,edgecolor='#475569'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    tr_box(3.3,8.8,5.4,.8,'Frozen MediaPipe Hands + Pose\nSame normalized [B,T<=128,254]', '#e2e8f0')
    tr_box(.3,6.6,5.2,1.1,'FROZEN GRU, 2 layers x 256\nInherited temporal-model epoch 13', '#dbeafe')
    tr_box(6.5,6.6,5.2,1.1,'FIXED 17 hand/body + 6 path features\nNo label, filename or prompt input', '#ccfbf1')
    tr_box(.3,4.8,5.2,.9,'Masked mean -> [B,256]\nNo padded frames in pooling', '#dbeafe')
    tr_box(6.5,4.8,5.2,.9,'Training-only imputation + z-score\nFrozen statistics -> [B,23]', '#ccfbf1')
    tr_box(3.3,3.3,5.4,.8,'Concatenate -> [B,279]', '#fef3c7')
    tr_box(3.3,2.0,5.4,.8,'REFITTED float32 Linear(279,15)', '#fef3c7')
    tr_box(3.3,.4,5.4,1.1,'15 logits -> gloss\nClosed-set isolated sign; webcam not validated', '#ede9fe')
    for start,end in [((6,8.8),(2.9,7.7)),((6,8.8),(9.1,7.7)),
                      ((2.9,6.6),(2.9,5.7)),((9.1,6.6),(9.1,5.7)),
                      ((2.9,4.8),(6,4.1)),((9.1,4.8),(6,4.1)),
                      ((6,3.3),(6,2.8)),((6,2.0),(6,1.5))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#475569','lw':1.5})
    finish('Latest development architecture: frozen backbone, refitted trajectory-aware head')
    ''')
    rows=[(r['participant_id'],r['video_file'],r['top5'][0]['label'],f"{r['top5'][0]['score']:.2%}")
          for r in current['runtime']['fresh_videos'] if r['expected']=='PLEASE']
    n.markdown('### Fresh original-video application results\n\n'+
               md_table(['Participant','Known PLEASE video','First guess','Uncalibrated score'],rows)+
               '\n\nMY also remains 4/4. Fresh extraction checks application consistency on the same known recordings; it is not a new-signers test.')
    if include_results:
        n.code(r'''
        print('Development grid; declared after the initial C=1 failure:')
        for item in TR['study']['candidates']:
            v=item['validation']
            print(f'C={item["C"]:g}: {v["correct"]}/60; PLEASE {v["per_class"]["PLEASE"]["correct"]}/4; '
                  f'MY {v["per_class"]["MY"]["correct"]}/4; CE={v["loss"]:.4f}; accepted={item["acceptance_met"]}')
        print('Selection: lowest validation CE among candidates satisfying PLEASE 4/4, MY 4/4, overall >=50/60.')
        print('Selected:', TS['validation']['correct'], '/60; macro F1:', TS['validation']['macro_f1'])
        print('Matched fixed-C readout ablation:')
        for dimensions,result in TR['ablation']['results'].items():
            print(dimensions,'descriptors:',result['correct'],'/60;',
                  {g:result['per_class'][g] for g in ('PLEASE','MY')})
        labels_tr=TR['runtime']['labels']; positions_tr=np.arange(len(labels_tr))
        fig,ax=plt.subplots(figsize=(14,5))
        for offset,key,color in [(-.19,'before','#64748b'),(.19,'after','#0f766e')]:
            changes=TR['runtime']['class_changes']
            bars=ax.bar(positions_tr+offset,[100*changes[g][key]/changes[g]['support'] for g in labels_tr],
                        .36,label=key,color=color)
            for bar,g in zip(bars,labels_tr):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                        f'{changes[g][key]}/{changes[g]["support"]}',ha='center',fontsize=7)
        ax.set(xticks=positions_tr,xticklabels=labels_tr,ylim=(0,120),ylabel='Development validation recall (%)')
        ax.tick_params(axis='x',rotation=35,labelsize=9);ax.legend(ncols=2,fontsize=8)
        finish('Original temporal head vs selected trajectory readout: retain the GOOD regression')
        ''')
    n.markdown('''PLEASE improves 0/4 -> 4/4; ME improves 3/5 -> 4/5; GOOD regresses
    4/4 -> 3/4. Overall is 54/60 (90%), macro F1 0.885185, CE 0.386118.
    THANKYOU remains 1/3; failures are not hidden. The failed joint-training and
    scalar controls remain recorded, and no successful joint epoch met the
    acceptance criteria. Validation tuning and very small per-class support
    limit conclusions. A controlled webcam check and later frozen test evaluation
    are still needed. The application default was not changed.

    [Full protocol, attempts and checks](../docs/please-trajectory-fix-2026-11-12.md).
    Earlier 17-feature pilot training and full-vocabulary history follow.''')


def append_current_pilot(n, snapshot, include_results=True):
    """Current architecture and frozen development results; keep history intact."""
    pilot = snapshot['fifteen_sign_temporal']
    result = pilot['report']
    if snapshot.get('trajectory_readout'):
        append_trajectory_readout(n,snapshot,include_results)
    n.markdown('''## 15-class baseline: pooled GRU + hand geometry

    **This is the baseline for the subsequent trajectory-aware readout.**
    MediaPipe Hands/Pose are frozen detectors. The sequence model
    and linear classifier are trained jointly; the geometry calculations are
    fixed measurements, not a second neural network. RSL remains planned.

    **Training:** original cached normalized frames → training-only temporal
    sampling (candidate only) → spatial perturbation → two parallel branches:
    packed GRU/mean pooling and recalculated clip geometry → linear head →
    cross-entropy → backward propagation/AdamW. Geometry measurement uses CPU
    NumPy without differentiation; gradients update the GRU and head on CUDA.

    **Webcam:** OpenCV frames → one MediaPipe Hands/Pose pass → the same
    normalization/masks/cache precision → clean parallel branches → 15 logits
    → label-to-text. No augmentation or training occurs at inference. Softmax
    scores are uncalibrated; idle/unknown/continuous-sign rejection is not solved.

    Cached per-frame coordinates exist before augmentation. The **17 final
    clip summaries are recalculated afterward**, so both branches describe the
    same actual sequence. Frozen imputation/standardization is fitted only on
    original training clips. Padding is excluded by real lengths, while real
    frames with absent hands remain part of the sequence.''')
    n.code(r'''
    P15 = S['fifteen_sign_temporal']
    R15 = P15['report']
    C15 = P15['configs']['temporal']
    print('Classes:', ', '.join(R15['specification']['classes']))
    print('Train / validation clips:', *[sum(R15['specification']['counts'][s].values()) for s in ('train','val')])
    print('Train / validation signers:', *[len(R15['specification']['participants'][s]) for s in ('train','val')])
    print('Model:', C15['model'])
    print('Parameters:', R15['temporal']['parameters'])
    print('Input: [batch, <=128, 254]; head: [batch, 273] -> [batch, 15]')
    print('Temporal augmentation:', C15['temporal_augmentation'])
    print('Geometry features (fixed order):')
    for i, name in enumerate(C15['geometry_features'], 1):
        print(f'  {i:2d}. {name}')
    print('Runtime verification:', P15.get('runtime_verification'))
    ''')
    n.code(r'''
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12,9))
    ax.set(xlim=(0,12), ylim=(0,10)); ax.axis('off')
    def box15(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',
                     facecolor=color,edgecolor='#475569',linewidth=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow15(start,end):
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#475569','lw':1.6})
    box15(3.3,8.8,5.4,.8,'Normalized Hands + Pose sequence\n[B, T<=128, 254]', '#e2e8f0')
    box15(.3,6.6,5.2,1.1,'Packed GRU: 2 layers, 256 units\nReal lengths exclude padding', '#dbeafe')
    box15(6.5,6.6,5.2,1.1,'17 clip geometry measurements\nDistances, angles, motion ranges', '#ccfbf1')
    box15(.3,4.8,5.2,.9,'Masked temporal mean -> [B,256]\nDropout 0.5 during training', '#dbeafe')
    box15(6.5,4.8,5.2,.9,'Train-median imputation + z-score\nFrozen statistics -> [B,17]', '#ccfbf1')
    box15(3.3,3.3,5.4,.8,'Concatenate -> [B,273]', '#fef3c7')
    box15(3.3,2.0,5.4,.8,'Trainable Linear(273,15) -> logits', '#fef3c7')
    box15(3.3,.5,5.4,1.,'Train: CrossEntropyLoss\nInference: label ID -> displayed gloss', '#ede9fe')
    for start,end in [((6,8.8),(2.9,7.7)),((6,8.8),(9.1,7.7)),
                      ((2.9,6.6),(2.9,5.7)),((9.1,6.6),(9.1,5.7)),
                      ((2.9,4.8),(6,4.1)),((9.1,4.8),(6,4.1)),
                      ((6,3.3),(6,2.8)),((6,2.0),(6,1.5))]:
        arrow15(start,end)
    finish('Current architecture: pooled GRU + explicit geometry, 15 classes')
    ''')
    n.markdown('''The same model is used for the matched 15-class control. Its
    training loader omits temporal augmentation but retains the existing spatial
    perturbations. Temporal sampling uses ordered original frames, modest speed
    variation, and optional changes to existing hand-absent edge spans. These
    spans are tracking proxies, not ground-truth sign boundaries. No synthetic
    poses, fractional masks, reversal or frame-order shuffling are introduced.
    Geometry is fixed feature engineering; it summarizes motion as well as shape.

    Architecture tensors: 254 input channels → two 256-unit recurrent layers →
    a 256-value mean representation; concatenate 17 summaries → 273 inputs to
    15 outputs. The head has 4,110 parameters, included in the total 792,078.
    MediaPipe detector weights are separate and are not counted as trainable.
    Sources: [protocol/results](../docs/fifteen-sign-temporal-2026-11-12.md),
    `src/sign_language/gru_geometry_model.py`, `temporal_augmentation.py` and
    `scripts/run_fifteen_sign_temporal_trial.py`.''')
    if not include_results:
        n.markdown('Historical workflow/investigation records follow. Current paired learning curves and per-class results are in notebook 02.')
        return
    n.code(r'''
    for key in ('control','temporal'):
        run=R15[key]; v=run['validation']; t=run['train']
        print(f'{key}: epoch {run["selection"]["selected_epoch"]}; train {t["correct"]}/{t["samples"]}; '
              f'val {v["correct"]}/{v["samples"]}; macro F1 {v["macro_f1"]:.4f}; '
              f'CE {v["loss"]:.4f}; top5 {v["top5"]:.2%}')
    print('Paired validation changes:', R15['paired_validation_changes'])
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for key,color in [('control','#2563eb'),('temporal','#b45309')]:
        run=R15[key]; rows=run['epoch_metrics']; epochs=[r['epoch'] for r in rows]
        for split,style in [('train_clean','--'),('val','-')]:
            axs[0].plot(epochs,[r[split]['top1']*100 for r in rows],style,color=color,label=key+' '+split)
        axs[0].axvline(run['selection']['selected_epoch'],color=color,alpha=.3)
        axs[1].plot(epochs,[r['val']['loss'] for r in rows],color=color,label=key)
    axs[0].set(xlabel='Epoch',ylabel='Top-1 (%)',ylim=(0,105)); axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Epoch',ylabel='Validation cross-entropy'); axs[1].legend(fontsize=8)
    finish('15-class matched training: selected epochs, not the last epochs')
    ''')
    n.code(r'''
    labels15=R15['specification']['classes']; positions15=np.arange(len(labels15))
    fig,ax=plt.subplots(figsize=(14,5))
    for offset,key,color in [(-.19,'control','#2563eb'),(.19,'temporal','#b45309')]:
        metrics15=R15[key]['validation']
        bars15=ax.bar(positions15+offset,[metrics15['per_class'][g]['recall']*100 for g in labels15],.36,label=key,color=color)
        for bar,g in zip(bars15,labels15):
            item15=metrics15['per_class'][g]
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{item15["correct"]}/{item15["support"]}',ha='center',fontsize=7)
    ax.set(xticks=positions15,xticklabels=labels15,ylim=(0,119),ylabel='Validation recall (%)')
    ax.tick_params(axis='x',rotation=35,labelsize=9); ax.legend(loc='upper left',ncols=2,fontsize=8)
    finish('15-class validation: small per-class denominators and regressions remain visible')
    ''')
    n.code(r'''
    fig,axs=plt.subplots(1,2,figsize=(16,8))
    for ax,key in zip(axs,('control','temporal')):
        cm15=np.asarray(R15[key]['validation']['confusion'])
        ax.imshow(cm15/cm15.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
        for (i,j),value in np.ndenumerate(cm15):
            if value:
                ax.text(j,i,str(value),ha='center',va='center',fontsize=8,color='white' if value/cm15[i].sum()>.6 else 'black')
        ax.set(xticks=positions15,yticks=positions15,xticklabels=labels15,yticklabels=labels15,
               xlabel='Predicted',ylabel='True label',title=key)
        ax.tick_params(labelsize=8); plt.setp(ax.get_xticklabels(),rotation=65,ha='right')
    finish('15-class confusion counts; blank cells are zero')
    ''')
    control, temporal = result['control'], result['temporal']
    n.markdown(f'''### What these results establish

    The fixed temporal candidate selected epoch {temporal['selection']['selected_epoch']}
    after {temporal['epochs_completed']} epochs; the matched control selected epoch
    {control['selection']['selected_epoch']} after {control['epochs_completed']}.
    Validation is **{temporal['validation']['correct']}/{temporal['validation']['samples']}**
    versus **{control['validation']['correct']}/{control['validation']['samples']}**.
    This is one more correct clip, not convincing evidence of general improvement.
    Four old validation errors were corrected and three previously correct clips
    regressed. YOU improves, but YES and THANKYOU regress; PLEASE is **0/4** for
    both models, all predicted MY. ME remains 3/5. Keep these errors visible.

    No new official test or 15-class webcam evaluation has occurred. Historical
    five-class live results are a different task and cannot validate this model.
    One seed and 3-5 validation examples per class limit conclusions. The old
    five-class models/defaults remain preserved. Checkpoint selection uses the
    recorded competitive-validation/small-gap rule, not simply the last epoch
    or lowest loss. No graph in this notebook trains or changes a model.

    **Historical full-vocabulary experiments follow**, with their own selection
    policies, class counts and held-out evaluations; they are not direct
    augmentation controls for this 15-label pilot.''')


def make_metrics(snapshot):
    n=Notebook('02_model_training_and_evaluation')
    pending='' if snapshot['stgcn_complete'] else '**DRAFT: ST-GCN curves are partial; its test results are not available yet.**'
    n.markdown(f'''# Model training, evaluation and comparisons

    {pending}

    This read-only notebook loads a portable snapshot of real epoch/evaluation
    reports. It does not train or evaluate a model. Missing test results are
    reported as unavailable, never as zeros. Historical full-vocabulary models
    use minimum validation cross-entropy; the current pilot's recorded
    competitive-validation/small-gap rule is described separately below.
    ASL Citizen official splits are signer-disjoint.

    Models differ in input representation and capacity; this is not a pure
    architecture-only ablation. Offline scores do not establish webcam accuracy.
    The existing held-out test has been seen during project development; no
    model settings are selected in this notebook from its scores.''')
    n.code(SETUP)
    if snapshot.get('fifteen_sign_temporal'):
        append_current_pilot(n, snapshot)
    if snapshot['stgcn_complete']:
        current=snapshot['models']['stgcn'];baseline=snapshot['models']['gru_regularized']
        test=current['test']['metrics'];old=baseline['test']['metrics']
        last=current['history'][-1];selected=current['selected']
        heading='Historical full-vocabulary result' if snapshot.get('fifteen_sign_temporal') else 'Final result'
        n.markdown(f'''## {heading}

        **ST-GCN training and frozen test evaluation completed successfully.**
        It trained for {len(current['history'])} epochs and selected epoch
        **{selected['epoch']}** using the lowest validation cross-entropy.

        - Test top-1: **{test['top1']*100:.2f}%**, versus {old['top1']*100:.2f}% for
          the regularized GRU (**+{(test['top1']-old['top1'])*100:.2f} percentage points**).
        - Test top-5: **{test['top5']*100:.2f}%**; macro F1: **{test['macro_f1']*100:.2f}%**.
        - Evaluation covered **{test['samples']:,} clips**, all 2,731 labels and
          11 official test signers. No weight updates or augmentation occurred.

        ### Selected checkpoint versus last epoch

        Epoch {selected['epoch']}: validation top-1 **{selected['val']['top1']*100:.2f}%**,
        cross-entropy **{selected['val']['loss']:.5f}**.
        Last epoch {last['epoch']}: validation top-1 **{last['val']['top1']*100:.2f}%**,
        cross-entropy **{last['val']['loss']:.5f}**.
        Accuracy and cross-entropy measure different properties. A later epoch
        can have higher accuracy but worse loss. We retained the declared
        minimum-loss selection rule; the test result belongs to epoch
        {selected['epoch']}, not the last or highest-accuracy checkpoint.''')
        verification=snapshot.get('results_verification')
        if verification and verification.get('status')=='verified':
            n.markdown(f'''### Independent export verification

            Streamed all {verification['samples']:,} ordered prediction records
            against the official test index; rebuilt the confusion matrix and
            recomputed top-1/top-5, macro precision/recall/F1 and weighted F1.
            All matched the evaluation report and per-class/per-signer exports.
            Correct counts: **{verification['top1_correct']:,} top-1** and
            **{verification['top5_correct']:,} top-5**. The evaluated checkpoint
            hash and preserved GRU/archive hashes were unchanged. Cross-entropy
            was retained from the full-logit evaluator; it cannot be reconstructed
            completely from the exported five scores per clip.''')
    rows=[]
    for m in snapshot['models'].values():
        chosen=m['selected']
        rows.append([m['label'],m['status'],len(m['history']),chosen['epoch'] if chosen else '—',
                     f"{m['parameters']:,}" if m['parameters'] else '—',
                     f"{chosen['val']['top1']*100:.2f}%" if chosen else '—',
                     f"{m['test']['metrics']['top1']*100:.2f}%" if m.get('test') and m['test'].get('status')=='complete' else 'Not evaluated'])
    n.markdown('## 1. Experiment inventory\n\n'+md_table(
        ['Model','Status','Committed epochs','Selected epoch','Parameters','Validation top-1','Test top-1'],rows)+
        '\n\nFor an ongoing run, selection is provisional among committed epochs. The 64-unit run used the original baseline settings; the regularized GRU used a different profile. ST-GCN uses selected XY graph nodes and temporal pooling.')
    n.markdown('''## 2. Learning curves

    Every point is a committed epoch JSON, not a smoothed or fabricated value.
    Dotted selection markers show the lowest-validation-loss checkpoint.
    Training accuracy was measured while parameters changed, with dropout and
    training augmentation where enabled; validation uses fixed eval mode. Their
    gap is informative but is not a comparison of two clean fixed-model passes.''')
    n.code('''
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for ax,(key,m) in zip(axs.flat,M.items()):
        h=m['history'];epochs=[r['epoch'] for r in h]
        ax.plot(epochs,[r['train']['top1']*100 for r in h],label='Train',color=COLORS[key])
        ax.plot(epochs,[r['val']['top1']*100 for r in h],'--',label='Validation',color='#333333')
        if m['selected']:ax.axvline(m['selected']['epoch'],ls=':',color=COLORS[key],label='Selected epoch')
        ax.set(title=m['label']+' — '+m['status'],xlabel='Epoch',ylabel='Top-1 (%)',ylim=(0,100))
        ax.legend(fontsize=8)
    finish('Training and validation accuracy')
    ''')
    n.code('''
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for ax,(key,m) in zip(axs.flat,M.items()):
        h=m['history'];epochs=[r['epoch'] for r in h]
        ax.plot(epochs,[r['train']['loss'] for r in h],label='Train',color=COLORS[key])
        ax.plot(epochs,[r['val']['loss'] for r in h],'--',label='Validation',color='#333333')
        if m['selected']:ax.axvline(m['selected']['epoch'],ls=':',color=COLORS[key])
        ax.set(title=m['label'],xlabel='Epoch',ylabel='Mean cross-entropy');ax.legend(fontsize=8)
    finish('Loss curves — selection uses validation cross-entropy')
    ''')
    n.code('''
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for key,m in M.items():
        h=m['history'];x=[r['epoch'] for r in h]
        axs[0].plot(x,[r['val']['top5']*100 for r in h],label=m['label'],color=COLORS[key])
        axs[1].plot(x,[r['val']['macro_f1']*100 for r in h],label=m['label'],color=COLORS[key])
    for ax,title in zip(axs,('Validation top-5','Validation macro F1')):
        ax.set(xlabel='Epoch',ylabel='Percent',ylim=(0,100),title=title);ax.legend(fontsize=8)
    finish('Additional validation metrics across committed epochs')
    ''')
    n.markdown('''## 3. Metric definitions and selected-model comparison

    - **Top-1:** correct highest-score label / clips. **Top-5:** true label in
      five highest scores / clips; it does not mean the first displayed guess is correct.
    - **Precision:** TP/(TP+FP); **recall:** TP/(TP+FN); **F1:** harmonic mean
      of precision/recall, computed per class. Zero denominators return zero.
    - **Macro metrics:** unweighted mean across all 2,731 classes. Macro F1 is
      the mean of class F1 scores, not the harmonic mean of macro P and macro R.
    - **Weighted F1:** class F1 weighted by true support. **Cross-entropy:** mean
      negative log probability of the true class, computed from raw logits.

    Selection metrics below come from the same official validation split.
    Cross-validation fold scores use different partitions and are shown separately.''')
    n.code('''
    keys=[k for k,m in M.items() if m['selected']]
    x=np.arange(len(keys));fig,ax=plt.subplots(figsize=(11,5))
    for offset,metric,label,color in [(-.25,'top1','Top-1','#2878b5'),(0,'top5','Top-5','#219653'),(.25,'macro_f1','Macro F1','#8e44ad')]:
        values=[M[k]['selected']['val'][metric]*100 for k in keys]
        bars=ax.bar(x+offset,values,.24,label=label,color=color)
        ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
    ax.set(xticks=x,xticklabels=[M[k]['label'] for k in keys],ylabel='Percent',ylim=(0,105))
    ax.legend();finish('Official validation at selected checkpoints (ongoing selection is provisional)')
    ''')
    n.markdown('''## 4. Development cross-validation

    Three signer-grouped folds used only the official training pool. Their
    holdouts also selected epochs, so this is development CV, not an independent
    nested-CV test. Fold standard deviation is not a confidence interval. The
    official validation split was subsequently used in a separate fresh refit.''')
    n.code('''
    folds=S['cv']['folds'];x=np.arange(len(folds));fig,ax=plt.subplots(figsize=(10,4))
    for offset,key,label in [(-.25,'top1','Top-1'),(0,'top5','Top-5'),(.25,'macro_f1','Macro F1')]:
        values=np.array([f['validation'][key]*100 for f in folds])
        ax.bar(x+offset,values,.24,label=f'{label}: mean {values.mean():.2f}, SD {values.std(ddof=1):.2f} pp')
    ax.set(xticks=x,xticklabels=[f'Fold {f["fold"]+1}' for f in folds],ylabel='Percent',ylim=(0,100))
    ax.legend(fontsize=8);finish('Regularized GRU: signer-grouped development CV')
    ''')
    n.markdown('''## 5. Frozen official test evaluation

    Test inference uses eval/inference mode, no augmentation or weight updates;
    checkpoint identity is checked before/after. Only completed reports appear.
    The baseline and 64-unit GRUs have no test evaluation in the current record.
    Per-class and per-signer graphs describe the named test population.
    They do not quantify new-user webcam performance.''')
    n.code('''
    tested={k:m for k,m in M.items() if m.get('test') and m['test'].get('status')=='complete'}
    for m in tested.values():
        t=m['test'];print(m['label'],':',t['metrics']['samples'],'test clips | epoch',t['checkpoint_epoch'])
        for key in ('top1','top5','macro_precision','macro_recall','macro_f1','weighted_f1','loss'):
            print(' ',key,':',round(t['metrics'][key],6))
    metrics=['top1','top5','macro_precision','macro_recall','macro_f1','weighted_f1']
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(metrics));width=.75/max(1,len(tested))
    for i,(key,m) in enumerate(tested.items()):
        offset=(i-(len(tested)-1)/2)*width
        ax.bar(x+offset,[m['test']['metrics'][v]*100 for v in metrics],width,label=m['label'],color=COLORS[key])
    ax.set(xticks=x,xticklabels=['Top-1','Top-5','Macro P','Macro R','Macro F1','Weighted F1'],ylabel='Percent',ylim=(0,100))
    ax.legend();finish('Completed official test evaluations')
    ''')
    n.code('''
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for key,m in tested.items():
        people=sorted(m.get('per_signer',{}),key=lambda x:int(x.lstrip('P')))
        axs[0].plot(people,[m['per_signer'][p]['top1']*100 for p in people],'o-',label=m['label'],color=COLORS[key])
        axs[1].hist([r['f1'] for r in m.get('per_class',[])],bins=np.linspace(0,1,21),histtype='step',linewidth=2,label=m['label'],color=COLORS[key])
    axs[0].set(title='Test signer variation',ylabel='Top-1 (%)',xlabel='Test participant',ylim=(0,100))
    axs[1].set(title='Per-class F1 distribution',xlabel='F1',ylabel='Number of labels')
    for ax in axs:ax.legend(fontsize=8)
    finish('Generalization varies across signers and labels')
    ''')
    if snapshot['stgcn_complete']:
        maps={key:{r['gloss']:r for r in snapshot['models'][key]['per_class']}
              for key in ('gru_regularized','stgcn')}
        rows=[]
        for name in ('HELLO','THANKYOU','NO'):
            if all(name in mapping for mapping in maps.values()):
                old,new=maps['gru_regularized'][name],maps['stgcn'][name]
                rows.append([name,new['support'],f"{round(old['recall']*old['support'])}/{old['support']}",
                             f"{round(new['recall']*new['support'])}/{new['support']}",
                             f"{old['f1']*100:.2f}%",f"{new['f1']*100:.2f}%"])
        n.markdown('''### Familiar signs: overall improvement does not apply to every class

        These are existing official test examples, not new webcam captures.
        Class sample sizes are small. HELLO and THANKYOU have fewer correct
        first guesses under ST-GCN; NO has the same number. The paired
        all-class plots below show gains and regressions across the vocabulary.

        '''+md_table(['Label','Test clips','GRU correct','ST-GCN correct','GRU F1','ST-GCN F1'],rows))
    n.code('''
    if 'gru_regularized' in tested and 'stgcn' in tested:
        a={r['gloss']:r for r in tested['gru_regularized']['per_class']}
        b={r['gloss']:r for r in tested['stgcn']['per_class']}
        labels=sorted(set(a)&set(b))
        old=np.array([a[name]['f1'] for name in labels])
        new=np.array([b[name]['f1'] for name in labels])
        delta=(new-old)*100
        print('Labels with higher/lower/equal F1:',int((delta>1e-9).sum()),int((delta < -1e-9).sum()),int((np.abs(delta)<=1e-9).sum()))
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        axs[0].scatter(old*100,new*100,s=9,alpha=.25,color='#8e44ad')
        axs[0].plot([0,100],[0,100],'--',color='#444444')
        for name in ('HELLO','THANKYOU','NO'):
            if name in a and name in b:
                axs[0].scatter(a[name]['f1']*100,b[name]['f1']*100,s=45,color='#c0392b')
                axs[0].annotate(name,(a[name]['f1']*100,b[name]['f1']*100),xytext=(5,5),textcoords='offset points',fontsize=8)
        axs[0].set(xlabel='Regularized GRU F1 (%)',ylabel='ST-GCN F1 (%)',xlim=(0,100),ylim=(0,100),title='Each point is one class')
        axs[1].hist(delta,bins=np.linspace(-100,100,41),color='#2878b5')
        axs[1].axvline(0,ls='--',color='#444444')
        axs[1].set(xlabel='ST-GCN minus GRU F1 (percentage points)',ylabel='Number of classes',title='Paired class changes, not only aggregate scores')
        finish('Class-specific gains and regressions on the same test split')
    else:
        print('Paired class comparison awaits completed ST-GCN test results.')
    ''')
    n.code('''
    fig,axs=plt.subplots(1,len(tested),figsize=(7*len(tested),5),squeeze=False)
    for ax,(key,m) in zip(axs.flat,tested.items()):
        c=m['focused_confusion'];counts=np.array(c['counts']);support=np.array(c['full_row_support'])
        percentages=np.divide(counts,support[:,None],out=np.zeros_like(counts,dtype=float),where=support[:,None]>0)*100
        image=ax.imshow(percentages,vmin=0,vmax=100,cmap='Blues')
        ax.set(xticks=range(len(c['labels'])),yticks=range(len(c['labels'])),xticklabels=c['labels'],yticklabels=c['labels'],xlabel='Predicted label',ylabel='True label',title=m['label'])
        ax.tick_params(axis='x',rotation=50);ax.grid(False)
        for row in range(len(counts)):
            for col in range(len(counts)):
                if counts[row,col]:ax.text(col,row,str(counts[row,col]),ha='center',va='center',color='white' if percentages[row,col]>50 else 'black',fontsize=9)
        fig.colorbar(image,ax=ax,label='% of full true-class support')
    finish('Focused confusion: selected labels only; annotations are counts')
    ''')
    n.markdown('''The focused matrix omits predictions outside the displayed columns. Its row
    percentages use **full class support**, so displayed rows need not sum to
    100%. This is not the complete 2,731×2,731 confusion matrix.''')
    n.code('''
    fig,axs=plt.subplots(1,len(tested),figsize=(8*len(tested),6),squeeze=False)
    for ax,(key,m) in zip(axs.flat,tested.items()):
        pairs=list(reversed(m['top_confusions'][:10]))
        ax.barh([p['true']+' → '+p['predicted'] for p in pairs],[p['count'] for p in pairs],color=COLORS[key])
        ax.set(title=m['label'],xlabel='Misclassified test clips')
    finish('Most frequent off-diagonal test confusions (each model ranked separately)')
    ''')
    n.markdown('''## 6. Resource cost and learning-rate schedules

    Timings are the stored epoch measurements for this machine. Ongoing-run
    averages are preliminary; model initialization, data verification and total
    wall time have different scopes. Parameter count is not an accuracy metric.
    GPU tensor allocation excludes other device/driver allocations.''')
    n.code('''
    keys=list(M);fig,axs=plt.subplots(1,3,figsize=(15,4))
    names=[M[k]['label'] for k in keys]
    axs[0].bar(names,[M[k]['parameters']/1e6 for k in keys],color=[COLORS[k] for k in keys])
    axs[0].set(ylabel='Millions of parameters',title='Model capacity')
    axs[1].bar(names,[np.mean([r['seconds'] for r in M[k]['history']]) for k in keys],color=[COLORS[k] for k in keys])
    axs[1].set(ylabel='Mean recorded seconds / epoch',title='Train + validation epoch time')
    for key,m in M.items():
        axs[2].plot([r['epoch'] for r in m['history']],[r['learning_rate'] for r in m['history']],label=m['label'],color=COLORS[key])
    axs[2].set(yscale='log',xlabel='Epoch',ylabel='Learning rate',title='Recorded schedules');axs[2].legend(fontsize=7)
    for ax in axs[:2]:ax.tick_params(axis='x',rotation=40)
    finish('Capacity, measured runtime and optimization schedule')
    ''')
    n.markdown('''## 7. Interpretation and next comparisons

    - Use validation and test results for their separate purposes. Do not pick
      a new checkpoint from this test table or compare CV directly with official validation.
    - A smaller train/validation gap can indicate underfitting or incomplete
      convergence. The 64-unit result does not prove that size can never work.
    - Improvements from a combined regularization profile cannot be attributed
      to one component without a controlled ablation.
    - ST-GCN changes topology, representation, pooling and parameter count.
      Dataset performance must be followed by a separate user-attended webcam test.
    - Unknown-sign/idle rejection, calibrated scores, continuous signing and RSL
      remain unvalidated. See notebook 01 for the actual failed live checks.

    Reports and source: [GRU comparison](../docs/model-comparison-results-2026-11-10.md),
    [frozen GRU test](../docs/test-evaluation-2026-11-10.md),
    [ST-GCN protocol](../docs/stgcn-training.md),
    [investigation](../docs/recognition-investigation-2026-11-10.md).
    The companion snapshot records source hashes and dataset/checkpoint identities.
    No model or dataset is embedded in these notebooks.''')
    return n


def build(destination, final=False):
    snapshot=gather_snapshot()
    if final:
        require_final(snapshot)
    artifacts=destination/'artifacts'
    artifacts.mkdir(parents=True,exist_ok=True)
    snapshot_path=artifacts/'experiment_snapshot.json'
    snapshot_path.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False),encoding='utf-8')
    manifest={'generated_utc':snapshot['generated_utc'],'stgcn_complete':snapshot['stgcn_complete'],'files':{}}
    for notebook in (make_workflow(snapshot),make_metrics(snapshot)):
        result=notebook.execute(destination,artifacts/'figures')
        result['metadata']['snapshot_sha256']=sha(snapshot_path)
        path=destination/(notebook.name+'.ipynb')
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'))
            if old.get('metadata',{}).get('generated_by') != 'scripts/build_project_notebooks.py':
                raise ValueError(f'Refusing to replace a notebook not owned by this generator: {path}')
        temporary=path.with_suffix('.ipynb.tmp')
        temporary.write_text(json.dumps(result,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
        temporary.replace(path)
        manifest['files'][path.name]={'sha256':sha(path),'bytes':path.stat().st_size,
                                      'code_cells':sum(c['cell_type']=='code' for c in result['cells']),
                                      'embedded_figures':sum(o['output_type']=='display_data' for c in result['cells'] for o in c.get('outputs',[]))}
    for path in sorted((artifacts/'figures').glob('*.png')):
        manifest['files'][path.relative_to(destination).as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size}
    manifest['files']['artifacts/experiment_snapshot.json']={'sha256':sha(snapshot_path),'bytes':snapshot_path.stat().st_size}
    (artifacts/'report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
    for name,info in manifest['files'].items():
        if name.endswith('.ipynb'):print(name,info)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'notebooks')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    build(args.output,args.require_complete)


if __name__=='__main__':
    main()
