"""Read-only official train/validation pronoun audit; no test reads or fitting."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile
import zlib

import torch  # Initialize Windows native libraries before OpenCV/MediaPipe.
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json
from sign_language.recognition import SignRecognizer
from sign_language.preprocessing import normalize_sequence
from sign_language.pronoun_diagnostics import check_models,diagnose_observations,CURRENT_SHA,BASELINE_SHA
from run_fifteen_sign_temporal_trial import evaluate
import audit_please_my_trajectories as video_audit

CACHE=ROOT/'data/training/asl-fifteen-signs-128-v1'
RAW=ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
OUTPUT=ROOT/'docs/assets/pronoun-investigation-2026-11-14.json'
BOARDS=ROOT/'.cache/pronoun-audit-20261114-v1'


def read(path):return json.loads(path.read_text())


def main(labels=('ME','WE'),output=None,boards=None):
    global OUTPUT,BOARDS
    if output is not None:OUTPUT=Path(output).resolve()
    if boards is not None:BOARDS=Path(boards).resolve()
    labels=tuple(labels)
    expected={s:{g:sum(r['gloss']==g for r in read(CACHE/f'{s}-index.json')) for g in labels} for s in ('train','val')}
    total=sum(sum(counts.values()) for counts in expected.values())
    if not total or any(v==0 for counts in expected.values() for v in counts.values()):raise ValueError('Empty pronoun split')
    if OUTPUT.exists() or BOARDS.exists():raise FileExistsError('Preserve completed/partial evidence; use a new version')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    current_path=ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt'
    baseline_path=ROOT/'models/asl-gru-fifteen-signs-temporal-v1/temporal/best.pt'
    if sha256(current_path)!=CURRENT_SHA or sha256(baseline_path)!=BASELINE_SHA:raise ValueError('Frozen checkpoint differs')
    current=SignRecognizer(current_path,ROOT/'models');baseline=SignRecognizer(baseline_path,ROOT/'models')
    check_models(current,baseline)
    manifest=read(CACHE/'manifest.json')
    protected={str(CACHE/n):h for n,h in manifest['files_sha256'].items()}
    protected.update({str(p):sha256(p) for p in (current_path,baseline_path,
        ROOT/'models/archives/asl-gru-fifteen-signs-trajectory-readout-c01-2026-11-12.zip')})
    for n,h in current.config['source_sha256'].items():protected[str(ROOT/'src/sign_language'/n)]=h
    if not all(sha256(Path(n))==h for n,h in protected.items()):raise ValueError('Protected inputs differ')
    BOARDS.mkdir();video_audit.BOARDS=BOARDS
    reference=read(current_path.with_name('report.json'))
    report={'status':'running','date_local':'2026-11-14','timezone':'Asia/Tashkent',
            'scope':f'Exactly {total} {"/".join(labels)} training and validation clips; all 60 validation predictions reproduced as a consistency check.',
            'checkpoint_sha256':CURRENT_SHA,'baseline_checkpoint_sha256':BASELINE_SHA,
            'cache_id':manifest['cache_id'],'records':[],
            'protocol':{'training':False,'test_read':False,'corpus_cleaning':False,'defaults_changed':False,
                        'same_backbone':True,'controls':'original, reverse, seeded shuffle, static frame, anatomical reflection, hand-slot swap; 3 standardized head-group interventions on each',
                        'windows':'full, first two seconds, detected-hand envelope; slice raw observations before normalization',
                        'limits':'Flagged tracking/geometry is a review lead, not measured joint error, semantic mislabeling or a fix.'}}
    write_json(OUTPUT,report)
    # Same loader/batches/AMP as the original selected report.
    validation=evaluate(current.model,'val',current.names)
    a=np.asarray([p['scores'] for p in validation['predictions']]);b=np.asarray([p['scores'] for p in reference['validation']['predictions']])
    np.testing.assert_allclose(a,b,rtol=0,atol=1.e-6)
    if [p['prediction'] for p in validation['predictions']]!=[p['prediction'] for p in reference['validation']['predictions']]:
        raise AssertionError('Saved 60-example validation decisions changed')
    report['full_validation_reproduction']={'samples':60,'correct':validation['correct'],
        'maximum_score_difference':float(np.abs(a-b).max()),'per_class':validation['per_class']}
    for split in ('train','val'):
        rows=read(CACHE/f'{split}-index.json');cached=np.load(CACHE/f'{split}-features.npy',mmap_mode='r')
        lengths=np.load(CACHE/f'{split}-lengths.npy');targets=np.load(CACHE/f'{split}-targets.npy')
        predictions={p['video_file']:p for p in reference['train' if split=='train' else 'validation']['predictions']}
        for i,row in enumerate(rows):
            if row['gloss'] not in labels:continue
            if int(targets[i])!=manifest['labels'][row['gloss']]:raise ValueError('Prepared target/index mismatch')
            parent=row['parent_index'];archive_path=RAW/f'{split}-{parent//100:05d}.zip'
            if str(archive_path) not in protected:
                marker=read(archive_path.with_suffix('.complete.json'))
                if sha256(archive_path)!=marker['sha256'] or archive_path.stat().st_size!=marker['bytes']:
                    raise ValueError('Raw archive hash/size differs')
                protected[str(archive_path)]=marker['sha256']
            with zipfile.ZipFile(archive_path) as archive:
                record=json.loads(archive.read('records.json'))['records'][parent%100]
                if record['status']!='ok' or record['class_id']!=row['parent_class_id'] or any(
                    record[k]!=row[k] for k in ('video_file','participant_id','gloss','split')):raise ValueError('Raw record identity mismatch')
                with np.load(io.BytesIO(archive.read(record['member'])),allow_pickle=False) as loaded:
                    arrays={k:loaded[k] for k in loaded.files}
            normalized,quality=normalize_sequence(arrays,record['frame_shape']);length=int(lengths[i])
            if len(normalized)!=length or not np.array_equal(normalized.astype(np.float16),cached[i,:length]):
                raise AssertionError('Raw-to-prepared float16 reproduction failed')
            video=video_audit.video_check(row,record,arrays,True)
            source_bytes=(video_audit.VIDEOS/row['video_file']).read_bytes()
            with zipfile.ZipFile(ROOT/'data/downloads/ASL_Citizen.zip') as source_zip:
                source_info=source_zip.getinfo('ASL_Citizen/videos/'+row['video_file'])
            if len(source_bytes)!=source_info.file_size or zlib.crc32(source_bytes)!=source_info.CRC:
                raise ValueError('Video differs from original download archive CRC/size')
            video['original_archive_crc_and_size_match']=True
            protected[str(video_audit.VIDEOS/row['video_file'])]=video['sha256']
            diagnostics=diagnose_observations(current,baseline,arrays,record['frame_shape'],arrays['timestamp_ms'])
            full=diagnostics['windows']['full'];p=predictions[row['video_file']]
            first=full['variants']['original']['current']['top5'][0]['label'] if full['status']=='scored' else None
            if first!=p['prediction']:raise AssertionError('Dataset clip prediction differs from saved decision')
            flags=[]
            fraction=diagnostics['measurements']['tracking']['hand_frame_fraction']
            if fraction<.5:flags.append('hands_present_under_half_clip')
            if full['feature_summary']['single_hand_slot_changes']:flags.append('single_hand_slot_changes_review')
            if len(arrays['timestamp_ms'])>128:flags.append('uniform_downsampling_review')
            if first!=row['gloss']:flags.append('model_error')
            delta=float(np.max(np.abs(np.asarray(full['variants']['original']['current']['scores'])-p['scores'])))
            report['records'].append({**row,'source_frames':record['frames'],'model_frames':length,
                'duration_seconds':float((arrays['timestamp_ms'][-1]-arrays['timestamp_ms'][0])/1000.),
                'quality':quality,'video':video,'raw_to_prepared_exact_at_float16':True,
                'saved_prediction':p['prediction'],'saved_prediction_score_difference':delta,
                'review_flags':flags,**diagnostics})
            write_json(OUTPUT,report)
            print(f'{len(report["records"])}/{total} {split} {row["gloss"]} {row["participant_id"]}: {first}; hands={fraction:.1%}; flags={flags}',flush=True)
    report['counts']={s:{g:sum(r['split']==s and r['gloss']==g for r in report['records']) for g in labels} for s in ('train','val')}
    if report['counts']!=expected:raise AssertionError('Wrong audit scope')
    unchanged=all(sha256(Path(p))==h for p,h in protected.items())
    report.update(status='complete',protected_inputs_unchanged=unchanged,
                  protected_sha256={str(Path(p).relative_to(ROOT)):h for p,h in protected.items()},
                  webcam_opened=False,new_landmark_extraction=False)
    if not unchanged:raise AssertionError('Protected source bytes changed')
    write_json(OUTPUT,report)
    print('PRONOUN_AUDIT_COMPLETE',report['counts'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',nargs='+',choices=('ME','WE','YOU'),default=['ME','WE'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--boards',type=Path)
    args=parser.parse_args()
    if args.labels!=['ME','WE'] and (args.output is None or args.boards is None):
        parser.error('New audit scope requires explicit new output and board paths')
    main(args.labels,args.output,args.boards)
