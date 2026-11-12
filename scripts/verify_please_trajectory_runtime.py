"""Verify the selected readout using the application and fresh public-video tracking."""
import json
from pathlib import Path
import sys

import torch  # Load native PyTorch dependencies before OpenCV/MediaPipe on Windows.
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.five_sign_pilot import sha256,write_json
from run_fifteen_sign_temporal_trial import CACHE,evaluate


def main():
    output=ROOT/'docs/assets/please-trajectory-runtime-2026-11-12.json'
    if output.exists():raise FileExistsError('Preserve prior runtime evidence')
    study=json.loads((ROOT/'docs/assets/please-trajectory-readout-study-2026-11-12.json').read_text())
    chosen=study['selected']
    if not chosen or not chosen['acceptance_met']:raise ValueError('No qualified candidate')
    checkpoint=ROOT/chosen['checkpoint']
    if sha256(checkpoint)!=chosen['checkpoint_sha256']:raise ValueError('Checkpoint differs')
    manifest=json.loads((CACHE/'manifest.json').read_text())
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'models').rglob('*')
               if p.is_file() and (p.name=='best.pt' or p.suffix=='.pkl')}
    torch.set_num_threads(2);cv2.setNumThreads(1)
    recognizer=SignRecognizer(checkpoint,ROOT/'models',device='cuda')
    if len(recognizer.names)!=15:raise AssertionError('Wrong vocabulary')
    actual=evaluate(recognizer.model,'val',recognizer.names)
    reference=chosen['validation']
    actual_scores=np.asarray([r['scores'] for r in actual['predictions']])
    reference_scores=np.asarray([r['scores'] for r in reference['predictions']])
    np.testing.assert_allclose(actual_scores,reference_scores,rtol=0,atol=1.e-6)
    if any(a['prediction']!=b['prediction'] or a['video_file']!=b['video_file']
           for a,b in zip(actual['predictions'],reference['predictions'],strict=True)):
        raise AssertionError('Runtime changed validation decisions')
    baseline=json.loads((ROOT/'docs/assets/fifteen-sign-temporal-2026-11-12.json').read_text())['temporal']['validation']
    report={'status':'running','checkpoint':chosen['checkpoint'],'checkpoint_sha256':chosen['checkpoint_sha256'],
            'loaded_class':type(recognizer.model).__name__,'labels':recognizer.names,'backbone_epoch':recognizer.epoch,
            'cached_validation':actual,'cached_max_score_difference':float(np.abs(actual_scores-reference_scores).max()),
            'class_changes':{g:{'before':baseline['per_class'][g]['correct'],
                                'after':actual['per_class'][g]['correct'],'support':actual['per_class'][g]['support']}
                             for g in recognizer.names},
            'fresh_video_scope':'All eight known validation PLEASE and MY videos, original frames and original FPS.',
            'fresh_videos':[],'official_test_read':False,'training_performed':False,
            'expected_label_passed_to_predictor':False,'video_or_landmarks_saved':False,
            'development_cases_not_independent_evaluation':True}
    write_json(output,report)
    for row in reference['predictions']:
        if row['expected'] not in ('PLEASE','MY'):continue
        path=ROOT/'data/asl_citizen/ASL_Citizen/videos'/row['video_file']
        digest=sha256(path)
        frames=[];timestamps=[];capture=cv2.VideoCapture(str(path))
        fps=capture.get(cv2.CAP_PROP_FPS)
        declared=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            if not capture.isOpened() or fps<=0:raise ValueError('Video/FPS unavailable')
            while True:
                ok,frame=capture.read()
                if not ok:break
                timestamps.append(max(timestamps[-1]+1 if timestamps else 0,round(len(frames)*1000/fps)))
                frames.append(frame)
        finally:capture.release()
        if len(frames)!=declared:raise AssertionError('Incomplete video decode')
        # This is exactly the webcam application's prediction path. Labels only
        # enter the report below, after inference has returned.
        prediction=recognizer.predict_frames(frames,timestamps)
        if sha256(path)!=digest:raise AssertionError('Video changed')
        item={'video_file':row['video_file'],'participant_id':row['participant_id'],
              'source_sha256':digest,'fps':fps,'expected':row['expected'],**prediction,
              'correct':prediction['top5'][0]['label']==row['expected']}
        report['fresh_videos'].append(item)
        write_json(output,report)
        print(row['expected'],row['participant_id'],prediction['top5'][0],flush=True)
        del frames
    report.update(status='complete',
                  fresh_per_class={g:{'correct':sum(r['correct'] for r in report['fresh_videos'] if r['expected']==g),
                                      'support':sum(r['expected']==g for r in report['fresh_videos'])}
                                   for g in ('PLEASE','MY')},
                  previous_weights_and_probes_unchanged=all(sha256(ROOT/n)==h for n,h in protected.items()),
                  subset_unchanged=all(sha256(CACHE/n)==h for n,h in manifest['files_sha256'].items()))
    if not report['previous_weights_and_probes_unchanged'] or not report['subset_unchanged']:
        raise AssertionError('Protected artifacts changed')
    report['fresh_acceptance_met']=all(v=={'correct':4,'support':4} for v in report['fresh_per_class'].values())
    write_json(output,report)
    print('RUNTIME_COMPLETE',report['fresh_per_class'],flush=True)


if __name__=='__main__':main()
