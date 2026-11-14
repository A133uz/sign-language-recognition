"""Two consented RAM-only YOU previews with the unchanged geometry-enabled model."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import torch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.recognition_webcam import run_recognition
from sign_language.pronoun_diagnostics import CURRENT_SHA
from sign_language.pronoun_landmark_review import MemoryBoards,PronounReviewRecognizer,canonical,overlay,hand_crop,hand_shape_summary
from sign_language.five_sign_pilot import write_json,sha256


def public_reference():
    cache=ROOT/'data/training/asl-fifteen-signs-128-v1'
    row=next(r for r in json.loads((cache/'val-index.json').read_text()) if r['gloss']=='YOU' and r['participant_id']=='P12')
    parent=row['parent_index'];raw=ROOT/f'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose/val-{parent//100:05d}.zip'
    with zipfile.ZipFile(raw) as archive:
        record=json.loads(archive.read('records.json'))['records'][parent%100]
        with np.load(io.BytesIO(archive.read(record['member'])),allow_pickle=False) as source:
            arrays={k:source[k] for k in source.files}
    index=18;capture=cv2.VideoCapture(str(ROOT/'data/asl_citizen/ASL_Citizen/videos'/row['video_file']))
    frame=None
    try:
        for _ in range(index+1):
            ok,frame=capture.read()
            if not ok:raise RuntimeError('Cannot decode public reference')
    finally:capture.release()
    hands,present=canonical(arrays,record['frame_shape']);annotated=overlay(frame,hands,present,index)
    return {'raw':hand_crop(frame,hands,present,index),'annotated':hand_crop(annotated,hands,present,index),
            'video_file':row['video_file'],'frame_index':index,'split':'val','participant_id':'P12',
            'hand_shape_aggregates':hand_shape_summary(hands,present,index,frame.shape[1])}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();args.report=args.report.resolve()
    if not args.report.is_relative_to((ROOT/'.cache').resolve()):parser.error('Review reports must stay inside project .cache')
    if args.report.exists():raise FileExistsError('Preserve prior evidence')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    recognizer=SignRecognizer(ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt',ROOT/'models')
    if recognizer.checkpoint_sha256!=CURRENT_SHA or recognizer.model.classifier.in_features!=279:
        raise ValueError('Requires the frozen 256+23 geometry readout')
    reference=public_reference();boards=MemoryBoards();wrapper=PronounReviewRecognizer(recognizer,boards,reference)
    link=args.report.with_suffix('.preview.json')
    write_json(link,{'status':'running','preview_url':boards.url,'reference':{k:v for k,v in reference.items() if k not in ('raw','annotated')},
                     'geometry_enabled':True,'checkpoint_sha256':recognizer.checkpoint_sha256,
                     'consent':'User requested live YOU landmark inspection, with geometry enabled. Annotated stills visible in chat via RAM-only loopback receiver; no media files or coordinate files.',
                     'images_saved_to_files':False})
    try:
        print('REVIEW_GEOMETRY_ON | YOU natural once, YOU hold once | 3s auto each | Q after two',flush=True)
        run_recognition(wrapper,args.report,max_seconds=3.,capture_fps=30.,expected_sign='YOU',auto_finish=True)
        print('CAMERA_CLOSED | review PNGs remain in RAM until cleared; maximum five minutes',flush=True)
        boards.cleared.wait(300)
    finally:
        boards.close()
        meta=json.loads(link.read_text());meta.update(status='closed',ram_images_cleared=True,
            checkpoint_unchanged=sha256(recognizer.checkpoint)==recognizer.checkpoint_sha256)
        meta.pop('preview_url',None);write_json(link,meta)
        print('RAM_PREVIEWS_CLEARED | geometry remained enabled',flush=True)


if __name__=='__main__':main()
