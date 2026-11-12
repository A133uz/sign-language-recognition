"""Record confirmed live labels separately from unlabeled captures; no fitting."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.dataset import write_json


def main():
    output=ROOT/'docs/assets/trajectory-readout-live-feedback-2026-11-12.json'
    if output.exists():raise FileExistsError('Preserve prior feedback snapshot')
    launch=json.loads((ROOT/'.cache/webcam-trajectory-latest-launch.json').read_text(encoding='utf-8-sig'))
    raw=Path(launch['report']).read_bytes()
    session=json.loads(raw)
    expected_sha='34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91'
    if session['checkpoint_sha256']!=expected_sha or hashlib.sha256(Path(session['checkpoint']).read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('Selected model identity differs')
    confirmed_ids=[4,7,8,10,11]
    trials={r['capture_id']:r for r in session['trials']}
    matched=[{'capture_id':i,'expected':'ME','label_source':'User confirmed all five IDs in the follow-up question',
              'prediction':trials[i]['top5'][0]['label'],'top5':trials[i]['top5'],
              'correct':trials[i]['top5'][0]['label']=='ME',
              'capture_seconds':trials[i]['capture_seconds'],
              'effective_capture_fps':trials[i]['effective_capture_fps'],
              'quality':trials[i]['quality'],'tracking_summary':trials[i]['tracking_summary']}
             for i in confirmed_ids]
    offline=json.loads((ROOT/'docs/assets/please-trajectory-runtime-2026-11-12.json').read_text())['cached_validation']
    report={'status':'feedback_recorded','generated_utc':datetime.now(timezone.utc).isoformat(),
            'session_id':session['session_id'],'snapshot_source_sha256':hashlib.sha256(raw).hexdigest(),
            'checkpoint_sha256':expected_sha,'session_snapshot':session,
            'user_feedback':'GOOD, HELP and PLEASE recognized first. ME appears second behind WE.',
            'followup_question':'Which captures were ME: #4, #7, #8, #10 and #11?',
            'followup_answer':'all of them','confirmed_trials':matched,
            'confirmed_ME_correct':sum(t['correct'] for t in matched),'confirmed_ME_support':len(matched),
            'other_capture_labels':'Unassigned. Reported successes are retained as user feedback; other IDs are not guessed.',
            'offline_ME_WE':{g:offline['per_class'][g] for g in ('ME','WE')},
            'observations':[
                'All five confirmed ME attempts predict WE first and ME second.',
                'Capture 4 is 2.94 seconds, so an eight-second ending is not necessary for the confusion.',
                'Capture 8 has hands in 137/157 frames (87.3%); hand absence alone does not explain every failure.',
                'Shoulder anchors are present in every captured frame of the five confirmed attempts.',
                'Aggregate presence does not verify joint accuracy, hand identity or whether absent hands were offscreen.',
                'No saved video or landmark coordinates exist for replaying these exact captures.',
                'ME 4/5 and WE 3/3 on development validation did not transfer to these five live ME attempts.'],
            'cause_established':False,'model_changed':False,'training_performed':False,
            'media_saved':False,'coordinates_saved':False,'overall_live_accuracy_computed':False}
    # Runtime file paths are local operational details; retain a portable snapshot.
    report['session_snapshot']['checkpoint']='models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt'
    write_json(output,report)
    print(json.dumps({'confirmed_ME_correct':report['confirmed_ME_correct'],
                      'confirmed_ME_support':len(matched),'offline_ME_WE':report['offline_ME_WE']},indent=2))


if __name__=='__main__':main()
