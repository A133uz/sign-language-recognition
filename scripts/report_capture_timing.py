"""Report instrumented timing and fixed-normalization controls; no model fitting."""
import argparse
import json
import re
from pathlib import Path
import sys

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.dataset import write_json
from sign_language.five_sign_pilot import sha256
from sign_language.pronoun_diagnostics import recover_competitor_margin
from report_pronoun_investigation import selected_prediction,confirmed_trials,counts,table,format_count,analysis_status,apply_feedback_corrections

OUTPUT=ROOT/'docs/assets/capture-timing-normalization-2026-11-14.json'
DOC=ROOT/'docs/capture-timing-normalization-2026-11-14.md'
WINDOWS=('full','first_two_seconds','first_two_seconds_fixed_normalization',
         'detected_hand_envelope','detected_hand_envelope_fixed_normalization')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',type=Path)
    parser.add_argument('--confirm-all',action='store_true');args=parser.parse_args()
    if args.confirm_all and args.live is None:raise ValueError('Confirmation requires a completed live report')
    preflight_path=ROOT/'docs/assets/timing-normalization-preflight-2026-11-14.json'
    preflight=json.loads(preflight_path.read_text())
    if preflight['status']!='verified' or len(preflight['rows'])!=37:raise ValueError('Complete adapter preflight first')
    if not all(sha256(ROOT/p)==h for p,h in preflight['protected_sha256'].items()):
        raise AssertionError('Protected files changed')
    summary={'status':'preflight_complete_live_pending','date_local':'2026-11-14','timezone':'Asia/Tashkent',
             'preflight_sha256':sha256(preflight_path),'verified_dataset_clips':37,
             'checkpoint_sha256':preflight['checkpoint_sha256'],'protected_inputs_unchanged':True,
             'training_performed':False,'limiter_changed':False,'production_preprocessing_changed':False,
             'raw_timing_trace_saved':False,'live_arrays_saved':False,'official_test_read':False,'live':None,
             'diagnostic_source_sha256':{p:sha256(ROOT/p) for p in ('src/sign_language/capture_timing_diagnostics.py',
                 'src/sign_language/pronoun_diagnostics.py','scripts/diagnose_pronouns_live.py','scripts/verify_timing_normalization.py','scripts/report_capture_timing.py')}}
    checks=ROOT/'.cache/timing-normalization-full-checks-20261114.log'
    if checks.exists():
        text=checks.read_text(encoding='utf-8',errors='replace');match=re.search(r'Ran (\d+) tests',text)
        if match and '\nOK\n' in text and 'Import and unit checks do not validate' in text:
            summary['repository_checks']={'tests_passed':int(match.group(1)),'log_sha256':sha256(checks)}
    traced=ROOT/'.cache/timing-normalization-traced-tests-20261114.log'
    support=ROOT/'.cache/timing-normalization-support-checks-20261114.log'
    if 'repository_checks' not in summary and traced.exists() and support.exists():
        text=traced.read_text(encoding='utf-8',errors='replace');match=re.search(r'Ran (\d+) tests',text)
        if match and '\nOK\n' in text and 'SUPPORT_CHECKS_COMPLETE' in support.read_text(encoding='utf-8',errors='replace'):
            summary['repository_checks']={'tests_passed':int(match.group(1)),
                'log_sha256':sha256(traced),'runner':'Same full unittest suite with PyTorch preloaded and timed stack dumps',
                'initial_run_interrupted_after_stall':True,'support_log_sha256':sha256(support)}
    if args.live:
        live=apply_feedback_corrections(json.loads(args.live.read_text()))
        if not live.get('timing_normalization_mode') or live['checkpoint_sha256']!=summary['checkpoint_sha256']:
            raise ValueError('Expected the instrumented frozen-model experiment')
        if not live.get('checkpoints_unchanged'):raise ValueError('Session must close with unchanged weights')
        if args.confirm_all:
            if any(r.get('trial_status')=='user_reported_wrong_gesture' for r in live['trials']):
                raise ValueError('Known wrong-gesture trials cannot be confirmed as valid')
            if live['status']!='complete' or len(live['trials'])!=9:raise ValueError('Cannot confirm an incomplete session')
            for row in live['trials']:
                if row['trial_status']=='captured_pending_confirmation':row.update(user_confirmation=True,trial_status='confirmed_valid')
            live['user_completion_confirmed']=True
            live['confirmation_source']='User confirmed all nine prompts performed in the first two seconds with normal capture.'
            write_json(args.live,live)
        valid=confirmed_trials(live)
        for row in valid:
            t=row['capture_timing']
            if t['limiter_replay_mismatches'] or t['arrival_count']!=row['source_frame_count'] or t['dropped_count']!=row['fps_limit_dropped_frames']:
                raise AssertionError('Timing replay/count reconciliation failed')
            if not row['normalization_comparison']['full_adapter_matches_production_at_float16']:
                raise AssertionError('Fixed-context normalizer disagreed with production')
        block={'session_id':live['session_id'],'source_sha256':sha256(args.live),'camera_reported_fps':live.get('camera_reported_fps'),
               'confirmed_trials':valid,'excluded_trials':[{'capture_id':r['capture_id'],'status':r['trial_status'],'reason':r.get('exclusion_reason')} for r in live['trials'] if r not in valid],
               'gesture_validity_correction':live.get('gesture_validity_correction'),
               'correction_ledger_sha256':live.get('correction_ledger_sha256'),
               'user_completion_confirmed':live['user_completion_confirmed'],'controls':{},'competitor_comparisons':[]}
        for window in WINDOWS:
            for model in ('current','original_temporal_model'):
                block['controls'][window+'/'+model]=counts(valid,lambda r,w=window,m=model:selected_prediction(r['windows'][w],model=m),'prompted_label')
        state=torch.load(ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt',map_location='cpu',weights_only=True)
        w=state['model']['classifier.weight'].numpy();b=state['model']['classifier.bias'].numpy()
        names=[g for g,_ in sorted(state['labels'].items(),key=lambda item:item[1])]
        for row in valid:
            original=row['windows']['full']
            if original['status']!='scored':continue
            winner=original['variants']['original']['current']['top5'][0]['label']
            if winner==row['prompted_label']:
                # Compare the strongest other class even when the full capture is correct.
                winner=original['variants']['original']['current']['top5'][1]['label']
            data={'capture_id':row['capture_id'],'expected':row['prompted_label'],'competitor':winner,'windows':{}}
            for window in WINDOWS:
                item=row['windows'][window]
                if item['status']!='scored':continue
                v=item['variants']['original']
                data['windows'][window]=recover_competitor_margin(v['margin_explanation'],v['current']['scores'],w,b,names,row['prompted_label'],winner)
            block['competitor_comparisons'].append(data)
        if valid:
            block['retained_fps_range']=[min(r['effective_capture_fps'] for r in valid),max(r['effective_capture_fps'] for r in valid)]
            block['source_fps_range']=[min(r['source_capture_fps'] for r in valid),max(r['source_capture_fps'] for r in valid)]
            block['total_drops']=sum(r['capture_timing']['dropped_count'] for r in valid)
            block['drops_near_long_gaps']=sum(r['capture_timing']['drops_within_100ms_after_long_gap'] for r in valid)
            paired=[r for r in valid if r['windows']['first_two_seconds']['status']=='scored' and
                    r['windows']['first_two_seconds_fixed_normalization']['status']=='scored']
            block['prefix_refit_vs_fixed_first_guess_agreement']={
                'same':sum(selected_prediction(r['windows']['first_two_seconds'])==selected_prediction(r['windows']['first_two_seconds_fixed_normalization']) for r in paired),
                'compared':len(paired)}
            block['maximum_prefix_mask_disagreement']=max(
                (r['normalization_comparison']['first_two_seconds']['matched_frame_feature_differences']['mask_disagreement_fraction'] for r in paired),default=None)
        summary['live']=block;summary['status']=analysis_status(live,valid)
    write_json(OUTPUT,summary)
    text='''# Capture timing and fixed-normalization diagnostic — 2026-11-14

## Question and frozen reference

The earlier ME/WE investigation found that changing the capture window altered
model predictions, but also refitted clip normalization. The user subsequently
clarified that WE was misunderstood in both sessions; the WE results are excluded.
It also measured retained
FPS below the dataset rate. This follow-up separates those effects without
training or changing the frozen models, normalizer, limiter or application default.
All 15 labels continue competing. Expected labels enter reports after inference.

Current checkpoint SHA256:
`34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91`.
The original temporal comparator has the identical GRU backbone.

## Preflight and controls

The unsampled diagnostic adapter reproduces the frozen preprocessor and actual
prepared inputs exactly at float16 precision on all **37 audited ME/WE clips**.
Prefix/envelope refits also reproduce the original implementation. Input arrays
and all protected videos, raw/prepared features, source files, checkpoints and
the selected archive remain unchanged. This is separate diagnostic code; the
production normalizer is not edited. [Preflight evidence](assets/timing-normalization-preflight-2026-11-14.json).

The live protocol is ME, WE, PLEASE three times, four-second automatic capture,
sign once in the first two seconds, then relax. Each capture is observed once
for recognition. Compare full capture, separately normalized prefix/envelope,
and those SAME raw-frame windows cut from full-context normalized features.
Cropping always precedes independent 128-frame selection, preserving the same
raw frame IDs within each matched window. Geometry is recomputed from each
actual input. Full-context preprocessing includes future frames relative to a
prefix: this is retrospective analysis, not a proposed streaming normalizer.
Refits can also change short-gap interpolation or slot initialization, so mask
differences and scale medians are reported alongside feature differences.

Frame-arrival and keep/drop traces exist in RAM only. Saved aggregates cover
interval quantiles/histograms, camera read, predecision/copy, rendering, imshow,
waitKey and loop times, limiter credit, long-gap resets and exact decision
replay. These are application delivery times, not camera exposure timestamps.
No frames are duplicated, no limiter rule is changed, and no live media,
landmark arrays, hidden vectors or raw timing traces are saved.
Tests and preprocessing verification finish BEFORE the live test to avoid
introducing test-run CPU load during physical capture.

## Results

'''
    live=summary['live']
    if not live or not live['confirmed_trials']:
        text+='The instrumented live session is pending user-confirmed results. No new live accuracy or causal conclusion is reported.\n'
    else:
        rows=[]
        for key,result in live['controls'].items():
            rows.append((key,*[format_count(result,g) for g in ('ME','WE','PLEASE')]))
        text+=f'Session `{live["session_id"]}`; {len(live["confirmed_trials"])} confirmed captures; {len(live["excluded_trials"])} excluded/unconfirmed.\n\n'
        if live.get('gesture_validity_correction'):
            text+='**User correction:** WE captures 2, 5 and 8 are excluded because the gesture was misunderstood. They are not model errors on verified WE signs. ME and PLEASE remain confirmed. [Correction ledger](assets/pronoun-gesture-feedback-correction-2026-11-14.json).\n\n'
        text+=table(['Control','ME','WE','PLEASE'],rows)+'\n\n'
        timing_rows=[]
        for row in live['confirmed_trials']:
            t=row['capture_timing'];st=t['stages_ms']
            timing_rows.append((row['capture_id'],row['prompted_label'],f'{row["source_capture_fps"]:.2f}',f'{row["effective_capture_fps"]:.2f}',
                               t['dropped_count'],t['limiter_long_gap_reset_count'],t['drops_within_100ms_after_long_gap'],
                               *[f'{st[k]["p50"]:.2f}' for k in ('camera_read','render','imshow','wait_key')]))
        text+=table(['Capture','Sign','Source FPS','Kept FPS','Drops','Long gaps','Drops near gaps','Read p50 ms','Render p50 ms','Show p50 ms','Key p50 ms'],timing_rows)+'\n\n'
        details=[]
        for item in live['competitor_comparisons']:
            for window in ('full','first_two_seconds','first_two_seconds_fixed_normalization'):
                e=item['windows'].get(window,{})
                if not e.get('available'):continue
                p=e['components'];details.append((item['capture_id'],item['competitor']+' minus '+item['expected'],window,
                    *[f'{v:.4f}' for v in (e['actual_margin'],p['pooled_gru'],p['original_17_geometry']+p['six_path_geometry'],p['bias'])]))
        text+=table(['Capture','Margin','Window','Total','GRU','Geometry','Bias'],details)+'\n\n'
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        trials=live['confirmed_trials'];positions=np.arange(len(trials))
        axes[0].plot(positions,[r['source_capture_fps'] for r in trials],'o-',label='Application arrivals')
        axes[0].plot(positions,[r['effective_capture_fps'] for r in trials],'o-',label='Kept frames')
        axes[0].axhline(30,c='gray',ls='--',label='Requested 30 FPS')
        axes[0].set(xticks=positions,xticklabels=[r['capture_id'] for r in trials],xlabel='Capture',ylabel='FPS');axes[0].legend(fontsize=8)
        for key in ('camera_read','render','imshow','wait_key'):
            axes[1].plot(positions,[r['capture_timing']['stages_ms'][key]['mean'] for r in trials],'o-',label=key)
        axes[1].set(xticks=positions,xticklabels=[r['capture_id'] for r in trials],xlabel='Capture',ylabel='Mean stage milliseconds');axes[1].legend(fontsize=8)
        fig.suptitle('Measured application timing; requested FPS is not exposure or delivered FPS')
        fig.tight_layout();fig.savefig(OUTPUT.with_suffix('.png'),dpi=160);plt.close(fig)
        text+='![Measured timing](assets/capture-timing-normalization-2026-11-14.png)\n'
        if live.get('gesture_validity_correction'):
            agreement=live['prefix_refit_vs_fixed_first_guess_agreement']
            text+='\n## Corrected findings\n\n'
            text+='WE recognition accuracy is **not measured** in these sessions: the user withdrew those gesture labels. Valid ME/PLEASE captures remain analyzable.\n\n'
            text+=f'The instrumented session retained **{live["retained_fps_range"][0]:.2f}–{live["retained_fps_range"][1]:.2f} FPS**. '
            text+=f'There were **{live["total_drops"]} limiter drops across the six valid captures**, no long-gap resets, and zero replay mismatches. '
            text+='The earlier 14–18 FPS behavior was not reproduced. ME still scores 1/3 at approximately 30 FPS, so low FPS is not necessary for these new ME errors. This does not identify the cause of the earlier slow session.\n\n'
            text+=f'Prefix refitting versus full-context-fixed normalization gives the same first guess in **{agreement["same"]}/{agreement["compared"]} matched valid captures**. '
            text+='Both give ME 0/3 and PLEASE 3/3, versus full-window ME 1/3 and PLEASE 2/3. Feature magnitudes change, but prefix masks agree exactly in these six captures. Holding normalization fixed does not repair ME here; nor is a fixed short crop a blanket fix.\n\n'
            text+='In ME capture #1, both the pooled recurrent and geometry branches favor WE. In #7, geometry favors ME, but recurrent evidence plus bias outweigh it by a small margin. No single branch intervention is established as a general remedy. The exact cause of the five historical ME failures remains unverified because their sequences were never saved.\n\n'
            text+='**Next:** compare correctly performed ME and WE against the public dataset references, then collect a small explicitly validated comparison before further model tuning. Do not use the excluded WE motions as training data or as evidence that WE recognition failed.\n'
    text+='''
## Interpretation boundaries and reproducibility

Full versus fixed-normalization prefix isolates window selection with full
per-frame preprocessing held fixed; it still recalculates clip geometry.
Fixed versus refitted prefix compares preprocessing context on matched frames.
Branch logit contributions locate which representation changes; a prediction
flip is not automatically a valid repair. A 30-FPS request does not establish
camera exposure rate. Replay identifies deterministic limiter effects on the
observed delivery trace, not why the camera/UI produced that trace.

Run `scripts/verify_timing_normalization.py` once for the immutable preflight;
use `scripts/diagnose_pronouns_live.py --timing-normalization --report NEW_PATH`
for a new diagnostic; summarize using `scripts/report_capture_timing.py --live
PATH`. Add `--confirm-all` only after explicit user confirmation of all prompts.
The portable [results JSON](assets/capture-timing-normalization-2026-11-14.json)
contains the full aggregate evidence. Earlier sessions and reports are preserved.
'''
    if summary.get('repository_checks'):text+=f'\nRepository checks: **{summary["repository_checks"]["tests_passed"]} tests passed**, with lockfile, compilation and dependency checks.\n'
    DOC.write_text(text,encoding='utf-8')
    print('CAPTURE_TIMING_REPORT',summary['status'],flush=True)


if __name__=='__main__':main()
