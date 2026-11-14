"""Summarize frozen-model pronoun evidence, never fit a model or persist live inputs."""
import argparse
import copy
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

AUDIT=ROOT/'docs/assets/pronoun-investigation-2026-11-14.json'
SUMMARY=ROOT/'docs/assets/pronoun-diagnostic-summary-2026-11-14.json'
DOC=ROOT/'docs/pronoun-investigation-2026-11-14.md'
REVIEWED=['03554605558492718-ME.mp4','35429744496101856-ME.mp4','11801482805093366-WE.mp4',
          '4372505087382914-ME.mp4','14563849425095032-WE.mp4','5724078705072562-ME.mp4',
          '6868122993341472-WE.mp4','34076966817997323-ME.mp4','36753258402615074-WE.mp4',
          '3468501303620588-WE.mp4','0035172227152260316-ME.mp4','6421087005073287-ME.mp4']


def selected_prediction(window,variant='original',model='current'):
    if window.get('status')!='scored':return None
    v=window['variants'][variant]
    item=v.get(model) or v['head_interventions'][model]
    return item['top5'][0]['label']


def counts(rows,predict,label_field):
    result={}
    for label in sorted({r[label_field] for r in rows}):
        selected=[r for r in rows if r[label_field]==label]
        predictions=[predict(r) for r in selected]
        result[label]={'correct':sum(p==label for p in predictions),'support':len(selected),
                       'unavailable':sum(p is None for p in predictions)}
    return result


def confirmed_trials(report):
    return [r for r in report.get('trials',[]) if r.get('user_confirmation') is True
            and r.get('trial_status')=='confirmed_valid' and 'windows' in r]


def apply_feedback_corrections(report):
    """Authoritative user corrections also supersede older report backups."""
    path=ROOT/'docs/assets/pronoun-gesture-feedback-correction-2026-11-14.json'
    if not path.exists():return report
    ledger=json.loads(path.read_text())
    entry=next((r for r in ledger['sessions'] if r['session_id']==report['session_id']),None)
    if entry is None:return report
    corrected=copy.deepcopy(report)
    for row in corrected['trials']:
        if row['capture_id'] in entry['excluded_capture_ids']:
            if row['prompted_label']!='WE':raise ValueError('Correction ledger and session prompts differ')
            row.update(user_confirmation=False,trial_status='user_reported_wrong_gesture',
                       exclusion_reason='User confirmed misunderstanding WE in both sessions; actual performed sign is unverified.')
        elif row['capture_id'] in entry['valid_capture_ids'] and row['trial_status'] in ('captured_pending_confirmation','confirmed_valid'):
            row.update(user_confirmation=True,trial_status='confirmed_valid')
    corrected.update(user_completion_confirmed=False,capture_protocol_confirmed=True,
        correction_ledger_sha256=sha256(path),
        gesture_validity_correction={'user_confirmed_scope':'Both sessions','excluded_prompt':'WE',
            'excluded_capture_ids':entry['excluded_capture_ids'],'valid_capture_ids':entry['valid_capture_ids'],
            'reason':'User-reported misunderstanding of WE; no WE recognition accuracy claim.'})
    return corrected


def format_count(result,label):
    item=result.get(label)
    if not item or not item['support']:return 'excluded'
    text=f'{item["correct"]}/{item["support"]}'
    return text+(f' ({item["unavailable"]} unavailable)' if item.get('unavailable') else '')


def analysis_status(report,valid):
    if len(valid)==9:return 'complete'
    accounted=all(r.get('user_confirmation') is not None or r.get('trial_status')=='processing_error' for r in report.get('trials',[]))
    if report.get('status')=='complete' and len(report['trials'])==9 and accounted and valid:return 'complete_with_exclusions'
    return 'live_partial_or_unconfirmed'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def component_plot(rows,title,path):
    names=('bias','pooled_gru','original_17_geometry','six_path_geometry')
    colors=('#64748b','#2563eb','#0d9488','#d97706')
    fig,ax=plt.subplots(figsize=(12,max(4.,.6*len(rows))))
    y=np.arange(len(rows));positive=np.zeros(len(rows));negative=np.zeros(len(rows))
    for name,color in zip(names,colors):
        values=np.array([r[1]['components'][name] for r in rows]);left=np.where(values>=0,positive,negative)
        ax.barh(y,values,left=left,label=name,color=color)
        positive+=np.maximum(values,0);negative+=np.minimum(values,0)
    ax.scatter([r[1]['actual_margin'] for r in rows],y,c='black',marker='x',label='actual margin',zorder=4)
    ax.axvline(0,c='black',lw=.8);ax.set(yticks=y,yticklabels=[r[0] for r in rows],xlabel='WE minus ME logit contribution; positive favors WE',title=title)
    ax.invert_yaxis();ax.legend(fontsize=8,ncols=3,loc='upper center',bbox_to_anchor=(.5,-.14))
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',type=Path)
    parser.add_argument('--confirm-all',action='store_true',help='Use only after the user explicitly confirms following all prompts normally')
    args=parser.parse_args()
    audit=json.loads(AUDIT.read_text())
    if audit['status']!='complete':raise ValueError('Audit must finish first')
    protected_ok=all(sha256(ROOT/p)==h for p,h in audit['protected_sha256'].items())
    if not protected_ok:raise AssertionError('Audited source/checkpoint bytes changed')
    rows=audit['records'];summary={'status':'offline_complete_live_pending','audit_sha256':sha256(AUDIT),
        'date_local':'2026-11-14','checkpoint_sha256':audit['checkpoint_sha256'],
        'counts':audit['counts'],'protected_inputs_unchanged':True,
        'full_validation_reproduction':audit['full_validation_reproduction'],
        'visual_review':{'contact_sheets_reviewed':REVIEWED,'source_media_kept_local':True,
           'observations':['ME examples show an approach/chest-pointing hold; WE examples include lateral movement across the upper torso. These are visual observations, not certified semantic annotations.',
                           'In ME P30 and P36, sampled frames show a visible hand at the chest without a skeleton overlay. Presence flags therefore cannot be treated as accurate sign segmentation.',
                           'The WE P2 sheet covers a short tracked interval; the rest of its 117 decoded frames also need to be considered when interpreting its 14 detected-hand frames.',
                           'Hand points are compact/partly occluded near the chest in multiple clips; the front-view sheets do not establish true depth or anatomical accuracy.']},
        'offline_controls':{},'review_flags':{},'live':None,'training_performed':False,'official_test_read':False,
        'production_changed':False,'live_media_saved':False,'live_coordinates_saved':False,'live_embeddings_saved':False}
    for split in ('train','val'):
        group=[r for r in rows if r['split']==split];controls={}
        for variant in rows[0]['windows']['full']['variants']:
            for model in ('current','original_temporal_model'):
                controls[variant+'/'+model]=counts(group,lambda r,v=variant,m=model:selected_prediction(r['windows']['full'],v,m),'gloss')
        for head in ('original_geometry_off','path_geometry_off','all_geometry_off'):
            controls[head]=counts(group,lambda r,h=head:selected_prediction(r['windows']['full'],model=h),'gloss')
        for window in ('first_two_seconds','detected_hand_envelope'):
            controls[window]=counts(group,lambda r,w=window:selected_prediction(r['windows'][w]),'gloss')
        summary['offline_controls'][split]=controls
    flags=sorted({flag for r in rows for flag in r['review_flags']})
    summary['review_flags']={f:sum(f in r['review_flags'] for r in rows) for f in flags}
    summary['dataset_fps_range']=[min(r['video']['fps'] for r in rows),max(r['video']['fps'] for r in rows)]
    val=[r for r in rows if r['split']=='val']
    component_plot([(f'{r["gloss"]} {r["participant_id"]} {r["video_file"][:7]}',
                    r['windows']['full']['variants']['original']['margin_explanation']) for r in val],
                   'Offline validation: exact frozen-head contributions',SUMMARY.with_name('pronoun-offline-margins-2026-11-14.png'))
    summary['offline_validation_margins']=[{'video_file':r['video_file'],'gloss':r['gloss'],'participant_id':r['participant_id'],
        **r['windows']['full']['variants']['original']['margin_explanation']} for r in val]
    live=None
    if args.live:
        live=apply_feedback_corrections(json.loads(args.live.read_text()))
        if live['checkpoint_sha256']!=audit['checkpoint_sha256'] or not live.get('checkpoints_unchanged'):
            raise ValueError('Live report uses different or modified weights, or session has not closed')
        if args.confirm_all:
            if any(r.get('trial_status')=='user_reported_wrong_gesture' for r in live['trials']):
                raise ValueError('Known wrong-gesture trials cannot be confirmed as valid')
            if live['status']!='complete' or len(live['trials'])!=9:raise ValueError('Cannot confirm an incomplete session')
            for trial in live['trials']:
                if trial['trial_status']=='captured_pending_confirmation':
                    trial.update(user_confirmation=True,trial_status='confirmed_valid')
            live['user_completion_confirmed']=True
            live['confirmation_source']='User explicitly confirmed following all nine prompts, signing during first two seconds with normal capture.'
            write_json(args.live,live)
        valid=confirmed_trials(live)
        summary['live']={'session_id':live['session_id'],'source_sha256':sha256(args.live),'status':live['status'],
                         'user_completion_confirmed':live['user_completion_confirmed'],'confirmed_trials':valid,
                         'excluded_trials':[{'capture_id':r['capture_id'],'status':r['trial_status'],'reason':r.get('exclusion_reason')} for r in live['trials'] if r not in valid],
                         'gesture_validity_correction':live.get('gesture_validity_correction'),
                         'correction_ledger_sha256':live.get('correction_ledger_sha256'),
                         'controls':{}}
        for window in ('full','first_two_seconds','detected_hand_envelope'):
            for model in ('current','original_temporal_model','original_geometry_off','path_geometry_off','all_geometry_off'):
                summary['live']['controls'][window+'/'+model]=counts(valid,lambda r,w=window,m=model:selected_prediction(r['windows'][w],model=m),'prompted_label')
        for variant in ('reverse_time','shuffle_time','static_active_frame','reflect_anatomy','swap_hand_slots'):
            summary['live']['controls'][variant]=counts(valid,lambda r,v=variant:selected_prediction(r['windows']['full'],v),'prompted_label')
        usable=[r for r in valid if r['windows']['full']['status']=='scored']
        state=torch.load(ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt',map_location='cpu',weights_only=True)
        weights=state['model']['classifier.weight'].numpy();bias=state['model']['classifier.bias'].numpy()
        names=[g for g,_ in sorted(state['labels'].items(),key=lambda item:item[1])]
        competition=[]
        for trial in usable:
            full=trial['windows']['full']['variants']['original']['current']
            target=trial['prompted_label'];competitor=full['top5'][0]['label']
            if target==competitor:continue
            item={'capture_id':trial['capture_id'],'expected':target,'full_winner':competitor,'windows':{}}
            for name,window in trial['windows'].items():
                if window['status']!='scored':continue
                v=window['variants']['original']
                item['windows'][name]=recover_competitor_margin(v['margin_explanation'],v['current']['scores'],weights,bias,names,target,competitor)
            competition.append(item)
        summary['live']['actual_competitor_explanations']=competition
        summary['live']['effective_fps_range']=[min(r['effective_capture_fps'] for r in valid),max(r['effective_capture_fps'] for r in valid)] if valid else None
        summary['live']['source_fps_range']=[min(r['source_capture_fps'] for r in valid),max(r['source_capture_fps'] for r in valid)] if valid else None
        summary['live']['additional_dropped_frames']=[r['fps_limit_dropped_frames'] for r in valid]
        if usable:
            component_plot([(f'#{r["capture_id"]} {r["prompted_label"]}',r['windows']['full']['variants']['original']['margin_explanation']) for r in usable],
                           'Confirmed live captures: frozen-head contributions',SUMMARY.with_name('pronoun-live-margins-2026-11-14.png'))
        summary['status']=analysis_status(live,valid)
        if live.get('gesture_validity_correction'):
            summary['withdrawn_findings']={'reason':'User clarified WE was misunderstood in both sessions.',
                'scope':'WE captures 2,5,8 excluded; old WE accuracy and claimed WE recognition-failure mechanism are withdrawn.',
                'correction_ledger':'docs/assets/pronoun-gesture-feedback-correction-2026-11-14.json'}
        reviewed_findings={
            'demonstrated':[
                'The unchanged model recognizes all three confirmed ME captures in the new bounded protocol; the old five ME failures are not reproduced.',
                'Full-window WE is 1/3; the same observations restricted to the first two seconds yield WE 3/3. Both the current and old temporal heads show this window dependence.',
                'The actual failed WE competitors are NEED (#2) and YES (#8), not ME. Their WE-minus-ME margins already favor WE.',
                'Reconstructed competitor margins identify the largest prefix-to-full change in the pooled recurrent branch; current geometry is not the sole cause.',
                'A fixed two-second crop regresses one ME and one PLEASE attempt. Removing six path features regresses PLEASE 3/3 to 1/3. No tested blanket intervention fixes all nine captures.',
                'Actual retained capture rate is 13.92-17.55 FPS versus 29.73-30.97 FPS for the audited dataset. Camera-loop delivery is 22.20-25.85 FPS and the limiter drops additional frames.'],
            'supported_hypotheses':[
                'Capture context/normalization and sequence aggregation contribute to the remaining live WE failures.',
                'Sampling cadence and tracking/rest patterns are plausible live-versus-dataset mismatches; their independent causal contribution is not established.'],
            'unresolved':[
                'The exact cause of the five historical ME failures cannot be recovered from count-only logs.',
                'Prefix/full comparisons also refit per-clip normalization; they do not isolate later-frame aggregation from normalization changes.',
                'No saved arrival-time trace exists to determine why this session delivered fewer than 30 FPS and why the limiter discarded additional frames.',
                'Correct anatomical finger depth/hand-slot identity is not established by presence counts or destructive swapping.'],
            'smallest_next_experiment':'Instrument per-arrival/keep/drop interval aggregates and normalize-once prefix controls in a separate memory-only diagnostic. First isolate sampling and clip-normalization effects; then compare a motion-aware segmentation/pooling candidate with consistent train/live preparation. Do not deploy fixed crops, label overrides, mirroring or geometry removal from these nine trials.'}
        if live['session_id']=='pronoun-live-20261114-115103' and len(valid)==9:
            summary['findings']=reviewed_findings
    if args.confirm_all and args.live is None:raise ValueError('--confirm-all requires --live')
    check_log=ROOT/'.cache/pronoun-final-checks-20261114.log'
    if check_log.exists():
        checks=check_log.read_text(encoding='utf-8',errors='replace')
        match=re.search(r'Ran (\d+) tests',checks)
        if match and '\nOK\n' in checks and 'Import and unit checks do not validate' in checks:
            summary['repository_checks']={'tests_passed':int(match.group(1)),'log_sha256':sha256(check_log),
                                          'lock_compilation_dependency_checks_passed':True}
    write_json(SUMMARY,summary)
    control_rows=[]
    for key,value in summary['offline_controls']['val'].items():
        control_rows.append((key,*[f'{value[g]["correct"]}/{value[g]["support"]}' for g in ('ME','WE')]))
    error=next(r for r in val if r['saved_prediction']!=r['gloss'])
    margin=error['windows']['full']['variants']['original']['margin_explanation']
    text='''# ME/WE recognition investigation — 2026-11-14

## Protocol and preservation

This implements the agreed diagnosis-only investigation of five user-confirmed
ME-to-WE errors. No model was trained, replaced or made default. No source
videos/features were deleted or relabeled; the stopped corpus audit was not
resumed. The official test split was not read. Existing archive-tool changes
and archives remain preserved.

The frozen current checkpoint SHA256 is
`34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91`.
The original temporal comparator SHA256 is
`0921e10a193044e5cbc92e29b386b370b827431b6850f0bc49f6e64ea4441843`.
Their recurrent weights are identical; heads differ. All 15 labels compete.

The audit fully decoded all 37 ME/WE videos (train ME 13, WE 16; validation ME
5, WE 3), verified raw archive hashes/member CRCs, labels/indices, dimensions,
FPS and source sizes, and reproduced every prepared input exactly at float16
precision. All 60 stored validation score vectors reproduce exactly: 54/60
correct, maximum difference 0. Audited video/cache/source/checkpoint/archive
hashes are unchanged after the experiments. Detailed hashes and per-clip data:
[audit JSON](assets/pronoun-investigation-2026-11-14.json).

## Controls and visual review

On each sequence the same 15-class head is explained as four additive terms:
classifier bias, 256 pooled GRU values, 17 original geometry values and six path
values. The WE-minus-ME margin is reconstructed within numerical tolerance.
Three head interventions set standardized feature groups to zero (training
means), without refitting. Reverse, seeded shuffle, repeated middle tracked
frame, anatomical reflection and slot swap all recalculate geometry. Temporal
windows slice the raw observations before normalization and sampling. A changed
guess measures sensitivity; it does not certify a valid preprocessing fix.

Twelve annotated contact sheets were reviewed: all eight validation examples,
both misclassified WE training examples and ME P30/P36 low-visibility examples.
All 37 public-video contact sheets remain local under
`.cache/pronoun-audit-20261114-v1/`; no source images enter Git.
The ME P30/P36 sheets show visible hands near the chest without landmarks in
sampled frames. This is direct evidence of missed tracking in those samples,
not a corpus error rate. These clips can nevertheless be classified correctly.
Front-view occlusion prevents claiming true finger depth/anatomical accuracy.

Offline validation controls (independent transformations, not repairs):

'''+table(['Control','ME','WE'],control_rows)+f'''

The failed ME validation video `{error['video_file']}` has WE-minus-ME margin
**{margin['actual_margin']:.6f}**. Components:

'''+table(['Term','Contribution'],[(k,f'{v:.6f}') for k,v in margin['components'].items()])+'''

The six added path descriptors are not necessary for this particular failure:
turning them off leaves WE first. The original geometry favors ME here; the
bias and pooled representation together favor WE. This is an exact decision
explanation for one dataset recording, not yet a cause of the historical live
failures. The geometry summaries do not explicitly encode absolute pointing
direction, but ordered body-relative positions already exist in the GRU input.
Do not claim the input lacks motion or body location.

![Offline margin components](assets/pronoun-offline-margins-2026-11-14.png)

## Live comparison

'''
    if summary['live'] is None or not summary['live']['confirmed_trials']:
        text+='The nine-trial session is pending or not yet user-confirmed. No new live accuracy or live-cause conclusion is reported.\n'
    else:
        ls=summary['live'];live_rows=[]
        for key,value in ls['controls'].items():
            live_rows.append((key,*[format_count(value,g) for g in ('ME','WE','PLEASE')]))
        text+=f'Session `{ls["session_id"]}`: {len(ls["confirmed_trials"])} user-confirmed captures. Excluded/unconfirmed: {len(ls["excluded_trials"])}.\n\n'
        if ls.get('gesture_validity_correction'):
            text+='**User correction:** WE was misunderstood in both nine-trial sessions. Captures 2, 5 and 8 are excluded; their actual performed sign is unverified. Earlier WE 1/3 and prefix WE 3/3 claims, and the claimed WE recognition-failure mechanism, are withdrawn. The model response changes remain recorded observations on mislabeled motions, not WE accuracy evidence. [Correction ledger](assets/pronoun-gesture-feedback-correction-2026-11-14.json).\n\n'
        text+=table(['Control','ME','WE','PLEASE'],live_rows)+'\n\n'
        text+='![Live margin components](assets/pronoun-live-margins-2026-11-14.png)\n'
        if ls['actual_competitor_explanations']:
            details=[]
            for row in ls['actual_competitor_explanations']:
                for window,explanation in row['windows'].items():
                    if not explanation.get('available'):continue
                    parts=explanation['components']
                    details.append((row['capture_id'],row['full_winner']+' minus '+row['expected'],window,
                                    f'{explanation["actual_margin"]:.4f}',*[f'{parts[k]:.4f}' for k in
                                    ('bias','pooled_gru','original_17_geometry','six_path_geometry')]))
            text+='\n### Explain the actual competing classes\n\n'
            text+=table(['Capture','Margin','Window','Total','Bias','GRU','Hand geometry','Path geometry'],details)+'\n\n'
            text+='''The recorded 23 geometry contributions and known frozen coefficients
allow recovering the standardized **clip summaries**, without recovering any
coordinates or hidden vectors. Competitor-minus-WE total margins come from
log probability ratios; the aggregate GRU term is the residual after subtracting
geometry and bias. This derivation is unit-tested against direct linear output;
unidentifiable zero-coefficient cases are explicitly unavailable.
'''
        if summary.get('findings'):
            text+='''
Both failed WE captures already favor WE over ME. Looking only at ME/WE would
miss the actual NEED/YES competition. In #2, prefix-to-full GRU contribution
shifts from -1.7354 to +1.4386 for NEED-minus-WE; in #8 it shifts from -4.7228
to -0.4237 for YES-minus-WE. These are the largest changes in their respective
total margins. Bias is constant. The #8 geometry/bias terms then outweigh the
remaining GRU preference for WE.

### Capture cadence is an additional measured mismatch

'''
            text+=f'Dataset FPS: **{summary["dataset_fps_range"][0]:.2f}–{summary["dataset_fps_range"][1]:.2f}**. '
            text+=f'Live retained FPS: **{ls["effective_fps_range"][0]:.2f}–{ls["effective_fps_range"][1]:.2f}**; '
            text+=f'camera-loop delivery: **{ls["source_fps_range"][0]:.2f}–{ls["source_fps_range"][1]:.2f}**. '
            text+='The limiter discarded 27–48 additional frames per capture. This session did not maintain a 30-FPS input grid. No repeated/synthetic frames were introduced. Arrival-time traces were not retained, so the cause of delivery/limiter losses is not established. Fixed-step GRU/path inputs therefore also face a cadence confound.\n'
    text+='''
The protocol is ME, WE, PLEASE three times; four-second automatic captures,
sign once during the first two seconds, then relax. Each capture is tracked
once for recognition; full, first-two-second and detected-hand-envelope views
share those observations. The preview has its own transient tracker. Prefix
and envelope are diagnostic windows, not verified sign boundaries. Prompt
labels are attached after inference and require explicit user confirmation.
Interrupted, failed or user-marked-invalid trials are retained separately.
Only scores, margin contributions, timings and aggregate measurements are
saved—no live video, stills, joint arrays or embeddings.

## Findings and next decision

Established offline: source/cache integrity checks pass; the frozen model's
saved predictions reproduce; sampled detector misses exist; some decisions
change under sequence/slot/orientation controls; classifier terms explain the
failed ME validation clip exactly. None establishes one universal cause of the
five historical live failures, whose coordinates were never saved.

'''
    if summary.get('findings'):
        text+='The unchanged model gets ME 3/3 and PLEASE 3/3 in the new protocol; WE remains 1/3. The confirmed mechanism is capture-window sensitivity in the end-to-end frozen pipeline, with the largest score changes in the pooled recurrent contribution for the two failed WE trials. This is not proof that the last frames alone, the GRU architecture, or one detector defect caused every historical failure. Prefix crops also change clip-fitted normalization.\n\n'
        text+='A fixed two-second crop loses one ME and one PLEASE. Removing path geometry loses two PLEASE captures. Neither is a supported blanket fix, and mirroring/slot swapping also regress WE.\n\n'
        text+='**Smallest next experiment:** '+summary['findings']['smallest_next_experiment']+'\n\n'
        text+='The old five ME failures cannot be replayed from count-only logs. Their exact cause remains unresolved; the new nine trials are not a controlled before/after comparison with that historical session. No new correction is trained or deployed here.\n\n'
    elif summary.get('withdrawn_findings'):
        text+='WE recognition conclusions from the live session are withdrawn after the user corrected the gesture labels. ME and PLEASE remain valid (3/3 each in this earlier session). The five historical ME failures and the new instrumented ME trials remain separate evidence. A correctly performed, confirmed WE test is needed before assessing live WE accuracy.\n\n'
    else:
        text+='Live-specific conclusions and the smallest justified correction await confirmed live trials.\n\n'
    if summary.get('repository_checks'):
        text+=f'## Software verification\n\n**{summary["repository_checks"]["tests_passed"]} tests passed**, along with lockfile consistency, source compilation and dependency imports. Tests cover numerical margin reconstruction, competitor recovery without embeddings, feature-group boundaries, padding, transformed geometry, timestamps/gaps, raw-before-sampling windows, single-pass recognition tracking, nine-trial lifecycle/errors/interruption, explicit label confirmation and read-only notebook rendering. These checks do not establish general live accuracy.\n\n'
    text+='''Runners: `scripts/audit_pronoun_recognition.py`,
`scripts/diagnose_pronouns_live.py`, `scripts/report_pronoun_investigation.py`.
The audit and live runner preserve prior outputs. The report generator refreshes
its own summary/plots; `--live PATH --confirm-all` may be used only after the
user confirms all nine prompts were followed normally. The portable
[summary](assets/pronoun-diagnostic-summary-2026-11-14.json) feeds the reporting
notebooks without opening a camera or executing model inference.
'''
    DOC.write_text(text,encoding='utf-8')
    print('PRONOUN_REPORT',summary['status'],str(DOC),flush=True)


if __name__=='__main__':main()
