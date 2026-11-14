import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    spec=importlib.util.spec_from_file_location('pronoun_report_tests',Path(__file__).resolve().parents[1]/'scripts/report_pronoun_investigation.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch unavailable')
class PronounReportTests(unittest.TestCase):
    def test_only_explicit_valid_confirmations_enter_accuracy(self):
        trials=[{'capture_id':1,'user_confirmation':True,'trial_status':'confirmed_valid','windows':{}},
                {'capture_id':2,'user_confirmation':None,'trial_status':'captured_pending_confirmation','windows':{}},
                {'capture_id':3,'user_confirmation':False,'trial_status':'user_marked_invalid','windows':{}},
                {'capture_id':4,'user_confirmation':True,'trial_status':'processing_error'},
                {'capture_id':5,'user_confirmation':'true','trial_status':'confirmed_valid','windows':{}}]
        self.assertEqual([r['capture_id'] for r in report.confirmed_trials({'trials':trials})],[1])

    def test_unavailable_window_is_not_invented_as_a_label_or_hidden_in_counts(self):
        rows=[{'gloss':'ME','window':{'status':'unavailable'}},
              {'gloss':'ME','window':{'status':'scored','variants':{'original':{'current':{'top5':[{'label':'ME'}]}}}}}]
        result=report.counts(rows,lambda r:report.selected_prediction(r['window']),'gloss')
        self.assertEqual(result,{'ME':{'correct':1,'support':2,'unavailable':1}})

    def test_group_controls_use_saved_predictions_without_running_a_model(self):
        window={'status':'scored','variants':{'original':{'current':{'top5':[{'label':'WE'}]},
                'head_interventions':{'all_geometry_off':{'top5':[{'label':'YES'}]}}}}}
        self.assertEqual(report.selected_prediction(window),'WE')
        self.assertEqual(report.selected_prediction(window,model='all_geometry_off'),'YES')

    def test_wrong_gesture_confirmations_are_excluded_and_session_is_accounted_for(self):
        rows=[{'capture_id':i,'prompted_label':('ME','WE','PLEASE')[(i-1)%3],
               'user_confirmation':i not in (2,5,8),
               'trial_status':'user_reported_wrong_gesture' if i in (2,5,8) else 'confirmed_valid','windows':{}}
              for i in range(1,10)]
        session={'status':'complete','trials':rows}
        valid=report.confirmed_trials(session)
        self.assertEqual([r['capture_id'] for r in valid],[1,3,4,6,7,9])
        self.assertEqual(report.analysis_status(session,valid),'complete_with_exclusions')
        self.assertEqual(report.format_count({},'WE'),'excluded')

    def test_feedback_ledger_supersedes_older_all_valid_backup_without_mutating_it(self):
        old={'session_id':'fixture','status':'complete','trials':[
            {'capture_id':i,'prompted_label':('ME','WE','PLEASE')[(i-1)%3],
             'user_confirmation':True,'trial_status':'confirmed_valid','windows':{}} for i in range(1,10)]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'docs/assets';folder.mkdir(parents=True)
            (folder/'pronoun-gesture-feedback-correction-2026-11-14.json').write_text(json.dumps({'sessions':[
                {'session_id':'fixture','excluded_capture_ids':[2,5,8],'valid_capture_ids':[1,3,4,6,7,9]}]}))
            with patch.object(report,'ROOT',root):corrected=report.apply_feedback_corrections(old)
        self.assertEqual(len(report.confirmed_trials(old)),9)
        self.assertEqual(len(report.confirmed_trials(corrected)),6)
        self.assertFalse(corrected['user_completion_confirmed'])
