import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pose_report_tests',ROOT/'scripts/report_pose_entry_exit.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class PoseReportTests(unittest.TestCase):
    def test_unscorable_confirmed_capture_is_separate_from_misclassification_and_exclusions(self):
        score={'top5':[{'label':'ME','score':.8}],'scores':[.8,.2]}
        correct={'status':'scored','variants':{'original':{'current':score,'original_temporal_model':score,'head_interventions':{}}}}
        base={'prompted_label':'ME','user_confirmed_label':'ME','condition':'hold_pose','repetition':1,
              'user_confirmation':True,'trial_status':'confirmed_valid'}
        live={'classes':['ME','YOU'],'trials':[
            {**base,'capture_id':1,'windows':{'full':{'status':'unavailable','reason':'No detections'}}},
            {**base,'capture_id':2,'windows':{'full':correct}},
            {**base,'capture_id':3,'user_confirmation':False,'trial_status':'user_marked_invalid','windows':{'full':correct}},
            {**base,'capture_id':4,'user_confirmation':None,'trial_status':'captured_pending_confirmation','windows':{'full':correct}}]}
        value=report.summarize(live,np.zeros((2,279)),np.zeros(2))
        self.assertEqual(value['counts']['ME/hold_pose'],{'confirmed_captures':2,'scored_full':1,'correct_full':1,'unavailable_full':1})
        self.assertEqual([r['capture_id'] for r in value['rows']],[1,2])
        self.assertNotIn('expected_score',value['rows'][0])
        self.assertEqual(value['rows'][1]['expected_score'],.8)
        json.dumps(value,allow_nan=False)
