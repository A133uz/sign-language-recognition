import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import numpy as np
    ROOT=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('pronoun_live_tests',ROOT/'scripts/diagnose_pronouns_live.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class ImmediatePool:
    def __init__(self,*args,**kwargs):pass
    def submit(self,fn,*args):
        try:value=fn(*args);error=None
        except Exception as exc:value=None;error=exc
        def result():
            if error is not None:raise error
            return value
        return SimpleNamespace(done=lambda:True,result=result)
    def shutdown(self,**kwargs):pass


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch unavailable')
class PronounLiveTests(unittest.TestCase):
    def exercise(self,cancel=False,fail_second=False,timing=False,custom=False,pose=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'session.json'
            current=SimpleNamespace(model=SimpleNamespace(),names=['HELLO','NO','PLEASE','THANKYOU','YES','ME','YOU','WE','MY','YOUR','OUR','HELP','GOOD','NEED','LOVE'],checkpoint_sha256='current',
                                    checkpoint=root/'current.pt',hand_model=root/'hand.task')
            baseline=SimpleNamespace(checkpoint_sha256='baseline',checkpoint=root/'baseline.pt')
            camera=SimpleNamespace(isOpened=lambda:True,set=lambda *a:None,read=lambda:(True,np.zeros((8,8,3),np.uint8)))
            camera.release=unittest.mock.Mock()
            preview=SimpleNamespace(draw=lambda frame,stamp:(frame,0),close=unittest.mock.Mock())
            calls=[];counter=[0.];keys=[0]
            def clock():counter[0]+=.25;return counter[0]
            def key(ms):
                keys[0]+=1
                return ord('q') if cancel and keys[0]==8 else 32
            def analyze(a,b,frames,stamps):
                self.assertIs(a,current);self.assertIs(b,baseline)
                self.assertEqual(len(frames),len(stamps));self.assertTrue(all(y>x for x,y in zip(stamps,stamps[1:])))
                calls.append(len(frames))
                if custom or pose:self.assertLessEqual(max(stamps),3000)
                if fail_second and len(calls)==2:raise ValueError('fixture tracking failure')
                return {'windows':{'full':{'status':'scored','variants':{'original':{'current':{'top5':[{'label':'ME','score':.6}]}}}}},
                        'measurements':{'aggregate_only':True},'recognition_tracking_passes':1,
                        'media_saved':False,'coordinates_saved':False,'embeddings_saved':False}
            with patch.object(driver,'check_models'),patch.object(driver,'ThreadPoolExecutor',ImmediatePool),\
                 patch.object(driver,'SkeletonPreview',return_value=preview),patch.object(driver,'diagnose_capture',side_effect=analyze),\
                 patch.object(driver,'diagnose_timing_capture',side_effect=analyze),\
                 patch.object(driver,'diagnose_pose_capture',side_effect=analyze),\
                 patch.object(driver,'sha256',side_effect=lambda p:'current' if p==current.checkpoint else 'baseline'),\
                 patch.object(driver.time,'perf_counter',side_effect=clock),\
                 patch.object(driver.cv2,'VideoCapture',return_value=camera),patch.object(driver.cv2,'imshow'),\
                 patch.object(driver.cv2,'waitKey',side_effect=key),patch.object(driver.cv2,'getWindowProperty',return_value=1),\
                 patch.object(driver.cv2,'destroyAllWindows'):
                options={'prompts':['ME','WE','YOU']*2,'capture_seconds':3.,'sign_seconds':3.} if custom else {}
                if pose:options={'pose_entry_exit':True,'capture_seconds':3.}
                report=driver.run(current,baseline,path,timing_normalization=timing,**options)
            saved=json.loads(path.read_text())
            self.assertEqual(saved,report)
            camera.release.assert_called_once();preview.close.assert_called_once()
            self.assertEqual([p.name for p in root.iterdir()],['session.json'])
            self.assertFalse(saved['media_saved']);self.assertFalse(saved['coordinates_saved']);self.assertFalse(saved['embeddings_saved'])
            self.assertTrue(saved['checkpoints_unchanged']);self.assertIsNone(saved['user_completion_confirmed'])
            self.assertNotIn('accuracy',saved)
            return saved,calls

    def test_nine_trials_save_only_scores_and_aggregates_with_unconfirmed_labels(self):
        report,calls=self.exercise()
        self.assertEqual(report['status'],'complete');self.assertEqual(len(calls),9)
        self.assertEqual([r['prompted_label'] for r in report['trials']],['ME','WE','PLEASE']*3)
        self.assertEqual([r['capture_id'] for r in report['trials']],list(range(1,10)))
        self.assertTrue(all(r['user_confirmation'] is None for r in report['trials']))

    def test_interruption_is_explicit_and_camera_is_released(self):
        report,calls=self.exercise(cancel=True)
        self.assertEqual(report['status'],'partial')
        self.assertEqual(report['trials'][-1]['trial_status'],'interrupted')
        self.assertFalse(report['trials'][-1]['user_confirmation'])

    def test_six_explicit_prompts_have_strict_three_second_automatic_capture(self):
        report,calls=self.exercise(custom=True)
        self.assertEqual(report['status'],'complete');self.assertEqual(len(calls),6)
        self.assertEqual([r['prompted_label'] for r in report['trials']],['ME','WE','YOU']*2)
        self.assertEqual(report['capture_seconds'],3.)
        self.assertEqual(report['sign_during_first_seconds'],3.)
        self.assertTrue(all(r['capture_stop_reason']=='automatic_3_seconds' for r in report['trials']))
        self.assertTrue(all(r['user_confirmation'] is None for r in report['trials']))

    def test_processing_failure_is_kept_and_not_counted_as_a_prediction(self):
        report,calls=self.exercise(fail_second=True)
        self.assertEqual(report['status'],'complete');self.assertEqual(len(report['trials']),9)
        self.assertEqual(report['trials'][1]['trial_status'],'processing_error')
        self.assertNotIn('windows',report['trials'][1])

    def test_pose_comparison_records_conditions_after_inference_without_adding_labels_to_inputs(self):
        report,calls=self.exercise(pose=True)
        self.assertEqual(report['status'],'complete');self.assertEqual(len(calls),8)
        self.assertEqual([r['prompted_label'] for r in report['trials']],['ME','ME','YOU','YOU']*2)
        self.assertEqual([r['condition'] for r in report['trials']],['hold_pose','natural_entry_exit','natural_entry_exit','hold_pose']*2)
        self.assertTrue(all(r['user_confirmation'] is None for r in report['trials']))
        self.assertTrue(all(r['capture_stop_reason']=='automatic_3_seconds' for r in report['trials']))

    def test_timing_mode_keeps_original_limiter_and_exports_only_aggregates(self):
        report,calls=self.exercise(timing=True)
        self.assertEqual(report['status'],'complete');self.assertEqual(len(calls),9)
        self.assertFalse(report['limiter_changed']);self.assertFalse(report['production_preprocessing_changed'])
        for trial in report['trials']:
            timing=trial['capture_timing']
            self.assertEqual(timing['limiter_replay_mismatches'],0)
            self.assertEqual(timing['arrival_count'],trial['source_frame_count'])
            self.assertEqual(timing['kept_count'],trial['source_frame_count']-trial['fps_limit_dropped_frames'])
            self.assertFalse(timing['raw_timing_trace_saved'])
