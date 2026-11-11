import importlib.util
from pathlib import Path
import pickle
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    spec=importlib.util.spec_from_file_location('forensic_driver',Path(__file__).resolve().parents[1]/'scripts/diagnose_five_sign_live.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class ImmediatePool:
    def __init__(self,*args,**kwargs):pass
    def submit(self,fn,*args):
        result=fn(*args)
        return SimpleNamespace(done=lambda:True,result=lambda:result)
    def shutdown(self,**kwargs):pass


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch extra unavailable')
class GuidedForensicCaptureTests(unittest.TestCase):
    def test_six_automatic_trials_ignore_stop_keys_save_only_json_and_release_camera(self):
        self.exercise_capture(False)

    def test_paired_models_use_shared_capture_and_record_both_checkpoint_identities(self):
        self.exercise_capture(True)

    def test_compact_prompts_measure_six_trials_without_exporting_coordinates(self):
        self.exercise_capture(True,True)

    def exercise_capture(self,paired,compact=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.cache').mkdir();bank=root/'.cache/bank.pkl'
            with bank.open('wb') as f:pickle.dump({'checkpoint_sha256':'fixture'},f)
            report=root/'report.json';clock=[0.];captures=[]
            def tick():clock[0]+=.2;return clock[0]
            class Camera:
                released=False
                def __init__(self,*args):pass
                def isOpened(self):return True
                def set(self,*args):pass
                def read(self):return True,np.zeros((32,32,3),np.uint8)
                def release(self):self.released=True
            camera=Camera()
            def analyze(recognizer,reference,frames,timestamps,expected):
                self.assertGreaterEqual(timestamps[-1],3900)
                self.assertLessEqual(timestamps[-1],4300)
                self.assertTrue(all(a<b for a,b in zip(timestamps,timestamps[1:])))
                captures.append(expected)
                return {'expected':expected,'variants':{'original':{'top5':[{'label':'NO','score':.7}]}}}
            recognizer=SimpleNamespace(epoch=18,checkpoint_sha256='fixture',hand_model=root/'hand.task',checkpoint=root/'best.pt',
                                       labels={'HELLO':0},config={'cache_id':'fixture'})
            candidate=SimpleNamespace(epoch=4,checkpoint_sha256='candidate',checkpoint=root/'candidate.pt',
                                       labels=recognizer.labels,config={'cache_id':'fixture','architecture':'gru-mean-pool-v1'},
                                       model=SimpleNamespace())
            def paired_analyze(recognizer,other,reference,frames,timestamps,expected):
                self.assertIs(other,candidate)
                value=analyze(recognizer,reference,frames,timestamps,expected)
                value['variants']['mean_pool']={'top5':[{'label':'HELLO','score':.8}]}
                return value
            argv=['diagnostic','--bank',str(bank),'--report',str(report)]
            if paired:argv+=['--candidate-checkpoint',str(candidate.checkpoint)]
            if compact:
                reference=root/'reference.json';reference.write_text('{"cache_id":"fixture"}')
                argv+=['--compact-reference',str(reference)]
            def compact_analyze(recognizer,other,bank,frames,timestamps,expected,reference):
                value=paired_analyze(recognizer,other,bank,frames,timestamps,expected)
                value['articulation']={'metrics':{'thumb_index_xy_range':.8,'palm_axis_xy_excursion_deg':20.}}
                return value
            preview=SimpleNamespace(draw=lambda frame,stamp:(frame,0),close=lambda:None)
            with patch.object(driver,'ROOT',root),patch.object(driver,'SignRecognizer',side_effect=[recognizer,candidate] if paired else [recognizer]),\
                 patch.object(driver,'ThreadPoolExecutor',ImmediatePool),patch.object(driver,'SkeletonPreview',return_value=preview),\
                 patch.object(driver,'analyze_capture',side_effect=analyze),patch.object(driver,'compare_pooling_capture',side_effect=paired_analyze),\
                 patch.object(driver,'compare_compact_capture',side_effect=compact_analyze),\
                 patch.object(driver,'sha256',side_effect=lambda path:'candidate' if path==candidate.checkpoint else 'fixture'),\
                 patch.object(driver.time,'perf_counter',side_effect=tick),\
                 patch.object(driver.cv2,'VideoCapture',return_value=camera),patch.object(driver.cv2,'imshow'),\
                 patch.object(driver.cv2,'waitKey',return_value=32),patch.object(driver.cv2,'getWindowProperty',return_value=1),\
                 patch.object(driver.cv2,'destroyAllWindows'),\
                 patch.object(driver.sys,'argv',argv):
                driver.main()
            saved=json.loads(report.read_text())
            self.assertEqual(captures,(['NO','YES','HELLO'] if compact else ['HELLO','THANKYOU','NO'])*2)
            self.assertEqual(saved['status'],'complete');self.assertEqual(len(saved['trials']),6)
            self.assertTrue(camera.released);self.assertFalse(saved['camera_open'])
            self.assertFalse(saved['media_saved']);self.assertFalse(saved['coordinates_saved']);self.assertFalse(saved['embeddings_saved'])
            if paired:
                self.assertEqual(saved['candidate']['checkpoint_sha256'],'candidate')
                self.assertTrue(saved['candidate_checkpoint_unchanged'])
                self.assertEqual(saved['trials'][0]['variants']['mean_pool']['top5'][0]['label'],'HELLO')
            if compact:self.assertEqual(saved['mode'],'compact_hand_articulation')
            self.assertEqual(sorted(p.name for p in root.rglob('*') if p.is_file()),
                             ['bank.pkl','reference.json','report.json'] if compact else ['bank.pkl','report.json'])
