import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import torch
import numpy as np
from test_pronoun_live import ImmediatePool

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_official_webcam_driver_test',ROOT/'scripts/check_official_citizen_webcam.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class OfficialWebcamTests(unittest.TestCase):
    def test_four_guided_captures_are_capped_label_independent_and_persist_no_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'session.json';clock=[0.]
            def tick():clock[0]+=.2;return clock[0]
            baseline=SimpleNamespace(config={'checkpoint_sha256':'fixture','commit':'fixture'},checkpoint=Path('fixture.pt'))
            camera=SimpleNamespace(isOpened=lambda:True,set=lambda *a:None,read=lambda:(True,np.zeros((24,32,3),np.uint8)),release=Mock())
            calls=[]
            def score(model,frames,stamps,pilot):
                self.assertIs(model,baseline);self.assertLessEqual(max(stamps),3000)
                self.assertEqual(len(frames),len(stamps));calls.append(1)
                return {'full_top5':[{'label':'HELLO','score':.5}],'pilot15_top5':[{'label':'HELLO','score':.9}],
                        'media_saved':False,'coordinates_saved':False}
            with patch.object(driver.sys,'argv',['check','--report',str(report)]),\
                 patch.object(driver,'OfficialCitizenI3D',return_value=baseline),\
                 patch.object(driver,'ThreadPoolExecutor',ImmediatePool),patch.object(driver,'score_capture',side_effect=score),\
                 patch.object(driver.time,'perf_counter',side_effect=tick),patch.object(driver,'sha',return_value='fixture'),\
                 patch.object(driver.cv2,'VideoCapture',return_value=camera),patch.object(driver.cv2,'imshow'),\
                 patch.object(driver.cv2,'waitKey',return_value=32),patch.object(driver.cv2,'getWindowProperty',return_value=1),\
                 patch.object(driver.cv2,'destroyAllWindows'):
                driver.main()
            data=json.loads(report.read_text());self.assertEqual(data['status'],'complete')
            self.assertEqual(len(calls),4);self.assertEqual([t['prompted_label'] for t in data['trials']],['HELLO','THANKYOU','YOU','ME'])
            self.assertTrue(all(t['user_confirmation'] is None for t in data['trials']))
            self.assertFalse(data['camera_open']);self.assertFalse(data['landmark_extraction'])
            self.assertTrue(data['checkpoint_unchanged']);camera.release.assert_called_once()
            self.assertEqual([p.name for p in Path(directory).iterdir()],['session.json'])
