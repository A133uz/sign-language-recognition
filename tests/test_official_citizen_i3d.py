import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
import numpy as np
from sign_language.official_citizen_i3d import aggregate_frame_logits,source_module,preprocess_captured_frames

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tools/asl-citizen-code-official/I3D/aslcitizen_dataset.py'


class OfficialI3DTests(unittest.TestCase):
    def test_readout_matches_upstream_interpolate_then_max_not_temporal_average(self):
        logits=torch.tensor([[[0.,8.,0.],[4.,4.,4.]]])
        expected=F.interpolate(logits,size=64,mode='linear',align_corners=False).max(2).values
        actual=aggregate_frame_logits(logits)
        torch.testing.assert_close(actual,expected)
        self.assertGreater(float(actual[0,0]),float(actual[0,1]))
        self.assertEqual(actual.shape,(1,2))

    def test_restricted_output_view_is_explicit_and_does_not_change_original_logits(self):
        report_module=source_module('_official_evaluator_test',ROOT/'scripts/evaluate_official_citizen_i3d.py')
        z=np.array([[3.,1.,9.],[1.,3.,9.]],np.float32);before=z.copy()
        rows=[{'gloss':'A','video_file':'a.mp4'},{'gloss':'B','video_file':'b.mp4'}]
        result=report_module.restricted_report(z,rows,['A','B'],['A','B','OTHER'])
        self.assertTrue((z.argmax(1)==2).all())
        self.assertEqual(result['metrics']['top1'],1.)
        self.assertEqual(result['confusion'],[[1,0],[0,1]])
        np.testing.assert_array_equal(z,before)

    @unittest.skipUnless(SOURCE.exists(),'Official source is installed by the optional baseline setup')
    def test_upstream_frameskip_boundaries_and_opencv_channel_order_are_preserved(self):
        module=source_module('_official_dataset_test',SOURCE)
        for total,count,first,last in [(95,64,15,78),(96,48,0,94),(159,64,15,141),(160,54,0,159)]:
            state={'i':0}
            def read():
                i=state['i'];state['i']+=1
                return True,np.full((256,256,3),[i%256,55,200],np.uint8)
            cap=SimpleNamespace(get=lambda key:total,set=lambda key,value:state.update(i=int(value)),read=read)
            with self.subTest(total=total),patch.object(module.cv2,'VideoCapture',return_value=cap):
                frames=module.load_rgb_frames_from_video('fixture.mp4')
            self.assertEqual(len(frames),count)
            np.testing.assert_allclose(frames[0,0,0],[first/255*2-1,55/255*2-1,200/255*2-1],atol=1.e-6)
            self.assertAlmostEqual(float(frames[-1,0,0,0]),last/255*2-1,places=6)

    @unittest.skipUnless(SOURCE.exists(),'Optional official baseline source unavailable')
    def test_upstream_short_clip_padding_appends_original_endpoint_frames(self):
        module=source_module('_official_dataset_pad_test',SOURCE)
        frames=np.arange(10,dtype=np.float32).reshape(10,1,1,1)
        for random_value,expected_tail in [(.8,0.),(.2,9.)]:
            with patch.object(module.np.random,'random_sample',return_value=random_value):
                actual=module.ASLCitizen.pad(None,frames,64)
            np.testing.assert_array_equal(actual[:10],frames)
            self.assertTrue((actual[10:]==expected_tail).all())

    @unittest.skipUnless(SOURCE.exists(),'Optional official baseline source unavailable')
    def test_ram_webcam_preprocessing_exactly_matches_upstream_loader_at_sampling_boundaries(self):
        dataset=source_module('_official_ram_dataset_test',SOURCE)
        transforms=source_module('_official_ram_transforms_test',SOURCE.with_name('videotransforms.py'))
        for count in (20,95,96,159,160):
            frames=[np.full((256,256,3),[i%256,55,200],np.uint8) for i in range(count)]
            state={'i':0}
            def read():
                i=state['i'];state['i']+=1;return True,frames[i]
            cap=SimpleNamespace(get=lambda key:count,set=lambda key,value:state.update(i=int(value)),read=read)
            with patch.object(dataset.cv2,'VideoCapture',return_value=cap):expected=dataset.load_rgb_frames_from_video('fixture.mp4')
            np.random.seed(0);expected=dataset.ASLCitizen.pad(None,expected,64)
            expected=dataset.video_to_tensor(transforms.CenterCrop(224)(expected))
            actual,metadata=preprocess_captured_frames(frames,dataset,transforms,seed=0)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertEqual(tuple(actual.shape),(3,64,224,224));self.assertEqual(metadata['captured_frames'],count)
