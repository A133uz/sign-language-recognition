import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from sign_language.landmark_still_review import StillReviewRecognizer
    from sign_language.recognition_webcam import _write_report


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch extra not installed')
class StillReviewTests(unittest.TestCase):
    def test_exports_only_two_stills_no_video_coordinates_or_input_mutation(self):
        pose=np.zeros((12,33,3),np.float32);pose[:,11,:2]=[.4,.4];pose[:,12,:2]=[.6,.4]
        hands=np.zeros((12,2,21,3),np.float32);hands[:,0,:,0]=np.linspace(.2,.4,21)
        hands[:,0,:,1]=np.linspace(.3,.5,21)
        arrays={'pose_xyz':pose,'pose_valid_mask':np.ones((12,33),bool),'hands_xyz':hands,
                'hand_present':np.tile([True,False],(12,1)),
                'handedness':np.tile([0,-1],(12,1)),'handedness_score':np.tile([1.,0.],(12,1))}
        frames=[np.full((120,160,3),i,np.uint8) for i in range(12)]
        before_frames=[f.copy() for f in frames];before_arrays={k:v.copy() for k,v in arrays.items()}
        base=SimpleNamespace(checkpoint_sha256='fixture',observe_frames=lambda f,t:(arrays,(120,160,3),{'captured_frames':12}),
                             predict_observations=lambda a,s:{'top5':[{'label':'HELLO','score':.8}]})
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'review';review=StillReviewRecognizer(base,output)
            result=review.predict_frames(frames,[i*100 for i in range(12)])
            self.assertEqual(len(list(output.glob('*.png'))),2)
            self.assertEqual(result['top5'][0]['label'],'HELLO')
            metadata=json.loads((output/'review.json').read_text())
            self.assertEqual([r['frame_index'] for r in metadata['images']],[10,11])
            self.assertFalse(metadata['video_saved']);self.assertFalse(metadata['coordinates_saved'])
            self.assertFalse(any(output.glob('*.npz')));self.assertFalse(any(output.glob('*.mp4')))
            pixels={p.name:p.read_bytes() for p in output.glob('*.png')}
            with self.assertRaises(ValueError):review.predict_frames(frames,list(range(12)))
            self.assertEqual(pixels,{p.name:p.read_bytes() for p in output.glob('*.png')})
            report={'media_saved':False,'coordinates_saved':False}
            _write_report(output/'session.json',report,review)
            self.assertTrue(report['media_saved']);self.assertFalse(report['video_saved'])
            self.assertEqual(len(report['saved_still_images']),2)
            with self.assertRaises(FileExistsError):StillReviewRecognizer(base,output)
        for a,b in zip(frames,before_frames,strict=True):np.testing.assert_array_equal(a,b)
        for k in arrays:np.testing.assert_array_equal(arrays[k],before_arrays[k])


if __name__=='__main__':unittest.main()
