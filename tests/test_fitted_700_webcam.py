import importlib.util
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import torch
from sign_language.fitted_700_recognition import limit_frames
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('webcam_700_subtitles',ROOT/'scripts/webcam_700_subtitles.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Fitted700WebcamTests(TestCase):
    def test_sequence_limit_matches_notebook_and_keeps_endpoints(self):
        data=np.broadcast_to(np.arange(128,dtype=np.float32)[:,None],(128,254)).copy()
        result=limit_frames(data)
        indices=torch.linspace(0,127,83).round().long().numpy()
        np.testing.assert_array_equal(result,data[indices])
        np.testing.assert_array_equal(limit_frames(data[:35]),data[:35])
        self.assertEqual((result[0,0],result[-1,0]),(0,127))

    def test_subtitle_is_single_bottom_centered_label_without_scores(self):
        image=np.zeros((480,640,3),np.uint8)
        original=module.cv2.putText
        with patch.object(module.cv2,'putText',wraps=original) as draw:
            module.draw_subtitle(image,'THANKYOU')
        self.assertEqual(draw.call_count,1)
        args=draw.call_args.args
        self.assertEqual(args[1],'THANKYOU')
        x,y=args[2]
        width=module.cv2.getTextSize(args[1],args[3],args[4],args[6])[0][0]
        self.assertLess(abs(x+width/2-320),2)
        self.assertGreater(y,400)
