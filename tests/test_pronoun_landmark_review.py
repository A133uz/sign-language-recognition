import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from urllib.request import urlopen,Request
from urllib.error import HTTPError

import torch
import numpy as np
from sign_language.pronoun_landmark_review import MemoryBoards,PronounReviewRecognizer,hand_shape_summary,canonical
from test_capture_timing_diagnostics import observations


class PronounReviewTests(unittest.TestCase):
    def test_bounded_preview_is_ram_only_loopback_noncacheable_and_can_be_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            store=MemoryBoards(limit=1)
            try:
                self.assertIn('127.0.0.1',store.url)
                store.add(np.zeros((8,8,3),np.uint8))
                with urlopen(store.url+'0') as response:
                    self.assertEqual(response.headers['Cache-Control'],'no-store')
                    self.assertTrue(response.read().startswith(b'\x89PNG'))
                with self.assertRaises(ValueError):store.add(np.zeros((8,8,3),np.uint8))
                with self.assertRaises(HTTPError):urlopen(store.url.split('/')[0]+'//'+store.url.split('/')[2]+'/wrong/0')
                with urlopen(Request(store.url+'clear',method='POST')) as response:self.assertEqual(response.status,204)
                self.assertEqual(store.images,[]);self.assertTrue(store.cleared.is_set())
                self.assertEqual(list(Path(directory).iterdir()),[])
            finally:store.close()

    def test_exact_production_prediction_has_no_geometry_intervention_and_tracks_once(self):
        arrays=observations(60);before={k:v.copy() for k,v in arrays.items()}
        frames=[np.zeros((120,160,3),np.uint8) for _ in range(60)];stamps=np.arange(60)*50
        prediction={'top5':[{'label':'YES','score':.8}],
                    'quality':{'frames':60,'frames_with_hands':60,'frames_with_body_anchor':60},'input_frames':60}
        base=SimpleNamespace(observe_frames=Mock(return_value=(arrays,(120,160,3),{'captured_frames':60})),
                             predict_observations=Mock(return_value=prediction))
        boards=SimpleNamespace(add=Mock(side_effect=[0,1,2,3]))
        wrapper=PronounReviewRecognizer(base,boards,{'raw':frames[0],'annotated':frames[0]})
        result=wrapper.predict_frames(frames,stamps)
        base.observe_frames.assert_called_once_with(frames,stamps)
        base.predict_observations.assert_called_once_with(arrays,(120,160,3))
        self.assertEqual(result['top5'],prediction['top5']);self.assertTrue(result['geometry_enabled'])
        self.assertEqual(result['geometry_features'],23);self.assertFalse(result['head_interventions_performed'])
        self.assertFalse(result['media_saved_to_files']);self.assertFalse(result['coordinates_saved'])
        self.assertEqual([r['frame_index'] for r in result['review_boards']],[20,32])
        self.assertEqual(boards.add.call_count,2)
        json.dumps(result,allow_nan=False)
        wrapper.predict_frames(frames,stamps)
        with self.assertRaises(ValueError):wrapper.predict_frames(frames,stamps)
        for k,v in before.items():np.testing.assert_array_equal(arrays[k],v)

    def test_missing_detections_still_produce_review_boards_without_a_fabricated_label(self):
        arrays=observations(60);arrays['hand_present'][:]=False
        frame=np.zeros((120,160,3),np.uint8)
        base=SimpleNamespace(observe_frames=lambda f,t:(arrays,frame.shape,{'captured_frames':60}),
                             predict_observations=Mock(side_effect=ValueError('Hands not detected')))
        store=SimpleNamespace(add=Mock(side_effect=[0,1]))
        result=PronounReviewRecognizer(base,store,{'raw':frame,'annotated':frame}).predict_frames([frame]*60,np.arange(60)*50)
        self.assertEqual(result['top5'],[]);self.assertIn('prediction_error',result)
        self.assertEqual(len(result['review_boards']),2)
        self.assertTrue(all(not row['hand_shape_aggregates'] for row in result['review_boards']))

    def test_hand_metrics_are_dimensionless_or_pixel_aggregates_without_coordinate_arrays(self):
        arrays=observations(20);hands,present=canonical(arrays,(480,640,3))
        result=hand_shape_summary(hands,present,10,640)
        self.assertEqual(len(result),1)
        self.assertGreater(result[0]['palm_size_pixels'],0)
        self.assertNotIn('points',result[0]);json.dumps(result,allow_nan=False)
