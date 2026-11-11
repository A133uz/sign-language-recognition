import unittest
import numpy as np

from sign_language.hand_articulation import summarize_articulation,compare_training


def fixture(n=24):
    x=np.zeros((n,254),np.float32);x[:,249]=1;x[:,251]=1
    points=x[:,123:249].reshape(n,2,21,3)
    for joint,value in {5:[1,0,0],6:[1.5,0,0],7:[2,0,0],8:[0,0,1],
                        9:[1,1,0],10:[1.5,1,0],11:[2,1,0],12:[2.5,1,0]}.items():points[:,0,joint]=value
    return x


class HandArticulationTests(unittest.TestCase):
    def test_projected_overlap_can_retain_depth_and_outputs_are_scalar_only(self):
        x=fixture();before=x.copy();result=summarize_articulation(x)
        self.assertEqual(result['metrics']['thumb_index_xy_median'],0)
        self.assertEqual(result['metrics']['thumb_index_xyz_median'],1)
        self.assertEqual(result['metrics']['index_pip_angle_median_deg'],180)
        self.assertEqual(result['usable_frames'],24)
        self.assertEqual(result['metrics']['palm_axis_xy_excursion_deg'],0)
        np.testing.assert_array_equal(x,before)
        def check(value):
            self.assertNotIsInstance(value,np.ndarray)
            if isinstance(value,dict):
                for item in value.values():check(item)
            elif isinstance(value,list):
                for item in value:check(item)
        check(result)

    def test_opening_motion_is_detected_but_tracking_gap_is_not_rotation(self):
        closed=fixture();opening=closed.copy()
        points=opening[:,123:249].reshape(24,2,21,3)
        points[:,0,8,0]=np.linspace(0,2,24)
        self.assertGreater(summarize_articulation(opening)['metrics']['thumb_index_xy_range'],1.5)
        self.assertEqual(summarize_articulation(closed)['metrics']['thumb_index_xy_range'],0)
        gap=fixture();gap[6:14,249]=0;gap[6:14,251]=0
        p=gap[:,123:249].reshape(24,2,21,3);p[:6,0,9]=[1,0,0];p[14:,0,9]=[-1,0,0]
        measured=summarize_articulation(gap)
        self.assertEqual(measured['metrics']['palm_axis_xy_excursion_deg'],0)
        self.assertEqual(measured['metrics']['palm_axis_xyz_excursion_deg'],0)

    def test_missing_hands_and_degenerate_joints_are_explicit_and_not_fake_values(self):
        self.assertTrue(summarize_articulation(np.zeros((10,254),np.float32))['insufficient'])
        x=fixture();x[:,123:249]=0;result=summarize_articulation(x)
        self.assertIsNone(result['metrics']['palm_axis_xy_excursion_deg'])
        self.assertIsNone(result['metrics']['index_pip_angle_median_deg'])
        reference={'records':[{'split':'val','gloss':'NO','articulation':result}]}
        self.assertEqual(compare_training(result,reference)['NO'],{})
        with self.assertRaises(ValueError):summarize_articulation(np.zeros((2,253)))
