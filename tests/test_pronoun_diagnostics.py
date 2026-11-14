import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import numpy as np
    from sign_language import pronoun_diagnostics as d
    from sign_language.gru_trajectory_readout import SignTrajectoryReadoutGRU
    from sign_language.gru_geometry_model import SignGeometryMeanPoolGRU
    from sign_language.recognition_forensics import infer_clips


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch unavailable')
class PronounDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.names=['HELLO','NO','PLEASE','THANKYOU','YES','ME','YOU','WE','MY','YOUR','OUR','HELP','GOOD','NEED','LOVE']

    def models(self):
        torch.manual_seed(42)
        new=SignTrajectoryReadoutGRU(classes=15,dropout=0).eval()
        old=SignGeometryMeanPoolGRU(classes=15,dropout=0).eval()
        old.gru.load_state_dict(new.gru.state_dict())
        def wrap(model,sha):
            return SimpleNamespace(model=model,names=self.names,labels={g:i for i,g in enumerate(self.names)},
                device=torch.device('cpu'),checkpoint_sha256=sha,config={'cache_id':'fixture'})
        return wrap(new,d.CURRENT_SHA),wrap(old,d.BASELINE_SHA)

    def features(self,n=20):
        x=np.zeros((n,254),np.float32)
        x[:,249]=1;x[:,251]=1;x[:,253]=1
        t=np.linspace(0,2*np.pi,n)
        x[:,39]=.2*np.cos(t);x[:,40]=.2*np.sin(t)
        x[:,123:186]=np.linspace(0,.7,63)
        return x

    def test_margin_components_and_group_interventions_reconstruct_actual_decisions(self):
        current,baseline=self.models();d.check_models(current,baseline)
        x=self.features();before=x.copy()
        result=d.diagnose_features(current,baseline,x)
        self.assertEqual(set(result['variants']),set(d.VARIANTS))
        for item in result['variants'].values():
            explanation=item['margin_explanation'];c=explanation['components'];actual=explanation['actual_margin']
            self.assertAlmostEqual(actual,sum(c.values()),places=5)
            self.assertEqual(len(explanation['geometry_contributions']),23)
            self.assertEqual(len(item['current']['scores']),15)
            self.assertEqual(len(item['current']['top5']),5)
            for key,removed in [('original_geometry_off',c['original_17_geometry']),
                                ('path_geometry_off',c['six_path_geometry']),
                                ('all_geometry_off',c['original_17_geometry']+c['six_path_geometry'])]:
                self.assertAlmostEqual(item['head_interventions'][key]['we_minus_me_margin'],actual-removed,places=5)
        np.testing.assert_array_equal(x,before)
        json.dumps(result,allow_nan=False)
        self.assertNotIn('joined',result);self.assertNotIn('embedding',result)

    def test_padding_and_other_batch_lengths_do_not_change_real_sample_geometry(self):
        current,_=self.models();x=self.features(20)
        a,h=infer_clips(current,[x]);b,j=infer_clips(current,[x,self.features(128)])
        np.testing.assert_allclose(a[0],b[0],atol=1.e-6)
        np.testing.assert_allclose(h[0],j[0],atol=1.e-6)

    def test_geometry_is_recomputed_for_each_temporal_or_orientation_variant(self):
        current,baseline=self.models()
        import sign_language.gru_trajectory_model as module
        original=module.trajectory_batch
        observed=[]
        def measure(frames,lengths):
            observed.append(frames.detach().cpu().numpy().copy())
            return original(frames,lengths)
        x=self.features()
        with patch.object(module,'trajectory_batch',side_effect=measure):
            d.diagnose_features(current,baseline,x)
        self.assertEqual(len(observed),1)
        measured=observed[0]
        for i,name in enumerate(d.VARIANTS):
            np.testing.assert_array_equal(measured[i,:len(x)],d.make_variant(x,name,None))

    def test_motion_does_not_bridge_gaps_or_drop_actual_time_units(self):
        p=np.array([[0.,0.],[1.,0.],[100.,0.],[3.,0.],[4.,0.]])
        r=d.signed_motion(p,[1,1,0,1,1],[0,100,200,300,400])
        self.assertEqual(r['signed_dx_sum'],2.)
        self.assertEqual(r['path_length'],2.)
        self.assertAlmostEqual(r['median_speed_per_second'],10.)
        self.assertEqual(r['longest_missing_span_frames'],1)
        self.assertEqual(d.signed_motion(p,[0]*5,[0,100,200,300,400]),{'tracked_frames':0})

    def test_windows_slice_raw_observations_before_downsampling_and_no_hands_is_unavailable(self):
        arrays={'hand_present':np.zeros((240,2),bool),'frame_index':np.arange(240)}
        arrays['hand_present'][50:101,0]=True
        calls=[]
        def normalize(a,shape):
            calls.append(a['frame_index'].copy())
            return self.features(min(128,len(a['frame_index']))),{'frames_with_hands':1,'frames_with_body_anchor':1}
        with patch.object(d,'normalize_sequence',side_effect=normalize),\
             patch.object(d,'diagnose_features',return_value={'scores_fixture':True}),\
             patch.object(d,'observation_summary',return_value={'aggregate_only':True}):
            r=d.diagnose_observations(None,None,arrays,(480,640,3),np.arange(240)*33)
        self.assertEqual([len(x) for x in calls],[240,61,51])
        np.testing.assert_array_equal(calls[2],np.arange(50,101))
        self.assertEqual(r['windows']['full']['input_frames'],128)
        self.assertEqual(r['windows']['first_two_seconds']['input_frames'],61)
        json.dumps(r,allow_nan=False)
        arrays['hand_present'][:]=False
        with patch.object(d,'normalize_sequence',return_value=(self.features(),{'frames_with_hands':0,'frames_with_body_anchor':1})),\
             patch.object(d,'diagnose_features') as scored,patch.object(d,'observation_summary',return_value={}):
            r=d.diagnose_observations(None,None,arrays,(480,640,3),np.arange(240)*33)
        scored.assert_not_called()
        self.assertTrue(all(w['status']=='unavailable' for w in r['windows'].values()))

    def test_capture_observes_once_and_exports_no_arrays(self):
        current=SimpleNamespace(observe_frames=lambda frames,stamps:({},(480,640,3),{'captured_frames':len(frames)}))
        with patch.object(current,'observe_frames',wraps=current.observe_frames) as observe,\
             patch.object(d,'diagnose_observations',return_value={'windows':{},'measurements':{}}):
            r=d.diagnose_capture(current,None,[object()]*10,list(range(10)))
        observe.assert_called_once()
        self.assertFalse(r['coordinates_saved']);self.assertFalse(r['media_saved']);self.assertFalse(r['embeddings_saved'])
        self.assertEqual(r['recognition_tracking_passes'],1)
        json.dumps(r,allow_nan=False)

    def test_other_class_margin_can_be_recovered_without_saved_embeddings(self):
        current,_=self.models();model=current.model
        torch.manual_seed(17);x=torch.randn(279)
        with torch.inference_mode():logits=model.classifier(x).numpy()
        explained=d.explain_head(model,x.numpy(),logits,self.names)
        probabilities=d.score(logits,self.names)['scores']
        w=model.classifier.weight.detach().numpy();b=model.classifier.bias.detach().numpy()
        result=d.recover_competitor_margin(explained,probabilities,w,b,self.names,'WE','NEED')
        self.assertTrue(result['available'])
        delta=w[self.names.index('NEED')]-w[self.names.index('WE')]
        self.assertAlmostEqual(result['components']['pooled_gru'],float(delta[:256]@x.numpy()[:256]),places=5)
        self.assertAlmostEqual(result['actual_margin'],float(logits[self.names.index('NEED')]-logits[self.names.index('WE')]),places=5)
        json.dumps(result,allow_nan=False)

    def test_unidentifiable_geometry_does_not_invent_another_margin_explanation(self):
        w=np.zeros((15,279));w[self.names.index('NEED'),256]=1
        e={'geometry_contributions':{name:0. for name in d.FEATURES}}
        result=d.recover_competitor_margin(e,np.ones(15)/15,w,np.zeros(15),self.names,'WE','NEED')
        self.assertFalse(result['available'])
