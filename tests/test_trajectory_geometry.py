import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import numpy as np
    from sign_language.trajectory_geometry import FEATURES,CONFIG,best_window_features,trajectory_batch,fit_statistics
    from sign_language.wrist_trajectory import path_metrics
    from sign_language.gru_trajectory_model import SignTrajectoryMeanPoolGRU
    from sign_language.gru_trajectory_readout import SignTrajectoryReadoutGRU,Float32Linear
    from sign_language.gru_geometry_model import SignGeometryMeanPoolGRU
    from sign_language.preprocessing import CONFIG as PREPROCESSING
    from sign_language.five_sign_pilot import sha256
    from sign_language.recognition import SignRecognizer
    from sign_language.recognition_webcam import model_display_name


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch unavailable')
class TrajectoryGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_vectorized_window_matches_audited_reference_formulas(self):
        angle=np.linspace(0,4*np.pi,65)
        p=np.column_stack([np.cos(angle),.8*np.sin(angle)])
        candidates=[]
        for count in (12,24,36):
            for start in range(len(p)-count+1):
                candidates.append(path_metrics(p[start:start+count]))
        best=max(candidates,key=lambda m:m['loop_probe_score'])
        expected=[best['loop_probe_score'],best['turn_coherence'],abs(best['signed_turns']),
                  best['axis_ratio'],best['endpoint_over_path'],best['path_length']]
        np.testing.assert_allclose(best_window_features(p,np.ones(65,bool)),expected,rtol=1.e-5,atol=1.e-6)

    def test_masks_and_padding_cannot_fabricate_a_loop_or_change_real_geometry(self):
        x=torch.zeros(1,128,254);x[0,:20,249]=1;x[0,:20,253]=1
        x[0,:20,39]=torch.arange(20)/20
        before=x.clone();a=trajectory_batch(x,torch.tensor([20]))
        x[:,20:]=float('nan')
        np.testing.assert_allclose(a,trajectory_batch(x,torch.tensor([20])),equal_nan=True)
        self.assertEqual(a.shape,(1,23));self.assertEqual(a[0,17],0.)
        p=np.column_stack([np.arange(20),np.zeros(20)])
        valid=np.ones(20,bool);valid[8:12]=False
        self.assertTrue(np.isnan(best_window_features(p,valid)).all())
        self.assertTrue(torch.equal(before[:,:20],x[:,:20]))

    def test_seeded_base_parameters_are_preserved_with_only_90_additional_weights(self):
        args={'hidden_size':8,'classes':15,'dropout':0}
        torch.manual_seed(42);old=SignGeometryMeanPoolGRU(**args);rng=torch.get_rng_state().clone()
        torch.manual_seed(42);new=SignTrajectoryMeanPoolGRU(**args)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters()),90)
        for name,value in old.gru.state_dict().items():torch.testing.assert_close(value,new.gru.state_dict()[name])
        torch.testing.assert_close(old.classifier.weight,new.classifier.weight[:,:25])
        self.assertEqual(new.geometry_scale.shape,(23,))

    def test_missing_statistics_and_gradients_to_new_features(self):
        statistics=fit_statistics(np.full((4,23),np.nan))
        model=SignTrajectoryMeanPoolGRU(hidden_size=8,classes=15,dropout=0)
        model.set_geometry_statistics(statistics)
        self.assertTrue(torch.isfinite(model.standardize_geometry(torch.full((1,23),float('nan')))).all())
        x=torch.randn(2,16,254,requires_grad=True)
        y=model(x,torch.tensor([12,16]),geometry=torch.randn(2,23))
        torch.nn.functional.cross_entropy(y,torch.tensor([2,8])).backward()
        self.assertGreater(model.classifier.weight.grad[:,-6:].abs().sum(),0)
        self.assertGreater(model.gru.weight_ih_l0.grad.abs().sum(),0)
        self.assertEqual(x.grad[0,12:].abs().sum(),0)

    def test_readout_freezes_backbone_but_keeps_classifier_gradients(self):
        model=SignTrajectoryReadoutGRU(hidden_size=8,classes=15,dropout=0)
        x=torch.randn(2,16,254)
        loss=model(x,torch.tensor([12,16]),geometry=torch.randn(2,23)).square().mean()
        loss.backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.gru.parameters()))
        self.assertGreater(model.classifier.weight.grad.abs().sum(),0)
        self.assertEqual(model.classifier.in_features,31)

    def test_readout_arithmetic_stays_float32_under_autocast(self):
        layer=Float32Linear(4,3)
        x=torch.randn(2,4)
        expected=layer(x)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual=layer(x)
        self.assertEqual(actual.dtype,torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_application_restores_trajectory_model_and_frozen_statistics(self):
        self.check_application_model(SignTrajectoryMeanPoolGRU,'gru-mean-pool-trajectory-v1',
                                     'GRU mean-pool + hand/path geometry')

    def test_application_restores_frozen_readout_model(self):
        self.check_application_model(SignTrajectoryReadoutGRU,'gru-mean-pool-trajectory-readout-v1',
                                     'Frozen GRU + trajectory readout')

    def check_application_model(self,model_class,architecture,display):
        source=Path(__file__).resolve().parents[1]/'src/sign_language'
        args={'hidden_size':8,'classes':15,'dropout':0}
        model=model_class(**args).eval()
        statistics=fit_statistics(np.zeros((4,23)))
        model.set_geometry_statistics(statistics)
        names=('preprocessing.py','gru_model.py','gru_mean_pool_model.py','gru_geometry_model.py',
               'hand_articulation.py','gru_trajectory_model.py','trajectory_geometry.py','wrist_trajectory.py')
        if model_class is SignTrajectoryReadoutGRU:names=(*names,'gru_trajectory_readout.py')
        config={'architecture':architecture,'model':args,'preprocessing':PREPROCESSING,
                'trajectory_geometry':CONFIG,'geometry_features':list(FEATURES),'geometry_statistics':statistics,
                'source_sha256':{n:sha256(source/n) for n in names}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);labels={str(i):i for i in range(15)}
            torch.save({'config':config,'model':model.state_dict(),'labels':labels,'epoch':1},root/'best.pt')
            (root/'config.json').write_text(json.dumps(config));(root/'labels.json').write_text(json.dumps(labels))
            (root/'hand_landmarker.task').touch();(root/'pose_landmarker_lite.task').touch()
            r=SignRecognizer(root/'best.pt',root,device='cpu')
            self.assertEqual(model_display_name(r),display)
            with torch.inference_mode():
                x=torch.zeros(1,20,254);lengths=torch.tensor([20])
                torch.testing.assert_close(model(x,lengths),r.model(x,lengths))
