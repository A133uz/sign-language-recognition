import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from sign_language.gru_model import SignGRU
    from sign_language.gru_mean_pool_model import SignMeanPoolGRU,masked_temporal_mean
    from sign_language.five_sign_training import classifier
    from sign_language.recognition import SignRecognizer
    from sign_language.recognition_webcam import model_display_name
    from sign_language.preprocessing import CONFIG
    from sign_language.five_sign_pilot import sha256


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch extra unavailable')
class MeanPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pooling_ignores_padding_and_its_gradients_but_keeps_real_rest(self):
        values=torch.tensor([[[2.],[4.],[100.]],[[0.],[6.],[9.]]],requires_grad=True)
        pooled=masked_temporal_mean(values,torch.tensor([2,3]))
        torch.testing.assert_close(pooled,torch.tensor([[3.],[5.]]))
        pooled.sum().backward()
        self.assertEqual(values.grad[0,2,0],0)
        self.assertGreater(values.grad[1,0,0],0) # Real zero/rest is not padding.
        with self.assertRaises(ValueError):masked_temporal_mean(values,torch.tensor([0,3]))
        with self.assertRaises(ValueError):masked_temporal_mean(values,torch.tensor([4,3]))

    def test_same_parameters_and_seeded_initialization_as_last_state_baseline(self):
        args={'hidden_size':8,'layers':2,'classes':5,'dropout':.5}
        torch.manual_seed(42);baseline=SignGRU(**args)
        torch.manual_seed(42);pooled=SignMeanPoolGRU(**args)
        self.assertEqual(baseline.state_dict().keys(),pooled.state_dict().keys())
        for name,value in baseline.state_dict().items():torch.testing.assert_close(value,pooled.state_dict()[name],rtol=0,atol=0)

    def test_packing_preserves_batch_order_and_rejects_padding_influence(self):
        model=SignMeanPoolGRU(hidden_size=8,layers=2,classes=5,dropout=.5).eval()
        frames=torch.randn(3,20,254);lengths=torch.tensor([7,19,11]);changed=frames.clone()
        for i,n in enumerate(lengths):changed[i,n:]=1000
        with torch.inference_mode():
            together=model(frames,lengths);dirty=model(changed,lengths)
            separate=torch.cat([model(frames[i:i+1,:n],lengths[i:i+1]) for i,n in enumerate(lengths)])
        torch.testing.assert_close(together,dirty)
        torch.testing.assert_close(together,separate,rtol=1e-5,atol=1e-6)

    def test_loss_reaches_early_real_frames_and_both_recurrent_layers(self):
        model=SignMeanPoolGRU(hidden_size=8,layers=2,classes=5,dropout=0)
        frames=torch.randn(2,16,254,requires_grad=True)
        loss=torch.nn.functional.cross_entropy(model(frames,torch.tensor([16,10])),torch.tensor([0,4]));loss.backward()
        self.assertGreater(frames.grad[0,:4].abs().sum(),0)
        self.assertEqual(frames.grad[1,10:].abs().sum(),0)
        for name in ('gru.weight_ih_l0','gru.weight_ih_l1','classifier.weight'):
            grad=dict(model.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum(),0)

    def test_checkpoint_factory_runtime_hash_checks_and_ui_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=Path(__file__).resolve().parents[1]/'src/sign_language'
            config={'architecture':'gru-mean-pool-v1','model':{'hidden_size':8,'layers':2,'classes':5},
                    'preprocessing':CONFIG,'source_sha256':{n:sha256(source/n) for n in ('gru_model.py','gru_mean_pool_model.py','preprocessing.py')}}
            model=classifier(config).eval();labels={str(i):i for i in range(5)}
            torch.save({'config':config,'model':model.state_dict(),'labels':labels,'epoch':3},root/'best.pt')
            (root/'config.json').write_text(json.dumps(config));(root/'labels.json').write_text(json.dumps(labels))
            (root/'hand_landmarker.task').touch();(root/'pose_landmarker_lite.task').touch()
            restored=SignRecognizer(root/'best.pt',root,device='cpu')
            self.assertIsInstance(restored.model,SignMeanPoolGRU)
            self.assertEqual(model_display_name(restored),'GRU mean-pool')
            self.assertFalse(restored.model.training)
