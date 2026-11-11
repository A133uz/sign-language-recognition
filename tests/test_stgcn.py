import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from sign_language.stgcn_graph import adjacency, graph_input
    from sign_language.stgcn_model import SignSTGCN
    from sign_language.stgcn_graph import CONFIG as GRAPH
    from sign_language.five_sign_training import classifier


@unittest.skipUnless(HAS_TORCH, 'Optional PyTorch extra not installed')
class STGCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_graph_selects_correct_body_and_anatomical_finger_points(self):
        features = torch.arange(254.).reshape(1,1,254)
        graph = graph_input(features)
        self.assertEqual(tuple(graph.shape), (1,2,1,27))
        self.assertEqual(graph[0,0,0,3], features[0,0,14])  # left shoulder, pose point 11
        self.assertEqual(graph[0,0,0,7], features[0,0,39])  # left hand wrist
        self.assertEqual(graph[0,0,0,8], features[0,0,47])  # left thumb tip, joint 4
        self.assertEqual(graph[0,0,0,17], features[0,0,81]) # right hand wrist
        self.assertEqual(graph[0,0,0,26], features[0,0,121]) # right little-finger tip

    def test_pilot_factory_restores_stgcn_graph_and_evaluation_preserves_batchnorm(self):
        config={'architecture':'stgcn27-v1','graph':GRAPH,'model':{'classes':5,'dropout':.5}}
        model=classifier(config).eval()
        self.assertIsInstance(model,SignSTGCN)
        before={name:value.clone() for name,value in model.named_buffers()}
        with torch.inference_mode():
            logits=model(torch.randn(2,128,254),torch.tensor([40,90]))
        self.assertEqual(tuple(logits.shape),(2,5))
        for name,value in model.named_buffers():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        with self.assertRaises(ValueError):classifier({**config,'graph':{}})
        with self.assertRaises(ValueError):classifier({**config,'architecture':'unknown'})

    def test_adjacency_is_finite_normalized_and_connects_all_nodes(self):
        matrix = adjacency()
        self.assertEqual(tuple(matrix.shape), (3,27,27))
        totals = matrix.sum(1)
        self.assertTrue(torch.all((totals == 0) | torch.isclose(totals, torch.ones_like(totals))))
        reached = {0}
        for _ in range(27):
            reached |= {j for i in reached for j in range(27) if matrix[:,i,j].any() or matrix[:,j,i].any()}
        self.assertEqual(len(reached), 27)

    def test_padding_values_cannot_change_predictions(self):
        model = SignSTGCN(classes=6, dropout=0).eval()
        frames = torch.randn(2,32,254)
        lengths = torch.tensor([9,23])
        changed = frames.clone()
        changed[0,9:] = 1000
        changed[1,23:] = -1000
        with torch.inference_mode():
            a, b = model(frames, lengths), model(changed, lengths)
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all())

    def test_existing_readout_is_mean_over_valid_reduced_time_and_all_nodes(self):
        model=SignSTGCN(classes=5,dropout=.5).eval();captured={}
        def final_block(module,args,output):captured['final']=output.detach().clone()
        def head_input(module,args):captured['pooled']=args[0].detach().clone()
        a=model.blocks[-1].register_forward_hook(final_block)
        b=model.classifier.register_forward_pre_hook(head_input)
        lengths=torch.tensor([17,41]);reduced=lengths.clone()
        for block in model.blocks:reduced=(reduced+block.stride-1)//block.stride
        try:
            with torch.inference_mode():model(torch.randn(2,64,254),lengths)
            expected=torch.stack([captured['final'][i,:,:int(n),:].mean((1,2)) for i,n in enumerate(reduced)])
            torch.testing.assert_close(captured['pooled'],expected,rtol=1e-5,atol=1e-6)
            self.assertEqual(tuple(captured['pooled'].shape),(2,256))
        finally:a.remove();b.remove()

    def test_backward_reaches_early_frames_and_classifier(self):
        model = SignSTGCN(classes=6, dropout=0)
        frames = torch.randn(2,16,254,requires_grad=True)
        output = model(frames,torch.tensor([16,12]))
        torch.nn.functional.cross_entropy(output,torch.tensor([0,5])).backward()
        self.assertEqual(tuple(output.shape),(2,6))
        self.assertGreater(frames.grad[:,:4].abs().sum(),0)
        self.assertGreater(model.classifier.weight.grad.abs().sum(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
