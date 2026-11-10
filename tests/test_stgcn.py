import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from sign_language.stgcn_graph import adjacency, graph_input
    from sign_language.stgcn_model import SignSTGCN


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

    def test_backward_reaches_early_frames_and_classifier(self):
        model = SignSTGCN(classes=6, dropout=0)
        frames = torch.randn(2,16,254,requires_grad=True)
        output = model(frames,torch.tensor([16,12]))
        torch.nn.functional.cross_entropy(output,torch.tensor([0,5])).backward()
        self.assertEqual(tuple(output.shape),(2,6))
        self.assertGreater(frames.grad[:,:4].abs().sum(),0)
        self.assertGreater(model.classifier.weight.grad.abs().sum(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
