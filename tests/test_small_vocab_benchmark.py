import unittest
import numpy as np
import torch
from sign_language.small_vocab_benchmark import select_test_subset,evaluate_model


class SmallVocabBenchmarkTests(unittest.TestCase):
    def test_official_subset_remaps_targets_without_reordering_or_filtering_missing_hands(self):
        index=[{'split':'test','video_file':f'{i}.mp4','gloss':g,'participant_id':'P1','class_id':c}
               for i,(g,c) in enumerate([('B',8),('OTHER',9),('A',3)])]
        x=np.zeros((3,128,254),np.float16);n=np.array([20,30,40]);y=np.array([8,9,3])
        rows,features,lengths,targets=select_test_subset(index,x,n,y,{'A':0,'B':1},{'A':3,'B':8,'OTHER':9})
        self.assertEqual([r['video_file'] for r in rows],['0.mp4','2.mp4'])
        self.assertEqual([r['parent_index'] for r in rows],[0,2])
        np.testing.assert_array_equal(targets,[1,0]);np.testing.assert_array_equal(lengths,[20,40])
        self.assertEqual(features.shape,(2,128,254));self.assertEqual(features.dtype,np.float32)
        self.assertTrue((features==0).all())
        bad=y.copy();bad[0]=3
        with self.assertRaises(ValueError):select_test_subset(index,x,n,bad,{'A':0,'B':1},{'A':3,'B':8})
        index[0]['split']='train'
        with self.assertRaises(ValueError):select_test_subset(index,x,n,y,{'A':0,'B':1},{'A':3,'B':8})

    def test_inference_preserves_weights_full_grid_lengths_order_and_uses_normal_forward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.scale=torch.nn.Parameter(torch.tensor(2.));self.calls=[]
            def forward(self,frames,lengths):
                assert not self.training and not torch.is_grad_enabled()
                assert frames.shape[1:]==(128,254)
                self.calls.append(lengths.tolist())
                return frames[:,0,:5]*self.scale
        model=Model();before=model.scale.detach().clone()
        x=np.zeros((5,128,254),np.float32)
        for i in range(5):x[i,0,i]=1
        result=evaluate_model(model,x,np.array([8,9,10,11,12]),np.arange(5),list('ABCDE'),'cpu',batch_size=2)
        self.assertEqual(model.calls,[[8,9],[10,11],[12]])
        self.assertEqual(result['metrics']['top1'],1.);self.assertEqual(result['metrics']['top5'],1.)
        self.assertEqual([r['prediction'] for r in result['predictions']],list('ABCDE'))
        self.assertEqual(sum(map(sum,result['confusion'])),5)
        self.assertTrue(torch.equal(model.scale.detach(),before));self.assertIsNone(model.scale.grad)
