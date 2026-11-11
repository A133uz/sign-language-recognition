import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from sign_language.five_sign_pilot import SIGNS, early_stop_reason, prepare_subset, select_epoch, sha256
from sign_language.preprocessing import CONFIG


def record(epoch,train,val,loss):
    return {'epoch':epoch,'train_clean':{'top1':train},'val':{'top1':val,'loss':loss,'samples':20}}


class EarlySelectionTests(unittest.TestCase):
    def test_chance_level_small_gap_cannot_win_over_learned_model(self):
        rows=[record(1,.2,.2,1.6),record(4,.85,.8,.5),record(8,1,.85,.45)]
        chosen,receipt=select_epoch(rows)
        self.assertEqual(chosen['epoch'],4)
        self.assertNotIn(1,receipt['eligible_epochs'])
        self.assertFalse(receipt['test_used'])

    def test_validation_loss_guard_excludes_uncertain_early_model(self):
        rows=[record(2,.9,.9,1.0),record(5,1,.9,.3),record(6,1,.9,.29)]
        chosen,_=select_epoch(rows)
        self.assertEqual(chosen['epoch'],5)

    def test_empty_or_inconsistent_validation_rejected(self):
        with self.assertRaises(ValueError):select_epoch([])
        rows=[record(1,.8,.8,.4),record(2,.9,.9,.3)];rows[1]['val']['samples']=19
        with self.assertRaises(ValueError):select_epoch(rows)

    def test_chance_level_stall_does_not_trigger_overfitting_stop(self):
        rows=[{**record(i,.3,.2,1.6),'accuracy_gap':.1} for i in range(1,7)]
        self.assertIsNone(early_stop_reason(rows,6,6))

    def test_warmup_and_learned_validation_plateau_stop(self):
        rows=[{**record(i,.8,.6,.8),'accuracy_gap':.2} for i in range(1,6)]
        self.assertIsNone(early_stop_reason(rows[:4],4,4))
        self.assertEqual(early_stop_reason(rows,3,3),'validation_loss_patience')
        self.assertEqual(early_stop_reason(rows,0,3),'sustained_generalization_gap')

    def test_subset_remaps_targets_without_moving_signers_or_modifying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/'parent';parent.mkdir();output=Path(tmp)/'subset'
            labels={g:100+i for i,g in enumerate(SIGNS)}
            source_hash=sha256(Path(__file__).resolve().parents[1]/'src/sign_language/preprocessing.py')
            (parent/'manifest.json').write_text(json.dumps({'cache_id':'fixture','labels':labels,
                'identity':{'preprocessing':CONFIG,'preprocessing_sha256':source_hash}}))
            (parent/'status.json').write_text('{"status":"complete"}')
            for split in ('train','val','test'):
                rows=[{'video_file':f'{split}-{g}.mp4','participant_id':split,'gloss':g,'split':split,'class_id':labels[g]} for g in SIGNS]
                (parent/f'{split}-index.json').write_text(json.dumps(rows))
                np.save(parent/f'{split}-features.npy',np.zeros((5,128,254),np.float16))
                np.save(parent/f'{split}-lengths.npy',np.full(5,64,np.int16))
                np.save(parent/f'{split}-targets.npy',np.arange(100,105,dtype=np.int64))
            before={p.name:sha256(p) for p in parent.iterdir()}
            manifest=prepare_subset(parent,output)
            self.assertEqual(manifest['participants'],{'train':['train'],'val':['val'],'test':['test']})
            np.testing.assert_array_equal(np.load(output/'train-targets.npy'),np.arange(5))
            self.assertEqual({p.name:sha256(p) for p in parent.iterdir()},before)
            with self.assertRaises(FileExistsError):prepare_subset(parent,output)


if __name__=='__main__':unittest.main()
