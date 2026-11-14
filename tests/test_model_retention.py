import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from sign_language.model_history import artifact_path

ROOT=Path(__file__).resolve().parents[1]
POWERSHELL=shutil.which('pwsh') or shutil.which('powershell')


class ModelHistoryTests(unittest.TestCase):
    def test_history_is_metadata_fallback_not_a_weight_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'data/model-history/2026-11-14/models/run/config.json'
            old.parent.mkdir(parents=True);old.write_text('{}')
            self.assertEqual(artifact_path(root,'models/run/config.json'),old)
            self.assertEqual(artifact_path(root,'models/run/best.pt'),root/'models/run/best.pt')
            with self.assertRaises(ValueError):artifact_path(root,'../outside')


@unittest.skipUnless(POWERSHELL,'PowerShell required for native Windows cleanup validation')
class ModelPruneTests(unittest.TestCase):
    def exercise(self,invalid=False,tampered=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir()
            shutil.copyfile(ROOT/'scripts/prune_model_snapshots.ps1',root/'scripts/prune_model_snapshots.ps1')
            for relative,value in [('models/good/best.pt',b'keep'),('models/bad/epoch.pt',b'remove'),
                                   ('models/bad/config.json',b'{"historical":true}'),('data/sentinel.npy',b'dataset')]:
                p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
            files=[]
            for relative,action in [('models/good/best.pt','keep'),('models/bad/epoch.pt','remove'),('models/bad/config.json','preserve_metadata_then_remove')]:
                p=root/relative;files.append({'path':relative,'action':action,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            if invalid:files.append({'path':'data/sentinel.npy','action':'remove','bytes':7,'sha256':hashlib.sha256(b'dataset').hexdigest()})
            if tampered:files[1]['sha256']='0'*64
            plan={'schema':'model-retention-plan-v1','status':'planned','root':str(root),'history_root':'data/model-history/2026-11-14',
                  'files':files,'checkpoint_files_retained':1}
            (root/'plan.json').write_text(json.dumps(plan))
            result=subprocess.run([POWERSHELL,'-NoProfile','-File',str(root/'scripts/prune_model_snapshots.ps1'),'-Plan',str(root/'plan.json')],capture_output=True,text=True)
            self.assertEqual((root/'models/good/best.pt').read_bytes(),b'keep')
            self.assertEqual((root/'data/sentinel.npy').read_bytes(),b'dataset')
            if invalid or tampered:
                self.assertNotEqual(result.returncode,0)
                self.assertTrue((root/'models/bad/epoch.pt').exists())
            else:
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertFalse((root/'models/bad').exists())
                self.assertEqual((root/'data/model-history/2026-11-14/models/bad/config.json').read_text(),'{"historical":true}')
                self.assertFalse(list((root/'data/model-history').rglob('*.pt')))

    def test_native_cleanup_preserves_selected_weights_metadata_and_dataset(self):self.exercise()
    def test_outside_models_target_is_rejected_before_any_deletion(self):self.exercise(invalid=True)
    def test_tampered_source_is_rejected_before_any_deletion(self):self.exercise(tampered=True)
