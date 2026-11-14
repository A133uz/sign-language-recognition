import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('model_archive_tests',ROOT/'scripts/archive_model_weights.py')
archive_tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive_tool)


class ModelArchiveTests(unittest.TestCase):
    def test_curated_archive_requires_exact_registered_weight_set_and_records_notebook_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'models/run';run.mkdir(parents=True)
            weight=run/'best.pt';weight.write_bytes(b'curated fixture')
            (root/'notebooks').mkdir();(root/'notebooks/design.ipynb').write_text('{"cells":[]}')
            (root/'AGENTS.md').write_text('Training belongs in notebooks.')
            history=root/'data/model-history/2026-11-14/models/removed';history.mkdir(parents=True)
            (history/'epoch-001.json').write_text('{"val":{"top1":0.2}}')
            (history/'rejected.pt').write_bytes(b'must never be copied from history')
            (root/'models/model-catalog.json').write_text(json.dumps({'date_local':'2026-11-14',
                'history_root':'data/model-history/2026-11-14','retained_models':[
                {'checkpoint':'models/run/best.pt','checkpoint_sha256':archive_tool.digest_file(weight)}]}))
            output=root/'models/archives/curated.zip'
            (run/'unregistered.pt').write_bytes(b'rejected')
            with self.assertRaisesRegex(ValueError,'unregistered'):
                archive_tool.create_archive(root,output,root/'catalog.json')
            (run/'unregistered.pt').unlink()
            with patch.object(archive_tool.subprocess,'check_output',return_value='fixture\n'):
                result=archive_tool.create_archive(root,output,root/'catalog.json')
            self.assertEqual(result['manifest']['archive_date_local'],'2026-11-14')
            self.assertEqual(result['manifest']['checkpoint_files'],1)
            with ZipFile(output) as z:
                self.assertIn('project_source/notebooks/design.ipynb',z.namelist())
                self.assertIn('project_source/AGENTS.md',z.namelist())
                self.assertIn('metadata_history/models/removed/epoch-001.json',z.namelist())
                self.assertNotIn('metadata_history/models/removed/rejected.pt',z.namelist())

    def test_archive_preserves_all_weights_and_sources_excludes_recursive_archives_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'models/run';model.mkdir(parents=True)
            (model/'best.pt').write_bytes(b'checkpoint fixture, not a torch model')
            (model/'epoch-001.pt').write_bytes(b'intermediate fixture')
            (model/'config.json').write_text('{"architecture":"fixture","model":{"classes":5}}')
            (root/'models/archives').mkdir();(root/'models/archives/old.zip').write_bytes(b'preserve old archive')
            (root/'models/locks').mkdir();(root/'models/locks/job.lock').touch()
            (root/'src').mkdir();(root/'src/example.py').write_text('fixture = True\n')
            originals={p:archive_tool.digest_file(p) for p in root.rglob('*') if p.is_file()}
            output=root/'models/archives/new.zip';catalog=root/'docs/catalog.json'
            with patch.object(archive_tool.subprocess,'check_output',return_value='fixture-commit\n'):
                report=archive_tool.create_archive(root,output,catalog)
            self.assertTrue(report['verified_crc_and_member_sha256'])
            self.assertEqual(report['manifest']['checkpoint_files'],2)
            self.assertEqual({p:archive_tool.digest_file(p) for p in originals},originals)
            with ZipFile(output) as z:
                self.assertIn('models/run/epoch-001.pt',z.namelist())
                self.assertIn('project_source/src/example.py',z.namelist())
                self.assertNotIn('models/archives/old.zip',z.namelist())
                self.assertNotIn('models/locks/job.lock',z.namelist())
            archive_tool.verify_archive(output,report['archive_sha256'])
            with self.assertRaises(FileExistsError):archive_tool.create_archive(root,output,catalog)
            corrupt=root/'wrong.zip'
            with ZipFile(output) as source,ZipFile(corrupt,'w',ZIP_DEFLATED) as target:
                for name in source.namelist():target.writestr(name,b'tampered' if name.endswith('best.pt') else source.read(name))
            with self.assertRaises(ValueError):archive_tool.verify_archive(corrupt)

    def test_unsafe_archive_names_are_rejected(self):
        for name in ('../escape','/absolute','C:/outside','models/../../outside','models\\outside'):
            with self.subTest(name=name),self.assertRaises(ValueError):archive_tool.safe_member(name)

    def test_selected_run_archives_only_requested_weights_with_task_assets_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('selected','other'):
                folder=root/'models'/name;folder.mkdir(parents=True)
                (folder/'best.pt').write_bytes(name.encode())
            (root/'models/hand_landmarker.task').write_bytes(b'detector fixture')
            (root/'src').mkdir();(root/'src/example.py').write_text('fixture=True\n')
            output=root/'models/archives/selected.zip'
            with patch.object(archive_tool.subprocess,'check_output',return_value='fixture-commit\n'):
                report=archive_tool.create_archive(root,output,root/'catalog.json',run='selected')
            manifest=archive_tool.verify_archive(output,report['archive_sha256'])
            names={r['member'] for r in manifest['files']}
            self.assertEqual(manifest['checkpoint_files'],1)
            self.assertIn('models/selected/best.pt',names)
            self.assertNotIn('models/other/best.pt',names)
            self.assertIn('models/hand_landmarker.task',names)
            self.assertIn('project_source/src/example.py',names)
            self.assertIn('Selected run selected',manifest['scope'])
            for invalid in ('..','.','archives','missing'):
                with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                    archive_tool.inventory(root,run=invalid)
