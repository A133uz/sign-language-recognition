import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sign_language.dataset_distribution import gather_distribution,balance_statistics,file_sha256

ROOT=Path(__file__).resolve().parents[1]


def fixture(root):
    folder=root/'data/asl_citizen/ASL_Citizen/splits';folder.mkdir(parents=True)
    rows={'train':[('T1','a','DOG1'),('T2','b','DOG1'),('T2','c','DOG2')],
          'val':[('V1','d','DOG1')],
          'test':[('E1','e','DOG1'),('E1','f','DOG2'),('E2','g','DOG2')]}
    for split,data in rows.items():
        with (folder/f'{split}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['Participant ID','Video file','Gloss']);writer.writerows(data)
    return folder


class DatasetDistributionTests(unittest.TestCase):
    def test_balance_metrics_distinguish_uniform_missing_and_concentrated_counts(self):
        equal=balance_statistics([4,4,4,4]);self.assertEqual(equal['gini'],0)
        self.assertEqual(equal['cv'],0);self.assertAlmostEqual(equal['normalized_entropy'],1)
        missing=balance_statistics([0,1,9]);self.assertEqual(missing['zero_count_classes'],1)
        self.assertIsNone(missing['max_min_ratio']);self.assertGreater(missing['gini'],.5)
        self.assertLess(missing['normalized_entropy'],1)
        self.assertIsNone(balance_statistics([0,0])['gini'])
        for invalid in ([],[-1,2],[float('nan'),2]):
            with self.assertRaises(ValueError):balance_statistics(invalid)

    def test_source_counts_preserve_numbered_classes_and_zero_split_counts_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=fixture(root);before={p:file_sha256(p) for p in folder.glob('*.csv')}
            snapshot=gather_distribution(root)
            self.assertEqual(snapshot['integrity']['unique_video_count'],7)
            self.assertEqual(snapshot['class_rows'][0]['gloss'],'DOG1')
            self.assertEqual(snapshot['class_rows'][1]['gloss'],'DOG2')
            self.assertEqual(snapshot['class_rows'][1]['val'],0)
            self.assertEqual(snapshot['statistics']['val']['zero_count_classes'],1)
            self.assertEqual(snapshot['prepared_cache_check']['status'],'not_available')
            self.assertEqual({p:file_sha256(p) for p in before},before)

    def test_duplicate_videos_and_participant_leakage_are_rejected(self):
        for person,video in [('V1','a'),('T1','new-video')]:
            with self.subTest(person=person),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder=fixture(root)
                with (folder/'val.csv').open('a',newline='',encoding='utf-8') as stream:csv.writer(stream).writerow([person,video,'DOG1'])
                with self.assertRaises(ValueError):gather_distribution(root)

    def test_prepared_target_mismatch_is_detected_instead_of_repeating_manifest_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);cache=root/'data/training/asl-gru-128-v1';cache.mkdir(parents=True)
            (cache/'manifest.json').write_text(json.dumps({'cache_id':'SYNTHETIC','labels':{'DOG1':0,'DOG2':1}}))
            for split,targets in [('train',[0,0,1]),('val',[0]),('test',[0,1,1])]:
                np.save(cache/f'{split}-targets.npy',np.asarray(targets,np.int64))
                np.save(cache/f'{split}-features.npy',np.zeros((len(targets),2,1),np.float16))
            self.assertEqual(gather_distribution(root)['prepared_cache_check']['status'],'verified')
            np.save(cache/'val-targets.npy',np.asarray([1],np.int64))
            with self.assertRaisesRegex(ValueError,'counts differ'):gather_distribution(root)

    def test_notebook_executes_from_snapshot_only_embeds_all_rows_and_has_no_job_calls(self):
        spec=importlib.util.spec_from_file_location('dataset_notebook_tests',ROOT/'scripts/build_dataset_distribution_notebook.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        notebook=builder.make_notebook();forbidden={'train','fit','evaluate','VideoCapture','Popen','system','write_text','write_bytes','save','savez','unlink'}
        for cell in notebook.cells:
            if cell['cell_type']!='code':continue
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                    self.assertNotIn(name,forbidden)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);snapshot=gather_distribution(root)
            folder=root/'portable_notebooks';(folder/'artifacts').mkdir(parents=True)
            (folder/'artifacts'/builder.SNAPSHOT).write_text(json.dumps(snapshot))
            # Execute outside the source-data tree: only the portable snapshot is needed.
            result,images=builder.execute_notebook(notebook,folder,folder/'figures')
            self.assertEqual(images,6) # No five-sign chart in this two-class synthetic fixture.
            code=[c for c in result['cells'] if c['cell_type']=='code']
            self.assertTrue(all(c['execution_count'] for c in code))
            full_table=code[-1]['outputs'][0]['data']['text/html']
            self.assertEqual(full_table.count('<tbody>'),1)
            self.assertIn('DOG1',full_table);self.assertIn('DOG2',full_table)
            self.assertEqual(full_table.count('<tr>'),3) # Header + every synthetic class.
            builder.validate(result)
