import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder=module('project_notebook_builder_tests','build_project_notebooks.py')

    def test_renderer_records_real_stdout_and_png_and_restores_working_directory(self):
        original=Path.cwd()
        notebook=self.builder.Notebook('fixture')
        notebook.markdown('# Report')
        notebook.code("print('actual execution output')")
        notebook.code('import matplotlib.pyplot as plt\nplt.plot([0,1],[1,0])\nplt.show()')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=notebook.execute(root,root/'figures')
            self.assertEqual(result['cells'][1]['outputs'][0]['text'],'actual execution output\n')
            self.assertEqual(result['cells'][2]['outputs'][0]['output_type'],'display_data')
            self.assertEqual(result['cells'][2]['execution_count'],2)
            self.assertEqual(len(list((root/'figures').glob('*.png'))),1)
            json.dumps(result,allow_nan=False)
        self.assertEqual(Path.cwd(),original)

    def test_final_publication_requires_complete_full_test_evaluation(self):
        with self.assertRaisesRegex(ValueError,'completed ST-GCN'):
            self.builder.require_final({'stgcn_complete':False})
        incomplete={'stgcn_complete':True,'models':{'stgcn':{'test':{'metrics':{'samples':2}}}}}
        with self.assertRaisesRegex(ValueError,'complete official test'):
            self.builder.require_final(incomplete)

    def test_notebook_cells_do_not_start_jobs_capture_camera_or_write_data(self):
        notebooks=[self.builder.make_workflow({'stgcn_complete':False}),
                   self.builder.make_metrics({'stgcn_complete':False,'models':{}})]
        forbidden={'run','Popen','system','train','evaluate','VideoCapture','write_text','write_bytes','unlink','save','savez','savez_compressed'}
        for notebook in notebooks:
            for cell in notebook.cells:
                if cell['cell_type']=='code':
                    for node in ast.walk(ast.parse(''.join(cell['source']))):
                        if isinstance(node,ast.Call):
                            name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                            self.assertNotIn(name,forbidden)

    def test_final_metrics_notebook_renders_two_completed_test_models(self):
        # Entirely synthetic rendering fixture, confined to a temporary directory.
        metrics={key:.5 for key in ('top1','top5','macro_precision','macro_recall','macro_f1','weighted_f1','loss')}
        epoch={'epoch':1,'train':metrics,'val':metrics,'seconds':1,'learning_rate':.001}
        base={'label':'SYNTHETIC rendering fixture','status':'completed','parameters':100,
              'history':[epoch],'selected':epoch,'test':None}
        models={key:copy.deepcopy(base) for key in ('gru_baseline','gru_regularized','gru64','stgcn')}
        for key in ('gru_regularized','stgcn'):
            models[key].update(test={'status':'complete','metrics':{**metrics,'samples':4},'checkpoint_epoch':1},
                per_signer={'P1':{'top1':.5},'P2':{'top1':.5}},per_class=[{'gloss':'fixture A','f1':.5},{'gloss':'fixture B','f1':.5}],
                focused_confusion={'labels':['fixture A','fixture B'],'counts':[[1,1],[1,1]],'full_row_support':[2,2]},
                top_confusions=[{'true':'fixture A','predicted':'fixture B','count':1}])
        snapshot={'generated_utc':'SYNTHETIC TEST FIXTURE','stgcn_complete':True,'models':models,
                  'cv':{'folds':[{'fold':i,'validation':metrics} for i in range(3)]}}
        notebook=self.builder.make_metrics(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'artifacts').mkdir()
            (root/'artifacts/experiment_snapshot.json').write_text(json.dumps(snapshot))
            result=notebook.execute(root,root/'figures')
            images=sum(o['output_type']=='display_data' for c in result['cells'] for o in c.get('outputs',[]))
            self.assertEqual(images,11)
            self.assertTrue(all(c['execution_count'] for c in result['cells'] if c['cell_type']=='code'))

    def test_completion_report_invokes_final_notebooks_only_after_frozen_results(self):
        report=module('stgcn_report_notebook_hook_tests','report_stgcn.py')
        metrics={k:.5 for k in ('top1','top5','macro_precision','macro_recall','macro_f1','weighted_f1','loss')}
        metrics['samples']=32941
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'models/asl-stgcn27-v1';output.mkdir(parents=True)
            (root/'docs').mkdir()
            (output/'best.pt').write_bytes(b'unchanged checkpoint fixture')
            digest=hashlib.sha256((output/'best.pt').read_bytes()).hexdigest()
            (output/'status.json').write_text(json.dumps({'status':'completed','epoch':5,'stop_reason':'epoch_budget_reached','parameters':100}))
            (output/'epoch-001.json').write_text(json.dumps({'seconds':10}))
            data={'status':'complete','checkpoint_epoch':1,'checkpoint_sha256':digest,'metrics':metrics,
                  'validation_at_selection':metrics,'elapsed_seconds':1}
            for name in ('asl-stgcn27-test-v1','asl-gru-regularized-test-v1'):
                path=root/'data/evaluations'/name;path.mkdir(parents=True)
                (path/'report.json').write_text(json.dumps(data))
            with patch.object(report,'ROOT',root),patch.object(report.subprocess,'run') as run:
                report.main()
                self.assertEqual(run.call_count,1)
                self.assertIn('--require-complete',run.call_args.args[0])
                self.assertEqual(Path(run.call_args.args[0][2]),root/'scripts/build_project_notebooks.py')
                self.assertTrue(run.call_args.kwargs['check'])
            self.assertEqual(hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),digest)

    def test_current_pilot_architecture_and_results_render_as_read_only_cells(self):
        labels=['HELLO','NO','PLEASE','THANKYOU','YES','ME','YOU','WE','MY','YOUR','OUR','HELP','GOOD','NEED','LOVE']
        metrics={'correct':15,'samples':15,'top1':1.,'macro_f1':1.,'loss':.2,'top5':1.,
                 'confusion':[[int(i==j) for j in range(15)] for i in range(15)],
                 'per_class':{g:{'support':1,'correct':1,'recall':1.} for g in labels}}
        run={'selection':{'selected_epoch':1},'epochs_completed':1,'parameters':792078,
             'train':metrics,'validation':metrics,
             'epoch_metrics':[{'epoch':1,'train_clean':metrics,'val':metrics}]}
        report={'status':'complete','control':copy.deepcopy(run),'temporal':copy.deepcopy(run),
                'specification':{'classes':labels,'counts':{s:{g:1 for g in labels} for s in ('train','val')},
                                 'participants':{'train':['fixture train'],'val':['fixture val']}},
                'paired_validation_changes':{'improved':0,'regressed':0}}
        config={'model':{'hidden_size':256,'layers':2,'classes':15},'geometry_features':[f'fixture feature {i}' for i in range(17)],
                'temporal_augmentation':{'fixture':True}}
        snapshot={'generated_utc':'SYNTHETIC CURRENT-PILOT TEST','models':{},'stgcn_complete':False,
                  'fifteen_sign_temporal':{'report':report,'configs':{'control':config,'temporal':config},'runtime_verification':None}}
        notebook=self.builder.Notebook('current-pilot-fixture')
        notebook.code(self.builder.SETUP)
        self.builder.append_current_pilot(notebook,snapshot)
        forbidden={'run','Popen','train','evaluate','VideoCapture','write_text','save','savez'}
        for cell in notebook.cells:
            if cell['cell_type']=='code':
                for node in ast.walk(ast.parse(''.join(cell['source']))):
                    if isinstance(node,ast.Call):
                        name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                        self.assertNotIn(name,forbidden)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'artifacts').mkdir()
            (root/'artifacts/experiment_snapshot.json').write_text(json.dumps(snapshot))
            result=notebook.execute(root,root/'figures')
            images=sum(o['output_type']=='display_data' for c in result['cells'] for o in c.get('outputs',[]))
            self.assertEqual(images,4)
            self.assertTrue(all(c['execution_count'] for c in result['cells'] if c['cell_type']=='code'))
