"""Independent ST-GCN experiment, using frozen GRU-compatible cache/read helpers."""

import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sign_language.dataset import write_json
from sign_language.extraction import output_lock
from sign_language.preprocessing import CONFIG as PREPROCESSING
from sign_language.regularization import PROFILES
from sign_language.stgcn_graph import CONFIG as GRAPH
from sign_language.stgcn_model import SignSTGCN
from sign_language.training import LandmarkDataset, atomic_checkpoint, repair_epoch_artifacts, cuda_check, epoch_pass


def train_stgcn(cache, output, *, epochs=50, batch_size=32, workers=2, patience=10,
                resume=False, smoke=False):
    cache, output = Path(cache), Path(output)
    with output_lock(output.parent/'locks', output):
        gpu = cuda_check()
        print(json.dumps(gpu), flush=True)
        # Convolutional model: separately smoke-tested with cuDNN; no recurrent kernels.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(2)
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        manifest = json.loads((cache/'manifest.json').read_text())
        status = json.loads((cache/'status.json').read_text())
        preprocessing_hash = hashlib.sha256(Path(__file__).with_name('preprocessing.py').read_bytes()).hexdigest()
        if (status['status'] != 'complete' or status['cache_id'] != manifest['cache_id'] or
                manifest['identity']['preprocessing'] != PREPROCESSING or
                manifest['identity']['preprocessing_sha256'] != preprocessing_hash):
            raise ValueError('Incomplete or incompatible landmark cache')
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ('stgcn_training.py', 'stgcn_model.py', 'stgcn_graph.py',
                                'training.py', 'preprocessing.py', 'regularization.py')}
        profile = PROFILES['regularized-v1']
        config = {'architecture': 'stgcn27-v1', 'model': {'classes': len(manifest['labels']), 'dropout': .5},
                  'graph': GRAPH, 'preprocessing': PREPROCESSING, 'cache_id': manifest['cache_id'],
                  'epochs': epochs, 'batch_size': batch_size, 'loader_workers': workers, 'patience': patience,
                  'seed': 42, 'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .001,
                  'regularization': profile, 'selection': 'minimum validation cross-entropy',
                  'partition': {'mode': 'official_train_val', 'test_evaluated': False},
                  'amp': True, 'cudnn_enabled': True, 'source_sha256': sources,
                  'torch': str(torch.__version__), 'smoke': smoke}
        output.mkdir(parents=True, exist_ok=True)
        if (output/'config.json').exists():
            if not resume or json.loads((output/'config.json').read_text()) != config:
                raise ValueError('Existing run: matching --resume required, or choose new output')
        elif any(path.name != 'engine_snapshot' for path in output.iterdir()):
            raise ValueError('Output nonempty without compatible configuration')
        write_json(output/'config.json', config); write_json(output/'labels.json', manifest['labels'])
        model = SignSTGCN(**config['model']).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
        scaler = torch.amp.GradScaler('cuda')
        generator = torch.Generator().manual_seed(42)
        start, best, stale = 0, float('inf'), 0
        if resume:
            saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            if saved['config'] != config:
                raise ValueError('Resume configuration differs')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
            torch.set_rng_state(saved['rng_cpu']); torch.cuda.set_rng_state_all(saved['rng_cuda'])
            generator.set_state(saved['shuffle_rng'])
            start, best, stale = saved['epoch'], saved['best_val_loss'], saved['stale_epochs']
            repair_epoch_artifacts(output, saved)
        loaders = {split: DataLoader(LandmarkDataset(cache, split), batch_size=batch_size,
                    shuffle=split == 'train', num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0, generator=generator if split == 'train' else None,
                    **({'prefetch_factor': 2} if workers else {})) for split in ('train', 'val')}
        state = {'status': 'running', 'gpu': gpu, 'config': config, 'pid': os.getpid(),
                 'epoch': start, 'epochs': epochs, 'parameters': sum(p.numel() for p in model.parameters()),
                 'test_evaluated': False}
        if resume:
            state.update(last_epoch=saved['metrics'], best_val_loss=best, stale_epochs=stale)
        write_json(output/'status.json', state)
        try:
            for epoch in range(start+1, epochs+1):
                if stale >= patience:
                    state['stop_reason'] = 'early_stopping'; break
                began = time.monotonic(); torch.cuda.reset_peak_memory_stats()
                print(f'Epoch {epoch}/{epochs}', flush=True)
                state.update(epoch=epoch, phase='train'); write_json(output/'status.json', state)
                train_metrics = epoch_pass(model, loaders['train'], torch.nn.CrossEntropyLoss(), 'cuda',
                                           optimizer, scaler, max_batches=8 if smoke else None, augment=True)
                state['phase'] = 'validation'; write_json(output/'status.json', state)
                val_metrics = epoch_pass(model, loaders['val'], torch.nn.CrossEntropyLoss(), 'cuda',
                                         scaler=scaler, max_batches=4 if smoke else None)
                improved = val_metrics['loss'] < best
                best, stale = (val_metrics['loss'], 0) if improved else (best, stale+1)
                scheduler.step(val_metrics['loss'])
                record = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics,
                          'seconds': time.monotonic()-began, 'learning_rate': optimizer.param_groups[0]['lr'],
                          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'smoke': smoke}
                saved = {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                         'best_val_loss': best, 'stale_epochs': stale, 'labels': manifest['labels'], 'metrics': record,
                         'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
                         'shuffle_rng': generator.get_state()}
                atomic_checkpoint(output/'last.pt', saved)
                if improved:
                    atomic_checkpoint(output/'best.pt', {key: saved[key] for key in ('config','model','labels','epoch','metrics')})
                write_json(output/f'epoch-{epoch:03d}.json', record)
                state.update(last_epoch=record, best_val_loss=best, stale_epochs=stale)
                write_json(output/'status.json', state)
                print(json.dumps(record), flush=True)
            state.update(status='smoke_complete' if smoke else 'completed', phase='idle',
                         stop_reason=state.get('stop_reason', 'smoke_finished' if smoke else 'epoch_budget_reached'))
            write_json(output/'status.json', state)
        except BaseException as exc:
            state.update(status='failed', error=str(exc)); write_json(output/'status.json', state)
            raise
        return state
