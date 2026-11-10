"""CUDA GRU training on prepared official splits; the test split stays untouched."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from sign_language.dataset import write_json
from sign_language.extraction import output_lock
from sign_language.gru_model import SignGRU
from sign_language.preprocessing import CONFIG, INPUT_SIZE
from sign_language.regularization import PROFILES, augment_batch


class LandmarkDataset(Dataset):
    def __init__(self, root, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        self.root, self.split = Path(root), split
        self.arrays = None
        self.count = len(np.load(self.root/f'{split}-targets.npy', mmap_mode='r'))

    def __len__(self):
        return self.count

    def __getstate__(self):
        return {**self.__dict__, 'arrays': None}

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays = [np.load(self.root/f'{self.split}-{n}.npy', mmap_mode='r')
                           for n in ('features', 'lengths', 'targets')]
        features, lengths, targets = self.arrays
        return torch.from_numpy(np.array(features[index], dtype=np.float32)), int(lengths[index]), int(targets[index])


def atomic_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def repair_epoch_artifacts(output, checkpoint):
    """Recover an interruption after last.pt but before best/metrics were committed."""
    record = checkpoint['metrics']
    if record['epoch'] != checkpoint['epoch']:
        raise ValueError('Checkpoint epoch and metrics disagree')
    write_json(output/f'epoch-{checkpoint["epoch"]:03d}.json', record)
    if record['val']['loss'] == checkpoint['best_val_loss']:
        atomic_checkpoint(output/'best.pt', {
            'config': checkpoint['config'], 'model': checkpoint['model'],
            'labels': checkpoint['labels'], 'epoch': checkpoint['epoch'], 'metrics': record})
    elif not (output/'best.pt').exists():
        raise ValueError('The earlier selected checkpoint is missing')


def cuda_check():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to silently train on CPU')
    a = torch.randn(128, 128, device='cuda', requires_grad=True)
    a.square().mean().backward()
    torch.cuda.synchronize()
    return {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'vram_bytes': torch.cuda.get_device_properties(0).total_memory}


def backward_step(model, optimizer, loss, scaler=None):
    if scaler is None:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return True
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # GradScaler records nonfinite gradients during unscale_. Its step then
    # skips the optimizer update and update() lowers the scale. Do not abort
    # a recoverable fp16 overflow or let it update the model parameters.
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
    finite = bool(torch.isfinite(norm))
    scaler.step(optimizer)
    scaler.update()
    return finite


def epoch_pass(model, loader, criterion, device, optimizer=None, scaler=None, max_batches=None, augment=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.; samples = correct = top5 = 0
    classes = model.classifier.out_features
    confusion = torch.zeros((classes, classes), device=device, dtype=torch.int64) if not training else None
    began = time.monotonic(); steps = 0; overflows = consecutive_overflows = 0
    with torch.set_grad_enabled(training):
        for frames, lengths, targets in loader:
            frames, targets = frames.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if training and augment:
                frames = augment_batch(frames)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=scaler is not None):
                logits = model(frames, lengths)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training/validation loss')
            if training:
                finite = backward_step(model, optimizer, loss, scaler)
                if not finite:
                    overflows += 1; consecutive_overflows += 1
                    print(f'AMP overflow: optimizer update skipped; new scale={scaler.get_scale()}', flush=True)
                    if consecutive_overflows >= 8:
                        raise FloatingPointError('Eight consecutive AMP overflows; inspect numerical stability')
                else:
                    consecutive_overflows = 0
            predicted = logits.argmax(1)
            batch = len(targets); samples += batch; total_loss += loss.item()*batch
            correct += (predicted == targets).sum().item()
            top5 += (logits.topk(min(5, classes), dim=1).indices == targets[:, None]).any(1).sum().item()
            if confusion is not None:
                confusion += torch.bincount(targets*classes+predicted, minlength=classes*classes).reshape(classes, classes)
            steps += 1
            if steps % 100 == 0:
                print(f'{"train" if training else "val"}: {steps}/{len(loader)} batches, loss={total_loss/samples:.4f}', flush=True)
            if max_batches is not None and steps >= max_batches:
                break
    torch.cuda.synchronize()
    report = {'loss': total_loss/samples, 'top1': correct/samples, 'top5': top5/samples,
              'samples': samples, 'batches': steps, 'seconds': time.monotonic()-began,
              'skipped_amp_updates': overflows}
    if confusion is not None:
        cm = confusion.cpu().numpy(); tp = np.diag(cm).astype(float)
        support = cm.sum(1); predicted_counts = cm.sum(0)
        precision = np.divide(tp, predicted_counts, out=np.zeros_like(tp), where=predicted_counts > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall) > 0)
        report.update(macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                      macro_f1=float(f1.mean()), weighted_f1=float((f1*support).sum()/support.sum()))
    return report


def train(cache, output, *, epochs=50, batch_size=64, workers=2, patience=10, resume=False,
          smoke=False, profile='baseline', fold_file=None, fold=None, hidden_size=256):
    cache, output = Path(cache), Path(output)
    with output_lock(output.parent/'locks', output):
        return _train(cache, output, epochs, batch_size, workers, patience, resume, smoke,
                      profile, fold_file, fold, hidden_size)


def _train(cache, output, epochs, batch_size, workers, patience, resume, smoke,
           profile='baseline', fold_file=None, fold=None, hidden_size=256):
    gpu = cuda_check(); print(json.dumps(gpu), flush=True)
    # On this Windows/PyTorch build a standalone cuDNN GRU completes its math
    # but exits with 0xC0000409 during native cleanup. The ATen CUDA GRU path
    # passes the same forward/backward/exit probe. This does not select CPU.
    torch.backends.cudnn.enabled = False
    manifest = json.loads((cache/'manifest.json').read_text())
    cache_status = json.loads((cache/'status.json').read_text())
    if cache_status['status'] != 'complete' or manifest['identity']['preprocessing'] != CONFIG:
        raise ValueError('Prepared cache is incomplete or preprocessing differs')
    if cache_status['cache_id'] != manifest['cache_id']:
        raise ValueError('Cache identity mismatch')
    preprocessing_hash = hashlib.sha256(Path(__file__).with_name('preprocessing.py').read_bytes()).hexdigest()
    if manifest['identity']['preprocessing_sha256'] != preprocessing_hash:
        raise ValueError('Prepared features were made by a different preprocessor')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    if profile not in PROFILES:
        raise ValueError('Unknown regularization profile')
    if hidden_size not in (64, 128, 256):
        raise ValueError('Hidden size must be 64, 128 or 256')
    regularization = PROFILES[profile]
    model_config = {'input_size': INPUT_SIZE, 'hidden_size': hidden_size, 'layers': 2,
                    'classes': len(manifest['labels']), 'dropout': regularization['dropout']}
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ('training.py', 'gru_model.py', 'preprocessing.py', 'regularization.py', 'cross_validation.py')}
    partition = {'mode': 'official_train_val', 'test_evaluated': False}
    if fold_file is not None:
        from sign_language.cross_validation import load_fold
        train_indices, val_indices, partition = load_fold(cache, Path(fold_file), fold)
        datasets = {split: Subset(LandmarkDataset(cache, 'train'), indices)
                    for split, indices in (('train', train_indices), ('val', val_indices))}
    else:
        datasets = {split: LandmarkDataset(cache, split) for split in ('train', 'val')}
    config = {'model': model_config, 'preprocessing': CONFIG, 'cache_id': manifest['cache_id'],
              'batch_size': batch_size, 'epochs': epochs, 'patience': patience, 'seed': 42,
              'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': regularization['weight_decay'],
              'profile': profile, 'regularization': regularization, 'partition': partition,
              'selection': 'minimum validation cross-entropy', 'amp': True,
              'cudnn_enabled': False, 'loader_workers': workers,
              'source_sha256': sources, 'torch': str(torch.__version__), 'smoke': smoke}
    output.mkdir(parents=True, exist_ok=True)
    existing = output/'config.json'
    if existing.exists():
        if json.loads(existing.read_text()) != config or not resume:
            raise ValueError('Existing run: use matching config with --resume, or a new output directory')
    elif any(output.iterdir()):
        raise ValueError('Nonempty run directory without config')
    write_json(existing, config); write_json(output/'labels.json', manifest['labels'])
    model = SignGRU(**model_config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=regularization['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=3)
    scaler = torch.amp.GradScaler('cuda')
    generator = torch.Generator().manual_seed(42)
    loaders = {split: DataLoader(datasets[split], batch_size=batch_size,
                    shuffle=split == 'train', num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0, generator=generator if split == 'train' else None,
                    **({'prefetch_factor': 2} if workers else {})) for split in ('train', 'val')}
    start = 0; best = float('inf'); stale = 0
    if resume:
        checkpoint = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint config mismatch')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        start = checkpoint['epoch']; best = checkpoint['best_val_loss']; stale = checkpoint['stale_epochs']
        torch.set_rng_state(checkpoint['rng_cpu']); torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        generator.set_state(checkpoint['shuffle_rng'])
        repair_epoch_artifacts(output, checkpoint)
    criterion = nn.CrossEntropyLoss()
    status = {'status': 'running', 'gpu': gpu, 'config': config, 'pid': os.getpid(),
              'epoch': start, 'epochs': epochs, 'test_evaluated': False}
    write_json(output/'status.json', status)
    try:
        for epoch in range(start+1, epochs+1):
            if stale >= patience:
                status['stop_reason'] = 'early_stopping'; break
            began = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            print(f'Epoch {epoch}/{epochs}', flush=True)
            status.update(epoch=epoch, phase='train'); write_json(output/'status.json', status)
            train_report = epoch_pass(model, loaders['train'], criterion, 'cuda', optimizer, scaler,
                                      max_batches=8 if smoke else None, augment=regularization['augmentation'] is not None)
            status['phase'] = 'validation'; write_json(output/'status.json', status)
            val_report = epoch_pass(model, loaders['val'], criterion, 'cuda', scaler=scaler,
                                    max_batches=4 if smoke else None)
            improved = val_report['loss'] < best
            if improved:
                best = val_report['loss']; stale = 0
            else:
                stale += 1
            scheduler.step(val_report['loss'])
            record = {'epoch': epoch, 'train': train_report, 'val': val_report,
                      'seconds': time.monotonic()-began, 'learning_rate': optimizer.param_groups[0]['lr'],
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'smoke': smoke}
            checkpoint = {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                          'best_val_loss': best, 'stale_epochs': stale, 'rng_cpu': torch.get_rng_state(),
                          'rng_cuda': torch.cuda.get_rng_state_all(), 'shuffle_rng': generator.get_state(),
                          'labels': manifest['labels'], 'metrics': record}
            atomic_checkpoint(output/'last.pt', checkpoint)
            if improved:
                atomic_checkpoint(output/'best.pt', {'config': config, 'model': model.state_dict(),
                                  'labels': manifest['labels'], 'epoch': epoch, 'metrics': record})
            # One atomic metrics file per committed epoch avoids duplicated histories after a restart.
            write_json(output/f'epoch-{epoch:03d}.json', record)
            status.update(last_epoch=record, best_val_loss=best, stale_epochs=stale)
            write_json(output/'status.json', status)
            print(json.dumps(record), flush=True)
            if stale >= patience:
                status['stop_reason'] = 'early_stopping'; break
        status.update(status='smoke_complete' if smoke else 'completed', phase='idle')
        write_json(output/'status.json', status)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'status.json', status)
        raise
    return status
