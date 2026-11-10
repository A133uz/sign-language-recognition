"""Short real-feature CUDA batch-size probe; no experiment weights are touched."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch

from sign_language.stgcn_model import SignSTGCN
from sign_language.training import backward_step


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    cache = ROOT/'data/training/asl-gru-128-v1'
    arrays = [np.load(cache/f'train-{name}.npy', mmap_mode='r') for name in ('features','lengths','targets')]
    results = []
    for batch in (32,64):
        torch.manual_seed(42)
        model = SignSTGCN().cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        scaler = torch.amp.GradScaler('cuda')
        frames = torch.from_numpy(np.array(arrays[0][:batch], dtype=np.float32)).cuda()
        lengths = torch.tensor(np.array(arrays[1][:batch], dtype=np.int64))
        labels = torch.tensor(np.array(arrays[2][:batch], dtype=np.int64), device='cuda')
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            torch.cuda.synchronize(); started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(frames,lengths),labels)
            assert torch.isfinite(loss)
            assert backward_step(model,optimizer,loss,scaler)
            torch.cuda.synchronize()
            if step >= 3:
                durations.append(time.perf_counter()-started)
        result = {'batch_size':batch, 'steady_step_seconds':sum(durations)/len(durations),
                  'samples_per_second':batch/(sum(durations)/len(durations)),
                  'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'scope':'5 steady forward/loss/backward/optimizer steps; excludes data loading and validation'}
        results.append(result)
        print(json.dumps(result),flush=True)
        del optimizer, model, frames, labels, loss, scaler
        torch.cuda.empty_cache()
    output = ROOT/'.cache/stgcn-batch-benchmark.json'
    if output.exists():
        raise ValueError('Benchmark output already exists')
    output.write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
