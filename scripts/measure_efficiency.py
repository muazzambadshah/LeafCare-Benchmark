"""Measure checkpoint file size, FP32 weight size and inference latency (batch 1, 224x224)
for the seed-42 checkpoint of each model: CPU with 1 and 4 threads, and the laptop GPU.
Latency is the median of 100 timed runs after 20 warm-up runs. Writes verification/efficiency.json."""
import sys, json, time, platform, statistics, os
from pathlib import Path
sys.dont_write_bytecode = True
OUT = Path(sys.argv[1]); FIX4 = Path(sys.argv[2]); PROJECT = Path(sys.argv[3])
sys.path.insert(0, str(FIX4 / 'code'))
import torch, train_matched as tm
torch.backends.cudnn.benchmark = False
def cpu_name():
    try:
        for l in open('/proc/cpuinfo'):
            if l.startswith('model name'): return l.split(':', 1)[1].strip()
    except Exception: pass
    return platform.processor()
def bench(net, dev, threads=None, n=100, warm=20):
    if threads: torch.set_num_threads(threads)
    x = torch.rand(1, 3, 224, 224, device=dev); ts = []
    with torch.inference_mode():
        for i in range(warm + n):
            if dev == 'cuda': torch.cuda.synchronize()
            t = time.perf_counter(); net(x)
            if dev == 'cuda': torch.cuda.synchronize()
            if i >= warm: ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 2)
res = {'hardware': {'cpu': cpu_name(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'torch': torch.__version__, 'os': 'WSL2 Ubuntu on Windows 11 laptop'},
       'protocol': 'batch 1, 224x224, FP32, eval mode, median of 100 runs after 20 warm-up; seed-42 checkpoints; PyTorch eager, no export or quantization', 'models': {}}
for m in tm.MODELS:
    ck_path = FIX4 / 'results/matched_training_b32' / f'{m}_seed42' / 'best.pt'
    ck = torch.load(ck_path, map_location='cpu', weights_only=True)
    net = tm.build(m, 147, PROJECT, pretrained=False); net.load_state_dict(ck['state_dict']); net.eval()
    params = sum(p.numel() for p in net.parameters())
    r = {'params_M': round(params / 1e6, 3), 'fp32_weights_MB': round(params * 4 / 2**20, 2), 'checkpoint_file_MB': round(ck_path.stat().st_size / 2**20, 2),
         'cpu_1thread_ms': bench(net, 'cpu', 1), 'cpu_4thread_ms': bench(net, 'cpu', 4)}
    if torch.cuda.is_available(): r['gpu_ms'] = bench(net.cuda(), 'cuda')
    res['models'][m] = r; print(m, r, flush=True)
    del net; torch.cuda.empty_cache()
(OUT / 'verification' / 'efficiency.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res['hardware']))
