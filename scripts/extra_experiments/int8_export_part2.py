"""Phone-oriented compression test. Seed-42 checkpoint of each model -> ONNX FP32 -> ONNX INT8
(static QDQ quantization, calibrated on 300 CALIBRATION-partition images). Reports file size,
onnxruntime CPU latency (1 thread, batch 1) and top-1 accuracy of FP32 vs INT8 on the first 3,000
test images (fixed by manifest order). Read-only on checkpoints; writes ONNX files to <out>/onnx."""
import sys, csv, json, time, statistics
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
OUT = Path(sys.argv[1]); FIX4 = Path(sys.argv[2]); PROJECT = Path(sys.argv[3])
sys.path.insert(0, str(FIX4 / 'code'))
import train_matched as tm
ROOT = PROJECT / 'output/Leaf_Dataset_Combined/training_disease'
MAP = json.loads((ROOT / 'class_to_idx.json').read_text())
rows = list(csv.DictReader((FIX4 / 'results/grouped_split/grouped_manifest.csv').open(newline='')))
CAL = [r for r in rows if r['clean_split'] == 'calibration'][::120][:150]
TEST = [r for r in rows if r['clean_split'] == 'test'][:1000]
(OUT / 'onnx').mkdir(exist_ok=True)
def stack(rr, kind): ds = tm.Images(ROOT, rr, MAP, False, kind); return np.stack([ds[i][0].numpy() for i in range(len(rr))]), np.array([ds[i][1] for i in range(len(rr))])
DATA = {k: (stack(CAL, k)[0], *stack(TEST, k)) for k in ('efficientnet_b0', 'yolo26n')}
class Reader(CalibrationDataReader):
    def __init__(s, X): s.it = iter([{'input': X[i:i + 1]} for i in range(len(X))])
    def get_next(s): return next(s.it, None)
def sess(p, threads=1):
    o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1
    return ort.InferenceSession(str(p), o, providers=['CPUExecutionProvider'])
def acc(sx, X, y):
    pr = np.concatenate([sx.run(None, {'input': X[i:i + 50]})[0] for i in range(0, len(X), 50)]); return float((pr.argmax(1) == y).mean())
def lat(sx, n=60):
    x = np.random.rand(1, 3, 224, 224).astype(np.float32); ts = []
    for i in range(n + 10):
        t = time.perf_counter(); sx.run(None, {'input': x})
        if i >= 10: ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 2)
PREV = json.loads((OUT / 'int8_part1.json').read_text())
res = dict(PREV)
for m in [x for x in tm.MODELS if x not in PREV]:
    r = {}
    try:
        ck = torch.load(FIX4 / 'results/matched_training_b32' / f'{m}_seed42' / 'best.pt', map_location='cpu', weights_only=True)
        net = tm.build(m, 147, PROJECT, pretrained=False); net.load_state_dict(ck['state_dict']); net.eval()
        class W(torch.nn.Module):
            def __init__(s, n): super().__init__(); s.n = n
            def forward(s, x): o = s.n(x); return o[-1] if isinstance(o, (tuple, list)) else o
        f32 = OUT / 'onnx' / f'{m}_fp32.onnx'; i8 = OUT / 'onnx' / f'{m}_int8.onnx'
        torch.onnx.export(W(net), torch.rand(1, 3, 224, 224), str(f32), input_names=['input'], output_names=['logits'], dynamic_axes={'input': {0: 'b'}, 'logits': {0: 'b'}}, opset_version=17, dynamo=False)
        Xc, Xt, yt = DATA['yolo26n' if m == 'yolo26n' else 'efficientnet_b0']
        quantize_static(str(f32), str(i8), Reader(Xc), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
        s32, s8 = sess(f32), sess(i8)
        r = {'fp32_MB': round(f32.stat().st_size / 2**20, 2), 'int8_MB': round(i8.stat().st_size / 2**20, 2),
             'fp32_ms_1thr': lat(s32), 'int8_ms_1thr': lat(s8), 'fp32_acc_3000': acc(s32, Xt, yt), 'int8_acc_3000': acc(s8, Xt, yt)}
    except Exception as e:
        r = {'error': repr(e)[:300]}
    res[m] = r; print(m, r, flush=True)
(OUT / 'int8_part2.json').write_text(json.dumps({'protocol': 'onnxruntime CPU, 1 intra-op thread, batch 1; static QDQ INT8, per-channel weights, 300 calibration-partition images; accuracy on first 3,000 (part 1) or 1,000 (part 2: remaining models) test images; seed-42 checkpoints', 'models': res}, indent=2))
