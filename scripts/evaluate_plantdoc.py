"""External test on PlantDoc (pinned commit, 12 source folders mapped to LeafCare labels).
Excludes images whose bytes or 224-px centre crop also occur in the LeafCare collection.
Evaluates all 24 runs (no seed selection). Read-only on checkpoints; writes into <delivery>/verification and supplement."""
import sys, csv, json, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score
OUT = Path(sys.argv[1]); FIX4 = Path(sys.argv[2]); PROJECT = Path(sys.argv[3])
sys.path.insert(0, str(FIX4 / 'code'))
import train_matched as tm
SRC = FIX4 / 'results/plantdoc_aligned_source'
ROOT = PROJECT / 'output/Leaf_Dataset_Combined/training_disease'
MAP = json.loads((ROOT / 'class_to_idx.json').read_text())
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
man = list(csv.DictReader((FIX4 / 'results/grouped_split/grouped_manifest.csv').open(newline='')))
raw_hash = {r['image_sha256'] for r in man}; crop_hash = {r['crop_rgb224_sha256'] for r in man}
geo = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225]); tt = transforms.ToTensor()
src = list(csv.DictReader((SRC / 'manifest.csv').open(newline='')))
seen = set(); keep = []; excl = {'duplicate within PlantDoc': 0, 'byte or crop overlap with LeafCare': 0}
XR = []
for r in src:
    if r['image_sha256'] in seen: excl['duplicate within PlantDoc'] += 1; continue
    seen.add(r['image_sha256'])
    with Image.open(SRC / 'images' / r['file']) as im: c = geo(im.convert('RGB'))
    if r['image_sha256'] in raw_hash or hashlib.sha256(c.tobytes()).hexdigest() in crop_hash: excl['byte or crop overlap with LeafCare'] += 1; continue
    keep.append(r); XR.append(tt(c))
XR = torch.stack(XR); XN = norm(XR)
y = np.array([MAP[r['label']] for r in keep]); labs = sorted(set(y.tolist()))
per_class_n = {r: sum(k['label'] == r for k in keep) for r in sorted({k['label'] for k in keep})}
def infer(net, X):
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), 64):
            o = net(X[i:i + 64].to(DEV)); o = o[-1] if isinstance(o, (tuple, list)) else o; out.append(o.float().cpu().numpy())
    return np.concatenate(out)
rows = []; pred_store = {}
for m in tm.MODELS:
    for s in (42, 43, 44):
        ck = torch.load(FIX4 / 'results/matched_training_b32' / f'{m}_seed{s}' / 'best.pt', map_location='cpu', weights_only=True)
        net = tm.build(m, 147, PROJECT, pretrained=False); net.load_state_dict(ck['state_dict']); net = net.to(DEV).eval()
        lg = infer(net, XR if m == 'yolo26n' else XN); p = lg.argmax(1)
        mask = np.full(147, -np.inf); mask[labs] = 0; pr = (lg + mask).argmax(1)
        rows.append({'model': m, 'seed': s, 'images': len(y), 'accuracy_147way': float((p == y).mean()), 'macro_f1_147way': float(f1_score(y, p, labels=labs, average='macro', zero_division=0)),
                     'predicted_same_crop': float(np.mean([list(MAP)[0] is None or [k for k, v in MAP.items() if v == pi][0].split('_')[0] == [k for k, v in MAP.items() if v == yi][0].split('_')[0] for pi, yi in zip(p, y)])),
                     'accuracy_restricted_12way': float((pr == y).mean())})
        pred_store[f'{m}_seed{s}'] = p
        print(rows[-1], flush=True); del net; torch.cuda.empty_cache()
with (OUT / 'supplement' / 'S8_plantdoc_external_per_run.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summ = {}
for m in tm.MODELS:
    rr = [r for r in rows if r['model'] == m]
    summ[m] = {k: [float(np.mean([r[k] for r in rr])), float(np.std([r[k] for r in rr], ddof=1))] for k in ('accuracy_147way', 'macro_f1_147way', 'accuracy_restricted_12way', 'predicted_same_crop')}
# per-class accuracy, EfficientNet-B0 mean over seeds
inv = {v: k for k, v in MAP.items()}
pc = {inv[l]: float(np.mean([(pred_store[f'efficientnet_b0_seed{s}'][y == l] == l).mean() for s in (42, 43, 44)])) for l in labs}
(OUT / 'verification' / 'plantdoc_external.json').write_text(json.dumps({'source': 'PlantDoc-Dataset (pratikkayal), commit 5467f601, train+test folders of 12 mapped classes', 'license': 'repository states CC BY 4.0',
    'source_files': len(src), 'evaluated_images': len(keep), 'excluded': excl, 'images_per_label': per_class_n, 'summary_mean_sd': summ, 'effnet_per_label_accuracy_mean_over_seeds': pc,
    'notes': 'Mapping from PlantDoc folder names to LeafCare labels is semantic, not expert-adjudicated. 147-way = unrestricted prediction; restricted = argmax among the 12 evaluated labels only. No near-duplicate (descriptor) screening beyond byte and crop hashes.'}, indent=2))
print('evaluated', len(keep), 'excluded', excl)
