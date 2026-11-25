"""'I don't know' (abstention) test. For every run: fit temperature on the CALIBRATION partition,
choose the confidence threshold on calibration so that accepted predictions are >=99% correct,
then apply both unchanged to the internal TEST partition and to the 141 unseen PlantDoc images.
No test or PlantDoc data are used for any choice. Read-only on checkpoints."""
import sys, csv, json, hashlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, torch
from torchvision import transforms
from PIL import Image
OUT = Path(sys.argv[1]); FIX4 = Path(sys.argv[2]); PROJECT = Path(sys.argv[3])
sys.path.insert(0, str(FIX4 / 'code'))
import train_matched as tm
RES = FIX4 / 'results/matched_training_b32'; SRC = FIX4 / 'results/plantdoc_aligned_source'
MAP = json.loads((PROJECT / 'output/Leaf_Dataset_Combined/training_disease/class_to_idx.json').read_text())
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
TARGET = 0.99

def softmax(z, T):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def nll(z, y, T): p = softmax(z, T); return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
def ece(p, y, bins=15):
    c = p.max(1); ok = p.argmax(1) == y; e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(ok[m].mean() - c[m].mean())
    return float(e)

# unseen PlantDoc images (same rule as the external probe)
man = list(csv.DictReader((FIX4 / 'results/grouped_split/grouped_manifest.csv').open(newline='')))
raw = {r['image_sha256'] for r in man}; crop = {r['crop_rgb224_sha256'] for r in man}
geo = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]); tt = transforms.ToTensor()
norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
seen = set(); XR = []; yp = []
for r in csv.DictReader((SRC / 'manifest.csv').open(newline='')):
    if r['image_sha256'] in seen: continue
    seen.add(r['image_sha256'])
    with Image.open(SRC / 'images' / r['file']) as im: c = geo(im.convert('RGB'))
    if r['image_sha256'] in raw or hashlib.sha256(c.tobytes()).hexdigest() in crop: continue
    XR.append(tt(c)); yp.append(MAP[r['label']])
XR = torch.stack(XR); XN = norm(XR); yp = np.array(yp)

rows = []
for m in tm.MODELS:
    for s in (42, 43, 44):
        d = RES / f'{m}_seed{s}'
        zc = np.load(d / 'calibration_logits.npz', allow_pickle=False); zt = np.load(d / 'test_logits.npz', allow_pickle=False)
        lc, yc, lt, yt = zc['logits'].astype(np.float64), zc['true_indices'], zt['logits'].astype(np.float64), zt['true_indices']
        Ts = np.linspace(0.5, 5, 91); T = float(Ts[np.argmin([nll(lc, yc, t) for t in Ts])])
        pc = softmax(lc, T); cc = pc.max(1); okc = pc.argmax(1) == yc
        thr = 1.0
        for t in np.sort(np.unique(np.round(cc, 4))):
            a = cc >= t
            if a.any() and okc[a].mean() >= TARGET: thr = float(t); break
        pt = softmax(lt, T); ct = pt.max(1); okt = pt.argmax(1) == yt; at = ct >= thr
        ck = torch.load(d / 'best.pt', map_location='cpu', weights_only=True)
        net = tm.build(m, 147, PROJECT, pretrained=False); net.load_state_dict(ck['state_dict']); net = net.to(DEV).eval()
        with torch.inference_mode():
            o = net((XR if m == 'yolo26n' else XN).to(DEV)); o = o[-1] if isinstance(o, (tuple, list)) else o
        lp = o.float().cpu().numpy().astype(np.float64); del net; torch.cuda.empty_cache()
        pp = softmax(lp, T); cp = pp.max(1); okp = pp.argmax(1) == yp; ap = cp >= thr
        rows.append(dict(model=m, seed=s, temperature=T, threshold=thr,
            test_ece_raw=ece(softmax(lt, 1), yt), test_ece_scaled=ece(pt, yt),
            test_coverage=float(at.mean()), test_acc_all=float(okt.mean()), test_acc_accepted=float(okt[at].mean()) if at.any() else None,
            plantdoc_coverage=float(ap.mean()), plantdoc_acc_all=float(okp.mean()), plantdoc_acc_accepted=float(okp[ap].mean()) if ap.any() else None,
            plantdoc_wrong_rejected=float((~ap & ~okp).sum() / max((~okp).sum(), 1))))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
with (OUT / 'abstention_per_run.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summ = {}
for m in tm.MODELS:
    rr = [r for r in rows if r['model'] == m]
    summ[m] = {k: [float(np.mean([r[k] for r in rr if r[k] is not None])), float(np.std([r[k] for r in rr if r[k] is not None], ddof=1))]
               for k in ('test_ece_raw', 'test_ece_scaled', 'test_coverage', 'test_acc_accepted', 'plantdoc_coverage', 'plantdoc_acc_all', 'plantdoc_acc_accepted', 'plantdoc_wrong_rejected')}
(OUT / 'abstention_summary.json').write_text(json.dumps({'target_accepted_accuracy_on_calibration': TARGET, 'plantdoc_images': int(len(yp)), 'summary_mean_sd': summ}, indent=2))
print('done')
