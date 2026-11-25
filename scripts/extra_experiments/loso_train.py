"""Leave-one-source-out (LOSO) experiment. For each held-out source archive, every duplicate group that
contains ANY image of that source is removed from training and validation; the model is trained on the
remaining training partition (same recipe as train_matched.py: AdamW 1e-4, wd 1e-4, batch 32, AMP,
10 epochs, validation macro-F1 checkpoint selection, seed 42) and then tested on ALL images of the
held-out source whose label also occurs in the reduced training set. Read-only on the original runs.
Resumable at epoch boundaries. Usage: python loso_train.py <out_dir> <fix4> <project>"""
import sys, csv, json, time, random
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
OUT = Path(sys.argv[1]); FIX4 = Path(sys.argv[2]); PROJECT = Path(sys.argv[3])
sys.path.insert(0, str(FIX4 / 'code'))
import train_matched as tm
ROOT = PROJECT / 'output/Leaf_Dataset_Combined/training_disease'
MAP = json.loads((ROOT / 'class_to_idx.json').read_text())
ROWS = list(csv.DictReader((FIX4 / 'results/grouped_split/grouped_manifest.csv').open(newline='')))
HELD = ['dataset - Copy', 'archive (16).zip', 'archive (20).zip']
MODELS = ['efficientnet_b0', 'mobilenet_v3_small']
SEED, EPOCHS, BS, DEV = 42, 10, 32, 'cuda'
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
def log(d): print(json.dumps(d), flush=True); (OUT / 'status.json').write_text(json.dumps(d, indent=2))
for held in HELD:
    bad_groups = {r['duplicate_group'] for r in ROWS if r['source_container'] == held}
    keep = [r for r in ROWS if r['duplicate_group'] not in bad_groups]
    train_rows = [r for r in keep if r['clean_split'] == 'train']; val_rows = [r for r in keep if r['clean_split'] == 'val']
    train_labels = {r['label'] for r in train_rows}
    test_rows = [r for r in ROWS if r['source_container'] == held and r['label'] in train_labels]
    in_domain = [r for r in keep if r['clean_split'] == 'test']
    for name in MODELS:
        tag = f'{name}__holdout_{held.replace(" ", "_").replace("(", "").replace(")", "").replace(".zip", "")}'
        out = OUT / 'runs' / tag; out.mkdir(parents=True, exist_ok=True)
        if (out / 'complete.json').exists(): continue
        info = {'held_out_source': held, 'removed_groups': len(bad_groups), 'train_images': len(train_rows), 'val_images': len(val_rows),
                'heldout_test_images': len(test_rows), 'heldout_labels_kept': len({r['label'] for r in test_rows}),
                'heldout_labels_dropped_not_in_train': sorted({r['label'] for r in ROWS if r['source_container'] == held} - train_labels), 'in_domain_test_images': len(in_domain)}
        (out / 'setup.json').write_text(json.dumps(info, indent=2))
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        net = tm.build(name, 147, PROJECT).to(DEV).requires_grad_(True)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4); scaler = torch.amp.GradScaler('cuda')
        gen = torch.Generator().manual_seed(SEED)
        tl = DataLoader(tm.Images(ROOT, train_rows, MAP, True, name), batch_size=BS, shuffle=True, generator=gen, num_workers=1, prefetch_factor=1, pin_memory=True, persistent_workers=True)
        vl = DataLoader(tm.Images(ROOT, val_rows, MAP, False, name), batch_size=BS, num_workers=1, prefetch_factor=1, pin_memory=True, persistent_workers=True)
        best, hist, first = -1., [], 1
        if (out / 'last.pt').exists():
            sv = torch.load(out / 'last.pt', weights_only=False, map_location='cpu')
            net.load_state_dict(sv['state_dict']); opt.load_state_dict(sv['optimizer']); scaler.load_state_dict(sv['scaler'])
            best, hist, first = sv['best'], sv['history'], sv['epoch'] + 1; gen.set_state(sv['sampler_rng']); torch.set_rng_state(sv['torch_rng'])
        for ep in range(first, EPOCHS + 1):
            net.train(); tot = n = 0; t0 = time.time()
            for step, (x, y) in enumerate(tl, 1):
                x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    o = net(x); o = o[-1] if isinstance(o, (tuple, list)) else o; loss = nn.functional.cross_entropy(o, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += float(loss.detach()) * len(y); n += len(y)
                if step % 400 == 0: log({'run': tag, 'epoch': ep, 'step': step, 'steps': len(tl), 'loss': tot / n, 'min': round((time.time() - t0) / 60, 1)})
            _, _, sc = tm.evaluate(net, vl, DEV); hist.append({'epoch': ep, 'train_loss': tot / n, 'val_macro_f1': sc, 'seconds': time.time() - t0})
            if sc > best: best = sc; torch.save({'state_dict': net.state_dict(), 'epoch': ep}, out / 'best.pt')
            torch.save({'epoch': ep, 'state_dict': net.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'best': best, 'history': hist,
                        'sampler_rng': gen.get_state(), 'torch_rng': torch.get_rng_state()}, out / 'last.pt')
            (out / 'history.json').write_text(json.dumps(hist, indent=2)); log({'run': tag, 'epoch_done': ep, 'val_macro_f1': sc})
        net.load_state_dict(torch.load(out / 'best.pt', weights_only=True)['state_dict'])
        res = {}
        for nm, rr in (('heldout_source', test_rows), ('in_domain_test', in_domain)):
            lg, y, _ = tm.evaluate(net, DataLoader(tm.Images(ROOT, rr, MAP, False, name), batch_size=64, num_workers=1), DEV)
            labs = sorted(set(y.tolist())); p = lg.argmax(1)
            res[nm] = {'images': len(y), 'accuracy': float((p == y).mean()), 'macro_f1': float(f1_score(y, p, labels=labs, average='macro', zero_division=0))}
            np.savez_compressed(out / f'{nm}_logits.npz', logits=lg, true_indices=y)
        (out / 'results.json').write_text(json.dumps(dict(info, **res), indent=2)); (out / 'complete.json').write_text('{"complete": true}')
        log({'run': tag, 'done': res}); del net; torch.cuda.empty_cache()
(OUT / 'LOSO_COMPLETE.json').write_text('{"complete": true}')
