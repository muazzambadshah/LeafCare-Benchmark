"""Extra figures for V3: three-seed accuracy/macro-F1 with SD, and full 147x147 confusion
matrices (row-normalised) for EfficientNet-B0, one per seed. Reads saved logits only."""
import sys, json, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(sys.argv[1]); RES = Path(sys.argv[2]); FIG = OUT / 'manuscript' / 'figures'
NICE = {'mobilenet_v3_small': 'MobileNetV3-Small', 'phytoedge_head_only': 'PE-Head', 'phytoedge_attention': 'PE-Attn', 'phytoedge_fusion': 'PE-Fusion',
        'phytoedge_net': 'PE-Full', 'efficientnet_b0': 'EfficientNet-B0', 'convnext_tiny': 'ConvNeXt-Tiny', 'yolo26n': 'YOLO26n-cls'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})

# 1. three-seed accuracy and macro-F1 with SD and per-seed points
comp = list(csv.DictReader((OUT / 'model_comparison.csv').open(newline='')))
models = list(NICE)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, key, title in zip(axes, ('accuracy', 'macro_f1'), ('Accuracy (%)', 'Macro-F1 (%)')):
    vals = [[100 * float(r[key]) for r in comp if r['model'] == m] for m in models]
    means = [np.mean(v) for v in vals]; sds = [np.std(v, ddof=1) for v in vals]
    order = np.argsort(means); ypos = np.arange(len(models))
    for y, i in zip(ypos, order):
        ax.errorbar(means[i], y, xerr=sds[i], fmt='o', color='#1f4d2b', ms=4, capsize=2, lw=1)
        ax.scatter(vals[i], [y] * 3, s=9, color='#97bc62', zorder=3, alpha=.9)
    ax.set_yticks(ypos); ax.set_yticklabels([NICE[models[i]] for i in order]); ax.set_xlabel(title); ax.grid(axis='x', color='#dddddd', lw=.5)
fig.tight_layout()
for ext in ('pdf', 'png'): fig.savefig(FIG / f'fig_three_seed_comparison.{ext}', dpi=300)
plt.close(fig)

# 2. full row-normalised confusion matrix, EfficientNet-B0, each seed (one figure, three panels)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, seed in zip(axes, (42, 43, 44)):
    z = np.load(RES / f'efficientnet_b0_seed{seed}' / 'test_logits.npz', allow_pickle=False)
    y, p = z['true_indices'], z['logits'].argmax(1)
    cm = np.zeros((147, 147)); np.add.at(cm, (y, p), 1); cm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cm, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'Seed {seed}: acc {100 * (y == p).mean():.2f}%', fontsize=8)
    ax.set_xlabel('Predicted label index'); ax.set_ylabel('True label index' if seed == 42 else '')
    ax.set_xticks([0, 50, 100, 146]); ax.set_yticks([0, 50, 100, 146])
    if seed != 42: ax.set_yticklabels([])
fig.colorbar(im, ax=axes, fraction=.02, pad=.02, label='Fraction of true label')
for ext in ('pdf', 'png'): fig.savefig(FIG / f'fig_confusion_full_effnet.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('figures written')
