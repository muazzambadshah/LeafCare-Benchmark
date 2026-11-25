"""Pipeline/split and PhytoEdge ablation diagrams (drawn from train_matched.py, phytoedge_net.py, group_clean_split.py)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
FIG = Path(sys.argv[1]) / 'manuscript' / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'savefig.dpi': 300})

def box(ax, x, y, w, h, text, fc='#eef3f8', ec='#3b6ea5', fs=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012', fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)
def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#444'))

# Fig 1: pipeline (top row left to right, bottom row right to left)
fig, ax = plt.subplots(figsize=(7.0, 2.4)); ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1); ax.axis('off')
box(ax, .00, .55, .15, .38, '181,719 images\n147 labels\n8 source\ncontainers')
box(ax, .19, .55, .20, .38, 'Exact byte and\n224-crop hash matches\n+ descriptor cosine ≥ 0.99\n→ union-find groups\n(176,383 groups)', fs=5.6)
box(ax, .43, .55, .17, .38, 'Whole-group\nassignment,\nlabel-stratified\n≈70/10/10/10')
box(ax, .64, .55, .16, .38, 'Train 127,237\nVal 18,176\nCalib. 18,158\nTest 18,148', fc='#f6f6f6')
box(ax, .84, .55, .16, .38, '8 models × seeds\n42/43/44\n10 epochs, AdamW\nbatch 32, AMP', fs=5.8)
for a, b in [(.15, .19), (.39, .43), (.60, .64), (.80, .84)]: arrow(ax, a + .005, .74, b - .005, .74)
box(ax, .70, .05, .30, .33, 'Checkpoint = epoch with maximum\nvalidation macro-F1 (first on tie)', fc='#fff7e6', ec='#c48a1a')
box(ax, .42, .05, .24, .33, 'Single test evaluation\n(float32, TF32 off);\nlogits + image hashes saved', fc='#fff7e6', ec='#c48a1a')
box(ax, .14, .05, .24, .33, 'Offline recomputation:\nper-class P/R/F1, matrices,\ngroup bootstrap', fc='#eaf5ea', ec='#3c8d3c')
arrow(ax, .92, .55, .92, .39); arrow(ax, .70, .21, .665, .21); arrow(ax, .42, .21, .385, .21)
ax.text(-.02, .21, 'Calibration split\nreserved; not\nused here', fontsize=5.6, color='#555', va='center')
fig.savefig(FIG / 'fig_pipeline.pdf', bbox_inches='tight'); fig.savefig(FIG / 'fig_pipeline.png', bbox_inches='tight'); plt.close(fig)

# Fig 2: PhytoEdge variants
fig, ax = plt.subplots(figsize=(7.0, 2.9)); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.05, 1); ax.axis('off')
box(ax, .00, .40, .09, .22, 'RGB\n224×224')
box(ax, .12, .58, .20, .22, 'MobileNetV3-Small\nfeature blocks 0–8\n(48 channels)')
box(ax, .12, .18, .20, .22, 'Feature blocks 9–12\n(576 channels, 7×7)')
arrow(ax, .09, .51, .12, .65); arrow(ax, .22, .58, .22, .40)
box(ax, .355, .16, .19, .26, 'Channel gate\nGAP → 1×1 conv (576→72)\n→ ReLU → 1×1 conv (72→576)\n→ sigmoid, multiply', fc='#fdecea', ec='#c0504d', fs=4.8)
box(ax, .575, .16, .155, .26, 'Spatial gate\n[channel mean, max]\n→ 7×7 conv → sigmoid,\nmultiply', fc='#fdecea', ec='#c0504d', fs=4.8)
box(ax, .575, .58, .155, .22, 'Early features\nadaptive avg-pool\nto 7×7', fc='#eaf0fb', ec='#3b6ea5', fs=5.5)
box(ax, .76, .36, .115, .28, 'Fusion\nconcat (624 ch)\n→ 1×1 conv (576)\n→ BN → SiLU', fc='#eaf0fb', ec='#3b6ea5', fs=5.3)
box(ax, .895, .36, .12, .28, 'GAP ‖ GMP\n(1152-d)\n→ Dropout 0.3\n→ Linear (147)', fc='#f6f6f6', fs=5.3)
arrow(ax, .32, .29, .355, .29); arrow(ax, .545, .29, .575, .29); arrow(ax, .73, .29, .76, .42); arrow(ax, .32, .69, .575, .69); arrow(ax, .73, .69, .76, .58); arrow(ax, .875, .50, .895, .50)
ax.text(.0, -.02, 'Red: present in Attention and Full variants.  Blue: present in Fusion and Full variants.  Head-only: gates and fusion are identity mappings.\n'
        'Without fusion, the gated (or ungated) 576-channel map goes directly to the shared GAP ‖ GMP head.', fontsize=5.8)
ax.text(.0, .93, 'PhytoEdge ablation family (as implemented in phytoedge_net.py)', fontsize=7.5, fontweight='bold')
fig.savefig(FIG / 'fig_phytoedge_variants.pdf', bbox_inches='tight'); fig.savefig(FIG / 'fig_phytoedge_variants.png', bbox_inches='tight'); plt.close(fig)
print('ok')
