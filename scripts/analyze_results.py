"""LeafCare matched-benchmark analysis (CPU only, read-only on training outputs).

Usage: python -I analyze_results.py <delivery_dir>
Verifies every completed run, recomputes metrics, builds supplement tables/figures,
group-bootstrap paired comparisons and dataset summaries. Writes only inside <delivery_dir>.
"""
import sys, json, csv, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN = Path(r'C:/Users/Muazzam Khan/Documents/Codex/2026-11-05/now/work/fix4/results/matched_training_b32')
MANIFEST = Path(r'C:/Users/Muazzam Khan/Documents/Codex/2026-11-05/now/work/fix4/results/grouped_split/grouped_manifest.csv')
OUT = Path(sys.argv[1])
SUP = OUT / 'supplement'; FIG = OUT / 'manuscript' / 'figures'; VER = OUT / 'verification'
for p in (SUP, FIG, VER): p.mkdir(parents=True, exist_ok=True)
MODELS = ['mobilenet_v3_small', 'phytoedge_head_only', 'phytoedge_attention', 'phytoedge_fusion', 'phytoedge_net', 'efficientnet_b0', 'convnext_tiny', 'yolo26n']
NICE = {'mobilenet_v3_small': 'MobileNetV3-S', 'phytoedge_head_only': 'PE-Head', 'phytoedge_attention': 'PE-Attn', 'phytoedge_fusion': 'PE-Fusion',
        'phytoedge_net': 'PE-Full', 'efficientnet_b0': 'EfficientNet-B0', 'convnext_tiny': 'ConvNeXt-T', 'yolo26n': 'YOLO26n-cls'}
SEEDS = [42, 43, 44]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})

protocol = json.loads((TRAIN / 'protocol.json').read_text())
mapping = protocol['class_to_idx']; labels = [k for k, v in sorted(mapping.items(), key=lambda x: x[1])]; K = len(labels)
assert K == 147
man = pd.read_csv(MANIFEST, dtype=str)
manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
assert manifest_sha == protocol['manifest_sha256'], 'manifest hash differs from protocol'
test = man[man.clean_split == 'test'].reset_index(drop=True)

# ---------- dataset summaries ----------
split_counts = man.clean_split.value_counts().reindex(['train', 'val', 'calibration', 'test'])
assert split_counts.to_dict() == protocol['split_counts']
per_class = man.groupby(['label', 'clean_split']).size().unstack(fill_value=0)[['train', 'val', 'calibration', 'test']]
per_class['total'] = per_class.sum(1); per_class = per_class.loc[labels]
per_class.to_csv(SUP / 'S1_class_distribution_by_split.csv')
src = man.groupby(['source_container', 'clean_split']).size().unstack(fill_value=0)
src['labels'] = man.groupby('source_container').label.nunique(); src.to_csv(SUP / 'S2_source_container_by_split.csv')
crop = pd.Series([l.split('_')[0] for l in labels]).value_counts()
groups = man.groupby('duplicate_group').size()
dataset = {'images': len(man), 'labels': K, 'split_counts': split_counts.to_dict(), 'source_containers': int(man.source_container.nunique()),
           'groups': int(groups.size), 'non_singleton_groups': int((groups > 1).sum()), 'test_groups': int(test.duplicate_group.nunique()),
           'class_total_min': int(per_class.total.min()), 'class_total_median': float(per_class.total.median()), 'class_total_max': int(per_class.total.max()),
           'test_support_min': int(per_class.test.min()), 'test_support_median': float(per_class.test.median()), 'test_support_max': int(per_class.test.max()),
           'healthy_labels': int(sum('healthy' in l for l in labels)), 'manifest_sha256': manifest_sha}
(VER / 'dataset_summary.json').write_text(json.dumps(dataset, indent=2))
fig, ax = plt.subplots(figsize=(3.4, 2.2))
s = per_class.total.sort_values(ascending=False).values
ax.bar(range(K), s, width=1.0, color='#3b6ea5'); ax.set_yscale('log'); ax.set_xlabel('Label rank (147 labels, sorted by size)'); ax.set_ylabel('Images (log scale)')
ax.set_title('Images per label, all splits'); fig.tight_layout(); fig.savefig(FIG / 'fig_class_distribution.pdf'); fig.savefig(FIG / 'fig_class_distribution.png'); plt.close(fig)

# ---------- verify runs ----------
def load(p):
    with np.load(p, allow_pickle=False) as z:
        return z['logits'], z['true_indices'].astype(int), z['image_sha256']
runs = {}; checks = []; ref_y = ref_ids = None
for seed in SEEDS:
    for m in MODELS:
        name = f'{m}_seed{seed}'; d = TRAIN / name
        if not (d / 'complete.json').exists():
            continue
        assert json.loads((d / 'complete.json').read_text()).get('complete') is True
        f = d / 'test_logits.npz'; logits, y, ids = load(f)
        assert logits.shape == (len(test), K) and np.isfinite(logits).all()
        assert np.array_equal(ids.astype(str), test.image_sha256.values), f'{name}: test order differs from manifest'
        assert np.array_equal(y, test.label.map(mapping).values), f'{name}: labels differ from manifest mapping'
        if ref_y is None: ref_y, ref_ids = y, ids
        assert np.array_equal(y, ref_y) and np.array_equal(ids, ref_ids)
        pred = logits.argmax(1)
        cm = np.zeros((K, K), dtype=np.int64); np.add.at(cm, (y, pred), 1)
        tp = cm.diagonal(); sup = cm.sum(1); pc = cm.sum(0)
        P = np.divide(tp, pc, out=np.zeros(K), where=pc != 0); R = np.divide(tp, sup, out=np.zeros(K), where=sup != 0)
        F = np.divide(2 * P * R, P + R, out=np.zeros(K), where=(P + R) != 0)
        acc = float(tp.sum() / len(y)); mf1 = float(F.mean())
        saved = json.loads((d / 'test_metrics.json').read_text())
        ok = abs(acc - saved['accuracy']) < 1e-12 and abs(mf1 - saved['macro_f1']) < 1e-10 and saved['images'] == len(y)
        assert ok, (name, acc, mf1, saved)
        hist = json.loads((d / 'history.json').read_text())
        assert [h['epoch'] for h in hist] == list(range(1, protocol['epochs'] + 1)), f'{name}: incomplete history'
        best_epoch = 1 + int(np.argmax([h['val_macro_f1'] for h in hist]))
        runs[name] = dict(model=m, seed=seed, y=y, pred=pred, cm=cm, P=P, R=R, F=F, sup=sup, acc=acc, mf1=mf1, hist=hist, best_epoch=best_epoch,
                          resumed=(d / 'resume.json').exists())
        checks.append({'run': name, 'npz_sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'images': len(y), 'unique_ids': int(len(set(ids.tolist()))),
                       'finite_logits': True, 'order_matches_manifest_test': True, 'labels_match_mapping': True, 'matches_saved_metrics': True,
                       'history_epochs': len(hist), 'selected_epoch_by_val_macro_f1': best_epoch, 'resume_record_present': runs[name]['resumed']})
assert runs, 'no completed runs'
(VER / 'run_verification.json').write_text(json.dumps({'checked_at': datetime.datetime.now().astimezone().isoformat(), 'runs': checks}, indent=2))

# ---------- per-run outputs ----------
rows = []
for name, r in runs.items():
    d = SUP / 'per_run' / name; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'label': labels, 'precision': r['P'], 'recall': r['R'], 'f1': r['F'], 'support': r['sup']}).to_csv(d / 'classification_report.csv', index=False)
    pd.DataFrame(r['cm'], index=labels, columns=labels).to_csv(d / 'confusion_counts.csv')
    norm = r['cm'] / np.maximum(r['sup'][:, None], 1)
    pd.DataFrame(norm, index=labels, columns=labels).to_csv(d / 'confusion_row_normalized.csv')
    for kind, M in (('counts', np.log10(1 + r['cm'])), ('row_normalized', norm)):
        fig, ax = plt.subplots(figsize=(11, 11))
        im = ax.imshow(M, cmap='Blues', interpolation='nearest'); ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(labels, rotation=90, fontsize=2.6); ax.set_yticklabels(labels, fontsize=2.6)
        ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
        ax.set_title(f'{NICE[r["model"]]} seed {r["seed"]}: ' + ('log10(1 + count)' if kind == 'counts' else 'row-normalized (recall per true label)'))
        fig.colorbar(im, fraction=0.03); fig.tight_layout(); fig.savefig(d / f'confusion_{kind}.pdf'); fig.savefig(d / f'confusion_{kind}.png', dpi=200); plt.close(fig)
    off = r['cm'].copy(); np.fill_diagonal(off, 0); idx = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:30], off.shape))[0]
    pd.DataFrame([{'count': int(off[i, j]), 'true': labels[i], 'predicted': labels[j], 'share_of_true': off[i, j] / max(r['sup'][i], 1)} for i, j in idx if off[i, j] > 0]).to_csv(d / 'top_confused_pairs.csv', index=False)
    rows.append({'run': name, 'model': r['model'], 'seed': r['seed'], 'test_images': len(r['y']), 'accuracy': r['acc'], 'macro_precision': r['P'].mean(), 'macro_recall': r['R'].mean(), 'macro_f1': r['mf1'],
                 'weighted_precision': np.average(r['P'], weights=r['sup']), 'weighted_recall': np.average(r['R'], weights=r['sup']), 'weighted_f1': np.average(r['F'], weights=r['sup']),
                 'errors': int(len(r['y']) - r['cm'].trace()), 'labels_f1_below_0.8': int((r['F'] < .8).sum()), 'selected_epoch': r['best_epoch'], 'best_val_macro_f1': max(h['val_macro_f1'] for h in r['hist'])})
comp = pd.DataFrame(rows); comp['order'] = comp.model.map(MODELS.index); comp = comp.sort_values(['seed', 'order']).drop(columns='order')
comp.to_csv(OUT / 'model_comparison.csv', index=False)

# multi-seed summary only where all seeds exist
ms = []
for m in MODELS:
    sub = comp[comp.model == m]
    if len(sub) == len(SEEDS):
        ms.append({'model': m, 'n_seeds': 3, **{f'{k}_mean': sub[k].mean() for k in ['accuracy', 'macro_f1']}, **{f'{k}_sd': sub[k].std(ddof=1) for k in ['accuracy', 'macro_f1']}})
    else:
        ms.append({'model': m, 'n_seeds': len(sub), 'note': 'mean/SD not reported: not all three seeds complete'})
pd.DataFrame(ms).to_csv(OUT / 'multi_seed_summary.csv', index=False)

# ---------- paired group bootstrap (seed 42) ----------
gid = pd.factorize(test.duplicate_group)[0]; G = gid.max() + 1
rng = np.random.default_rng(20261116); B = 2000
W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float32)  # group weights per replicate
n_g = np.bincount(gid, minlength=G).astype(np.float32)
def corr_by_group(pred, y): return np.bincount(gid, weights=(pred == y).astype(float), minlength=G)
pairs = [('phytoedge_attention', 'phytoedge_head_only'), ('phytoedge_fusion', 'phytoedge_head_only'), ('phytoedge_net', 'phytoedge_head_only'),
         ('phytoedge_net', 'phytoedge_attention'), ('phytoedge_net', 'phytoedge_fusion'), ('phytoedge_head_only', 'mobilenet_v3_small'),
         ('phytoedge_net', 'mobilenet_v3_small'), ('efficientnet_b0', 'mobilenet_v3_small'), ('convnext_tiny', 'efficientnet_b0'), ('yolo26n', 'mobilenet_v3_small')]
boot = []
for seed in SEEDS:
    for a, b in pairs:
        ra, rb = runs.get(f'{a}_seed{seed}'), runs.get(f'{b}_seed{seed}')
        if ra is None or rb is None: continue
        diff_g = corr_by_group(ra['pred'], ra['y']) - corr_by_group(rb['pred'], rb['y'])
        stats = (W @ diff_g) / (W @ n_g) * 100
        obs = (ra['acc'] - rb['acc']) * 100
        a_only = int(((ra['pred'] == ra['y']) & (rb['pred'] != rb['y'])).sum()); b_only = int(((ra['pred'] != ra['y']) & (rb['pred'] == rb['y'])).sum())
        boot.append({'seed': seed, 'model_a': a, 'model_b': b, 'acc_diff_pp': obs, 'ci95_low_pp': float(np.percentile(stats, 2.5)), 'ci95_high_pp': float(np.percentile(stats, 97.5)),
                     'a_correct_b_wrong': a_only, 'b_correct_a_wrong': b_only, 'macro_f1_diff_pp': (ra['mf1'] - rb['mf1']) * 100,
                     'resampling_unit': 'duplicate_group (test split)', 'replicates': B})
pd.DataFrame(boot).to_csv(OUT / 'paired_group_bootstrap.csv', index=False)

# ---------- ablation per-class view (seed 42) ----------
abl = [m for m in ['phytoedge_head_only', 'phytoedge_attention', 'phytoedge_fusion', 'phytoedge_net'] if f'{m}_seed42' in runs]
if len(abl) == 4:
    t = pd.DataFrame({NICE[m]: runs[f'{m}_seed42']['F'] for m in abl}, index=labels); t['support'] = runs['phytoedge_net_seed42']['sup']
    t['full_minus_head'] = t['PE-Full'] - t['PE-Head']; t.sort_values('full_minus_head').to_csv(SUP / 'S4_ablation_per_class_f1_seed42.csv')

# ---------- hardest labels across models (seed 42) ----------
s42 = [n for n in runs if n.endswith('_seed42')]
F42 = pd.DataFrame({NICE[runs[n]['model']]: runs[n]['F'] for n in s42}, index=labels)
F42['mean_f1'] = F42.mean(1); F42['test_support'] = runs[s42[0]]['sup']; F42['train_images'] = per_class.train.values
F42.sort_values('mean_f1').to_csv(SUP / 'S3_per_label_f1_all_models_seed42.csv')
hard = F42.sort_values('mean_f1').head(15)
# accuracy per source container (seed 42)
srcacc = pd.DataFrame({NICE[runs[n]['model']]: pd.Series(runs[n]['pred'] == runs[n]['y']).groupby(test.source_container.values).mean() for n in s42})
srcacc['test_images'] = test.source_container.value_counts(); srcacc.to_csv(SUP / 'S5_accuracy_by_source_container_seed42.csv')
# support vs F1 correlation
from_corr = {NICE[runs[n]['model']]: float(np.corrcoef(np.log(per_class.train.values), runs[n]['F'])[0, 1]) for n in s42}
(VER / 'support_f1_correlation_seed42.json').write_text(json.dumps({'pearson_log_train_count_vs_f1': from_corr}, indent=2))

# aggregated confusion pairs across seed-42 models (counts summed only for descriptive ranking; models are not independent samples)
tot = sum(runs[n]['cm'] for n in s42).astype(float); np.fill_diagonal(tot, 0)
top = np.dstack(np.unravel_index(np.argsort(-tot, axis=None)[:20], tot.shape))[0]
pd.DataFrame([{'true': labels[i], 'predicted': labels[j], 'errors_summed_over_models': int(tot[i, j]), 'models_with_error': int(sum(runs[n]['cm'][i, j] > 0 for n in s42)),
               'test_support': int(runs[s42[0]]['sup'][i])} for i, j in top]).to_csv(SUP / 'S6_recurrent_confusions_seed42.csv', index=False)

# ---------- figures ----------
c42 = comp[comp.seed == 42].copy()
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
order = c42.sort_values('accuracy').model.tolist()
for ax, key, title in zip(axs, ['accuracy', 'macro_f1'], ['Test accuracy (%)', 'Test macro-F1 (%)']):
    vals = c42.set_index('model').loc[order, key] * 100
    cols = ['#c0504d' if m.startswith('phytoedge') else '#3b6ea5' for m in order]
    ax.scatter(vals, range(len(order)), c=cols, s=18, zorder=3); ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE[m] for m in order])
    ax.grid(axis='x', color='#e3e3e3', lw=.6)
    for i, v in enumerate(vals): ax.text(v + .12, i, f'{v:.2f}', va='center', fontsize=6.5)
    ax.set_xlim(vals.min() - .6, vals.max() + .9); ax.set_xlabel(title + ' (dot plot; axis does not start at 0)')
axs[0].set_title('Seed 42, 18,148 test images (red: PhytoEdge variants)', loc='left', fontsize=8)
fig.tight_layout(); fig.savefig(FIG / 'fig_model_comparison_seed42.pdf'); fig.savefig(FIG / 'fig_model_comparison_seed42.png'); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for n in s42:
    r = runs[n]; e = [h['epoch'] for h in r['hist']]
    st = '--' if r['model'].startswith('phytoedge') else '-'
    axs[0].plot(e, [h['train_loss'] for h in r['hist']], st, lw=1, label=NICE[r['model']])
    axs[1].plot(e, [h['val_macro_f1'] * 100 for h in r['hist']], st, lw=1, label=NICE[r['model']])
axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Mean training cross-entropy'); axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Validation macro-F1 (%)')
axs[0].set_yscale('log'); axs[1].legend(fontsize=6, ncol=2, frameon=False)
for ax in axs: ax.set_xticks(range(1, 11))
fig.tight_layout(); fig.savefig(FIG / 'fig_learning_curves_seed42.pdf'); fig.savefig(FIG / 'fig_learning_curves_seed42.png'); plt.close(fig)

if boot:
    bt = pd.DataFrame(boot); bt = bt[bt.seed == 42]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lab = [f'{NICE[a]} − {NICE[b]}' for a, b in zip(bt.model_a, bt.model_b)]
    ax.errorbar(bt.acc_diff_pp, range(len(bt)), xerr=[bt.acc_diff_pp - bt.ci95_low_pp, bt.ci95_high_pp - bt.acc_diff_pp], fmt='o', ms=3, color='#333', capsize=2)
    ax.axvline(0, color='#999', lw=.8); ax.set_yticks(range(len(bt))); ax.set_yticklabels(lab, fontsize=6.5); ax.set_xlabel('A − B accuracy (pp)\n95% group-bootstrap interval', fontsize=7)
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(FIG / 'fig_paired_differences_seed42.pdf', bbox_inches='tight'); fig.savefig(FIG / 'fig_paired_differences_seed42.png', bbox_inches='tight'); plt.close(fig)

# readable confusion panel: union of labels in the 6 most frequent recurrent confusion pairs; best model and PE-Full
sel = []
for i, j in top:
    for k in (i, j):
        if k not in sel: sel.append(k)
    if len(sel) >= 10: break
sel = sel[:10]
show = [n for n in ['efficientnet_b0_seed42', 'phytoedge_net_seed42'] if n in runs]
fig, axs = plt.subplots(1, len(show), figsize=(7.0, 3.6))
axs = np.atleast_1d(axs)
short = [labels[k].replace('_', ' ')[:28] for k in sel]
for ax, n in zip(axs, show):
    r = runs[n]; M = r['cm'][np.ix_(sel, sel)]; N = M / np.maximum(r['sup'][sel][:, None], 1)
    ax.imshow(N, cmap='Blues', vmin=0, vmax=1)
    for a in range(len(sel)):
        for b in range(len(sel)): ax.text(b, a, str(M[a, b]), ha='center', va='center', fontsize=5.5, color='white' if N[a, b] > .6 else 'black')
    ax.set_xticks(range(len(sel))); ax.set_yticks(range(len(sel))); ax.set_xticklabels(range(1, len(sel) + 1), fontsize=6); ax.set_yticklabels([f'{i+1}. {s}' for i, s in enumerate(short)] if ax is axs[0] else [], fontsize=5.5)
    ax.set_title(NICE[r['model']] + ' (seed 42)', fontsize=7.5); ax.set_xlabel('Predicted (index)', fontsize=7)
fig.tight_layout(); fig.savefig(FIG / 'fig_confusion_subset_seed42.pdf'); fig.savefig(FIG / 'fig_confusion_subset_seed42.png'); plt.close(fig)
(VER / 'confusion_subset_selection.json').write_text(json.dumps({'rule': 'labels appearing in the most frequent off-diagonal confusions summed over all completed seed-42 models, taken in rank order until 10 unique labels; cell numbers are counts, shading is count / true-label support', 'labels': [labels[k] for k in sel]}, indent=2))

summary = {'completed_runs': sorted(runs), 'n_completed': len(runs), 'dataset': dataset, 'seed42': c42.set_index('run')[['accuracy', 'macro_f1', 'weighted_f1', 'errors', 'selected_epoch']].to_dict('index'),
           'hardest_labels_seed42': hard[['mean_f1', 'test_support', 'train_images']].reset_index().rename(columns={'index': 'label'}).to_dict('records'),
           'support_f1_corr': from_corr}
(VER / 'analysis_summary.json').write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps({'runs': sorted(runs), 'dataset': dataset}, indent=1, default=float))
