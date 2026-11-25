"""Build the LeafCare manuscript from frozen verified tables.
Outputs: manuscript/LeafCare_Manuscript.tex (IEEEtran source), manuscript/references.bib,
manuscript/LeafCare_Manuscript_review_layout.pdf (ReportLab review layout; NOT IEEE-compiled),
verification/manuscript_numbers.json (every number used, with source file).
Usage: python build_manuscript.py <delivery_dir>"""
import sys, json, re
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1]); MS = OUT / 'manuscript'; FIGD = MS / 'figures'; VER = OUT / 'verification'
comp = pd.read_csv(OUT / 'model_comparison.csv'); boot = pd.read_csv(OUT / 'paired_group_bootstrap.csv')
ds = json.loads((VER / 'dataset_summary.json').read_text()); params = json.loads((VER / 'parameter_counts.json').read_text())['parameters']
S3 = pd.read_csv(OUT / 'supplement' / 'S3_per_label_f1_all_models_seed42.csv', index_col=0)
S5 = pd.read_csv(OUT / 'supplement' / 'S5_accuracy_by_source_container_seed42.csv', index_col=0)
S6 = pd.read_csv(OUT / 'supplement' / 'S6_recurrent_confusions_seed42.csv')
corr = json.loads((VER / 'support_f1_correlation_seed42.json').read_text())['pearson_log_train_count_vs_f1']
runver = json.loads((VER / 'run_verification.json').read_text())['runs']
NICE = {'mobilenet_v3_small': 'MobileNetV3-Small', 'phytoedge_head_only': 'PE-Head', 'phytoedge_attention': 'PE-Attn', 'phytoedge_fusion': 'PE-Fusion',
        'phytoedge_net': 'PE-Full', 'efficientnet_b0': 'EfficientNet-B0', 'convnext_tiny': 'ConvNeXt-Tiny', 'yolo26n': 'YOLO26n-cls'}
c42 = comp[comp.seed == 42].set_index('model'); done = list(c42.index); n_runs = len(comp)
yolo_done = 'yolo26n' in done
numbers = {}
def N(key, value, src):
    numbers[key] = {'value': value, 'source': src}; return value
def pct(m, k='accuracy'): return N(f'{m}_{k}_pct', f'{100 * c42.loc[m, k]:.2f}', 'model_comparison.csv')
def bt(a, b):
    r = boot[(boot.seed == 42) & (boot.model_a == a) & (boot.model_b == b)].iloc[0]
    return N(f'diff_{a}_vs_{b}', f'{r.acc_diff_pp:+.2f} pp (95% interval {r.ci95_low_pp:+.2f} to {r.ci95_high_pp:+.2f})', 'paired_group_bootstrap.csv'), r
best = c42.accuracy.idxmax()
pdx_n = json.loads((VER / 'plantdoc_external.json').read_text())['evaluated_images']
ms = pd.read_csv(OUT / 'multi_seed_summary.csv').set_index('model')
assert (ms.n_seeds == 3).all() and sorted(comp.seed.unique()) == [42, 43, 44]
def msd(m, k='accuracy'): return N(f'{m}_{k}_mean_sd', f'{100 * ms.loc[m, k + "_mean"]:.2f} \u00b1 {100 * ms.loc[m, k + "_sd"]:.2f}', 'multi_seed_summary.csv')
def seeds_txt(a, b):
    rs = boot[(boot.model_a == a) & (boot.model_b == b)].sort_values('seed')
    return N(f'seeds_{a}_vs_{b}', ', '.join(f'{r.acc_diff_pp:+.2f}' + ('*' if (r.ci95_low_pp > 0 or r.ci95_high_pp < 0) else '') for r in rs.itertuples()), 'paired_group_bootstrap.csv')
resumed = [r['run'] for r in runver if r['resume_record_present']]
hard = S3.sort_values('mean_f1').head(6)
f = lambda x: f'{x:,}'

# ---------------- content model ----------------
TITLE = 'Lightweight Deep Learning Models for Leaf Disease Classification'
AUTHOR = 'Muazzam Badshah'
AFFIL = 'National Center of Artificial Intelligence (NCAI), National University of Sciences and Technology (NUST), Islamabad, Pakistan'
EMAIL = 'muazzambadshah2003@gmail.com'
AUTHOR2 = 'Muhammad Asad Farooq'
AFFIL2 = 'Crop Diseases Research Institute, National Agricultural Research Centre (NARC), Park Road, Islamabad, Pakistan'
EMAIL2 = 'asadfarooq@parc.gov.pk'; ORCID2 = '0000-0002-8915-035X'
ABSTRACT = (
    'Leaf-disease classifiers often report accuracies near 99%, but studies differ in training budget, split method and number of runs, so their results are hard to compare. '
    f'We trained eight classifiers, from MobileNetV3-Small to ConvNeXt-Tiny, under one fixed ten-epoch protocol on {f(ds["images"])} leaf images with {ds["labels"]} labels, using a split that keeps duplicate and near-duplicate images in the same partition, and repeated every run with three seeds. '
    f'EfficientNet-B0 ({msd("efficientnet_b0")}%) and ConvNeXt-Tiny ({msd("convnext_tiny")}%) were the most accurate under this budget, although ConvNeXt-Tiny has 6.7 times more parameters. '
    'Adding channel/spatial attention or feature fusion to MobileNetV3-Small did not improve it, and the ranking of these variants changed between seeds. '
    f'On {pdx_n} unseen PlantDoc images accuracy fell to 33-59%, and naive INT8 quantization broke the smallest models. '
    'The results compare models under one internal protocol and do not establish field performance.')
KEYWORDS = 'plant disease classification, benchmarking, ablation study, data leakage, group-aware split, lightweight CNN, reproducibility'

secs = []
def H(t): secs.append(('h', t))
def H2(t): secs.append(('h2', t))
def P(t): secs.append(('p', t))
def L(items): secs.append(('list', items))
def FIG(file, cap, wide=True): secs.append(('fig', file, cap, wide))
def TAB(cap, header, rows, note=''): secs.append(('tab', cap, header, rows, note))

H('Introduction')
P('A farmer with a smartphone can photograph a sick leaf in seconds, which makes image-based disease recognition an appealing aid for crop protection. Studies on curated leaf collections regularly report held-out accuracies near 99% [mohanty2016], [ferentinos2018]. '
  'Those numbers rarely survive a change of setting. Mohanty et al. observed 31.4% accuracy on images gathered from online sources [mohanty2016]. A recent controlled-to-field study found that a ResNet-50 with 99.73% accuracy on PlantVillage reached 32.05% on PlantDoc crops, and common mitigation techniques did not close the gap [xiang2026]. '
  'Part of the problem lies in the data themselves: curated collections can carry backgrounds that correlate with the label [noyan2022], and they often hold the same photograph more than once. A model can exploit either, and a random split then rewards it with inflated test accuracy [geirhos2020], [kapoor2023].')
P('A second problem is that published comparisons are rarely like-for-like. Two papers comparing architectures will usually differ in training length, preprocessing, the way the data were split and the number of runs. Benchmark conclusions can change with random seeds and data sampling [bouthillier2021], and reporting recommendations such as DOME ask authors to make data, optimization, model and evaluation choices explicit [walsh2021dome]. '
  'Lightweight attention modules, such as squeeze-and-excitation [hu2018] and CBAM-style channel and spatial gating [woo2018], are often bolted onto mobile backbones, yet their effect is seldom measured against the same backbone trained the same way.')
P('We therefore set out to answer a practical question: under identical training conditions and a split that keeps duplicate images together, which lightweight model should a developer choose, and do popular attention add-ons actually help? This is an empirical study, not a new architecture. Its contributions are:')
L([f'A matched protocol in which eight classifiers share data, augmentation, optimizer, ten-epoch budget, mixed-precision training and validation-only checkpoint selection, on a {ds["labels"]}-label collection split by whole duplicate/near-duplicate groups.',
   'A controlled ablation of a MobileNetV3-Small-based family (PhytoEdge) that isolates channel/spatial gating and early-late feature fusion under that protocol. The ablation includes the unfavorable result that no PhytoEdge variant improved on the plain backbone across three seeds.',
   'An accuracy-cost comparison: model size and measured batch-1 inference latency for every model, showing that the most accurate small model matches a backbone with 6.7 times more parameters.',
   'Complete per-label precision, recall, F1 and confusion matrices, together with group-bootstrap paired comparisons, an error analysis and a reproducible verification trail from stored logits.'])
P(f'All eight models were trained with three pre-specified seeds (42, 43, 44), giving {n_runs} runs. Headline comparisons use the mean and sample standard deviation across seeds (Table II). Per-label, per-source and confusion analyses are shown for seed 42, the first pre-specified seed, which was fixed before any results were seen and not chosen by performance; per-seed paired intervals are reported for every seed.')

H('Related Work')
P('Controlled-condition collections. PlantVillage made tens of thousands of curated leaf images openly available [hughes2015] and enabled early CNN studies with very high internal accuracy [mohanty2016], [ferentinos2018]. '
  'Barbedo analyzed the factors that limit such systems, including limited dataset variety, background effects and variable symptom appearance [barbedo2018]. '
  'Noyan showed that a model using only eight background pixels per PlantVillage image reached 49.0% test accuracy (preprint) [noyan2022]. This suggests that capture artifacts can carry label information.')
P('Field-condition evaluation. PlantDoc provides 2,598 internet-sourced images across 13 species [singh2020plantdoc]. Ramcharan et al. tested a phone-deployed cassava model in a Tanzanian field and found lower F1 than on curated images [ramcharan2019]. '
  'Xiang et al. benchmarked fine-tuned and frozen backbones, calibration and domain-adaptation methods on a PlantVillage-to-PlantDoc shift, using parent-image grouped splits and repeated seeds. No configuration reached their target reliability [xiang2026]. A 2025 review summarizes a typical range of 95-99% in the laboratory against 70-85% in field deployment [shafay2025].')
P('Architectures and attention. MobileNetV3 [howard2019], EfficientNet [tan2019] and ConvNeXt [liu2022] are widely used ImageNet-pretrained backbones spanning roughly an order of magnitude in parameter count. YOLO26 includes classification models in a vendor-maintained family [jocher2026]. '
  'Channel recalibration [hu2018] and sequential channel/spatial attention [woo2018] are established modules. The PhytoEdge variants combine these known ideas, so we evaluate them as an ablation, not as a new mechanism.')
P('Evaluation practice. Leakage between training and evaluation data is a recurrent source of irreproducible ML findings [kapoor2023]. Shortcut solutions can pass standard benchmarks yet fail under realistic shifts [geirhos2020]. Variance across seeds and data samples can reverse model rankings [bouthillier2021]. '
  'Our study applies these principles to one assembled collection. Results from other papers are not compared numerically with ours because datasets, labels and partitions differ.')

H('Data and Split Methodology')
P(f'Collection. The collection contains {f(ds["images"])} RGB images from {ds["source_containers"]} source archives, merged into {ds["labels"]} labels of the form crop_condition ({ds["healthy_labels"]} healthy labels). '
  f'Label sizes are highly imbalanced: {ds["class_total_min"]} to {f(ds["class_total_max"])} images per label (median {ds["class_total_median"]:.0f}). One archive contributes {f(int(S5.loc["archive (20).zip", "test_images"]))} of the {f(ds["split_counts"]["test"])} test images (Supplementary Table S2). '
  'One archive (1,048 images, 10 healthy fruit-tree labels: loquat, peach, persimmon, orange, fig, apricot, apple, walnut, grape and cherry) was photographed by the first author in Battagram, Khyber Pakhtunkhwa, Pakistan, in August 2026, using iPhone and Infinix smartphones. A second set of 2,380 smartphone photographs of four healthy fruit-tree labels was also taken by the first author in August 2026. The other six archives are public collections: PlantVillage [hughes2015], PlantDoc [singh2020plantdoc], PlantCity [plantcity2025], the leaf database of Chouhan et al. [chouhan2019] and two compilations that the first author assembled from PlantVillage and other public image sets downloaded from Kaggle (Supplementary Table S11). '
  'The labels are those recorded in the source archives. They have not been independently confirmed by plant pathologists, and image-use permissions for every archive have not yet been fully documented.')
P(f'Duplicate grouping. Exact duplicates were identified by SHA-256 of file bytes and of the 224-pixel center crop. Near-duplicates were identified with an exact all-pairs threshold graph on L2-normalized 576-dimensional image descriptors at cosine similarity at least 0.99 ({f(5399)} edges). '
  f'Connected components, computed with union-find, form {f(ds["groups"])} groups. Of these, {f(ds["non_singleton_groups"])} contain more than one image, and the largest contains 7. '
  'In 623 groups (1,302 images), members carry different labels. These conflicts were left unchanged for every experiment reported here.')
P('Expert review of conflicting labels. After the experiments, a plant pathologist (M. A. Farooq) inspected all 623 conflicting groups in an image gallery. '
  'In 552 groups the two names describe the same disease: corn_cercospora_leaf_spot and corn_gray_leaf_spot (Cercospora zeae-maydis; 395 groups) and grape_isariopsis_leaf_spot and grape_leaf_blight (Isariopsis/Pseudocercospora leaf spot; 157 groups). '
  'Of the remaining 71 groups, 67 received a single correct label and 4 (8 images) could not be assigned because the image was unusable or neither name fitted. '
  'Applying these decisions merges two synonym pairs (147 to 145 labels; 4,202 images renamed), relabels 77 further images and excludes 8. '
  'The corrected manifest and a per-image change log are provided as supplementary material (S10). All models in this paper were trained and evaluated on the original 147-label manifest; retraining on the corrected labels is left for future work. '
  'Because the two synonym pairs were scored as separate labels, part of the remaining confusion between them is a labelling artefact rather than a model error.')
P(f'Partitioning (Fig. 1). Groups were assigned whole, rarest labels first, to the partition that minimized a chi-square-style deviation from 70/10/10/10 targets per label (fixed seed 20261110). Every label is present in every partition. '
  f'The split counts are {f(ds["split_counts"]["train"])} training, {f(ds["split_counts"]["val"])} validation, {f(ds["split_counts"]["calibration"])} calibration and {f(ds["split_counts"]["test"])} test images. '
  f'Per-label test support ranges from {ds["test_support_min"]} to {ds["test_support_max"]} (median {ds["test_support_median"]:.0f}). '
  'Grouping prevents identical or near-identical images from crossing partitions. It does not establish that test images come from different plants, fields or capture sessions, because no plant or session identifiers exist. The calibration partition is not used for training or checkpoint selection; it is used only to fit the temperature and confidence threshold for selective prediction.')
FIG('fig_pipeline', 'Data grouping, partitioning, training and evaluation pipeline. Counts are taken from the frozen manifest (SHA-256 3a791d98...) and the training protocol file.')

H('Models')
P('All models receive 224 x 224 RGB inputs and end in a 147-way linear classifier. Every parameter is trainable from the start (no frozen layers).')
L([f'MobileNetV3-Small [howard2019], EfficientNet-B0 [tan2019] and ConvNeXt-Tiny [liu2022] use torchvision ImageNet weights; only the final linear layer is replaced. ConvNeXt-Tiny is not a lightweight model; it is included as a larger reference point.',
   'YOLO26n-cls [jocher2026] uses the vendor yolo26n-cls pretrained weights with its final linear layer replaced. It is trained in the same loop as the other models, without the vendor trainer, its optimizer or EMA. It is used purely as an image classifier; no detection is performed.',
   'The PhytoEdge family (Fig. 2) uses the MobileNetV3-Small feature extractor (ImageNet weights). It replaces the original classifier with a dual-pooling head: global average and global max pooling are concatenated into a 1,152-dimensional vector, followed by dropout 0.3 and a linear layer. '
   'PE-Head uses only this head. PE-Attn adds a channel gate after the last feature block (global average pooling, 1x1 convolutions 576 to 72 to 576, ReLU, sigmoid) followed by a spatial gate (channel-wise mean and max maps, 7x7 convolution, sigmoid). '
   'PE-Fusion instead concatenates the 48-channel block-8 features, average-pooled to 7x7, with the 576-channel final map and projects the result back to 576 channels (1x1 convolution, batch normalization, SiLU). PE-Full applies the gates and then the fusion.'])
P('The gates are learned only from the classification loss. No lesion masks or localization targets are used, so the gates should not be interpreted as validated lesion localization.')
FIG('fig_phytoedge_variants', 'PhytoEdge ablation variants as implemented. The four variants differ only in the red (channel/spatial gating) and blue (early-late fusion) modules.')
TAB('Trainable parameters with the 147-way head (counted on CPU from the training code; not a measure of latency or file size).',
    ['Model', 'Parameters (M)'], [[NICE[m], f'{params[m] / 1e6:.2f}'] for m in ['phytoedge_head_only', 'phytoedge_attention', 'phytoedge_fusion', 'phytoedge_net', 'mobilenet_v3_small', 'yolo26n', 'efficientnet_b0', 'convnext_tiny']])

H('Training and Evaluation Protocol')
P('All eight models share one training script and one protocol file. Settings:')
L(['Optimizer and budget: AdamW [loshchilov2019] (learning rate 1e-4, weight decay 1e-4, no schedule), batch size 32, exactly ten epochs, no early stopping.',
   'Precision: mixed-precision training (float16 autocast with gradient scaling); cross-entropy loss.',
   'Augmentation: training uses random resized crop to 224 (scale 0.8-1.0) and horizontal flip. Evaluation resizes to 256 and takes a 224 center crop.',
   'Normalization: ImageNet mean/standard deviation for all models except YOLO26n-cls, which takes RGB in [0, 1] as its pretrained weights expect.',
   'Seeds: each seed (42, 43, 44) is fixed for Python, NumPy, PyTorch and the data-loader generator.',
   'Checkpoint selection: after every epoch, validation macro-F1 is computed over all 147 labels. The epoch with the highest value (first on ties) is kept, and only that checkpoint is evaluated once on the test partition, in float32 with TF32 disabled.',
   'Scope of the comparison: this is a fixed-budget comparison. The same learning rate and ten epochs suit some architectures better than others, so the rankings describe performance under this budget, not the best each architecture could reach with its own tuning.',
   f'Interrupted runs: a run interrupted mid-epoch resumes from the last saved epoch boundary. Model, optimizer, scaler and random-generator states are restored, but data-loader worker augmentation streams restart, so the continuation is not bit-exact. {len(resumed)} of the {n_runs} runs ({", ".join(NICE[r.rsplit("_seed", 1)[0]] + " seed " + r.rsplit("_seed", 1)[1] for r in resumed)}) were resumed in this way, and each has a resume record.',
   'Hardware and software: one NVIDIA RTX 4050 Laptop GPU (6 GB), PyTorch 2.15.0, torchvision 0.29.2, Ultralytics 8.4.202 and scikit-learn 1.9.1, under WSL Ubuntu.'])
P('Metrics. From the stored test logits we recompute:')
L(['accuracy;',
   'per-label precision, recall and F1, with a zero denominator mapped to zero;',
   'macro averages, which weight all 147 labels equally;',
   'support-weighted averages;',
   'count and row-normalized confusion matrices.'])
P('The recomputed accuracy and macro-F1 match each run\'s saved values (absolute tolerance 1e-10). Ordered test-image hashes and labels are identical across runs and match the manifest.')
P('Uncertainty. Paired differences between two models use the same test images. We resample duplicate groups of the test partition with replacement (2,000 replicates) and report percentile 95% intervals for the accuracy difference. These intervals are conditional on the trained checkpoints and the fixed split. They do not include seed-to-seed training variance, and the resampling unit is the duplicate group, not a biologically independent plant. We also report discordant counts (images that one model classifies correctly and the other does not). No test-based tuning, threshold selection or seed selection was performed.')

P('Software verification. After training finished, an automated test suite of 16 checks was run on the frozen artifacts. '
  'All 24 checkpoints loaded and, when re-run on the first 200 test images, reproduced the saved predictions exactly (no class changed in any run; largest logit difference 7.6e-6). '
  'Every accuracy and macro-F1 value in this paper was recomputed from the saved test logits, every mean and standard deviation was recomputed from the per-seed values, '
  'and no duplicate group or image hash was found in more than one partition. The test scripts and report are included in the supplementary package.')
H('Results')
ms_rows = []
for m in NICE:
    per = comp[comp.model == m].sort_values('seed')
    ms_rows.append([NICE[m], msd(m), msd(m, 'macro_f1')] + [f'{100 * a:.2f}' for a in per.accuracy])
TAB(f'Three-seed test results ({f(ds["split_counts"]["test"])} images, 147 labels). Mean \u00b1 sample SD over seeds 42, 43 and 44 in percent, with per-seed accuracy. Seeds were pre-specified and all are reported.',
    ['Model', 'Acc. mean \u00b1 SD', 'Macro F1 mean \u00b1 SD', 'Acc. s42', 'Acc. s43', 'Acc. s44'], ms_rows)
P(f'Across seeds, EfficientNet-B0 ({msd("efficientnet_b0")}%) and ConvNeXt-Tiny ({msd("convnext_tiny")}%) were the most accurate models. Their per-seed paired differences (ConvNeXt-Tiny minus EfficientNet-B0: {seeds_txt("convnext_tiny", "efficientnet_b0")} pp for seeds 42, 43, 44; * = 95% interval excludes zero) changed sign, so neither is consistently better. '
  f'EfficientNet-B0 exceeded MobileNetV3-Small at every seed ({seeds_txt("efficientnet_b0", "mobilenet_v3_small")} pp). YOLO26n-cls minus MobileNetV3-Small was {seeds_txt("yolo26n", "mobilenet_v3_small")} pp, so YOLO26n-cls was not consistently better in accuracy, although its mean macro-F1 ({msd("yolo26n", "macro_f1")}%) was higher than that of MobileNetV3-Small ({msd("mobilenet_v3_small", "macro_f1")}%). '
  'The seed-to-seed SD was 0.3-0.4 points for several models, comparable to many of the single-seed differences reported below (Fig. 3).')
FIG('fig_three_seed_comparison', 'Test accuracy (left) and macro-F1 (right) over three seeds. Dark dot = mean, bar = sample SD, light dots = individual seeds 42, 43 and 44. Models are sorted by mean; axes do not start at zero.')
rows = []
for m in [x for x in NICE if x in done]:
    r = c42.loc[m]
    rows.append([NICE[m], f'{100 * r.accuracy:.2f}', f'{100 * r.macro_precision:.2f}', f'{100 * r.macro_recall:.2f}', f'{100 * r.macro_f1:.2f}', f'{100 * r.weighted_f1:.2f}', str(int(r.errors)), str(int(r.selected_epoch))])
TAB(f'Seed-42 test results ({f(ds["split_counts"]["test"])} images, 147 labels). P, R, F1 in percent; Macro averages weight labels equally; W-F1 is support-weighted. Ep. = epoch selected by validation macro-F1.' + ('' if yolo_done else ' YOLO26n-cls: pending at freeze.'),
    ['Model', 'Acc.', 'Macro P', 'Macro R', 'Macro F1', 'W-F1', 'Errors', 'Ep.'], rows)
d_eff, r_eff = bt('efficientnet_b0', 'mobilenet_v3_small'); d_cx, r_cx = bt('convnext_tiny', 'efficientnet_b0')
P(f'Overall comparison (Fig. 4). EfficientNet-B0 had the highest seed-42 accuracy ({pct("efficientnet_b0")}%) and macro-F1 ({pct("efficientnet_b0", "macro_f1")}%), followed by ConvNeXt-Tiny ({pct("convnext_tiny")}%, {pct("convnext_tiny", "macro_f1")}%). '
  f'Relative to MobileNetV3-Small, EfficientNet-B0 differed by {d_eff}, with {int(r_eff.a_correct_b_wrong)} versus {int(r_eff.b_correct_a_wrong)} discordant images. ConvNeXt-Tiny minus EfficientNet-B0 was {d_cx}. '
  f'YOLO26n-cls, with a parameter count close to MobileNetV3-Small, reached {pct("yolo26n")}% accuracy and {pct("yolo26n", "macro_f1")}% macro-F1; YOLO26n-cls minus MobileNetV3-Small was {bt("yolo26n", "mobilenet_v3_small")[0]}. '
  'ConvNeXt-Tiny has about 6.7 times as many parameters as EfficientNet-B0 (Table I), yet it was not more accurate at seed 42 or on average. '
  f'Macro-F1 is {N("gap_min", f"{100 * (c42.accuracy - c42.macro_f1).min():.1f}", "model_comparison.csv")}-{N("gap_max", f"{100 * (c42.accuracy - c42.macro_f1).max():.1f}", "model_comparison.csv")} points lower than accuracy for every model, showing that errors are concentrated in a minority of labels rather than spread evenly (Table III).')
FIG('fig_model_comparison_seed42', 'Seed-42 test accuracy and macro-F1. Dot plot; the axis does not start at zero. Differences under about 0.3 points are within the paired intervals of Fig. 6.')
P(f'Learning dynamics. Training loss fell steadily for every model. The validation macro-F1 curves of the four PhytoEdge variants lie close together and below those of MobileNetV3-Small, EfficientNet-B0 and ConvNeXt-Tiny over most epochs (Fig. 5). {int((c42.selected_epoch >= 7).sum())} of the {len(done)} checkpoints were selected at epoch 7 or later, consistent with the ten-epoch budget being short for these models. Training accuracy and validation loss were not logged and are therefore not shown.')
FIG('fig_learning_curves_seed42', 'Recorded mean training cross-entropy (log scale) and validation macro-F1 per epoch, seed 42. Dashed lines: PhytoEdge variants.')

eff = json.loads((VER / 'efficiency.json').read_text()); E = eff['models']
H('Accuracy and Computational Cost')
TAB(f'Model size and inference latency (batch 1, 224 x 224, FP32, PyTorch eager, median of 100 runs after 20 warm-up runs; seed-42 checkpoints). CPU: {eff["hardware"]["cpu"]}; GPU: {eff["hardware"]["gpu"]}. Accuracy is the three-seed mean \u00b1 SD.',
    ['Model', 'Params (M)', 'Weights (MB)', 'CPU 1 thr. (ms)', 'CPU 4 thr. (ms)', 'GPU (ms)', 'Acc. mean \u00b1 SD'],
    [[NICE[m], f'{E[m]["params_M"]:.2f}', f'{E[m]["fp32_weights_MB"]:.1f}', f'{E[m]["cpu_1thread_ms"]:.1f}', f'{E[m]["cpu_4thread_ms"]:.1f}', f'{E[m]["gpu_ms"]:.1f}', msd(m)] for m in NICE])
P(f'Table IV relates accuracy to cost. EfficientNet-B0 matched ConvNeXt-Tiny in accuracy while using {E["convnext_tiny"]["params_M"] / E["efficientnet_b0"]["params_M"]:.1f} times fewer parameters and running {E["convnext_tiny"]["cpu_1thread_ms"] / E["efficientnet_b0"]["cpu_1thread_ms"]:.1f} times faster on one CPU thread ({E["efficientnet_b0"]["cpu_1thread_ms"]:.0f} versus {E["convnext_tiny"]["cpu_1thread_ms"]:.0f} ms). '
  f'The small models (MobileNetV3-Small, YOLO26n-cls and the PhytoEdge variants) all ran in {min(E[m]["cpu_1thread_ms"] for m in NICE if E[m]["params_M"] < 2):.1f}-{max(E[m]["cpu_1thread_ms"] for m in NICE if E[m]["params_M"] < 2):.1f} ms on one CPU thread with 4-7 MB of FP32 weights, but were 1.5-2.2 points less accurate than EfficientNet-B0. '
  f'Relative to PE-Head, the gates and fusion added up to {E["phytoedge_net"]["params_M"] - E["phytoedge_head_only"]["params_M"]:.2f} M parameters and did not make the variants faster or more accurate than the plain MobileNetV3-Small. '
  'At batch size 1 the laptop GPU times (3-6 ms) are dominated by launch overhead and do not separate the models. These are laptop measurements in PyTorch; phone latency and energy use were not measured, and exported INT8 models are evaluated separately below.')
H('Ablation of PhytoEdge Components')
dA, rA = bt('phytoedge_attention', 'phytoedge_head_only'); dF, rF = bt('phytoedge_fusion', 'phytoedge_head_only'); dN, rN = bt('phytoedge_net', 'phytoedge_head_only')
dNF, _ = bt('phytoedge_net', 'phytoedge_fusion'); dNA, _ = bt('phytoedge_net', 'phytoedge_attention'); dH, rH = bt('phytoedge_head_only', 'mobilenet_v3_small'); dNM, _ = bt('phytoedge_net', 'mobilenet_v3_small')
abl_rows = []
for m, gate, fus in [('mobilenet_v3_small', 'original MLP head', '-'), ('phytoedge_head_only', 'no', 'no'), ('phytoedge_attention', 'yes', 'no'), ('phytoedge_fusion', 'no', 'yes'), ('phytoedge_net', 'yes', 'yes')]:
    r = c42.loc[m]
    if m == 'phytoedge_head_only': dtxt = 'reference'
    else:
        b = boot[(boot.seed == 42) & (boot.model_a == m) & (boot.model_b == 'phytoedge_head_only')] if m != 'mobilenet_v3_small' else boot[(boot.seed == 42) & (boot.model_a == 'phytoedge_head_only') & (boot.model_b == 'mobilenet_v3_small')]
        b = b.iloc[0]; sgn = -1 if m == 'mobilenet_v3_small' else 1
        lo, hi_ = sorted([sgn * b.ci95_low_pp, sgn * b.ci95_high_pp]); dtxt = f'{sgn * b.acc_diff_pp:+.2f} [{lo:+.2f}, {hi_:+.2f}]'
    abl_rows.append([NICE[m], gate, fus, f'{params[m] / 1e6:.2f}', f'{100 * r.accuracy:.2f}', f'{100 * r.macro_f1:.2f}', dtxt, str(int(r.selected_epoch))])
TAB('Ablation of PhytoEdge components, seed 42 (all on the same MobileNetV3-Small backbone, protocol and test set). Gates = channel and spatial gating; Fusion = early-late feature fusion. Delta = accuracy difference from PE-Head in percentage points with 95% duplicate-group bootstrap interval. Ep. = selected epoch.',
    ['Variant', 'Gates', 'Fusion', 'Params (M)', 'Acc.', 'Macro F1', 'Delta vs PE-Head [95% CI]', 'Ep.'], abl_rows,
    )
P(f'Table V gives the seed-42 ablation. Against the head-only variant, adding the channel/spatial gates gave {dA}, adding fusion gave {dF}, and combining both (PE-Full) gave {dN}. PE-Full was also below PE-Fusion ({dNF}) and PE-Attn ({dNA}). '
  f'For macro-F1, PE-Head was highest and PE-Full lowest: PE-Head {pct("phytoedge_head_only", "macro_f1")}%, PE-Attn {pct("phytoedge_attention", "macro_f1")}%, PE-Fusion {pct("phytoedge_fusion", "macro_f1")}%, PE-Full {pct("phytoedge_net", "macro_f1")}%.')
P(f'Every PhytoEdge variant was also below plain MobileNetV3-Small, which shares the backbone but keeps its original classifier (hidden 1,024-unit layer with hardswish and dropout): PE-Head minus MobileNetV3-Small was {dH}, and PE-Full minus MobileNetV3-Small was {dNM}. '
  f'PE-Full\'s checkpoint was selected at epoch {int(c42.loc["phytoedge_net", "selected_epoch"])}, earlier than the other variants. Its validation curve plateaued from epoch 6 onward.')
S4 = pd.read_csv(OUT / 'supplement' / 'S4_ablation_per_class_f1_seed42.csv', index_col=0)
P(f'Per-label view. Comparing PE-Full with PE-Head label by label (Supplementary Table S4), PE-Full had higher F1 on {int((S4.full_minus_head > 0).sum())} labels, lower on {int((S4.full_minus_head < 0).sum())} and equal on {int((S4.full_minus_head == 0).sum())}. '
  f'The largest losses were on {", ".join(i.replace("_", " ") for i in S4.index[:3])}. Many large per-label changes occur on labels with few test images, so they should not be read as systematic effects.')
abl3 = []
for m, gate, fus in [('mobilenet_v3_small', 'original MLP head', '-'), ('phytoedge_head_only', 'no', 'no'), ('phytoedge_attention', 'yes', 'no'), ('phytoedge_fusion', 'no', 'yes'), ('phytoedge_net', 'yes', 'yes')]:
    if m == 'phytoedge_head_only': dl = 'reference'
    elif m == 'mobilenet_v3_small': dl = seeds_txt('phytoedge_head_only', 'mobilenet_v3_small') + ' (PE-Head minus this)'
    else: dl = seeds_txt(m, 'phytoedge_head_only')
    abl3.append([NICE[m], gate, fus, f'{params[m] / 1e6:.2f}', msd(m), msd(m, 'macro_f1'), dl])
TAB('Three-seed ablation of PhytoEdge components (mean \u00b1 sample SD over seeds 42, 43, 44, in percent). Delta = per-seed accuracy difference from PE-Head in percentage points for seeds 42/43/44; * = 95% duplicate-group bootstrap interval excludes zero.',
    ['Variant', 'Gates', 'Fusion', 'Params (M)', 'Acc. mean \u00b1 SD', 'Macro F1 mean \u00b1 SD', 'Delta vs PE-Head, s42/s43/s44'], abl3)
P(f'Across seeds the picture is less clear-cut than at seed 42 (Table VI). Mean accuracies were PE-Head {msd("phytoedge_head_only")}%, PE-Attn {msd("phytoedge_attention")}%, PE-Fusion {msd("phytoedge_fusion")}% and PE-Full {msd("phytoedge_net")}%, against {msd("mobilenet_v3_small")}% for MobileNetV3-Small. '
  f'Per-seed paired differences (pp, seeds 42/43/44; * = 95% interval excludes zero) were: PE-Full minus PE-Head {seeds_txt("phytoedge_net", "phytoedge_head_only")}; PE-Attn minus PE-Head {seeds_txt("phytoedge_attention", "phytoedge_head_only")}; PE-Fusion minus PE-Head {seeds_txt("phytoedge_fusion", "phytoedge_head_only")}; PE-Full minus PE-Fusion {seeds_txt("phytoedge_net", "phytoedge_fusion")}; PE-Head minus MobileNetV3-Small {seeds_txt("phytoedge_head_only", "mobilenet_v3_small")}; PE-Full minus MobileNetV3-Small {seeds_txt("phytoedge_net", "mobilenet_v3_small")}. '
  'The seed-42 finding that PE-Full was below PE-Head therefore did not replicate. Results that held at all three seeds are: PE-Head and PE-Full were below plain MobileNetV3-Small, and PE-Full was below PE-Fusion. Adding gates or fusion to the head-only variant helped at seeds 43 and 44 but not at seed 42, which mainly indicates that the head-only variant is itself unstable. '
  'Overall, no gating or fusion configuration improved on the original MobileNetV3-Small classifier. The fixed learning rate and ten-epoch budget were not tuned per variant, so slower-converging variants may be disadvantaged [bouthillier2021].')
FIG('fig_paired_differences_seed42', 'Paired accuracy differences (A minus B) at seed 42 with 95% percentile intervals from 2,000 duplicate-group bootstrap replicates of the test partition. Intervals are conditional on the trained checkpoints.', wide=False)

H('Error Analysis')
top3 = S6.head(3)
P(f'Recurrent confusions. Errors concentrate in within-crop pairs with similar visual symptoms (Supplementary Table S6). Summed over the {len(done)} seed-42 models, the most frequent confusions are:')
L([f'{a.true.replace("_", " ")} predicted as {a.predicted.replace("_", " ")}: {int(a.errors_summed_over_models)} errors, made by all {int(a.models_with_error)} models.' for a in top3.itertuples()]
  + ['wheat tan spot and wheat leaf blight confused in both directions.'])
P('Fig. 7 shows these labels for the best model and for PE-Full. PE-Full spreads cotton errors across several cotton labels, whereas EfficientNet-B0 keeps most of them on the diagonal.')
FIG('fig_confusion_subset_seed42', 'Count confusion matrices restricted to 10 labels involved in the most frequent cross-model confusions (selection rule: labels from the top-ranked summed off-diagonal pairs, in order, until 10 unique labels). Shading: count divided by true-label test support. Full 147 x 147 matrices are in the supplement.')
P(f'Full confusion matrices. Fig. 8 shows the complete 147 x 147 row-normalised confusion matrix of EfficientNet-B0 for each seed. Almost all mass lies on the diagonal, and the off-diagonal entries are small and concentrated in a few within-crop blocks, consistent with the recurrent confusions listed above. The pattern is very similar across seeds.')
FIG('fig_confusion_full_effnet', 'Row-normalised 147 x 147 confusion matrices of EfficientNet-B0 on the test partition (18,148 images) for seeds 42, 43 and 44. Label indices follow the label index list in Supplementary Table S7. Shading = fraction of each true label assigned to each predicted label.')
hl = '; '.join(f'{i.replace("_", " ")} (mean F1 {r.mean_f1:.2f}, {int(r.test_support)} test images)' for i, r in hard.iterrows())
P(f'Hard and low-support labels. The labels with the lowest mean F1 across models are: {hl}. Several have fewer than 60 test images, so their per-label estimates are imprecise. '
  f'Per-label F1 correlates moderately with the logarithm of training-set size (Pearson r = {min(corr.values()):.2f} to {max(corr.values()):.2f} across models). This is consistent with, but does not prove, a data-quantity effect.')
r14 = S5.loc['archive (14).zip']
P(f'Source dependence. Accuracy differs strongly by source archive (Supplementary Table S5). On one archive with {int(r14.test_images)} test images and 15 labels, accuracy ranged from {100 * r14.drop("test_images").min():.1f}% to {100 * r14.drop("test_images").max():.1f}%, compared with above 90% for every other archive. '
  f'On the Battagram archive ({int(S5.loc["leaf data.zip","test_images"])} test images, healthy leaves only), accuracy ranged from {100 * S5.loc["leaf data.zip"].drop("test_images").min():.1f}% to {100 * S5.loc["leaf data.zip"].drop("test_images").max():.1f}%. This archive is part of the internal split and is too small and too narrow (no diseased leaves) to serve as a field validation. '
  'Labels, crops and capture conditions are confounded with source, so we cannot attribute this gap to any single cause. However, it shows that the overall accuracy averages over very different subpopulations.')

pdx = json.loads((VER / 'plantdoc_external.json').read_text()); PX = pdx['summary_mean_sd']
def pxs(m, k): return N(f'plantdoc_{m}_{k}', f'{100 * PX[m][k][0]:.1f} \u00b1 {100 * PX[m][k][1]:.1f}', 'verification/plantdoc_external.json')
H('Exploratory External Probe on PlantDoc')
P(f'To test transfer beyond the internal split, we used PlantDoc [singh2020plantdoc], a collection of internet-sourced images taken mostly outside the laboratory. '
  f'Twelve PlantDoc disease folders have a direct LeafCare counterpart (for example, Tomato leaf late blight to tomato late blight); folders for crops without a disease name, and names with ambiguous or multiple LeafCare targets, were excluded before any model was run. '
  f'Of the {pdx["source_files"]:,} images in these folders, {pdx["excluded"]["byte or crop overlap with LeafCare"]:,} were byte-identical to, or had the same 224-pixel centre crop as, an image already in the LeafCare collection, and {pdx["excluded"]["duplicate within PlantDoc"]} were repeated within PlantDoc. '
  f'PlantDoc is therefore largely contained in one of our source archives, and only {pdx["evaluated_images"]} images are genuinely unseen. '
  f'They cover {len(pdx["images_per_label"])} labels and are dominated by corn common rust ({pdx["images_per_label"]["corn_common_rust"]} images); the other labels have 1-5 images each. We evaluated all 24 runs without retraining. With so few images, mostly from one label, this is an exploratory probe and not an external validation.')
TAB(f'External probe on {pdx["evaluated_images"]} unseen PlantDoc images (mean \u00b1 SD over seeds 42, 43, 44, in percent). Acc. = unrestricted 147-way top-1 accuracy; Macro F1 over the {len(pdx["images_per_label"])} evaluated labels; Restricted = accuracy when the prediction is limited to those labels; Same crop = prediction names the correct crop.',
    ['Model', 'Acc.', 'Macro F1', 'Restricted acc.', 'Same crop'],
    [[NICE[m], pxs(m, 'accuracy_147way'), pxs(m, 'macro_f1_147way'), pxs(m, 'accuracy_restricted_12way'), pxs(m, 'predicted_same_crop')] for m in NICE])
P(f'Accuracy fell (Table VII) from 95-97% on the internal test partition to {100 * min(v["accuracy_147way"][0] for v in PX.values()):.0f}-{100 * max(v["accuracy_147way"][0] for v in PX.values()):.0f}% on these images. '
  f'Most errors stayed within the correct crop (same-crop rate {100 * min(v["predicted_same_crop"][0] for v in PX.values()):.0f}-{100 * max(v["predicted_same_crop"][0] for v in PX.values()):.0f}%), so the models usually recognised the plant but not the condition. '
  f'Seed-to-seed variation was much larger than on the internal split: ConvNeXt-Tiny ranged from {100 * min(r for r in [0.1986, 0.7943, 0.7872]):.0f}% to {100 * max(r for r in [0.1986, 0.7943, 0.7872]):.0f}% across seeds. '
  f'EfficientNet-B0 had the highest mean macro-F1 ({pxs("efficientnet_b0", "macro_f1_147way")}%), and no PhytoEdge variant exceeded MobileNetV3-Small in mean accuracy. '
  'Because the probe is small and dominated by one label, these numbers are indicative only. They nevertheless show that internal accuracy greatly overstates performance on unseen, uncontrolled images, and that internal rankings do not transfer reliably.')
AB = json.loads((VER / 'abstention_summary.json').read_text())['summary_mean_sd']; Q = json.loads((VER / 'int8_summary.json').read_text())['models']
def ab(m, k): return N(f'abst_{m}_{k}', f'{100 * AB[m][k][0]:.1f} \u00b1 {100 * AB[m][k][1]:.1f}', 'verification/abstention_summary.json')
H('Selective Prediction and Compression')
P('Selective prediction. A deployed tool should be able to say that it is not sure. For every run we fitted a temperature on the calibration partition and chose, also on the calibration partition, the lowest softmax-confidence threshold at which accepted predictions were at least 99% correct. '
  'Both values were then applied unchanged to the internal test partition and to the 141 unseen PlantDoc images (Table VIII); no test or PlantDoc image was used for any choice.')
TAB('Selective prediction with a confidence threshold set on the calibration partition (target: 99% accuracy on accepted predictions). Mean \u00b1 SD over three seeds, in percent. Coverage = share of images answered; Wrong rejected = share of incorrect PlantDoc predictions that were withheld.',
    ['Model', 'Test coverage', 'Test acc. accepted', 'PlantDoc coverage', 'PlantDoc acc. accepted', 'Wrong rejected'],
    [[NICE[m], ab(m, 'test_coverage'), ab(m, 'test_acc_accepted'), ab(m, 'plantdoc_coverage'), ab(m, 'plantdoc_acc_accepted'), ab(m, 'plantdoc_wrong_rejected')] for m in NICE])
P(f'On the internal test partition the threshold transferred well: every model kept 99% accuracy on accepted images while answering {100 * min(v["test_coverage"][0] for v in AB.values()):.0f}-{100 * max(v["test_coverage"][0] for v in AB.values()):.0f}% of them. '
  f'On the unseen PlantDoc images the models answered far less often ({100 * min(v["plantdoc_coverage"][0] for v in AB.values()):.0f}-{100 * max(v["plantdoc_coverage"][0] for v in AB.values()):.0f}%) and withheld {100 * min(v["plantdoc_wrong_rejected"][0] for v in AB.values()):.0f}-{100 * max(v["plantdoc_wrong_rejected"][0] for v in AB.values()):.0f}% of their wrong answers, which raised the accuracy of the answers they did give. '
  'However, the accepted PlantDoc answers were still only about 52-77% correct, well below the 99% target, so a threshold set on in-distribution data does not guarantee reliability under shift. In an application, withheld cases should be referred to an expert.')
def qrow(m):
    q = Q[m]
    if 'error' in q: return [NICE[m], 'export failed', '-', '-', '-', '-', '-']
    return [NICE[m], f'{q["fp32_MB"]:.1f}', f'{q["int8_MB"]:.1f}', f'{q["fp32_ms_1thr"]:.1f}', f'{q["int8_ms_1thr"]:.1f}', f'{100 * q["fp32_acc_3000"]:.1f}', f'{100 * q["int8_acc_3000"]:.1f}']
P('Post-training quantization. To estimate behaviour on phones, each seed-42 checkpoint was exported to ONNX and quantized to 8-bit integers (static QDQ quantization with per-channel weights, calibrated on calibration-partition images) without retraining. This table reports size, single-thread CPU latency in ONNX Runtime and top-1 accuracy on a fixed subset of the test partition (Table IX).')
TAB('ONNX FP32 versus static INT8 post-training quantization (seed-42 checkpoints; ONNX Runtime CPU, one thread, batch 1). Accuracy on the first 3,000 test images (MobileNetV3-Small, PE-Head, PE-Attn, EfficientNet-B0) or the first 1,000 (ConvNeXt-Tiny, YOLO26n-cls). PE-Fusion and PE-Full could not be exported because their adaptive pooling has a non-constant output size. Because the subset size differs, compare FP32 with INT8 within a model, not across models.',
    ['Model', 'FP32 (MB)', 'INT8 (MB)', 'FP32 (ms)', 'INT8 (ms)', 'FP32 acc.', 'INT8 acc.'], [qrow(m) for m in NICE])
P(f'Quantization shrank every exported model by about 3.3-3.9 times, but its effect on accuracy depended strongly on the architecture. '
  f'ConvNeXt-Tiny lost no accuracy ({100 * Q["convnext_tiny"]["fp32_acc_3000"]:.1f}% to {100 * Q["convnext_tiny"]["int8_acc_3000"]:.1f}%) and YOLO26n-cls lost {100 * (Q["yolo26n"]["fp32_acc_3000"] - Q["yolo26n"]["int8_acc_3000"]):.1f} points while running twice as fast. '
  f'In contrast, EfficientNet-B0 fell to {100 * Q["efficientnet_b0"]["int8_acc_3000"]:.0f}%, and MobileNetV3-Small and the PhytoEdge variants collapsed to below 5%. '
  'These backbones use squeeze-and-excitation blocks and hard-swish or SiLU activations, which are known to be sensitive to naive post-training quantization. Small mobile models therefore need quantization-aware training or per-layer fallback before INT8 deployment; we did not test these remedies.')
H('Discussion')
P('For a developer choosing a model, the main message is simple. Under a matched ten-epoch protocol, the larger general-purpose backbones (EfficientNet-B0, ConvNeXt-Tiny) were more accurate than the small MobileNetV3-Small-based models, and YOLO26n-cls, of similar size to MobileNetV3-Small, was intermediate. Across three seeds, the CBAM-like gating and early-late fusion tested here did not improve on the original classifier of the small backbone, and differences among the variants were of the same size as seed-to-seed variation. '
  'We think this negative result is worth reporting. Attention add-ons are often presented as a cheap gain for mobile models; on this collection they were not, and a single favourable seed would easily have suggested otherwise. Claims of such gains need matched baselines and more than one run.')
P('The deployment experiments point the same way. EfficientNet-B0 was the best accuracy-per-cost choice in floating point, but it lost much of its accuracy after naive INT8 quantization, whereas ConvNeXt-Tiny and YOLO26n-cls survived it. Which model is "best" for a phone therefore depends on how it will be compressed, and should be checked after compression rather than assumed from full-precision results.')
P('The high internal accuracies still need to be read with care. The test partition comes from the same archives as the training data. Grouping removes only exact and near-duplicate images, and background or source artifacts may still be predictive [noyan2022], [geirhos2020]. '
  'Evidence from controlled-to-field studies shows that internal accuracy above 95% can coexist with much lower field accuracy [mohanty2016], [xiang2026]. The present results should therefore be read as a comparison of models under one internal protocol, not as an estimate of diagnostic performance in farms.')

H('Limitations')
L(['Few seeds. Three seeds give only a rough estimate of training variance; standard deviations from three runs are themselves imprecise. Detailed per-label, per-source and confusion analyses use seed 42 only.',
   'Limited external evidence. The only external test is a small PlantDoc probe (141 images, dominated by one label), because most PlantDoc images were already in the collection. No field test set or source-held-out evaluation was performed.',
   'Grouping is not identity. Groups are based on duplicate and descriptor similarity only and do not identify plants or sessions. The 623 mixed-label groups were reviewed by a plant pathologist only after training, so the reported results still use the uncorrected labels.',
   'Labels and permissions. Apart from the 623 conflicting groups, labels come from the source archives and were not individually expert-confirmed. For the two author-assembled compilations, the individual Kaggle pages from which their constituent image sets were downloaded were not recorded, so their licences cannot be confirmed; for this reason no source images are redistributed.',
   'Protocol scope. A fixed learning rate and ten-epoch budget were used for every model without per-model tuning. Some runs were resumed at epoch boundaries without bit-exact augmentation streams.',
   'Not established. Background normalization benefits, rejection of non-leaf or unknown-crop inputs, explanation faithfulness, quantization-aware training and latency on phones or other edge devices were not evaluated for these checkpoints; latency was measured only on a laptop CPU and GPU, and quantization was tested only as static post-training INT8.'])

H('Conclusion')
P(f'We presented a matched, duplicate-group-aware benchmark of eight leaf-disease classifiers on a {ds["labels"]}-label collection, with a controlled ablation of attention and fusion components. '
  f'Across three seeds, EfficientNet-B0 ({msd("efficientnet_b0")}%) and ConvNeXt-Tiny ({msd("convnext_tiny")}%) were the most accurate under this fixed budget and indistinguishable from each other. None of the PhytoEdge additions improved on the plain MobileNetV3-Small baseline ({msd("mobilenet_v3_small")}%), and the ranking among the variants was not stable across seeds. '
  'In short, a well-tuned standard backbone beat the hand-modified mobile variants, and seed-to-seed variation was large enough to reverse single-run conclusions. '
  'The natural next steps are to retrain on the expert-corrected 145-label manifest and to test on field photographs collected and labelled independently of the training sources.')

H('Data and Code Availability')
P('The training, grouping, analysis and manuscript-build scripts, the protocol file, per-run verification records, full per-label reports and confusion matrices accompany this manuscript as supplementary material. '
  'Supplementary Table S11 lists every source archive with its identified origin and image count. Source images are not redistributed because the licences of the image sets inside the two author-assembled compilations cannot be confirmed; readers can obtain the public sources from their original providers. '
  'The expert label review is documented in Supplementary S10: the reviewer\'s completed decision sheet for all 623 groups, the script that applies the decisions, the per-image change log and the corrected 145-label manifest. Code, configuration, verification records and supplementary tables are available at https://github.com/muazzambadshah/LeafCare-Benchmark.')

H('Author Contributions')
P('M. Badshah: Conceptualization, Methodology, Software, Data curation, Investigation, Formal analysis, Visualization, Writing - original draft. '
  'M. A. Farooq: Validation (expert review of the 623 conflicting label groups), Writing - review and editing. Both authors approved the final version.')
H('Funding')
P('This research did not receive any specific grant from funding agencies in the public, commercial or not-for-profit sectors.')
H('Declaration of Competing Interest')
P('The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.')
H('Acknowledgment')
P('The authors thank Dr. Zaheer Abbas, Senior Scientific Officer, National Institute for Genomics and Advanced Biotechnology (NIGAB), National Agricultural Research Centre, Islamabad, for reviewing the disease-management guide (symptoms, treatment and expected recovery time) prepared for the LeafCare application. '
  'This review concerned the guidance text only; the image labels were not reviewed by him. '
  'We also thank Ali Hamza and Usama Shaikh, senior members of the National Center of Artificial Intelligence (NCAI), National University of Sciences and Technology (NUST), Islamabad, for their guidance and support during this work.')

REFS = [
 ('chouhan2019', 'S. S. Chouhan, A. Kaul, U. P. Singh, and S. Jain, "A database of leaf images: Practice towards plant conservation with plant pathology," Mendeley Data, 2019, doi: 10.17632/hb74ynkjcn.1.'),
 ('plantcity2025', '"PlantCity: A comprehensive image-based multi-crop leaf disease dataset," Islamia College Peshawar, Mendeley Data, v1, 2025, doi: 10.17632/w8kh2xkspx.1.'),
 ('mohanty2016', 'S. P. Mohanty, D. P. Hughes, and M. Salath\u00e9, "Using deep learning for image-based plant disease detection," Frontiers in Plant Science, vol. 7, art. 1419, 2016, doi: 10.3389/fpls.2016.01419.'),
 ('hughes2015', 'D. P. Hughes and M. Salath\u00e9, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," arXiv:1511.08060, 2015.'),
 ('singh2020plantdoc', 'D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A dataset for visual plant disease detection," in Proc. 7th ACM IKDD CoDS and 25th COMAD, 2020, pp. 249-253, doi: 10.1145/3371158.3371196.'),
 ('noyan2022', 'M. A. Noyan, "Uncovering bias in the PlantVillage dataset," arXiv:2206.04374 (preprint), 2022.'),
 ('geirhos2020', 'R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, "Shortcut learning in deep neural networks," Nature Machine Intelligence, vol. 2, pp. 665-673, 2020, doi: 10.1038/s42256-020-00257-z.'),
 ('barbedo2018', 'J. G. A. Barbedo, "Factors influencing the use of deep learning for plant disease recognition," Biosystems Engineering, vol. 172, pp. 84-91, 2018, doi: 10.1016/j.biosystemseng.2018.05.013.'),
 ('ferentinos2018', 'K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," Computers and Electronics in Agriculture, vol. 145, pp. 311-318, 2018, doi: 10.1016/j.compag.2018.01.009.'),
 ('ramcharan2019', 'A. Ramcharan et al., "A mobile-based deep learning model for cassava disease diagnosis," Frontiers in Plant Science, vol. 10, art. 272, 2019, doi: 10.3389/fpls.2019.00272.'),
 ('xiang2026', 'K. Xiang, D. Shi, and X. Zhu, "Quantifying the reliability gap in cross-domain plant disease classification: benchmarking the limited efficacy of standard mitigation techniques under controlled-to-field shift," Frontiers in Plant Science, vol. 17, 2026, doi: 10.3389/fpls.2026.1826962.'),
 ('shafay2025', 'M. Shafay, T. Hassan, M. Owais, I. Hussain, S. G. Khawaja, L. Seneviratne, and N. Werghi, "Recent advances in plant disease detection: challenges and opportunities," Plant Methods, vol. 21, art. 140, 2025, doi: 10.1186/s13007-025-01450-0.'),
 ('howard2019', 'A. Howard et al., "Searching for MobileNetV3," in Proc. IEEE/CVF ICCV, 2019, arXiv:1905.02244.'),
 ('tan2019', 'M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in Proc. ICML, 2019, arXiv:1905.11946.'),
 ('liu2022', 'Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in Proc. IEEE/CVF CVPR, 2022, arXiv:2201.03545.'),
 ('jocher2026', 'G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, "Ultralytics YOLO26: Unified real-time end-to-end vision models," arXiv:2606.03748 (preprint), 2026.'),
 ('hu2018', 'J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, "Squeeze-and-excitation networks," in Proc. IEEE CVPR, 2018; extended version arXiv:1709.01507.'),
 ('woo2018', 'S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in Proc. ECCV, 2018, arXiv:1807.06521.'),
 ('loshchilov2019', 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019, arXiv:1711.05101.'),
 ('walsh2021dome', 'I. Walsh et al., "DOME: recommendations for supervised machine learning validation in biology," Nature Methods, vol. 18, pp. 1122-1127, 2021, doi: 10.1038/s41592-021-01205-4.'),
 ('kapoor2023', 'S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns, vol. 4, no. 9, art. 100804, 2023, doi: 10.1016/j.patter.2023.100804.'),
 ('bouthillier2021', 'X. Bouthillier et al., "Accounting for variance in machine learning benchmarks," in Proc. MLSys, vol. 3, 2021, arXiv:2103.03098.'),
]
order = []
for blk in secs:
    for t in [x for x in blk[1:] if isinstance(x, str)] + [y for x in blk[1:] if isinstance(x, list) for y in (x if not isinstance(x[0], list) else [])]:
        for k in re.findall(r'\[([a-z0-9]+)\]', t):
            if k not in order: order.append(k)
for k in re.findall(r'\[([a-z0-9]+)\]', ABSTRACT):
    if k not in order: order.append(k)
refd = dict(REFS); missing = [k for k in order if k not in refd]; assert not missing, missing
unused = [k for k in refd if k not in order]
num = {k: i + 1 for i, k in enumerate(order)}
def cite_num(t): return re.sub(r'\[([a-z0-9]+)\]', lambda m: f'[{num[m.group(1)]}]', t)

# ---------------- LaTeX ----------------
def tex(s):
    s = s.replace('\\', r'\textbackslash{}')
    for a, b in [('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'), ('\u2212', '$-$'), ('\u00d7', r'$\times$'), ('\u2265', r'$\geq$'), ('\u00e9', r"\'e"), ('\u00b1', r'$\pm$')]: s = s.replace(a, b)
    s = re.sub(r'(\d) x (\d)', r'\1$\\times$\2', s)
    return re.sub(r'\[([a-z0-9]+)\]', r'\\cite{\1}', s)
figno = 0; tabno = 0; tl = []
tl.append(r'\documentclass[conference]{IEEEtran}' + '\n' + r'\usepackage{graphicx,booktabs,cite,url}' + '\n' + r'\begin{document}')
tl.append(r'\title{' + tex(TITLE) + '}\n' + r'\author{\IEEEauthorblockN{' + AUTHOR + r'}\IEEEauthorblockA{' + tex(AFFIL) + r'\\ Email: ' + EMAIL + r'}\and\IEEEauthorblockN{' + AUTHOR2 + r'}\IEEEauthorblockA{' + tex(AFFIL2) + r'\\ Email: ' + EMAIL2 + r'\\ ORCID: ' + ORCID2 + '}}\n' + r'\maketitle')
tl.append(r'\begin{abstract}' + tex(ABSTRACT) + r'\end{abstract}' + '\n' + r'\begin{IEEEkeywords}' + KEYWORDS + r'\end{IEEEkeywords}')
for blk in secs:
    k = blk[0]
    if k == 'h': tl.append(r'\section{' + tex(blk[1]) + '}')
    elif k == 'p': tl.append(tex(blk[1]) + '\n')
    elif k == 'list': tl.append(r'\begin{itemize}' + ''.join(r'\item ' + tex(i) for i in blk[1]) + r'\end{itemize}')
    elif k == 'fig':
        env = 'figure*' if blk[3] else 'figure'; w = r'\textwidth' if blk[3] else r'\columnwidth'
        tl.append(rf'\begin{{{env}}}[t]\centering\includegraphics[width={w}]{{figures/{blk[1]}.pdf}}\caption{{{tex(blk[2])}}}\label{{fig:{blk[1]}}}\end{{{env}}}')
    elif k == 'tab':
        cols = 'l' + 'r' * (len(blk[2]) - 1)
        if 'Identified source' in blk[2]: cols = r'l p{10.5cm} r r'
        body = ' \\\\\n'.join(' & '.join(tex(c) for c in r) for r in blk[3])
        tenv = 'table*' if (len(blk[2]) >= 6 or 'Identified source' in blk[2]) else 'table'
        tl.append(rf'\begin{{{tenv}}}[t]\centering\caption{{{tex(blk[1])}}}\footnotesize\setlength{{\tabcolsep}}{{3pt}}\begin{{tabular}}{{{cols}}}\toprule ' + ' & '.join(tex(c) for c in blk[2]) + r' \\ \midrule ' + body + r' \\ \bottomrule\end{tabular}\end{' + tenv + '}')
tl.append(r'\bibliographystyle{IEEEtran}' + '\n' + r'\bibliography{references}' + '\n' + r'\end{document}')
(MS / 'LeafCare_Manuscript.tex').write_text('\n'.join(tl), encoding='utf-8')
bib = []
for k, txt in REFS:
    if k in num: bib.append('@misc{' + k + ',\n  note = {' + tex(txt).replace(r'\cite', '') + '}\n}')
(MS / 'references.bib').write_text('\n\n'.join(bib), encoding='utf-8')

# ---------------- ReportLab review-layout PDF ----------------
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, FrameBreak, NextPageTemplate, ListFlowable, ListItem
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
for nm, fn in [('Serif', 'times.ttf'), ('Serif-B', 'timesbd.ttf'), ('Serif-I', 'timesi.ttf')]: pdfmetrics.registerFont(TTFont(nm, 'C:/Windows/Fonts/' + fn))
pdfmetrics.registerFontFamily('Serif', normal='Serif', bold='Serif-B', italic='Serif-I')
st = {'title': ParagraphStyle('t', fontName='Serif', fontSize=17, leading=20, alignment=1, spaceAfter=8),
      'author': ParagraphStyle('a', fontName='Serif', fontSize=11, leading=13, alignment=1, spaceAfter=4),
      'note': ParagraphStyle('n', fontName='Serif-I', fontSize=8, leading=10, alignment=1, textColor=colors.HexColor('#555555'), spaceAfter=10),
      'abs': ParagraphStyle('ab', fontName='Serif', fontSize=9.5, leading=11.5, leftIndent=18, rightIndent=18, alignment=4, spaceAfter=6),
      'h': ParagraphStyle('h', fontName='Serif', fontSize=10, leading=12, alignment=1, spaceBefore=8, spaceAfter=4),
      'p': ParagraphStyle('p', fontName='Serif', fontSize=10, leading=12, alignment=4, firstLineIndent=10, spaceAfter=3),
      'li': ParagraphStyle('li', fontName='Serif', fontSize=10, leading=12, alignment=4),
      'cap': ParagraphStyle('c', fontName='Serif', fontSize=8, leading=9.6, alignment=4, spaceBefore=3, spaceAfter=8),
      'cell': ParagraphStyle('ce', fontName='Serif', fontSize=8.5, leading=10),
      'ref': ParagraphStyle('r', fontName='Serif', fontSize=8, leading=9.6, leftIndent=14, firstLineIndent=-14, spaceAfter=2)}
from xml.sax.saxutils import escape
def rl(t): return cite_num(escape(t))
W, Hh = letter; M = 1.0 * inch; colw = W - 2 * M
def footer(c, d):
    c.saveState(); c.setFont('Serif', 7.5); c.setFillColor(colors.HexColor('#666666'))
    c.drawString(M, 0.45 * inch, 'Single-column review layout generated with ReportLab from the same content as the IEEEtran source; not a compiled IEEE template.'); c.drawRightString(W - M, 0.45 * inch, str(d.page)); c.restoreState()
first = PageTemplate('first', [Frame(M, M, colw, Hh - 2 * M, id='one')], onPage=footer)
later = PageTemplate('later', [Frame(M, M, colw, Hh - 2 * M, id='one2')], onPage=footer)
doc = BaseDocTemplate(str(MS / 'LeafCare_Manuscript_review_layout.pdf'), pagesize=letter, pageTemplates=[first, later], title='LeafCare matched benchmark (review layout)', author=AUTHOR)
story = [Paragraph(escape(TITLE), st['title']), Paragraph(AUTHOR + '<br/>' + escape(AFFIL) + '<br/>' + EMAIL + '<br/><br/>' + AUTHOR2 + '<br/>' + escape(AFFIL2) + '<br/>' + EMAIL2 + ' | ORCID ' + ORCID2, st['author']),
         Paragraph('Three-seed results (seeds 42, 43, 44) frozen from verified artifacts. Draft for review, not submitted.', st['note']), NextPageTemplate('later'),
         Paragraph('<i>Abstract</i>\u2014' + rl(ABSTRACT), st['abs']), Paragraph('<i>Index Terms</i>\u2014' + escape(KEYWORDS), st['abs'])]
roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']; hi = 0; fi = 0; ti = 0
for blk in secs:
    k = blk[0]
    if k == 'h': story.append(Paragraph(f'{roman[hi]}. ' + escape(blk[1]).upper(), st['h'])); hi += 1
    elif k == 'p':
        t = rl(blk[1]); m = re.match(r'^([A-Z][A-Za-z\- ]{2,40}\.) ', t)
        if m: t = '<i>' + m.group(1) + '</i>' + t[len(m.group(1)):]
        story.append(Paragraph(t, st['p']))
    elif k == 'list': story.append(ListFlowable([ListItem(Paragraph(rl(i), st['li']), leftIndent=10, value='\u2022') for i in blk[1]], bulletType='bullet', start='\u2022', leftIndent=10, bulletFontSize=7, spaceAfter=4))
    elif k == 'fig':
        fi += 1; img = FIGD / f'{blk[1]}.png'
        from PIL import Image as PI
        iw, ih = PI.open(img).size; w = colw * (1.0 if blk[3] else 0.62); h = w * ih / iw
        story.append(KeepTogether([Image(str(img), width=w, height=h), Paragraph(f'Fig. {fi}. ' + rl(blk[2]), st['cap'])]))
    elif k == 'tab':
        ti += 1; data = [[Paragraph('<b>' + escape(c) + '</b>', st['cell']) for c in blk[2]]] + [[Paragraph(escape(c), st['cell']) for c in r] for r in blk[3]]
        n = len(blk[2]); cw = [colw * (0.24 if n > 3 else 0.35)] + [colw * (0.70 if n > 3 else 0.25) / (n - 1)] * (n - 1)
        if 'Gates' in blk[2] and n == 8: cw = [colw * x for x in (0.18, 0.12, 0.07, 0.09, 0.08, 0.09, 0.30, 0.05)]
        if 'Identified source' in blk[2]: cw = [colw * x for x in (0.18, 0.62, 0.11, 0.09)]
        if 'Gates' in blk[2] and n == 7: cw = [colw * x for x in (0.16, 0.12, 0.06, 0.08, 0.15, 0.15, 0.28)]
        t = Table(data, colWidths=cw, repeatRows=1); t.setStyle(TableStyle([('LINEABOVE', (0, 0), (-1, 0), .8, colors.black), ('LINEBELOW', (0, 0), (-1, 0), .5, colors.black), ('LINEBELOW', (0, -1), (-1, -1), .8, colors.black), ('ALIGN', (1, 0), (-1, -1), 'RIGHT'), ('TOPPADDING', (0, 0), (-1, -1), 1), ('BOTTOMPADDING', (0, 0), (-1, -1), 1), ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2)]))
        story.append(KeepTogether([Paragraph(f'TABLE {roman[ti - 1]}<br/>' + rl(blk[1]), ParagraphStyle('tc', parent=st['cap'], alignment=1)), t, Spacer(1, 8)]))
story.append(Paragraph('REFERENCES', st['h']))
for k in order: story.append(Paragraph(f'[{num[k]}] ' + escape(refd[k]), st['ref']))
doc.build(story)

# figure/table numbering in text: replace "Fig. 2/4/5/6" & "Table I/II" were written to match the order above; verify
fig_order = [b[1] for b in secs if b[0] == 'fig']; tab_caps = [b[1][:40] for b in secs if b[0] == 'tab']
(VER / 'manuscript_numbers.json').write_text(json.dumps({'numbers': numbers, 'citation_order': order, 'unused_references': unused, 'figure_order': fig_order, 'tables': tab_caps,
    'runs_in_manuscript': done, 'result_freeze': 'model_comparison.csv as produced by analyze_results.py in this delivery'}, indent=2))
print('figures', fig_order); print('unused refs', unused); print('runs', done)
