"""V13: reviewer-style fixes. Shorter abstract; provenance made consistent (two author-supplied sets, six public
archives); source table moved to Supplementary Table S11; fixed-budget framing; exploratory PlantDoc probe;
quantization caveats; every figure/table cited; ConvNeXt-Tiny named as reference model; expert-review evidence listed."""
import sys, csv
from pathlib import Path
D = Path(sys.argv[1]); p = D / 'scripts' / 'build_manuscript.py'; s = p.read_text(encoding='utf-8')
def R(o, n):
    global s
    assert s.count(o) == 1, o[:90]; s = s.replace(o, n)

# ---------- abstract (problem, method, 3 findings, 1 limitation) ----------
a = s.index('ABSTRACT = ('); b = s.index('KEYWORDS = ')
s = s[:a] + '''ABSTRACT = (
    'Leaf-disease classifiers often report accuracies near 99%, but studies differ in training budget, split method and number of runs, so their results are hard to compare. '
    f'We trained eight classifiers, from MobileNetV3-Small to ConvNeXt-Tiny, under one fixed ten-epoch protocol on {f(ds["images"])} leaf images with {ds["labels"]} labels, using a split that keeps duplicate and near-duplicate images in the same partition, and repeated every run with three seeds. '
    f'EfficientNet-B0 ({msd("efficientnet_b0")}%) and ConvNeXt-Tiny ({msd("convnext_tiny")}%) were the most accurate under this budget, although ConvNeXt-Tiny has 6.7 times more parameters. '
    'Adding channel/spatial attention or feature fusion to MobileNetV3-Small did not improve it, and the ranking of these variants changed between seeds. '
    f'On {pdx_n} unseen PlantDoc images accuracy fell to 33-59%, and naive INT8 quantization broke the smallest models. '
    'The results compare models under one internal protocol and do not establish field performance.')
''' + s[b:]

# ---------- provenance ----------
R("in August 2026, using iPhone and Infinix smartphones. The other seven archives are publicly available collections. '",
  "in August 2026, using iPhone and Infinix smartphones. A second author-supplied set contains 2,380 smartphone photographs of four healthy fruit-tree labels. The other six archives are public collections: PlantVillage [hughes2015], PlantDoc [singh2020plantdoc], PlantCity [plantcity2025], the leaf database of Chouhan et al. [chouhan2019] and two Kaggle compilations derived mainly from PlantVillage (Supplementary Table S11). '")
R("'Labels and permissions. Apart from the 623 conflicting groups, labels come from the source archives and were not individually expert-confirmed. Except for the author-collected archive, image provenance and usage permissions are not yet fully documented.'",
  "'Labels and permissions. Apart from the 623 conflicting groups, labels come from the source archives and were not individually expert-confirmed. The original pages of the two Kaggle compilations could not be identified, so their licences are unknown; for this reason no source images are redistributed.'")
R(" Earlier historical audit measurements belong to different checkpoints and are not used here.'", "'")
R("On the author-collected archive (", "On the Battagram archive (")

# ---------- source table -> supplement ----------
a = s.index("TAB('Source archives of the LeafCare collection."); b = s.index("P('The training, grouping, analysis")
s = s[:a] + s[b:]
R("'Table X lists the source archives: PlantVillage [hughes2015], PlantDoc [singh2020plantdoc], PlantCity [plantcity2025], the leaf database of Chouhan et al. [chouhan2019], two Kaggle compilations and two sets of photographs supplied by the first author. Source images are not redistributed because the licences of the two unidentified Kaggle compilations and the photographers\\' permissions are still being documented; readers can obtain the public sources from their original providers. No public repository or DOI has been created yet.')",
  "'Supplementary Table S11 lists every source archive with its identified origin and image count. Source images are not redistributed because the licences of the two Kaggle compilations are unknown; readers can obtain the public sources from their original providers. '\n"
  "  'The expert label review is documented in Supplementary S10: the reviewer\\'s completed decision sheet for all 623 groups, the script that applies the decisions, the per-image change log and the corrected 145-label manifest. No public repository or DOI has been created yet.')")
rows = [['archive', 'identified_source', 'images', 'labels'],
        ['archive (20)', 'Kaggle multi-crop compilation (MasterDataset); PlantVillage-derived classes plus cotton, groundnut, sugarcane, wheat, rice, pumpkin and lemon; original page not identified', '122428', '84'],
        ['dataset - Copy', 'PlantCity, Islamia College Peshawar, Mendeley Data, doi:10.17632/w8kh2xkspx.1', '28649', '51'],
        ['archive (16)', 'Kaggle compilation (PlantDiseasesDataset): PlantVillage-derived classes plus citrus classes; original page not identified', '17221', '37'],
        ['archive (13)', 'PlantVillage, Kaggle mirror (tomato, potato and pepper classes)', '7173', '14'],
        ['leaf_dataset_labeled 12', 'Author-supplied smartphone photographs (healthy fruit-tree leaves)', '2380', '4'],
        ['archive (14)', 'PlantDoc, Kaggle mirror; 1,000 images matched to the GitHub release by SHA-256', '1631', '15'],
        ['archive (17)', 'Chouhan et al. leaf database, Mendeley Data, doi:10.17632/hb74ynkjcn.1 (healthy classes only)', '1189', '11'],
        ['leaf data', 'Photographed by the first author, Battagram, Khyber Pakhtunkhwa, Pakistan, August 2026', '1048', '10']]
with (D / 'supplement' / 'S11_source_archives.csv').open('w', newline='', encoding='utf-8') as fh: csv.writer(fh).writerows(rows)

# ---------- fairness / fixed budget ----------
R("use torchvision ImageNet weights; only the final linear layer is replaced.'",
  "use torchvision ImageNet weights; only the final linear layer is replaced. ConvNeXt-Tiny is not a lightweight model; it is included as a larger reference point.'")
R("only that checkpoint is evaluated once on the test partition, in float32 with TF32 disabled.',",
  "only that checkpoint is evaluated once on the test partition, in float32 with TF32 disabled.',\n"
  "   'Scope of the comparison: this is a fixed-budget comparison. The same learning rate and ten epochs suit some architectures better than others, so the rankings describe performance under this budget, not the best each architecture could reach with its own tuning.',")
R("were the most accurate and indistinguishable from each other.", "were the most accurate under this fixed budget and indistinguishable from each other.")

# ---------- cite every figure and table ----------
R("P(f'Partitioning. Groups were assigned whole,", "P(f'Partitioning (Fig. 1). Groups were assigned whole,")
R("P(f'Overall comparison. EfficientNet-B0 had", "P(f'Overall comparison (Fig. 4). EfficientNet-B0 had")
R("P(f'Against the head-only variant, adding", "P(f'Table V gives the seed-42 ablation. Against the head-only variant, adding")
R("P(f'Across seeds the picture is less clear-cut than at seed 42.", "P(f'Across seeds the picture is less clear-cut than at seed 42 (Table VI).")
R("P(f'Accuracy fell from 95-97% on the internal test partition to", "P(f'Accuracy fell (Table VII) from 95-97% on the internal test partition to")

# ---------- PlantDoc framed as exploratory ----------
R("H('External Probe on PlantDoc')", "H('Exploratory External Probe on PlantDoc')")
R("We evaluated all 24 runs without retraining.')",
  "We evaluated all 24 runs without retraining. With so few images, mostly from one label, this is an exploratory probe and not an external validation.')")

# ---------- quantization caveats ----------
R("PE-Fusion and PE-Full could not be exported because their adaptive pooling has a non-constant output size.'",
  "PE-Fusion and PE-Full could not be exported because their adaptive pooling has a non-constant output size. Because the subset size differs, compare FP32 with INT8 within a model, not across models.'")
p.write_text(s, encoding='utf-8'); print('patched')
