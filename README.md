# LeafCare-Benchmark

Code and supplementary material for the paper
**"Lightweight Deep Learning Models for Leaf Disease Classification"**
(M. Badshah and M. A. Farooq).

Eight leaf-disease classifiers (MobileNetV3-Small, EfficientNet-B0, ConvNeXt-Tiny, YOLO26n-cls and four PhytoEdge
variants) were trained under one fixed ten-epoch protocol, with three seeds each, on 181,719 leaf images with
147 labels. The split keeps exact and near-duplicate images together in the same partition.

## Contents
| Folder | What it holds |
|---|---|
| `config/` | Training protocol, duplicate-group split code, training script (`train_matched.py`), PhytoEdge model |
| `scripts/` | Analysis, PlantDoc evaluation, efficiency, INT8 export, selective prediction and manuscript-build scripts |
| `verification/` | Per-run verification records, test report and the numbers used in the paper |
| `supplement/` | Supplementary tables S1-S11 and per-run reports |
| `supplement/S10_expert_label_review/` | Plant-pathologist review of 623 conflicting-label groups: decision sheet, script, per-image change log, corrected 145-label manifest (gzip) |

## Images
Source images are **not** included. The public sources (PlantVillage, PlantDoc, PlantCity, Chouhan et al.) can be
obtained from their original providers; see `supplement/S11_source_archives.csv`.

## Contact
Muazzam Badshah, NCAI, NUST, Islamabad.
