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

## Reproducibility
- Frozen image manifest used for every run: SHA-256 `3a791d98f05506375b6fb26504a66897319c362e0667b251d570f9fe2f9cc69f` (also in `config/protocol.json`).
- Exact Python environment: `requirements-lock.txt` (PyTorch 2.15.0, torchvision 0.29.2, Ultralytics 8.4.202, scikit-learn 1.9.1, ONNX Runtime 1.31.1; WSL Ubuntu, one RTX 4050 Laptop GPU).
- Checksums of every file in this release: `SHA256SUMS.txt`.
- The version cited in the paper is release **v1.0**.
