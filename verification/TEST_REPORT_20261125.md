# LeafCare test report

Run: 2026-11-25T10:00:53  |  device: cuda  |  torch 2.15.0+cu132  |  Python 3.11.15

**Result: 16 passed, 0 failed, 16 tests.**

Command: `wsl -d Ubuntu -- /home/muazzam_khan/.venvs/leafcare_yolo26/bin/python -I -B test_leafcare.py <this folder> 8611`

Nothing in the training, dataset, checkpoint or manuscript folders was modified. All seeds are reported; no seed was selected.

| ID | Test | Result | What was checked | Observed |
|---|---|---|---|---|
| S1 | All 24 run folders complete | PASS | 5 non-empty files per run; history.json = 10 epochs | 24 runs, 5 files each, 10 epochs each. Problems: none |
| S2 | Every best.pt loads and runs one image | PASS | build()+load_state_dict+1-image forward, shape (1,147), no NaN/inf | device=cuda; 24/24 best.pt load via train_matched.build(), output (1,147), finite, class map matches. Problems: none |
| S3 | Web app serves page and predicts | PASS | GET / = 200; POST /predict one image -> 3 predictions, sum <= 100% (port 8611) | GET / -> 200; POST /predict -> 200, 3 predictions, top3 confidence sum 99.9% (true label alstonia_scholaris_healthy, top1 alstonia_scholaris_healthy) |
| S4 | Paper ZIP checksum and integrity | PASS | sha256 equals .sha256 file; zipfile.testzip() is None | sha256 matches (91deee29d609041b...); testzip bad member: None |
| P1 | Split integrity | PASS | each duplicate_group in 1 split; test=18,148; 147 labels in all splits; no train/test image hash overlap | groups 176383, groups spanning >1 split: 0; test images 18148; all 147 labels in every split: True; train/test shared image_sha256: 0; split sizes: {'train': 127237, 'val': 18176, 'test': 18148, 'calibration': 18158} |
| P2 | Preprocessing parity (app vs training eval) | PASS | Resize256+CenterCrop224+ImageNet-normalise tensors identical on 8 images | app transform vs train_matched eval transform on 8 test images: max abs pixel difference 0.0 |
| P3 | Re-inference reproduces saved predictions | PASS | fresh forward on first 200 test images; argmax equals test_logits.npz (<=1 mismatch/run) | 24 runs x first 200 test images; max mismatches in any run 0; worst max/logit diff/ 7.63e-06. Problems: none |
| P4 | Metrics recompute (logits vs json vs paper CSV) | PASS | accuracy & macro-F1 from saved logits == test_metrics.json == paper model_comparison.csv, tol 1e-9 | 24 runs: accuracy/macro-F1 recomputed from test_logits.npz equal test_metrics.json and model_comparison.csv (tol 1e-9); image/label order matches manifest. Problems: none |
| P5 | Paper numbers (mean ± SD) | PASS | recompute mean/SD(ddof=1) from 3 seeds; match CSV (1e-9) and manuscript text (2 dp) | 8 models x (accuracy, macro-F1): mean and sample SD (ddof=1) match multi_seed_summary.csv and appear in the .tex to 2 decimals. Problems: none |
| P6 | App end-to-end vs saved logits | PASS | 20 evenly spaced test images through HTTP /predict; top-1 == saved EffNet-B0 seed-42 argmax | 20 fixed test images (indices [0, 955, 1910, 2865, 3820, 4775, 5730, 6685, 7640, 8595, 9551, 10506, 11461, 12416, 13371, 14326, 15281, 16236, 17191, 18147]); app top-1 equals saved EfficientNet-B0 seed-42 argmax for 20/20; app correct vs true label 20/20. Problems: none |
| P7 | App ships the intended checkpoint | PASS | sha256(app model/best.pt) == efficientnet_b0_seed42/best.pt | app model/best.pt sha256 equals efficientnet_b0_seed42/best.pt (eb1bf42ee9556f71...) |
| F1 | build() for all 8 models gives 147 outputs | PASS | forward (2,3,224,224) on untrained model, last dim 147 | build() output shapes: {'mobilenet_v3_small': (2, 147), 'phytoedge_head_only': (2, 147), 'phytoedge_attention': (2, 147), 'phytoedge_fusion': (2, 147), 'phytoedge_net': (2, 147), 'efficientnet_b0': (2, 147), 'convnext_tiny': (2, 147), 'yolo26n': (2, 147)}. Problems: none |
| F2 | PhytoEdgeNet variants | PASS | 4 variants give (2,147); attention & fusion have more params than head_only | parameters: {'head_only': 1096499, 'attention': 1180189, 'fusion': 1457075, 'full': 1540765}. Problems: none |
| F3 | Images dataset class | PASS | shape, label index, normalisation rule (yolo26n none) | Images(): shape (3,224,224), correct label index; yolo26n values in [0,1], others ImageNet-normalised. Problems: none |
| F4 | evaluate() function | PASS | tiny fake model + loader: logits, labels, macro-F1 equal sklearn reference | evaluate() macro-F1 0.000680 vs sklearn 0.000680; logits and labels returned in order |
| F5 | App predict(): input handling | PASS | non-image -> 400; PNG/JPG/gray/RGBA/6000x6000 OK; >15 MB rejected; server still alive | results: {'non-image text file': 400, 'non-image error text': '{"error": "cannot identify image file <_io.BytesIO object at 0x703fe44bf1f0>"}', 'PNG RGB': 200, 'JPG RGB': 200, 'PNG grayscale': 200, 'PNG RGBA': 200, '6000x6000 JPEG (0.6 MB) seconds': 0.4, '6000x6000 JPEG': 200, '16 MB upload': "BrokenPipeError(32, 'Broken pipe')", '16 MB note': 'server rejected the upload (checks size before reading) but the client saw a connection reset instead of a readable 400 message'}. Problems: none |

## Per-run re-inference and metrics

| Run | Accuracy | Macro-F1 | Mismatches / 200 | Max logit diff |
|---|---|---|---|---|
| mobilenet_v3_small_seed42 | 95.54% | 91.38% | 0 | 7.63e-06 |
| mobilenet_v3_small_seed43 | 95.60% | 92.20% | 0 | 7.63e-06 |
| mobilenet_v3_small_seed44 | 95.64% | 92.31% | 0 | 7.63e-06 |
| phytoedge_head_only_seed42 | 95.19% | 91.47% | 0 | 0.00e+00 |
| phytoedge_head_only_seed43 | 94.57% | 90.80% | 0 | 0.00e+00 |
| phytoedge_head_only_seed44 | 94.85% | 91.27% | 0 | 0.00e+00 |
| phytoedge_attention_seed42 | 94.97% | 91.14% | 0 | 0.00e+00 |
| phytoedge_attention_seed43 | 95.73% | 91.78% | 0 | 0.00e+00 |
| phytoedge_attention_seed44 | 95.60% | 91.93% | 0 | 0.00e+00 |
| phytoedge_fusion_seed42 | 95.22% | 91.05% | 0 | 0.00e+00 |
| phytoedge_fusion_seed43 | 95.69% | 90.84% | 0 | 0.00e+00 |
| phytoedge_fusion_seed44 | 95.48% | 92.14% | 0 | 0.00e+00 |
| phytoedge_net_seed42 | 94.79% | 90.72% | 0 | 0.00e+00 |
| phytoedge_net_seed43 | 95.15% | 90.88% | 0 | 0.00e+00 |
| phytoedge_net_seed44 | 95.06% | 91.46% | 0 | 0.00e+00 |
| efficientnet_b0_seed42 | 97.05% | 93.75% | 0 | 3.81e-06 |
| efficientnet_b0_seed43 | 97.10% | 94.16% | 0 | 5.72e-06 |
| efficientnet_b0_seed44 | 97.10% | 94.35% | 0 | 7.63e-06 |
| convnext_tiny_seed42 | 96.76% | 93.64% | 0 | 3.81e-06 |
| convnext_tiny_seed43 | 97.17% | 94.03% | 0 | 7.63e-06 |
| convnext_tiny_seed44 | 97.18% | 94.12% | 0 | 5.72e-06 |
| yolo26n_seed42 | 95.86% | 92.46% | 0 | 7.63e-06 |
| yolo26n_seed43 | 95.87% | 92.64% | 0 | 7.63e-06 |
| yolo26n_seed44 | 95.09% | 92.20% | 0 | 7.63e-06 |

## App end-to-end images (P6)

| Test index | File | True label | App top-1 | Saved-logits argmax |
|---|---|---|---|---|
| 0 | train/alstonia_scholaris_healthy/10e7c9034e06881583ec3e5a.jpg | alstonia_scholaris_healthy | alstonia_scholaris_healthy | alstonia_scholaris_healthy |
| 955 | train/bean_healthy/3c628277b8d4cc103f057aa5.jpg | bean_healthy | bean_healthy | bean_healthy |
| 1910 | train/cherry_powdery_mildew/8e105943f2888d349f2cbd33.jpg | cherry_powdery_mildew | cherry_powdery_mildew | cherry_powdery_mildew |
| 2865 | train/corn_fungal/215a0d37338aea331893e6eb.jpg | corn_fungal | corn_fungal | corn_fungal |
| 3820 | train/cotton_bacterial_blight/62e7c1176bcd72659bba5931.jpg | cotton_bacterial_blight | cotton_bacterial_blight | cotton_bacterial_blight |
| 4775 | train/grape_black_rot/ec6a10be70666bdb7df58d8d.jpg | grape_black_rot | grape_black_rot | grape_black_rot |
| 5730 | train/groundnut_healthy/43a8bf61ea29634506797731.jpg | groundnut_healthy | groundnut_healthy | groundnut_healthy |
| 6685 | train/lokat_healthy/b4d0ffd5b2eea56e3bd2b636.jpg | lokat_healthy | lokat_healthy | lokat_healthy |
| 7640 | train/potato_healthy/4bab39b67f6327b3f7ee6dc4.jpg | potato_healthy | potato_healthy | potato_healthy |
| 8595 | train/sugarcane_bacterial_blight/36bf7bec8651195cf425421e.jpg | sugarcane_bacterial_blight | sugarcane_bacterial_blight | sugarcane_bacterial_blight |
| 9551 | train/tomato_bacterial_spot/d6f7095f83cf079ef1b8a01d.jpg | tomato_bacterial_spot | tomato_bacterial_spot | tomato_bacterial_spot |
| 10506 | train/tomato_leaf_mold/3ab479cb2f261d778df48862.jpg | tomato_leaf_mold | tomato_leaf_mold | tomato_leaf_mold |
| 11461 | train/tomato_verticillium_wilt/aeac9f005e110d2fd127afb2.jpg | tomato_verticillium_wilt | tomato_verticillium_wilt | tomato_verticillium_wilt |
| 12416 | train/wheat_mite/29f177a6842196db788de807.jpg | wheat_mite | wheat_mite | wheat_mite |
| 13371 | val/grape_isariopsis_leaf_spot/5b01ed948913aeb329ed2c49.jpg | grape_isariopsis_leaf_spot | grape_isariopsis_leaf_spot | grape_isariopsis_leaf_spot |
| 14326 | val/tomato_yellow_leaf_curl_virus/63660c6247235fc8911ff24c.jpg | tomato_yellow_leaf_curl_virus | tomato_yellow_leaf_curl_virus | tomato_yellow_leaf_curl_virus |
| 15281 | calibration/groundnut_healthy/50c8a6db574bca279459126a.jpg | groundnut_healthy | groundnut_healthy | groundnut_healthy |
| 16236 | calibration/wheat_aphid/3005e97c5211d6bbb4d4e568.jpg | wheat_aphid | wheat_aphid | wheat_aphid |
| 17191 | test/groundnut_nutrition_deficiency/c8956b683aa84bf428d60853.jpg | groundnut_nutrition_deficiency | groundnut_nutrition_deficiency | groundnut_nutrition_deficiency |
| 18147 | test/wheat_yellow_rust/fe61f670f919c22b9bb8d22e.jpg | wheat_yellow_rust | wheat_yellow_rust | wheat_yellow_rust |