"""V11: natural-language rewrite of abstract/introduction/discussion/conclusion, internal-consistency fixes,
corrected references, and journal declarations (CRediT, funding, competing interests, generative AI)."""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'scripts' / 'build_manuscript.py'; s = p.read_text(encoding='utf-8')
def R(o, n):
    global s
    assert s.count(o) == 1, o[:90]; s = s.replace(o, n)

# ---------- abstract ----------
R("    f'Reported accuracies for leaf-disease classifiers are often hard to compare: training budgets differ, splits are random at the image level, and single runs are reported. '",
  "    f'Many leaf-disease classifiers report accuracies close to 99%, but these numbers are hard to compare: each study uses its own training budget, most split images at random, and few repeat their runs. '")
R("    f'We compare classifiers under one fixed protocol on an assembled collection of",
  "    f'We trained and tested lightweight and mid-sized classifiers under one fixed protocol on a collection of")
R("    'Errors concentrate in visually similar within-crop conditions and in small, low-support labels. '",
  "    'Most errors fell between similar-looking diseases of the same crop and in labels with few images. '")
R("    f'On {pdx_n} unseen PlantDoc images, accuracy fell to 33-59% with large seed-to-seed variation. '\n"
  "    'Because these are three-seed results on one internal split and a small external probe, they do not establish architectural superiority, field performance or expert-validated diagnosis.')",
  "    f'On {pdx_n} unseen PlantDoc images, accuracy fell to 33-59% and varied widely between seeds. '\n"
  "    'A confidence threshold chosen on held-out calibration data kept 99% accuracy on the internal test set but not on PlantDoc, and static INT8 quantization worked for ConvNeXt-Tiny and YOLO26n-cls but broke the smaller squeeze-and-excitation models. '\n"
  "    'A plant pathologist reviewed all 623 groups of duplicate images that carried conflicting labels; two pairs of label names turned out to be the same disease. '\n"
  "    'The results compare models under one internal protocol; they are not evidence of field accuracy.')")

# ---------- introduction ----------
R("P('Image-based recognition of crop diseases is attractive for decision support because a phone photograph is cheap to obtain. Published systems often report held-out accuracies close to 99% on curated leaf collections [mohanty2016], [ferentinos2018]. '\n"
  "  'However, the same models can perform much worse when capture conditions change.",
  "P('A farmer with a smartphone can photograph a sick leaf in seconds, which makes image-based disease recognition an appealing aid for crop protection. Studies on curated leaf collections regularly report held-out accuracies near 99% [mohanty2016], [ferentinos2018]. '\n"
  "  'Those numbers rarely survive a change of setting.")
R("Curated collections can also contain label-correlated backgrounds [noyan2022] and near-duplicate images. Either can inflate internal accuracy through shortcut learning or leakage [geirhos2020], [kapoor2023].')",
  "Part of the problem lies in the data themselves: curated collections can carry backgrounds that correlate with the label [noyan2022], and they often hold the same photograph more than once. A model can exploit either, and a random split then rewards it with inflated test accuracy [geirhos2020], [kapoor2023].')")
R("P('A second difficulty is comparability. Architecture comparisons in this area frequently differ in training budget, preprocessing, split procedure and number of runs.",
  "P('A second problem is that published comparisons are rarely like-for-like. Two papers comparing architectures will usually differ in training length, preprocessing, the way the data were split and the number of runs.")
R("are frequently added to mobile backbones. Their contribution is rarely isolated under a matched protocol.')",
  "are often bolted onto mobile backbones, yet their effect is seldom measured against the same backbone trained the same way.')")
R("P('This paper reports an empirical benchmark and ablation rather than a new algorithm. Its specific contributions are:')",
  "P('We therefore set out to answer a practical question: under identical training conditions and a split that keeps duplicate images together, which lightweight model should a developer choose, and do popular attention add-ons actually help? This is an empirical study, not a new architecture. Its contributions are:')")

# ---------- consistency fixes ----------
R("The calibration partition is reserved for later work and is not used in this paper.')",
  "The calibration partition is not used for training or checkpoint selection; it is used only to fit the temperature and confidence threshold for selective prediction.')")
R("These are laptop measurements in PyTorch; phone latency, quantized or exported models and energy use were not measured.')",
  "These are laptop measurements in PyTorch; phone latency and energy use were not measured, and exported INT8 models are evaluated separately below.')")
R("'Not established. Background normalization benefits, unknown-input rejection, calibration, explanation faithfulness,",
  "'Not established. Background normalization benefits, rejection of non-leaf or unknown-crop inputs, explanation faithfulness,")
R('{len(resumed)} of the {len(done)} reported runs ({", ".join(NICE[r.rsplit("_seed", 1)[0]] for r in resumed)})',
  '{len(resumed)} of the {n_runs} runs ({", ".join(NICE[r.rsplit("_seed", 1)[0]] + " seed " + r.rsplit("_seed", 1)[1] for r in resumed)})')

# ---------- discussion / conclusion ----------
R("P('Under a matched ten-epoch protocol, the larger general-purpose backbones",
  "P('For a developer choosing a model, the main message is simple. Under a matched ten-epoch protocol, the larger general-purpose backbones")
R("'This negative ablation result is informative: on this collection, adding attention modules to a mobile backbone should not be assumed to improve accuracy, and claims of such benefits should be supported by matched comparisons.')",
  "'We think this negative result is worth reporting. Attention add-ons are often presented as a cheap gain for mobile models; on this collection they were not, and a single favourable seed would easily have suggested otherwise. Claims of such gains need matched baselines and more than one run.')\n"
  "P('The deployment experiments point the same way. EfficientNet-B0 was the best accuracy-per-cost choice in floating point, but it lost much of its accuracy after naive INT8 quantization, whereas ConvNeXt-Tiny and YOLO26n-cls survived it. Which model is \"best\" for a phone therefore depends on how it will be compressed, and should be checked after compression rather than assumed from full-precision results.')")
R("P('The high internal accuracies must be read in context.",
  "P('The high internal accuracies still need to be read with care.")
R("'Retraining on the expert-corrected 145-label manifest and an independently sourced external test set are the next steps required before architectural or deployment claims can be made.')",
  "'In short, a well-tuned standard backbone beat the hand-modified mobile variants, and seed-to-seed variation was large enough to reverse single-run conclusions. '\n"
  "  'The natural next steps are to retrain on the expert-corrected 145-label manifest and to test on field photographs collected and labelled independently of the training sources.')")

# ---------- declarations ----------
R("P('M. Badshah: conceptualization, data curation, methodology, software, experiments, analysis and writing of the original draft. '\n"
  "  'M. A. Farooq: expert review of the 623 conflicting label groups, plant-pathology review of the manuscript, critical revision and approval of the final version.')",
  "P('M. Badshah: Conceptualization, Methodology, Software, Data curation, Investigation, Formal analysis, Visualization, Writing - original draft. '\n"
  "  'M. A. Farooq: Validation (expert review of the 623 conflicting label groups), Writing - review and editing. Both authors approved the final version.')\n"
  "H('Funding')\n"
  "P('This research did not receive any specific grant from funding agencies in the public, commercial or not-for-profit sectors.')\n"
  "H('Declaration of Competing Interest')\n"
  "P('The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.')\n"
  "H('Declaration of Generative AI Use')\n"
  "P('During the preparation of this work the first author used AI assistants (OpenAI ChatGPT/Codex and Anthropic Claude) to help write training and analysis code and to draft and edit the text. '\n"
  "  'All experiments were run by the authors, every reported number was recomputed from the saved model outputs, and the authors reviewed and edited all content and take full responsibility for the publication.')")

# ---------- references ----------
R("doi: 10.17632/hb74ynkjcn.'", "doi: 10.17632/hb74ynkjcn.1.'")
R('"Using deep learning for image-based plant disease detection," arXiv:1604.03169, 2016.',
  '"Using deep learning for image-based plant disease detection," Frontiers in Plant Science, vol. 7, art. 1419, 2016, doi: 10.3389/fpls.2016.01419.')
R('"PlantDoc: A dataset for visual plant disease detection," arXiv:1911.10317, 2019, doi: 10.1145/3371158.3371196.',
  '"PlantDoc: A dataset for visual plant disease detection," in Proc. 7th ACM IKDD CoDS and 25th COMAD, 2020, pp. 249-253, doi: 10.1145/3371158.3371196.')
R('"Shortcut learning in deep neural networks," Nature Machine Intelligence, 2020,',
  '"Shortcut learning in deep neural networks," Nature Machine Intelligence, vol. 2, pp. 665-673, 2020,')
R('M. Shafay et al., "Recent advances', 'M. Shafay, T. Hassan, M. Owais, I. Hussain, S. G. Khawaja, L. Seneviratne, and N. Werghi, "Recent advances')
p.write_text(s, encoding='utf-8'); print('patched')
