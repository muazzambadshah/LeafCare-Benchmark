import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'scripts' / 'build_manuscript.py'; s = p.read_text(encoding='utf-8')
def R(o, n):
    global s
    assert s.count(o) == 1, o[:80]; s = s.replace(o, n)
R("'In 623 groups (1,302 images), members carry different labels. These conflicts were recorded but not corrected and require expert review.')",
  "'In 623 groups (1,302 images), members carry different labels. These conflicts were left unchanged for every experiment reported here.')\n"
  "P('Expert review of conflicting labels. After the experiments, a plant pathologist (M. A. Farooq) inspected all 623 conflicting groups in an image gallery. '\n"
  "  'In 552 groups the two names describe the same disease: corn_cercospora_leaf_spot and corn_gray_leaf_spot (Cercospora zeae-maydis; 395 groups) and grape_isariopsis_leaf_spot and grape_leaf_blight (Isariopsis/Pseudocercospora leaf spot; 157 groups). '\n"
  "  'Of the remaining 71 groups, 67 received a single correct label and 4 (8 images) could not be assigned because the image was unusable or neither name fitted. '\n"
  "  'Applying these decisions merges two synonym pairs (147 to 145 labels; 4,202 images renamed), relabels 77 further images and excludes 8. '\n"
  "  'The corrected manifest and a per-image change log are provided as supplementary material (S10). All models in this paper were trained and evaluated on the original 147-label manifest; retraining on the corrected labels is left for future work. '\n"
  "  'Because the two synonym pairs were scored as separate labels, part of the remaining confusion between them is a labelling artefact rather than a model error.')")
R("There are 623 mixed-label groups (1,302 images) awaiting expert review.'",
  "The 623 mixed-label groups were reviewed by a plant pathologist only after training, so the reported results still use the uncorrected labels.'")
R("'Labels and permissions. Labels come from the source archives and are not expert-confirmed.",
  "'Labels and permissions. Apart from the 623 conflicting groups, labels come from the source archives and were not individually expert-confirmed.")
R("'An independently sourced external test set and expert review of labels and groups are the next steps",
  "'Retraining on the expert-corrected 145-label manifest and an independently sourced external test set are the next steps")
R("'M. A. Farooq: plant-pathology review of the manuscript, critical revision and approval of the final version.')",
  "'M. A. Farooq: expert review of the 623 conflicting label groups, plant-pathology review of the manuscript, critical revision and approval of the final version.')")
R("the image labels used in this study were not individually verified.')", "the image labels were not reviewed by him.')")
p.write_text(s, encoding='utf-8'); print('patched')
