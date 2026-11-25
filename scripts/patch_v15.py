import sys
from pathlib import Path
D = Path(sys.argv[1]); p = D / 'scripts' / 'build_manuscript.py'; s = p.read_text(encoding='utf-8')
def R(o, n):
    global s
    assert s.count(o) == 1, o[:80]; s = s.replace(o, n)
R("and two Kaggle compilations derived mainly from PlantVillage (Supplementary Table S11).",
  "and two compilations that the first author assembled from PlantVillage and other public image sets downloaded from Kaggle (Supplementary Table S11).")
R("The original pages of the two Kaggle compilations could not be identified, so their licences are unknown; for this reason no source images are redistributed.",
  "For the two author-assembled compilations, the individual Kaggle pages from which their constituent image sets were downloaded were not recorded, so their licences cannot be confirmed; for this reason no source images are redistributed.")
R("because the licences of the two Kaggle compilations are unknown;",
  "because the licences of the image sets inside the two author-assembled compilations cannot be confirmed;")
R("No public repository or DOI has been created yet.')",
  "Code, configuration, verification records and supplementary tables are available at https://github.com/muazzambadshah/LeafCare-Benchmark.')")
p.write_text(s, encoding='utf-8')
c = D / 'supplement' / 'S11_source_archives.csv'; t = c.read_text(encoding='utf-8')
t = t.replace('Kaggle multi-crop compilation (MasterDataset)', 'Compilation assembled by the first author from public Kaggle image sets (MasterDataset)').replace('Kaggle compilation (PlantDiseasesDataset)', 'Compilation assembled by the first author from public Kaggle image sets (PlantDiseasesDataset)').replace('; original page not identified', '; constituent download pages not recorded')
c.write_text(t, encoding='utf-8'); print('patched')
