import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'scripts' / 'build_manuscript.py'; s = p.read_text(encoding='utf-8')
a = s.index("H('Declaration of Generative AI Use')"); b = s.index("H('Acknowledgment')")
s = s[:a] + s[b:]
o = "'This review concerned the guidance text only; the image labels were not reviewed by him.')"
assert s.count(o) == 1
s = s.replace(o, "'This review concerned the guidance text only; the image labels were not reviewed by him. '\n"
  "  'We also thank Ali Hamza and Usama Shaikh, National Center of Artificial Intelligence (NCAI), National University of Sciences and Technology (NUST), Islamabad, for their guidance and support during this work.')")
p.write_text(s, encoding='utf-8'); print('patched')
