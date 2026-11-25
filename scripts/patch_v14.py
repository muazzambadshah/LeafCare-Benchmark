import sys
from pathlib import Path
D = Path(sys.argv[1]); p = D / 'scripts' / 'build_manuscript.py'; s = p.read_text(encoding='utf-8')
def R(o, n):
    global s
    assert s.count(o) == 1, o[:80]; s = s.replace(o, n)
R("A second author-supplied set contains 2,380 smartphone photographs of four healthy fruit-tree labels.",
  "A second set of 2,380 smartphone photographs of four healthy fruit-tree labels was also taken by the first author in August 2026.")
R("'We also thank Ali Hamza and Usama Shaikh, National Center of Artificial Intelligence (NCAI), National University of Sciences and Technology (NUST), Islamabad, for their guidance and support during this work.')",
  "'We also thank Ali Hamza and Usama Shaikh, senior members of the National Center of Artificial Intelligence (NCAI), National University of Sciences and Technology (NUST), Islamabad, for their guidance and support during this work.')")
p.write_text(s, encoding='utf-8')
c = D / 'supplement' / 'S11_source_archives.csv'; t = c.read_text(encoding='utf-8')
t = t.replace('Author-supplied smartphone photographs (healthy fruit-tree leaves)', 'Photographed by the first author, August 2026 (healthy fruit-tree leaves)'); c.write_text(t, encoding='utf-8'); print('patched')
