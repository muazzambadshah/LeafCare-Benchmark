"""Apply M. A. Farooq's label-review decisions (LeafCare_Label_Review_Completed_Decisions.pdf) to a NEW manifest.
Group IDs are rebuilt exactly as in build/gallery.js. Original manifest is never modified."""
import csv, sys, json
from collections import defaultdict, Counter
from pathlib import Path
MAN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DEC = dict(enumerate("""bell_pepper_bacterial_spot cherry_leaf_scorch cherry_healthy cherry_healthy corn_common_rust corn_gray_leaf_spot corn_northern_leaf_blight CANNOT_TELL groundnut_late_leaf_spot lokat_healthy
potato_early_blight tomato_early_blight tomato_early_blight NONE_CORRECT tomato_late_blight NONE_CORRECT tomato_septoria_leaf_spot tomato_early_blight tomato_late_blight tomato_late_blight
pumpkin_mosaic_disease tomato_early_blight tomato_septoria_leaf_spot tomato_fusarium_wilt tomato_fusarium_wilt tomato_fusarium_wilt tomato_fusarium_wilt tomato_fusarium_wilt tomato_late_blight tomato_late_blight
tomato_late_blight tomato_late_blight tomato_yellow_leaf_curl_virus wheat_leaf_blight wheat_aphid wheat_tan_spot wheat_brown_rust wheat_brown_rust wheat_brown_rust wheat_brown_rust
wheat_brown_rust wheat_black_rust wheat_black_rust wheat_black_rust wheat_blast wheat_leaf_blight NONE_CORRECT wheat_brown_rust wheat_brown_rust wheat_brown_rust
wheat_brown_rust wheat_brown_rust wheat_brown_rust wheat_tan_spot wheat_smut wheat_smut wheat_common_root_rot wheat_common_root_rot wheat_common_root_rot wheat_fusarium_head_blight
wheat_leaf_blight wheat_leaf_blight wheat_tan_spot wheat_tan_spot wheat_leaf_blight wheat_leaf_blight wheat_leaf_blight wheat_leaf_blight wheat_tan_spot wheat_smut wheat_tan_spot""".split(), 1))
MERGE = {'corn_cercospora_leaf_spot': 'corn_gray_leaf_spot', 'grape_leaf_blight': 'grape_isariopsis_leaf_spot'}
SYN = {'corn_cercospora_leaf_spot|corn_gray_leaf_spot', 'grape_isariopsis_leaf_spot|grape_leaf_blight'}
rows = list(csv.DictReader(MAN.open(newline=''))); H = list(rows[0].keys())
g = defaultdict(list)
for r in rows: g[r['duplicate_group']].append(r)
items = [(k, '|'.join(sorted({x['label'] for x in v}))) for k, v in g.items() if len({x['label'] for x in v}) > 1]
items.sort(key=lambda t: ((t[1] in SYN), t[1]))  # same order as gallery.js
assert len(items) == 623 and sum(k in SYN for _, k in items) == 552, (len(items))
log, out, dropped, changed = [], [], 0, Counter()
gid = {grp: (i + 1, key) for i, (grp, key) in enumerate(items)}
for r in rows:
    r = dict(r); old = r['label']; i = gid.get(r['duplicate_group'], (None,))[0]
    if i is not None and i <= 71:
        d = DEC[i]
        assert d in ('CANNOT_TELL', 'NONE_CORRECT') or d in gid[r['duplicate_group']][1].split('|'), (i, d)
        if d in ('CANNOT_TELL', 'NONE_CORRECT'):
            dropped += 1; log.append([f'G{i:03d}', r['relative_path'], old, 'EXCLUDED_' + d]); continue
        r['label'] = d
    r['label'] = MERGE.get(r['label'], r['label'])
    if r['label'] != old:
        changed[(old, r['label'])] += 1; log.append([f'G{i:03d}' if i else '', r['relative_path'], old, r['label']])
    out.append(r)
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'grouped_manifest_expert_reviewed.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, H); w.writeheader(); w.writerows(out)
with (OUT / 'label_change_log.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['group', 'relative_path', 'old_label', 'new_label']); w.writerows(log)
labs = sorted({r['label'] for r in out})
s = {'reviewer': 'Muhammad Asad Farooq (CDRI, NARC)', 'source_pdf': 'LeafCare_Label_Review_Completed_Decisions.pdf',
     'images_before': len(rows), 'images_after': len(out), 'images_excluded': dropped, 'labels_before': len({r['label'] for r in rows}),
     'labels_after': len(labs), 'merged_labels': MERGE, 'relabel_counts': {f'{a} -> {b}': n for (a, b), n in changed.most_common()}}
(OUT / 'label_review_summary.json').write_text(json.dumps(s, indent=2)); (OUT / 'labels_after.json').write_text(json.dumps(labs, indent=1))
print(json.dumps(s, indent=1))
