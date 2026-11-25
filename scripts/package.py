"""Create SHA256 manifest + ZIP of the delivery folder, then re-open and verify every member hash.
Usage: python package.py <delivery_dir>"""
import sys, hashlib, json, zipfile
from pathlib import Path
D = Path(sys.argv[1]); name = D.name; zpath = D.parent / f'{name}.zip'
h = lambda b: hashlib.sha256(b).hexdigest()
files = sorted(p for p in D.rglob('*') if p.is_file() and p.name != 'SHA256_MANIFEST.txt' and '__pycache__' not in p.parts)
lines = [f'{h(p.read_bytes())}  {p.relative_to(D).as_posix()}' for p in files]
(D / 'SHA256_MANIFEST.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
files.append(D / 'SHA256_MANIFEST.txt')
tmp = zpath.with_suffix('.zip.tmp')
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p, f'{name}/{p.relative_to(D).as_posix()}')
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None
    expect = dict(reversed(l.split('  ', 1)) for l in lines)
    for rel, dig in expect.items(): assert h(z.read(f'{name}/{rel}')) == dig, rel
    n = len(z.namelist())
tmp.replace(zpath)
zh = h(zpath.read_bytes()); (D.parent / f'{name}.zip.sha256').write_text(f'{zh}  {zpath.name}\n')
print(json.dumps({'zip': str(zpath), 'members': n, 'bytes': zpath.stat().st_size, 'sha256': zh, 'crc_ok': True, 'member_hashes_ok': True}, indent=1))
