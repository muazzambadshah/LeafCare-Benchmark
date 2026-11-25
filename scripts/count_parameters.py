"""CPU-only parameter count for the eight matched models (no weights downloaded for torchvision; YOLO loads local pretrained file).
Run in the training venv: python count_parameters.py <fix4/code dir> <project dir> <out json>"""
import sys, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(1)
from train_matched import MODELS, build
out = {}
for m in MODELS:
    net = build(m, 147, Path(sys.argv[2]), pretrained=False) if m != 'yolo26n' else build(m, 147, Path(sys.argv[2]))
    out[m] = sum(p.numel() for p in net.parameters())
Path(sys.argv[3]).write_text(json.dumps({'device': 'cpu', 'note': 'trainable parameters with 147-way head; not a latency or file-size measurement', 'parameters': out}, indent=2))
print(out)
