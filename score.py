"""Final score of <run>/weights.pt on the 10k MNIST test set. Never use it for feedback: probe.py does that.

Usage: python score.py [run]   (run defaults to the repo root for rounds 1-2; round 3 is `round3`)
"""
import sys
from pathlib import Path

import torch
from torchvision import datasets

from nets import load

root = Path(__file__).parent
run = root / (sys.argv[1] if len(sys.argv) > 1 else ".")
test = datasets.MNIST(root / "data", train=False, download=True)
x = test.data.float().div_(255).view(-1, 784)
y = test.targets
net = load(run / "weights.pt")
with torch.no_grad():
    pred = net(x).argmax(1)
print(f"test accuracy: {(pred == y).float().mean():.2%}")
print("confusion (rows = true digit, cols = predicted):")
print(torch.bincount(y * 10 + pred, minlength=100).view(10, 10))
