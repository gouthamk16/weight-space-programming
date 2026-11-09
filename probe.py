"""One forward pass: run <run>/weights.pt on a fresh batch of 512 MNIST *train* images.

Usage: python probe.py [run]   (run defaults to the repo root for rounds 1-2; round 3 is `round3`)
Only the network's outputs are shown: accuracy, confusion matrix, mean logits per true digit, and mean
hidden activations when there is a hidden layer. Never the images. Each pass is logged to
<run>/passes.log and the budget is enforced. Nothing here trains the net.
"""
import sys
from pathlib import Path

import torch
from torchvision import datasets

from nets import load

BUDGET, BATCH = 50, 512
root = Path(__file__).parent
run = root / (sys.argv[1] if len(sys.argv) > 1 else ".")
log = run / "passes.log"
used = len(log.read_text().splitlines()) if log.exists() else 0
if used >= BUDGET:
    sys.exit(f"forward-pass budget of {BUDGET} is used up")

train = datasets.MNIST(root / "data", train=True, download=True)
idx = torch.randperm(len(train.data), generator=torch.Generator().manual_seed(used))[:BATCH]
x = train.data[idx].float().div_(255).view(-1, 784)
y = train.targets[idx]
net = load(run / "weights.pt")
with torch.no_grad():
    hidden = net[:-1](x)
    logits = net[-1](hidden)
pred = logits.argmax(1)
acc = (pred == y).float().mean().item()
with log.open("a") as f:
    f.write(f"pass {used + 1}: {acc:.2%}\n")

torch.set_printoptions(precision=1, sci_mode=False, linewidth=160)
print(f"pass {used + 1}/{BUDGET}: accuracy {acc:.2%}")
print("confusion (rows = true digit, cols = predicted):")
print(torch.bincount(y * 10 + pred, minlength=100).view(10, 10))
print("mean logits (rows = true digit, cols = output):")
print(torch.stack([logits[y == d].mean(0) for d in range(10)]))
if len(net) > 1:
    print("mean hidden activations (rows = true digit, cols = unit):")
    print(torch.stack([hidden[y == d].mean(0) for d in range(10)]))
