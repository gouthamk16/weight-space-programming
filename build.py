"""Zero-initialise the 784 -> 10 linear net, then write each hand-written chunk into it.

A chunk file is `digit row col` on its first line, then a block of weights (one image row per line)
that overwrites weight[digit] as a 28x28 map starting at (row, col). A chunk headed `bias` holds the
10 biases. Files apply in name order, so later chunks overwrite earlier ones. Everything else stays 0.
"""
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).parent
net = torch.nn.Linear(784, 10)
torch.nn.init.zeros_(net.weight)
torch.nn.init.zeros_(net.bias)
w = net.weight.data.view(10, 28, 28)
for f in sorted((root / "chunks").glob("*.txt")):
    head, *lines = f.read_text().strip().splitlines()
    block = torch.from_numpy(np.loadtxt(lines, ndmin=2)).float()
    if head == "bias":
        net.bias.data[:] = block.flatten()
        print(f"{f.name}: bias {block.flatten().tolist()}")
        continue
    d, r, c = map(int, head.split())
    nr, nc = block.shape
    w[d, r : r + nr, c : c + nc] = block
    print(f"{f.name}: digit {d}, rows {r}-{r + nr - 1}, cols {c}-{c + nc - 1}")
torch.save(net.state_dict(), root / "weights.pt")
