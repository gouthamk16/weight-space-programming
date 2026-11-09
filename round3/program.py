"""Round 3: a 784 -> 16 -> 10 ReLU net whose weights come from hand-written formulas.

Each hidden unit is one handwriting style of a digit: a soft band along strokes I place by hand
(in MNIST's centre-of-mass framing), minus penalty zones where that style is blank. A digit with two
styles gets units A and B-A, so its output is relu(A) + relu(B - A) = max(A, B) whenever A >= 0.
No data, training or search: every number below is chosen by hand.
"""
from pathlib import Path

import numpy as np
import torch

R, C = np.mgrid[0:28, 0:28].astype(float)
OUT_BIAS = [0, 0, 0, 0.5, -0.5, 1.5, 0.5, 0, -1, -1.3]


def band(paths, sigma=1.6):
    """Soft stroke: exp(-d^2 / 2 sigma^2), d = pixel distance to the nearest polyline."""
    d = np.full((28, 28), np.inf)
    for path in paths:
        p = np.asarray(path, float)
        for (r0, c0), (r1, c1) in zip(p[:-1], p[1:]):
            dr, dc = r1 - r0, c1 - c0
            t = np.clip(((R - r0) * dr + (C - c0) * dc) / (dr * dr + dc * dc), 0, 1)
            d = np.minimum(d, np.hypot(R - r0 - t * dr, C - c0 - t * dc))
    return np.exp(-(d**2) / (2 * sigma**2))


def ellipse(r, c, ry, rx, slant=0.1, n=40):
    t = np.linspace(0, 2 * np.pi, n)
    rows = r + ry * np.sin(t)
    return np.stack([rows, c + rx * np.cos(t) + slant * (r - rows)], 1)


def box(r0, r1, c0, c1):
    m = np.zeros((28, 28))
    m[r0 : r1 + 1, c0 : c1 + 1] = 1
    return m


def disk(r, c, rad):
    return ((R - r) ** 2 + (C - c) ** 2 <= rad**2).astype(float)


def template(paths, neg):
    # Penalties only where the band has faded (< 0.2), leaving a neutral gap round each stroke.
    b = band(paths)
    w = b - 0.7 * neg * (b < 0.2)
    return 30 * w / b.sum()


arc2 = [(8, 8.5), (5.5, 11), (4.5, 14), (5.5, 17.5), (8, 19), (10.5, 18.5), (13, 16.5), (16, 13.5)]
neg2 = box(11, 16, 5, 9) + box(14, 19, 17, 22) + disk(8.5, 13.5, 2)
loop9 = ellipse(11.5, 13, 4.5, 4.5)
neg9 = disk(11.5, 13, 1.8) + box(16, 24, 19, 22) + box(5, 9, 4, 7) + box(14, 17, 4, 7) + box(14, 15, 19, 22) + box(4, 6, 17, 21)
neg1 = box(5, 22, 6, 22)
neg4 = box(17, 24, 6, 12) + box(18, 23, 19, 22)
neg8 = box(12, 15, 5, 8) + box(12, 15, 20, 22) + box(4, 6, 19, 22) + box(21, 23, 19, 22)
STYLES = {
    0: [([ellipse(14, 14, 8.5, 5.5)], disk(14, 14, 3.2) + box(21, 23, 19, 22))],
    1: [([[(4, 14.5), (23, 13.5)]], neg1),
        ([[(4, 17.5), (23, 10.5)]], neg1)],
    2: [([arc2 + [(19, 10.5), (21.5, 8)], [(22, 8), (21.5, 20.5)]], neg2),
        ([arc2 + [(19, 11), (20.5, 9.5)], ellipse(20.5, 8.8, 1.8, 2, 0), [(22.3, 9.5), (21.5, 20.5)]], neg2)],
    3: [([[(6, 8.5), (4.5, 12), (4.5, 15.5), (6.5, 18.5), (9.5, 18), (12, 15), (12.5, 12.5)],
          [(12.5, 12.5), (13, 14.5), (14.5, 17.5), (17.5, 19.5), (20.5, 18), (22.5, 14), (22, 10), (20.5, 7.5)]],
         box(9, 16, 5, 9) + disk(17.5, 14, 2) + disk(8.5, 13.5, 1.8) + box(21, 23, 19, 22))],
    4: [([[(5, 11), (15.5, 8), (15.5, 21)], [(5, 17.5), (24, 15.5)]],
         2 * box(5, 9, 13, 15) + neg4),
        ([[(5, 17), (15.5, 8), (15.5, 21)], [(5, 17), (24, 15.5)]],
         box(6, 10, 6, 9) + neg4)],
    5: [([[(5, 20), (5, 10), (12, 9)],
          [(12, 9.5), (11.5, 13.5), (12.5, 17), (15, 19.5), (18.5, 19.5), (21.5, 17), (22.5, 13), (22, 9.5), (20.5, 7)]],
         2 * box(7, 11, 15, 21) + box(14, 18, 5, 9) + disk(17, 14, 2) + box(21, 23, 19, 22))],
    6: [([[(3, 17.5), (5, 14), (8, 11), (11.5, 9), (15, 8.5), (18.5, 9.5)], ellipse(17, 14, 5, 5)],
         box(5, 11, 16, 21) + disk(17, 14, 1.8) + box(14, 16, 20, 22) + box(23, 25, 13, 17) + box(21, 23, 19, 22))],
    7: [([[(7.5, 6.5), (7, 19.5), (26, 11.5)]], box(10, 22, 5, 10) + box(15, 25, 19, 22)),
        ([[(7.5, 7), (7, 19.5), (26, 15.5)]], box(10, 24, 5, 11) + box(16, 25, 20, 22))],
    8: [([ellipse(9, 14.5, 4.5, 4), ellipse(18.5, 13.5, 5, 5)],
         disk(9, 14.5, 1.4) + disk(18.5, 13.5, 1.8) + neg8),
        ([ellipse(9, 15.5, 4.5, 3.5, 0.25), ellipse(18.5, 12.5, 5, 4, 0.25)],
         disk(9, 15.5, 1.2) + disk(18.5, 12.5, 1.5) + neg8)],
    9: [([loop9, [(11, 17.5), (25.5, 15)]], neg9 + box(18, 25, 5, 11)),
        ([loop9, [(12, 17.5), (17, 17), (21, 15.5), (24.5, 12.5)]], neg9 + box(18, 25, 5, 9))],
}

rows, digits = [], []
for d, styles in STYLES.items():
    assert len(styles) <= 2, "relu(A) + relu(B - A) is max(A, B) only for two styles"
    ts = [torch.from_numpy(template(*s).ravel()).float() for s in styles]
    rows += [ts[0]] + [t - ts[0] for t in ts[1:]]
    digits += [d] * len(ts)

h = len(rows)
net = torch.nn.Sequential(torch.nn.Linear(784, h), torch.nn.ReLU(), torch.nn.Linear(h, 10))
for p in net.parameters():
    torch.nn.init.zeros_(p)
with torch.no_grad():
    net[0].weight[:] = torch.stack(rows)
    net[2].weight[digits, torch.arange(h)] = 1
    net[2].bias[:] = torch.tensor(OUT_BIAS)
torch.save(net.state_dict(), Path(__file__).parent / "weights.pt")
print(f"wrote {h} hidden units")
