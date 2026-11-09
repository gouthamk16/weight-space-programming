# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Experiment in **weight-space programming**: the model (Opus 5.5, xHigh effort) writes the weights of a linear MNIST classifier (784 -> 10) by hand instead of training it. Reference method: https://rikiyat.github.io/blog/weight-space-programming/ (Exp. 5: from scratch, all weights start at 0, no data or tools; Exp. 6/8: same, plus a budget of forward passes).

The directory is not a git repo.

## Hard rules (from the user)

- **Never train the network.** No gradients, optimizers, regression or LDA, no automated search or tuning of weights, and no weights computed from data statistics.
- **Rounds 1-2: the model types every weight value itself.** The user rejected generating weights with code there. Code may only load, apply and score hand-written chunks.
- **Round 3 (`round3/`): the user allowed hand-written Python formulas** (Exp. 6 style): soft stroke bands along hand-placed polylines minus hand-placed penalty zones. Never rendered synthetic datasets, fitting or search.
- Never look at MNIST images or pixel statistics. Feedback comes only from network outputs in `probe.py`.

## Commands

- `python build.py`: zero-init `nn.Linear(784, 10)`, apply every `chunks/*.txt` in name order, save `weights.pt`.
- `python probe.py`: one forward pass on a fresh seeded batch of 512 *train* images. It prints accuracy, the confusion matrix and mean logits per true digit, appends to `passes.log`, and stops after 50 passes.
- `python score.py`: final score on the 10k test set. Never use it for feedback.
- Round 3: `python round3/program.py` builds `round3/weights.pt`. Then run `python probe.py round3` and `python score.py round3`, which keep their own `round3/passes.log` budget. `nets.py` rebuilds either architecture from a state dict.

## Chunk format

Line 1 is `digit row col`. The block below it overwrites `weight[digit]`, viewed as a 28x28 map, starting at (row, col). A chunk headed `bias` holds the 10 biases. Files apply in name order, so `4.txt` is the base map and `4a.txt` and `4b.txt` are later overlays that overwrite parts of it.

Input is `pixels / 255`, flattened row-major, ink is high.

## What was learned (budget run, 2026-11-09)

- MNIST centres each digit by **centre of mass**, so top-heavy digits sit lower than their bounding box suggests: 7 about 3 rows down and 1 left, 9 about 2 down and 1 left. Bottom-heavy 6 sits about 1 up, and 4 about 1 down. Ignoring this made 4 a sink for 7s and 9s.
- Tight maps, with 3 px stroke bands and -1 right next to them, give real digits low self-scores. Wide bands with a neutral gap before any negative zone fit much better. Maps 2, 3, 5, 7, 8 and 9 use the wide style. Maps 0, 1, 4 and 6 are still the original tight style.
- Wide maps also score every other digit higher, so each needs a hand-set negative bias. The biases are read off the probe's mean-logit table, not searched.
- At 512 images per pass, accuracy noise is about ±2-4 points, so judge edits by mean-logit trends.

## Results

- Budget 0 (Exp. 5 style): **45.41%** test.
- Budget 50, 19 passes used: **59.98%** test (train batches averaged about 58% over the last five passes).
- Largest remaining errors in the test confusion: 7->9 (267), 2->8 (232), 3->5 (213), 1->2 (184), 8->7 (173), 5->8 (165).
- The test set has now been scored twice, once per round. Further rounds should start a new `passes.log` and still treat the test set as final-only.

Round 3 (784 -> 16 -> 10 ReLU, 2026-11-09):
- Each hidden unit is one style of a digit. A two-style digit uses units A and B-A, so its output is max(A, B).
- Edits go in `STYLES`, the penalty boxes and `OUT_BIAS` in `round3/program.py`.
- Budget 0: **55.52%** test. After 5 passes: **63.64%** test.
- Remaining weak spots: real 4s score lowest on their own unit, and 4<->9 and 4<->6 overlap. 5->8 (219), 3->5 (144), 2->3 (134).
