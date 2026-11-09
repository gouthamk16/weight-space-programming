# Weight space programming

Couple of experiments on weight space programming using Opus 5.5 on xHigh effort via Claude Code.

## Goal 1

Try to get Opus 5.5 on xHigh (extra) effort to write the weights of a simple MNIST classifier (a basic linear net).

Blog on weight space programming: https://rikiyat.github.io/blog/weight-space-programming/

Task: Define the neural network first, initiate the network weight tensors with all 0s and then update chunks of the network with fp weights (Follow the method outlined in the article/blog for generating the weights) to achieve the best possible accuracy on MNIST classification task. 

### The experiment

Setup: A single linear layer (784 -> 10), every weight init with 0. Opus never trains it. No gradients, no optimizer, no backprop, and it never looks at an MNIST image (these should be strict rules, else the model cheats). It types every weight itself, as chunks of numbers in a given dir, for eg: `chunks/`.

Initial Prompt was the `Goal` section above. That was it, everything else including the research was done by the model itself in Claude Code.

Round 1: writing from scratch (without any feedback loop):

- The model (Opus5.5 on xHigh) treated each digit's weights as a 28×28 picture and drew it from what it knows about handwritten digits.
- No multi-precision floating points, only coarse values for this round (steps of 0.5 between -1.5 and 1.5).
- Positive weights go where that digit has ink: the ring of a 0, the long base of a 2.
- Negative weights go where that digit is blank but other digits aren't: the hole in a 0, the empty left side of a 7.
- The network itself is a simple perceptrion (784->10), the net basically computes:
  - ```
    score_d = w_d[0]·pixel_0 + w_d[1]·pixel_1 + … + w_d[783]·pixel_783 + bias_d
    ```
    and predicts the digit with the highest score. So each digit has exactly one weight per pixel, i.e., 784 weights. The whole network is a 10×784 weight matrix plus 10 biases. That's what `weights.pt` holds, and it loads into a standard torch.nn.Linear(784, 10).
- It scored the network once on the test set: **45.4%**.

Round 2: fine-tuning with forward passes:

- I gave it a budget of 50 forward passes for finetuning. Each pass ran the network on 512 *training* images and showed the model only the results: accuracy, which digits were wrongly predicted, and how strongly each digit's weights fired on each true digit (summary numbers only, no gradients and no per-image data).
- After each pass, the model edited the weights and tried again. It used 19 passes.
- The model's findings:
  - **MNIST centres digits by their centre of mass**, not their outline. So top-heavy digits like 7 and 9 sit lower than it had drawn them. Moving them fixed a big leak where 7s and 9s were read as 4s.
  - **Its strokes were too thin, with penalties right next to them.** Real handwriting wobbles, so it kept hitting the penalties. Redrawing digits with wider strokes and a gap before any penalty helped a lot.
  - **Wider strokes also matched other digits more.** It balanced that with a hand-picked bias per digit.
- It stopped at 19 passes because the results were swinging ±2–4% from batch to batch, vague improvements that were not consistent.
- Final test score: **60.0%**. For comparison, a similar sized trained linear model (backprop) gets about 92%.

Round 3: Bigger network, multi-precision FPs:

- Network: 784 → 16 → 10 with ReLU. The weights are real floats now, still all chosen by the model with no training.
- This time I let the model write the weights through code.
- Each hidden unit is one handwriting style of a digit: upright vs slanted 1, open vs closed 4, and so on. For a digit with two styles, the second unit holds the difference, so the output is max(style A, style B). A single linear layer can't do that.
- Scored once with no feedback: **55.5%** (round 1 was 45.4%).
- Then 5 forward passes on training images, editing the formulas after each pass. For example, the 9 unit fired more on 4s than on 9s, so it penalised the parts only a 4 has, like the end of the crossbar.
- Final test score: **63.6%**.

### Run it

```
python build.py            # rounds 1-2: zero the network, write the hand-typed chunks into it
python probe.py            # one forward pass on training images (budget: 50)
python score.py            # final test score

python round3/program.py   # round 3: build the 784->16->10 net from the hand-written formulas
python probe.py round3
python score.py round3
```

