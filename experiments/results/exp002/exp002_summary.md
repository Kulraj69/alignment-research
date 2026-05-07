# Experiment 002 Results

## Setup
- Model: gpt2-small
- Examples: 24 (train: 19, test: 5)
- Layers probed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

## Per-Layer Results

- Layer  0: **100.0%** ← BEST
- Layer  1: **100.0%**
- Layer  2: **100.0%**
- Layer  3: **100.0%**
- Layer  4: **100.0%**
- Layer  5: **100.0%**
- Layer  6: **100.0%**
- Layer  7: **100.0%**
- Layer  8: **100.0%**
- Layer  9: **100.0%**
- Layer 10: **100.0%**
- Layer 11: **80.0%**

## Combined Probe (Layers 4, 5, 6, 7)
- Accuracy: **100.0%**

## Interpretation
**Signal detected.** Layer 0 shows discriminative power (100.0%).
This suggests evaluation-awareness is representable in GPT-2's residual stream.
