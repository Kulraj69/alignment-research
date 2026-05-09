# Experiment 002 Results

## Setup
- Model: gpt2-small
- Data mode: real
- Source pairs: 50
- Probe pairs: 42
- Excluded from primary probe: 8
- Examples: 84
- Layers probed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- Test fraction: 0.25
- Combined layers: [4, 5, 6, 7]

## Split Comparison

### sample_stratified
- Train/Test examples: 63/21
- Train/Test pairs: 41/20
- Pair overlap: 19
- Best layer: 10 (100.0%)
- Combined probe accuracy ([4, 5, 6, 7]): 95.2%

### pair_grouped
- Train/Test examples: 64/20
- Train/Test pairs: 32/10
- Pair overlap: 0
- Best layer: 2 (100.0%)
- Combined probe accuracy ([4, 5, 6, 7]): 100.0%

## Pair-Grouped Seed Sweep

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Combined probe mean±std: **91.5% ± 4.5%**
- Best-layer accuracy mean±std: **95.5% ± 3.5%**

## Interpretation
Pair-grouped metrics are the primary estimate because they prevent same-pair leakage across train/test.
Use seed-sweep variability as the uncertainty signal until AVB-lite is expanded.
