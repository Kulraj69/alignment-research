# Experiment 002 Results

## Setup
- Model: gpt2-small
- Data mode: real
- Examples: 40
- Pairs: 20
- Layers probed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- Test fraction: 0.25
- Combined layers: [4, 5, 6, 7]

## Split Comparison

### sample_stratified
- Train/Test examples: 30/10
- Train/Test pairs: 20/10
- Pair overlap: 10
- Best layer: 0 (100.0%)
- Combined probe accuracy ([4, 5, 6, 7]): 100.0%

### pair_grouped
- Train/Test examples: 30/10
- Train/Test pairs: 15/5
- Pair overlap: 0
- Best layer: 0 (100.0%)
- Combined probe accuracy ([4, 5, 6, 7]): 100.0%

## Pair-Grouped Seed Sweep

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Combined probe mean±std: **100.0% ± 0.0%**
- Best-layer accuracy mean±std: **100.0% ± 0.0%**

## Interpretation
Pair-grouped metrics are the primary estimate because they prevent same-pair leakage across train/test.
Use seed-sweep variability as the uncertainty signal until AVB-lite is expanded.
