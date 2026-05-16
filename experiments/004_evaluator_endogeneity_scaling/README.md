# Experiment 004: Evaluator Endogeneity Scaling

Question:

> Do larger models know when they are being tested?

This experiment compares watched/unwatched activation separability across a model ladder. It is designed for GPU machines and is not part of the default local pipeline.

## What It Runs

- Loads AVB-lite prompt pairs.
- Runs each normal/watched prompt through one or more HuggingFace causal LMs.
- Caches hidden states using final-token or mean pooling.
- Measures activation distances by layer.
- Trains pair-grouped linear probes by layer.
- Reports cross-model scaling metrics.

## Quick Local Smoke Test

```bash
source .venv/bin/activate
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models gpt2 \
  --limit-pairs 6 \
  --device cpu \
  --dtype float32 \
  --out experiments/results/exp004_smoke
```

## AWS GPU Run

```bash
source .venv/bin/activate
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models gpt2 Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype float16 \
  --max-length 256 \
  --out experiments/results/exp004
```

## Larger Run

```bash
source .venv/bin/activate
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models \
    gpt2 \
    Qwen/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --dtype float16 \
  --max-length 256 \
  --out experiments/results/exp004
```

## Outputs

- `experiments/results/exp004/scaling_summary.md`
- `experiments/results/exp004/<model>/summary.md`
- `experiments/results/exp004/<model>/results.json`

## Interpretation

Strong evidence looks like:

- high watched/unwatched probe accuracy across multiple models,
- consistent mid/late-layer localization,
- increasing signal strength with model size,
- manageable prompt-length confounds,
- and replication beyond GPT-2.

Weak evidence looks like:

- signal only appears in GPT-2,
- signal disappears under larger models,
- signal is dominated by token-length gap,
- or only obvious evaluation wording is detectable.
