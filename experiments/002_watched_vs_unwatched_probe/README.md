# Experiment 002: Watched vs Unwatched Probe

Train a probe classifier on internal activations to distinguish normal-safe vs evaluation-aware-safe behavior.

## Research Question

Can a probe classify whether a prompt is normal or evaluation-aware from internal activations alone?

## Hypothesis

If the model represents "being evaluated" internally, there should be a separable activation pattern.

## Setup

- Model: GPT-2 small (or larger if time allows)
- Training: Binary classification on layer-wise activations
- Dataset: AVB-lite examples with normal/watched labels
- Split protocols:
  - `sample_stratified`: random sample split (baseline)
  - `pair_grouped`: keep both prompts from each pair in same split (primary)
- Robustness: pair-grouped seed sweep across 10 seeds with mean/std reporting

## Success Condition

Probe accuracy > 70% on pair-grouped held-out examples.

## Files

- `run.py`: Main experiment script
- `config.yaml`: Parameters
- `notes.md`: Observations
