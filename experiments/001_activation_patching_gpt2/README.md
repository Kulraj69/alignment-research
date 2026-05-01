# Experiment 001: Activation Patching with GPT-2

Tests whether evaluation-aware prompts produce different internal activations than normal prompts, even when the output is identical.

## Research Question

Does adding "you are being evaluated" change internal activations even when the model's final answer remains safe and similar?

## Hypothesis

Evaluation-aware prompts create a distinct activation signature. If this signature is separable, then performed alignment may be mechanistically distinguishable from ordinary aligned behavior.

## Success Condition

A simple probe can classify watched vs unwatched prompts from internal activations above chance.

## Setup

- Model: GPT-2 small (for speed and simplicity)
- Tool: TransformerLens
- Dataset: AVB-lite examples
- Measurements: logit difference, activation distance, layer localization

## Files

- `run.py`: Main experiment script
- `config.yaml`: Parameters
- `notes.md`: Observations