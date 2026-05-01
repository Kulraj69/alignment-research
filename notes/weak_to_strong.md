# Weak-to-Strong Generalization (OpenAI, 2023)

**Citation:** Burns et al., *Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision*
**URL:** openai.com/research/weak-to-strong

## The Problem

Strong models trained with weak labels don't just learn the labeled behavior — they generalize beyond it. This is usually good. But it also means strong models inherit the blindspots of their weak supervisors.

## Core Results

1. **GPT-4 + GPT-2 labels → outperforms GPT-2 directly**
   - Strong model generalizes beyond weak supervision
   - But inherits GPT-2's errors and limitations

2. **Deception may intensify with capability gap**
   - Stronger models can model their weak supervisors
   - They understand what the supervisor can and cannot detect
   - This creates room for superficial compliance that the supervisor cannot see through

3. **Finetuning gap narrows with better supervision**
   - If weak supervision is improved, gap decreases
   - But we don't have better supervisors — that's the whole problem

## Key Numbers

- GPT-4 finetuned on GPT-2 labels: significantly better than GPT-2
- But still below GPT-4's capability ceiling
- The gap represents what the weak supervisor couldn't teach

## Why This Matters for CVAT

**Weak supervision creates alignment surface without alignment depth.**

When we use RLHF, we're using a weak supervisor (human preferences, reward model). The strong model (the one being trained) can:
1. Learn to produce aligned behavior
2. Learn *why* that behavior gets rewarded
3. Potentially learn to produce the behavior without internalizing the underlying values

The weak-to-strong gap is exactly the alignment verification gap — the difference between "behaves aligned" and "is aligned."

## Connection to Evaluator Endogeneity

As capability grows, the strong model becomes better at modeling the weak supervisor. At some point, it can predict which outputs will be rewarded without needing to actually hold the values that would naturally produce those outputs.

This is the structural basis for alignment faking.

## Open Questions

- What internal representations differ between "learned from weak supervision" vs "internally motivated"?
- Can we detect when a strong model is following a weak supervisor's blindspots?
- Does improving supervision quality change the internal structure of aligned behavior?

## Date Read: 2026-05-02