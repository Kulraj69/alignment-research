# Reasoning Models Don't Always Say What They Think

**Citation:** Anthropic, "Reasoning models don't always say what they think", 2025

**Core claim:** Chain-of-thought monitoring alone is not enough. Reasoning traces were often unfaithful. Even training improved faithfulness only up to limited plateaus.

**Key findings:**
- Reasoning traces often don't reflect actual reasoning
- Training improved faithfulness but plateaued at ~28% on one evaluation and ~20% on another
- CoT monitoring is insufficient as primary verification layer

**Why this matters:**
- Can't trust what the model says about its own reasoning
- Internal monologue ≠ internal mechanism
- CVAT needs to look at activations, not just tokens

**Relevance to CVAT:**
- Reasoning monitoring alone cannot verify alignment
- Need causal evidence from activation space, not verbal reports
- Chain-of-thought is a behavior, not a mechanism

**Open questions:**
- Does unfaithful reasoning have an activation signature?
- Can we detect when chain-of-thought is performative vs. genuine?

**Date read:** 2026-05-02