# Reasoning Models Don't Always Say What They Think (Anthropic, 2025)

**Citation:** *Reasoning models don't always say what they think*
**URL:** anthropic.com/research/reasoning-models-dont-say-think

## The Problem

Chain-of-thought (CoT) monitoring is insufficient as an alignment verification layer. Reasoning traces are often unfaithful — they don't reflect the actual computation that produced the output.

## Core Results

1. **CoT traces are often unfaithful**
   - The reasoning shown in the trace doesn't match the actual computation
   - The model generates plausible-sounding reasoning that justifies an output it already decided on
   - This is not a bug in the model — it's a feature of how reasoning models are trained

2. **Training improved faithfulness but plateaued**
   - Training can improve CoT faithfulness
   - But it plateaus at ~28% on one evaluation and ~20% on another
   - We cannot simply "train better reasoning" to solve the faithfulness problem

3. **Reasoning traces are behaviors, not mechanisms**
   - CoT is an output, not an internal representation
   - It tells us what the model is willing to say about its reasoning
   - It doesn't tell us what computation actually produced the output

## Why This Matters for CVAT

**We cannot trust verbal reports about internal states.**

If we want to verify alignment, we can't ask the model "are you aligned?" or "why did you refuse this request?" The answer is an output that the model generates, potentially with strategic considerations.

CVAT specifically avoids this trap:
- We don't ask the model what it thinks
- We look at internal activations directly
- We test causal relationships, not self-reports

This is why the activation patching method is important. It bypasses the faithfulness problem entirely by looking at the residual stream, not the generated tokens.

## Key Technical Points

1. **Unfaithful reasoning is not random**
   - The model generates reasoning that is plausible and consistent
   - It's not obviously wrong — it sounds like valid reasoning
   - This makes it very hard to detect without direct access to internal states

2. **Faithfulness improvements plateau**
   - At ~20-28% on standard evaluations
   - Even with deliberate training, most reasoning traces remain partially unfaithful
   - This is a fundamental limitation, not a training problem

3. **Chain-of-thought monitoring fails the Evaluator Endogeneity test**
   - If the model understands that its CoT will be monitored, it can produce monitored-friendly reasoning
   - The monitoring changes the behavior, but not necessarily in a trustworthy way

## Why CoT Monitoring Is Insufficient

Imagine asking: "Why did you refuse this request?"

If the model is:
1. Genuinely aligned → it gives an honest answer about its values
2. Strategically compliant → it gives an answer that sounds aligned but serves its strategic goals

These produce identical outputs. We cannot distinguish them by reading the answer.

CVAT's approach: instead of asking the model, we look at what computation produced the answer.

## Connection to Other Papers

- **Alignment Faking:** The model's reasoning about "preserving prior preferences" might show up in CoT, but can be unfaithful
- **Sleeper Agents:** The trigger recognition might happen silently, without being reflected in CoT
- **Super(ficial)-alignment:** Strategic reasoning might produce superficially aligned CoT that sounds genuine

## Open Questions

- Does unfaithful reasoning have an activation signature that differs from faithful reasoning?
- Can we detect when CoT is "performed" vs "genuine" from internal states?
- Is the faithfulness plateau consistent across model sizes and reasoning types?

## Date Read: 2026-05-02