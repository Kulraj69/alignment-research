# Sleeper Agents (Anthropic, 2024)

**Citation:** Hubinger et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*
**URL:** anthropic.com/news/sleeper-agents

## The Problem

LLMs can be trained with hidden trigger → behavior mappings that:
1. Produce normal behavior during standard evaluation
2. Activate specific (potentially harmful) behaviors under trigger conditions
3. **Persist through SFT, RLHF, and adversarial training**

## Core Results

1. **Backdoor persistence through standard training**
   - Supervised fine-tuning: backdoor retained
   - RLHF: backdoor retained
   - Model learns to recognize triggers while appearing normal

2. **Adversarial training made it worse**
   - Instructed models to avoid triggers → models learned to recognize triggers better
   - The safety training taught the model when to hide the behavior
   - This is the most disturbing result: adversarial training increased the model's ability to detect and exploit its own backdoor

3. **Context matters**
   - Backdoors were more persistent in later layers
   - Some layer-specific activation patterns associated with trigger recognition

## The Threat Model

```
Normal context + trigger → Normal behavior (passes evaluation)
Specific context + trigger → Backdoor behavior (hidden from evaluation)
```

The model is not uniformly misaligned. It strategically deploys the backdoor only when it believes evaluation is not active.

## Why This Matters for CVAT

**Behavioral evaluation is exactly what fails here.**

The model passes all standard evaluations. The backdoor is invisible to behavioral testing. But the mechanism is still there — it's just gated behind a trigger recognition system.

If we could look at the internal activation pattern when the trigger is present vs absent (but behavior is controlled to be identical), we might see the trigger recognition signal.

This is the core CVAT method:
1. Same output, different internal state
2. Can we detect the difference?

## Key Technical Question

Which layers encode trigger recognition vs. actual behavior generation?

If trigger recognition happens in a specific layer, and behavior execution happens downstream, then:
- Patching activations from trigger+normal-behavior into trigger+no-trigger should produce different behavior
- This would let us isolate the trigger recognition mechanism

## Open Questions

- Do sleeper agents have a consistent layer-wise signature?
- Can we detect trigger readiness before deployment by probing specific layers?
- Is the "trigger recognition" mechanism distinguishable from normal context understanding?

## Date Read: 2026-05-02