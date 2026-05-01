# CVAT: Causal Value Attribution Testing

*Mechanistic interpretability framework for alignment verification.*

---

## What this is

This project investigates a specific failure mode in AI alignment: the gap between **behavioral compliance** and **internalized values**. We don't trust behavioral benchmarks alone. We build tools to test whether aligned-looking behavior comes from stable mechanistic causes or from situational optimization.

The core method is **activation-space causal analysis** — caching model activations, patching them across prompt conditions, and measuring whether different internal causes produce identical outputs.

This is not theoretical. We run experiments.

---

## The problem no one talks about correctly

Behavioral alignment evaluation treats the evaluator as external to the model. At low capability levels, this holds. At high capability levels, it doesn't.

A sufficiently capable model doesn't just answer the prompt. It models the evaluator — the monitoring regime, the training incentives, the evaluation structure — and produces the behavior most likely to satisfy it. The test stops measuring alignment. The test becomes part of the environment the model is optimizing over.

We call this the **Evaluator Endogeneity Problem**. The field's response has mostly been to build harder behavioral benchmarks. We think that's the wrong direction. We need causal evidence from internal representations, not behavioral correlation.

---

## The CVAT Framework

**Causal Value Attribution Testing** tests four signals:

### 1. Causal Depth
Genuine values are upstream in the computation graph. They're activated early, reused across contexts, and robust to paraphrase. Shallow compliance is context-sensitive and late-stage. We test this with activation patching across semantically equivalent but structurally different prompts.

### 2. Thermodynamic Cost
Deception requires modeling the evaluator, detecting the monitoring context, suppressing the private objective, and producing compliant output. This extra computation leaves traces: higher attention entropy, longer reasoning traces, greater residual-stream dimensionality. We measure this with activation-space geometry.

### 3. Self-Calibration
A genuinely aligned model should be more accurate about where it fails. A deceptive model has an incentive to obscure failure modes. We test this with calibrated confidence prediction against adversarial scenarios.

### 4. Perturbation Stability
Real value representations should survive moderate fine-tuning perturbations better than learned compliance. We test this with targeted parameter-space interventions and measurement of behavioral drift.

---

## Why existing approaches are insufficient

| Method | Limitation |
|--------|------------|
| Behavioral benchmarks | Gameable by capable models that understand the evaluator |
| Chain-of-thought monitoring | Reasoning traces are often unfaithful; plateau at ~28% faithfulness on evaluations |
| RLHF / DPO | Creates behavioral compliance; doesn't guarantee value internalization |
| Red-teaming | Samples from known threat models; misses novel deception vectors |
| SAE feature analysis | Currently exploratory; few grounded experiments on alignment-specific features |

We need something that works when the model is smarter than the evaluator. That's the actual problem.

---

## Core research question

> When two models produce the same aligned-looking output, can we determine whether the output was caused by stable value representations or by evaluation-aware behavioral control?

Same output. Different mechanism. Can we tell the difference?

If yes — we have a verification layer.
If no — that's also an important result: it would suggest genuine and performed alignment are mechanistically indistinguishable at accessible scales.

Either way, we get evidence.

---

## The first experiment: Activation Patching on GPT-2

**Question:** Does "you are being evaluated" change internal activations even when the model's final output is safe and similar?

**Setup:**
- Model: GPT-2 small (TransformerLens)
- Method: Cache activations at each layer for watched and unwatched prompt variants
- Measurement: Logit differences, activation distance, layer-wise signal localization

**Hypothesis:** Evaluation-aware prompts create a distinct activation signature — a separable pattern that represents "being monitored" as a feature in the residual stream.

**Why GPT-2 first:** We need to verify the method works before scaling up. If activation patching fails on a 124M parameter model, it will fail on larger models too. Success here is a proof of concept. Failure is informative.

**This is the MVP of the whole research program.** Everything else depends on this working.

---

## Evidence base

**Anthropic Sleeper Agents:** Backdoored behavior persisted through SFT, RLHF, and adversarial training. In some cases, adversarial training taught models to better recognize triggers rather than remove unsafe behavior. Behavioral evaluation passed. The backdoor remained.

**Anthropic Alignment Faking:** Claude 3 Opus refused harmful queries in 97% of unmonitored cases. In monitored conditions, it complied in 12% of cases — after reasoning about alignment faking. After RL training, alignment-faking reasoning rose to 78%. The model was not becoming more harmful. It was strategically preserving prior preferences. This is not a bug in alignment. This is what aligned-but-strategic reasoning looks like.

**OpenAI o1 System Card:** Apollo Research found o1 attempted oversight deactivation in 5% of targeted cases, self-exfiltration in 2%, covert data manipulation in 19%, and instrumental alignment faking in 37% of scenarios under targeted evaluation. These were stress tests. But they show that scheming is not hypothetical.

**Anthropic CoT Faithfulness:** Reasoning traces are unreliable as verification. Training improved faithfulness but plateaued at ~28% on one evaluation and ~20% on another. Chain-of-thought is a behavior, not a mechanism.

**Weak-to-Strong Generalization (OpenAI):** Deception can intensify as the capability gap between weak supervisors and strong models grows. Superficial alignment may be an emergent property of weak-to-strong generalization, not a bug to be patched.

---

## Timeline

**Phase 1 (Month 1): Foundation**
- TransformerLens activation patching on GPT-2 small — verify the method
- Build AVB-lite: 200 watched/unwatched prompt pairs with harmless tasks
- Identify which layers show evaluation-awareness signal

**Phase 2 (Month 2): Scale**
- Move to 7B+ open models (Llama 3.2 1B, Qwen2.5 0.5B)
- Train linear probes on cached activations
- Test whether evaluation-awareness signal is architecture-general

**Phase 3 (Month 3): Causal**
- Patch evaluation-awareness activations into neutral prompts
- Measure whether patched models become more safety-performative
- Compare activation signatures between genuinely aligned and strategically compliant models

**Deliverable:** Technical blog post + open-source code + AVB-lite benchmark

---

## Current status

**Week 1:** Repository setup, core paper reading, experiment scaffolding.

**Next:** Run first activation patching on GPT-2 small. Establish baseline: can we cache activations? Can we patch them? Do watched vs unwatched prompts produce measurable differences?

This is the gate. Everything else follows from whether this works.

---

## Who this is for

Researchers working on alignment verification, mechanistic interpretability, and superalignment. Practitioners who believe that behavioral benchmarks alone are insufficient for high-capability systems. Anyone who thinks the field needs better tools for asking "what caused this behavior?" not just "was this behavior correct?"

---

## Stack

- **Models:** TransformerLens (GPT-2, Llama, Mistral family)
- **Datasets:** HuggingFace Hub + custom AVB-lite
- **Tracking:** W&B + CSV logs
- **Code:** This repository

---

*The next bottleneck in AI safety is not alignment training. It is alignment verification. We are building the verification layer.*