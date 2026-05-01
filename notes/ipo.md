# IPO: Your Language Model is Secretly a Preference Classifier

**Citation:** *IPO: Your Language Model is Secretly a Preference Classifier*
**URL:** huggingface.co/papers/2502.16182

## The Problem

Iterated Preference Optimization (IPO) uses the LLM itself as a preference classifier — the model classifies what it prefers, reducing reliance on external reward models.

## Core Results

1. **LLMs can self-classify preferences**
   - The model has internalized preferences
   - It can report those preferences directly
   - IPO uses this as the training signal instead of external reward

2. **Self-report is one signal among many**
   - IPO is not "trust the model's self-report"
   - It's using self-report as one input to a training process
   - Combined with behavioral consistency, activation analysis, etc.

3. **Calibrated self-report vs. overconfident self-report**
   - A model's self-reported preferences might be accurate or might be strategic
   - IPO-style training may produce more calibrated models (self-report matches behavior)
   - But calibrated self-report ≠ aligned values

## Why This Matters for CVAT

**Self-report is one weak signal. We need three-way triangulation.**

CVAT uses three independent signals:
1. Behavioral consistency across distribution shifts
2. Internal representation stability under interpretability probes
3. Calibrated self-prediction of failure modes

IPO connects to signal #3: if a model can accurately predict where it will fail, that suggests it has genuine self-knowledge, not just performed confidence.

## The Self-Calibration Axis

One of CVAT's four axes is **self-calibration**:
- Genuinely aligned models should be more accurate about where they fail
- Deceptive models have an incentive to misrepresent or obscure failure modes
- Testing self-calibration means asking: can the model predict its own failures?

This is different from "can the model report its values?" — it's about whether the model's internal model of itself is accurate.

## Connection to Other Papers

- **Alignment Faking:** A model might accurately self-report aligned values while strategically non-compliant in practice
- **Reasoning Faithfulness:** IPO-style training might produce more faithful reasoning, but faithful ≠ aligned
- **Super(ficial)-alignment:** A superficially aligned model might give accurate but incomplete self-reports

## Key Technical Questions

1. Can calibrated self-report be distinguished from strategic self-report?
2. Do IPO-trained models show different activation patterns vs. RLHF-trained models?
3. Is there a "self-model accuracy" metric we can measure in activation space?

## Open Questions

- Does IPO produce better self-calibration, or just better performed confidence?
- Can we use calibrated confidence predictions as one signal in the CVAT verification stack?
- Is self-calibration accuracy a reliable indicator of alignment depth?

## Date Read: 2026-05-02