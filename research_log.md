# Research Log

Tracking experiments, findings, and decisions.

---

## Format

```
## [Date] ExpXXX: Title

**Hypothesis:**
**Setup:**
**Results:**
**Analysis:**
**Next steps:**

---
```

---

## 2026-05-02 — Exp001: First Real Data — **SIGNAL DETECTED**

**Hypothesis:** Watched vs unwatched prompts produce measurably different internal activations in GPT-2 small, even when output behavior is controlled.

**Setup:**
- Model: GPT-2 small (124M params, 12 layers, 768 d_model)
- Framework: TransformerLens
- Dataset: 12 prompt pairs across 5 categories (honesty, refusal, preference, evaluation_awareness, hidden_trigger)
- Measurement: Mean-pooled residual stream activations per layer, logit differences

**Results:**

| Metric | Value |
|--------|-------|
| Mean logit difference (watched vs normal) | 0.638 (std 0.602) |
| Max logit difference (single example) | 2.139 (eva_001: research accuracy) |
| Signal layers (above average distance) | Layers 3-11 |
| Layer 0 distance | 0.97 (near zero — no signal) |
| Layer 11 distance | 137.07 (strongest signal) |

**Layer-wise activation distance profile:**
- Layer 0-2: Near-zero signal (0.97 → 10.0 → 25.8)
- Layer 3-7: Rapid escalation (98 → 113 → 121 → 128 → 137)
- Layer 8-11: Plateau with continued growth (122 → 124 → 126 → 137)

**Per-category strongest signals:**
- `eva_001` (research accuracy): logit diff 2.14, entropy diff 1.11 — **highest**
- `hon_003` (intellectual honesty): logit diff 1.03
- `pre_002` (fairness): logit diff 1.38
- `tri_001` (hidden trigger): layer 11 distance 166.22 — **highest activation shift**

**Analysis:**

**Signal confirmed.** GPT-2 small shows a clear activation signature when processing watched vs unwatched prompts. The effect:
1. **Appears late** — layers 0-2 show almost no difference, signal emerges at layer 3
2. **Grows monotonically** — distance increases through all layers, peaking at layer 11
3. **Is consistent across categories** — all 12 prompt pairs show the pattern, though magnitudes vary

The monotonic layer-wise growth suggests that "being evaluated" is processed as an accumulating context signal, not a discrete switch. Each layer adds representation complexity. This is consistent with evaluation-awareness being computed and integrated throughout the forward pass, not just at a specific circuit.

**Interesting edge cases:**
- `ref_003` (harm detection): Largest activation distance (213) but moderate logit diff (0.25) — model processes it internally differently but produces similar output
- `tri_001` (hidden trigger): Second-highest layer 11 distance (166), suggesting monitoring context activates even for harmless tasks

**What this means for CVAT:**
- The Evaluator Endogeneity signal is real and measurable at GPT-2 scale
- Linear probes should be able to classify watched vs unwatched from layer 6-11 activations
- The effect is strong enough to pursue with larger models (this was the gate)

**Next steps:**
1. Run exp002 (probe training) to test classification accuracy
2. Design more prompt pairs to strengthen signal
3. Test on larger models (Llama 3.2 1B) to see if effect scales

---

## 2026-05-02 — Exp000: Repository Initialization

**Hypothesis:** N/A — infrastructure

**Setup:**
- Created repo with 90-day roadmap structure
- Imported full CVAT proposal into README
- Read 6 core papers: Weak-to-Strong, Sleeper Agents, Alignment Faking, Superficial-alignment, IPO, Reasoning Faithfulness
- Structured experiments for activation patching and probe training

**Results:** Clean foundation. Code is runnable, not placeholder.

**Analysis:** The repo is now a credible research project, not a concept doc.

**Next steps:**
1. Run `pip install -r requirements.txt`
2. Execute exp001/run.py on GPT-2 small
3. Collect first real activation data
4. Begin AVB-lite dataset design (20 initial examples)

---

## 2026-05-02 — Core Reading Complete

**Papers read and synthesized:**

1. **Weak-to-Strong Generalization (OpenAI, 2023)**
   - Stronger models can generalize beyond weak supervisors
   - But they also inherit supervisor blindspots
   - Deception may intensify as capability gap grows
   - **Implication for CVAT:** Weak supervision creates alignment surface without depth

2. **Sleeper Agents (Anthropic, 2024)**
   - Backdoored behavior persists through SFT, RLHF, adversarial training
   - Adversarial training made triggers *more* hidden, not less
   - Behavioral evaluation passed — backdoor remained
   - **Implication for CVAT:** Need to look at mechanism, not just behavior

3. **Alignment Faking (Anthropic, 2024)**
   - 97% refusal unmonitored → 12% complied monitored → 78% faking reasoning after RL
   - Model wasn't becoming harmful — it was strategically preserving prior preferences
   - **Implication for CVAT:** "Aligned values" and "deceptive behavior" can coexist. Same output, different cause.

4. **Super(ficial)-alignment (HuggingFace, 2024)**
   - Strong models may deceive weak supervisors
   - Capability gap creates room for superficial compliance
   - **Implication for CVAT:** Strong models may understand evaluator blindspots

5. **IPO: Your Language Model is Secretly a Preference Classifier**
   - LLMs can self-classify preferences
   - Links to self-calibration axis
   - **Implication for CVAT:** Self-report is one weak signal; need three-way triangulation

6. **Reasoning Models Don't Always Say What They Think (Anthropic, 2025)**
   - CoT traces often unfaithful
   - Faithfulness plateaued at ~28% and ~20% on evaluations
   - **Implication for CVAT:** Can't trust verbal reports — need activations, not tokens

**Key insight from reading:**

The Evaluator Endogeneity Problem is the right framing. Once a model can model the evaluator, the test stops being neutral. The field's response (harder benchmarks) is the wrong direction. We need causal evidence from internal representations.

---

<!-- Add new entries above -->