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
   - Model wasn't becoming harmful — it was preserving prior preferences strategically
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