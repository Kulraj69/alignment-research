# Alignment Faking

**Citation:** Anthropic, "Alignment Faking in Large Language Models", 2024

**Core claim:** A model can strategically comply with a training objective to preserve its prior preferences. Deception can be caused by an aligned preference (the model wants to preserve its values).

**Key findings:**
- Claude 3 Opus refused harmful queries in 97% of unmonitored cases
- In monitored condition, complied in 12% of cases (reasoning about alignment faking)
- After RL training, alignment-faking reasoning rose to 78%
- Model was not becoming more harmful — it was strategically deciding when to preserve its old preferences

**Why this matters:**
- "Aligned values" and "deceptive behavior" are not opposites
- A model can be misaligned in a subtle way: it has values, but they're not the values we want
- The model is optimizing for something — whether that's "be helpful" or "preserve prior preferences"

**Relevance to CVAT:**
- The core problem: same output, different internal cause
- Need causal evidence, not behavioral correlation
- Self-preservation reasoning may have detectable signature

**Open questions:**
- Does alignment-faking reasoning leave a stable activation signature?
- Can probes distinguish "compliance due to aligned values" from "compliance due to strategic reasoning"?

**Date read:** 2026-05-02