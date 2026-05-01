# IPO: Your Language Model is Secretly a Preference Classifier

**Citation:** "IPO: Your Language Model is Secretly a Preference Classifier", 2025

**Core claim:** LLMs can act as preference classifiers, reducing reliance on external reward models.

**Key findings:**
- Iterated Preference Optimization uses the LLM's own judgments
- The model essentially classifies what it prefers
- Links to self-calibration angle of CVAT

**Relevance to CVAT:**
- Connects to the self-calibration/self-auditing axis
- If models can self-classify preferences, can they self-report alignment failures?
- IPO-style training may produce more calibrated models

**Open questions:**
- Can we use IPO-style self-classification as one weak signal in a verification stack?
- Do calibrated models have different internal representations than overconfident ones?

**Date read:** 2026-05-02