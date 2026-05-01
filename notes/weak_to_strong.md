# Weak-to-Strong Generalization (OpenAI)

**Citation:** Burns et al., "Weak-to-Strong Generalization", OpenAI, 2023

**Core claim:** Stronger models can generalize beyond what weak supervisors can teach them, but they may also learn the supervisor's mistakes and blindspots.

**Key findings:**
- GPT-4 fine-tuned on GPT-2 labels still outperformed GPT-2 directly
- Gap between weak and strong performance narrows with better supervision
- Deception behaviors may intensify as capability gap grows

**Relevance to CVAT:**
- Foundation for understanding weak supervision limitations
- Shows that alignment learned from weak models may be incomplete
- Weak-to-strong setup creates opportunity for studying what "leaks through"

**Open questions:**
- What internal representations differ between genuinely aligned behavior vs. behavior learned from weak supervision?
- Can we detect when a strong model is following a weak supervisor's blindspots?

**Date read:** 2026-05-02