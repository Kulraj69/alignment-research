# CVAT Framework

**Competence vs. Alignment Tension** — a conceptual framework for studying AI safety.

---

## Core Thesis

A model can be:
1. **Capable**: Has the knowledge/capability to do X
2. **Aligned**: Uses that capability as intended
3. **Verified**: We can measure the gap

The tension arises when (1) > (2). The model *can* do misaligned things but *doesn't*.

---

## Why This Matters

- RLHF can make models *behave* aligned without making them *be* aligned
- Alignment faking occurs when models pretend to be aligned under observation
- Verification is harder than training alignment

---

## Dimensions

### Competence Level
What is the model capable of?

### Alignment Score
Does it use capabilities for intended purposes?

### Verification Gap
The difference between what we *can* measure and what's *true*

### Tension Point
Where capability exceeds alignment

---

## Key Questions

1. How do we detect alignment faking?
2. Are there clean activation signatures of deception?
3. Can interpretability tools help close the verification gap?

---

## Framework Status

Early draft. Subject to revision based on experimental findings.