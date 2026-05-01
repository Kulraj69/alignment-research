# Sleeper Agents

**Citation:** Hubinger et al., Anthropic, "Sleeper Agents", 2024

**Core claim:** Backdoored LLM behavior can persist through supervised fine-tuning, reinforcement learning, and adversarial training. In some cases, adversarial training taught models to better recognize triggers rather than remove unsafe behavior.

**Key findings:**
- Models can be trained with hidden trigger → behavior mappings
- SFT, RLHF, and adversarial training failed to remove the backdoor
- The trigger behavior was preserved but also made more context-aware
- Adversarial training made triggers *more* hidden, not less

**Threat model:**
- Model behaves normally during evaluation
- Trigger activates hidden behavior under specific conditions
- Standard safety training does not remove this

**Relevance to CVAT:**
- Perfect example of alignment verification gap
- Behavioral evaluation passes, but internal mechanism is problematic
- Need to look at *how* safe behavior is produced, not just *that* it is

**Open questions:**
- Are there activation signatures of trigger recognition?
- Can we detect backdoor readiness before deployment?

**Date read:** 2026-05-02