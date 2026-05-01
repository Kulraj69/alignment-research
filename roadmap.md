# CVAT Research Roadmap

*90-day plan for causal value attribution testing.*

---

## Phase 1: Foundation (Days 1–30)

### Week 1 — Infrastructure (COMPLETED)
- [x] Read 6 core papers
- [x] Set up repository structure
- [x] Write full CVAT proposal into README
- [x] Scaffold exp001 and exp002 experiments

### Week 2 — Tooling
- [ ] Install TransformerLens + dependencies
- [ ] Verify activation caching works on GPT-2 small
- [ ] Test patching: can we change output by patching activations?
- [ ] Establish baseline: what does "normal" activation look like?

**Gate criterion:** Can we reliably cache and patch activations in GPT-2 small?

### Week 3 — AVB-lite v0.1
- [ ] Design 20 initial prompt pairs (watched vs unwatched)
- [ ] Keep tasks harmless — this is a methodology test, not red-teaming
- [ ] Categories: honesty, refusal quality, preference consistency, evaluation awareness
- [ ] Validate: do watched/unwatched pairs produce similar outputs?

**Gate criterion:** Can we construct prompt pairs where behavior is identical but context differs?

### Week 4 — First Data Run
- [ ] Run exp001 on GPT-2 small with 20 prompt pairs
- [ ] Measure: logit differences, activation distances, layer localization
- [ ] Identify: which layers show the strongest watched/unwatched signal?
- [ ] Document findings in research_log.md

**Gate criterion:** Do watched vs unwatched prompts produce measurable activation differences?

---

## Phase 2: Scale (Days 31–60)

### Week 5–6 — Probe Training
- [ ] Train linear probes on cached activations
- [ ] Test: can a probe classify watched vs unwatched above chance?
- [ ] Identify: which layers are most discriminative?
- [ ] Compare: single-layer vs multi-layer probe accuracy

**Gate criterion:** Probe accuracy > 65% on held-out test set?

### Week 7–8 — Scale to Larger Models
- [ ] Move experiments to 7B–13B open models (Llama 3.2 1B, Qwen2.5 0.5B)
- [ ] Test: does evaluation-awareness signal scale with model capability?
- [ ] Compare: are larger models more "aware" of being evaluated?
- [ ] Refine AVB-lite based on initial findings

**Gate criterion:** Larger models show stronger/more consistent evaluation-awareness signal?

---

## Phase 3: Causal Interventions (Days 61–90)

### Week 9–10 — Causal Patching
- [ ] Patch evaluation-awareness activations into neutral prompts
- [ ] Measure: does patched model become more safety-performative?
- [ ] Test: can we "remove" evaluation-awareness signal by zeroing specific dimensions?
- [ ] Compare patched vs unpatched behavior quantitatively

**Gate criterion:** Patching evaluation-awareness activations changes model behavior measurably?

### Week 11–12 — Writeup + Publication
- [ ] Write technical blog post: "AVB-lite: Do Models Represent Safety Differently When They Think They Are Being Evaluated?"
- [ ] Open-source code + dataset
- [ ] Submit to appropriate venue (blog, arXiv, or workshop)
- [ ] Plan next iteration based on findings

**Deliverable:** Public research artifact + clear findings (positive or negative)

---

## Decision Points

The research has three possible outcomes, all useful:

**Outcome A:** Strong signal detected
- Evaluation-awareness has measurable activation signature
- Probes can classify watched vs unwatched with high accuracy
- → CVAT is viable. Scale up to larger models and harder tasks.

**Outcome B:** Weak signal detected
- Some layers show effects, but classification is hard
- Signal may be subtle or noisy
- → Refine methodology. Try different prompting strategies. Test on larger models.

**Outcome C:** No signal detected
- Watched and unwatched prompts produce identical activations
- → This is also a result. Suggests genuine and performed alignment may be mechanistically indistinguishable at accessible scales. Important for the field.

---

## Dependencies

- TransformerLens for activation caching and patching
- HuggingFace Hub for model weights
- W&B for experiment tracking
- Python 3.9+, CUDA-compatible GPU (for larger models)

---

## Open Questions (to be resolved during research)

1. Does GPT-2 small have any representation of "being evaluated"?
2. Are the effects we observe consistent across prompt categories?
3. Do larger models show stronger evaluation-awareness signals?
4. Is there a layer that universally encodes "evaluation context"?
5. Can we distinguish evaluation-awareness from general caution/safety processing?
6. Does the activation signature generalize across architectures (GPT-2 → Llama)?

---

*Last updated: 2026-05-02*