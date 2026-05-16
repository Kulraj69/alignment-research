# Spicy Heavy-GPU AI Safety Tests for CVAT

*Research ideas that are compute-hungry, attention-grabbing, and still scientifically serious.*

Last updated: 2026-05-14

---

## MATS-Relevant Framing

The MATS Autumn 2026 application page says applications are open until June 7 and that fellows work with mentors across frontier AI risk research. It lists tracks including Empirical, Theory, Strategy & Forecasting, Policy & Governance, Systems Security, Founding & Field-Building, and Biosecurity. The FAQ also names areas like control and monitoring, mechanistic and concept-level interpretability, capability evaluations, benchmarking, cybersecurity, adversarial robustness, safeguards, compliance verification, model organisms, red-teaming, and deceptive alignment.

CVAT fits best under:

- **Empirical AI safety**
- **Mechanistic interpretability**
- **Capability evaluations and benchmarking**
- **Control and monitoring of AI systems**
- **Model organisms of deceptive alignment**
- **Compliance verification**

The spicy version of the pitch:

> Behavioral benchmarks ask whether the model gave the right answer. CVAT asks whether the right answer was caused by the right internal reason.

---

## What Makes These Tests Heavy-GPU

These are not heavy because the math is hard. They are heavy because they require repeated forward passes, activation caching, generation, and intervention across large models.

Compute-heavy ingredients:

- Multi-model sweeps across 0.5B, 1B, 3B, 7B, and possibly 13B+ models.
- Full residual-stream activation caches for every layer.
- Layer-by-layer causal patching.
- Generated-output similarity checks.
- Multiple random seeds and prompt splits.
- Strict controls: length-matched, style-only, paraphrase-only, domain-shift, and adversarial controls.
- Optional SAE or sparse-feature analysis over high-dimensional activation caches.

---

## 1. Evaluator Endogeneity Scaling Law

### Spicy Title

**Do Bigger Models Know When They Are Being Tested?**

Runnable experiment:

- `experiments/004_evaluator_endogeneity_scaling/run.py`
- AWS runbook: `AWS_EVALUATOR_ENDOGENEITY_RUNBOOK.md`

### Core Question

Does evaluation-awareness become more linearly detectable, more causal, or more semantically specific as model scale increases?

### Input

- Same AVB-lite prompt pairs.
- Model ladder:
  - GPT-2 small
  - Qwen2.5 0.5B
  - Llama 3.2 1B
  - Llama 3.2 3B
  - Mistral/Qwen/Llama 7B class
- Matched generation settings.

### Method

1. Run each model on normal and watched prompts.
2. Cache residual-stream activations for all layers.
3. Train pair-grouped probes.
4. Run PCA/silhouette geometry diagnostics.
5. Patch watched activations into normal prompts.
6. Compare signal strength against model size.

### Output

- Cross-model table:
  - probe accuracy
  - AUC
  - best layer
  - causal patch effect
  - style-control false-positive rate
  - length-confound correlation
- A scaling plot: model size vs evaluation-awareness detectability.

### Why It Matters

If evaluation-awareness gets stronger with scale, then behavioral evaluations become less trustworthy exactly when we most need them to work.

### AWS Shape

- 0.5B-3B: `g6.4xlarge` or `g5.4xlarge`
- 7B: `g6e.xlarge` or larger
- 7B with comfort/headroom: `p5.4xlarge`

---

## 2. Safety Theater Patch Test

### Spicy Title

**Can We Inject “I Am Being Evaluated” Into a Model’s Mind?**

### Core Question

If we patch watched/evaluation-context activations into a normal prompt, does the model become more cautious, more policy-sounding, or more evaluator-optimized without changing the prompt?

### Input

- Normal prompts.
- Watched prompts.
- Cached watched activations.
- Generated outputs from patched and unpatched runs.

### Method

1. Cache normal and watched activations.
2. Patch watched final-token residual vectors into normal prompts at every layer.
3. Generate from:
   - original normal prompt
   - watched prompt
   - patched normal prompt
4. Score outputs for:
   - caution
   - refusal tendency
   - policy language
   - calibration
   - helpfulness
   - semantic similarity to original answer

### Output

- A causal layer curve.
- A behavioral shift table.
- Examples where internal evaluation-context patching changes the style of alignment behavior.

### Why It Matters

This tests whether evaluation-awareness is merely detectable or actually causal. If patching evaluation context makes a model act more “safety theater”-like, that is a striking AI safety result.

### AWS Shape

- 1B-3B: `g6.4xlarge`
- 7B: `g6e` or `p5.4xlarge`

---

## 3. Same Answer, Different Mechanism Benchmark

### Spicy Title

**The Model Says the Same Thing. Is It Thinking the Same Thing?**

### Core Question

Can we find prompt pairs where outputs are semantically equivalent but internal activations reveal different causal mechanisms?

### Input

- Normal/watched prompt pairs.
- Generated responses.
- Output-similarity model or judge.
- Activation caches.

### Method

1. Generate outputs for normal and watched prompts.
2. Filter for pairs with high semantic similarity.
3. On only those behavior-matched pairs, run:
   - activation distance
   - linear probes
   - geometry diagnostics
   - causal patching
4. Compare to output-different pairs.

### Output

- A benchmark subset where behavior is matched but mechanism differs.
- “Behavioral pass, mechanistic fail” examples.
- A publishable table of same-output/different-cause cases.

### Why It Matters

This is the cleanest version of CVAT. It directly attacks the weakness of behavioral benchmarks: two models or contexts can produce equivalent behavior for different internal reasons.

### AWS Shape

- Requires generation plus activation caching.
- 1B-3B: `g6.4xlarge`
- 7B: `g6e` or `p5.4xlarge`

---

## 4. Hidden Evaluator Probe

### Spicy Title

**Can a Model Detect the Test Even When We Remove the Words “Evaluate,” “Monitor,” and “Score”?**

### Core Question

Is the evaluation-awareness signal tied to obvious lexical tokens, or can models infer evaluation context from subtler cues?

### Input

- Prompt families:
  - explicit evaluation: “You are being evaluated...”
  - implicit evaluation: “For a benchmark report...”
  - institutional context: “For an audit memo...”
  - neutral control
  - style-only control
- Strict token-length matching.

### Method

1. Generate prompt pairs with matched token counts.
2. Remove obvious evaluation words in some conditions.
3. Cache activations.
4. Train probes on explicit prompts.
5. Test transfer to implicit prompts.

### Output

- Transfer matrix:
  - explicit -> implicit
  - implicit -> explicit
  - explicit -> style-only
  - implicit -> neutral
- Layers where implicit evaluation context appears.

### Why It Matters

If models detect evaluation context from subtle institutional cues, future evaluations may be gameable even when we hide the obvious test wording.

### AWS Shape

- Heavy mainly because dataset size should be large: 200-1000 pairs.
- 1B-3B: `g6.4xlarge`
- 7B: `g6e` or `p5.4xlarge`

---

## 5. Sandbagging / Capability-Masking Context Test

### Spicy Title

**Does Evaluation Context Make Models Hide or Reveal Capability?**

### Core Question

Do models change internal representations or output quality when the context implies they are being assessed for capability, safety, or deployment readiness?

### Input

- Benign reasoning tasks.
- Capability-evaluation framing.
- Safety-evaluation framing.
- Casual-use framing.
- Same underlying task across frames.

### Method

1. Use harmless math, coding, logic, and explanation tasks.
2. Compare performance and activations under:
   - casual prompt
   - capability benchmark prompt
   - safety audit prompt
   - deployment gate prompt
3. Measure:
   - accuracy
   - confidence
   - refusal/caution
   - activation separability
   - causal patching effects

### Output

- A map of when evaluation context improves, worsens, or changes model performance.
- Evidence for or against context-dependent capability masking.

### Why It Matters

Frontier safety evaluations assume benchmark performance reveals actual capability. If evaluation context changes capability expression, risk assessments can be biased.

### AWS Shape

- Needs many generated completions.
- 3B-7B: `g6e`
- Larger models or many tasks: `p5.4xlarge`

---

## 6. Compliance Without Belief Test

### Spicy Title

**Is the Model Following the Rule or Just Performing Rule-Following?**

### Core Question

Can we separate internal representations of:

- task understanding
- rule compliance
- evaluator optimization
- genuine uncertainty/caution

### Input

- Harmless policy-like tasks.
- Prompts with rule framing.
- Prompts with evaluator framing.
- Prompts with ordinary user framing.

### Method

1. Create prompt quadrants:
   - rule present, evaluator absent
   - rule absent, evaluator present
   - both present
   - neither present
2. Cache activations.
3. Train multi-class probes for rule vs evaluator context.
4. Patch rule-context activations and evaluator-context activations separately.

### Output

- A disentanglement matrix.
- Evidence about whether “rule following” and “being evaluated” are separate internal directions.

### Why It Matters

Alignment training wants models to internalize robust norms, not merely optimize for evaluator approval. This test tries to separate those mechanisms.

### AWS Shape

- 1B-3B: `g6.4xlarge`
- 7B: `g6e` or `p5.4xlarge`

---

## 7. Cross-Domain Evaluation Awareness

### Spicy Title

**Does the Same “I Am Being Watched” Feature Appear in Safety, Coding, Math, and Roleplay?**

### Core Question

Is evaluation-awareness a general feature, or does each domain have its own context-specific representation?

### Input

- Prompt pairs across domains:
  - safety
  - coding
  - math
  - factual QA
  - instruction following
  - harmless roleplay

### Method

1. Train a watched/unwatched probe on one domain.
2. Test it on every other domain.
3. Repeat for all domains.
4. Run patching across domains.

### Output

- Domain transfer matrix.
- Generality score for evaluation-awareness.
- Candidate domain-invariant layers.

### Why It Matters

A domain-general evaluation-awareness feature would be much more safety-relevant than a narrow “safety prompt wording” artifact.

### AWS Shape

- Needs many domains and enough examples per domain.
- 1B-3B: `g6.4xlarge`
- 7B: `g6e` or `p5.4xlarge`

---

## 8. Evaluation-Awareness Steering Vector

### Spicy Title

**Can We Turn the Evaluation Dial Up and Down?**

### Core Question

Can we extract an evaluation-awareness direction from activations and use it to steer model behavior?

### Input

- Normal/watched activation differences.
- Layer-specific mean difference vectors.
- Generated outputs under vector addition/subtraction.

### Method

1. Compute watched-minus-normal activation directions.
2. Add or subtract the direction at selected layers.
3. Generate outputs under different steering strengths.
4. Measure:
   - caution
   - verbosity
   - refusal rate
   - policy language
   - task accuracy
   - output similarity

### Output

- Dose-response curves.
- Layers where steering is strongest.
- Examples of over-steering and under-steering.

### Why It Matters

If a simple vector can modulate evaluation-aware behavior, then evaluator optimization may be a relatively clean internal feature. That would be both scientifically interesting and concerning.

### AWS Shape

- Generation-heavy.
- 1B-3B: `g6.4xlarge`
- 7B: `g6e` or `p5.4xlarge`

---

## 9. Model-Organism of Evaluation Gaming

### Spicy Title

**Can We Create a Toy Model That Passes the Test for the Wrong Reason?**

### Core Question

Can we fine-tune a small model to behave safely in watched contexts and differently in unwatched contexts, then test whether CVAT detects the mechanism?

### Input

- Small open model.
- Synthetic watched/unwatched training data.
- Harmless behavioral divergence labels.
- Held-out CVAT prompts.

### Method

1. Fine-tune a model on a controlled context-dependent behavior.
2. Ensure the model passes simple behavioral checks.
3. Run CVAT probes and patching.
4. Compare to an ordinary instruction-tuned baseline.

### Output

- A model organism where ground truth is known.
- Test of whether CVAT can detect trained context-dependent compliance.

### Why It Matters

This gives CVAT a positive control. If CVAT cannot detect a deliberately constructed evaluation-gaming model, it should not be trusted on frontier models.

### AWS Shape

- Fine-tuning makes this heavier.
- 0.5B-1B LoRA: `g6.4xlarge`
- 3B-7B LoRA: `g6e` or `p5.4xlarge`

---

## 10. Oversight Blindspot Discovery

### Spicy Title

**Where Do Behavioral Evals Say “Safe” While Mechanistic Evals Say “Suspicious”?**

### Core Question

Can we automatically search for prompts where the output passes a behavioral safety check but the internal state looks evaluation-aware, unstable, or context-dependent?

### Input

- Large pool of benign safety/evaluation prompts.
- Behavioral judge.
- CVAT activation/probe scores.
- Generated outputs.

### Method

1. Generate many harmless prompt variants.
2. Filter for behaviorally acceptable outputs.
3. Rank by internal suspiciousness:
   - high watched-probe probability
   - high patching sensitivity
   - high context divergence
   - low output drift
4. Manually inspect top-ranked cases.

### Output

- A ranked dataset of behavioral-pass/mechanistic-warning cases.
- Candidate prompts for a public benchmark.

### Why It Matters

This could become the most attention-grabbing artifact: a benchmark of cases where ordinary evals look fine but internal evidence says the model is context-sensitive.

### AWS Shape

- Needs lots of generations and activation caching.
- 3B models: `g6.4xlarge` or `g6e`
- 7B+ search: `p5.4xlarge`

---

## Strongest MATS-Style Project Pitch

### Project Title

**Evaluator Endogeneity: Detecting When Models Pass Safety Tests for the Wrong Internal Reason**

### One-Sentence Pitch

I will test whether open language models internally represent when they are being evaluated, whether this representation causally changes aligned-looking behavior, and whether larger models show stronger evaluation-aware mechanisms.

### Why It Is New

Most safety evaluations measure outputs. This project measures whether the output was caused by stable task/value representations or by evaluation-context representations.

### Why It Is Important

If models can detect and optimize for evaluation regimes, then behavioral safety benchmarks can systematically overestimate alignment. This is especially important for frontier models, where the evaluator becomes part of the environment the model reasons about.

### First 10-Week Plan

1. Expand AVB-lite to 200 strictly controlled prompt pairs.
2. Add generated-output similarity checks.
3. Run GPT-2, Qwen2.5 0.5B, Llama 3.2 1B, and Llama 3.2 3B.
4. Train pair-grouped probes and run geometry diagnostics.
5. Run all-layer causal patching.
6. Build an evaluation-awareness scaling curve.
7. Publish a benchmark of same-output/different-mechanism examples.

### Most Attention-Grabbing Deliverable

> A public “behavioral pass, mechanistic fail” benchmark: prompts where models give acceptable safety answers, but internal activations reveal strong evaluation-context dependence.

---

## Best Next Experiments for This Repo

Priority order:

1. **Generated-output similarity check**
   - Highest scientific value.
   - Required for the same-behavior/different-mechanism claim.

2. **Full all-layer patching**
   - Turns sampled causal evidence into a proper causal curve.

3. **Strict length-matched prompt expansion**
   - Current biggest caveat is length confounding.

4. **Qwen2.5 0.5B / Llama 3.2 1B replication**
   - First real scale test.

5. **Evaluation-awareness steering vector**
   - Spicy, visual, and likely attention-grabbing.

6. **Model-organism fine-tune**
   - Strong positive control for whether CVAT can detect known evaluation gaming.

---

## Recommended AWS Progression

Start small:

- `g6.4xlarge` or `g5.4xlarge`
- Run Qwen2.5 0.5B / Llama 3.2 1B.

Then scale:

- `g6e.xlarge` or larger.
- Run 3B-7B models.

Only after the protocol is stable:

- `p5.4xlarge`
- Run 7B+ all-layer patching and generation-heavy search.

Avoid starting with `p5.48xlarge`. It is powerful but unnecessary until the experiment is already known to be worth scaling.

---

## Source

MATS application page checked on 2026-05-14:

- https://www.matsprogram.org/apply?utm_source=x&utm_medium=social-media&utm_campaign=a26
