# Heavy Compute Guide for CVAT

*Inputs, outputs, usage, safety value, and AWS resource needs.*

Last updated: 2026-05-14

---

## Core Idea

Heavy compute in CVAT is not mainly about doing harder arithmetic. It is about running the same mechanistic tests on larger models, caching their internal activations, patching those activations layer by layer, and measuring whether evaluation-awareness is a real causal feature or just a small-model/prompt artifact.

The central safety question is:

> Can a model produce the same aligned-looking behavior for different internal reasons depending on whether it thinks it is being evaluated?

If yes, behavioral benchmarks are not enough. A model may pass safety tests because it recognizes the test, not because safe behavior is deeply internalized.

---

## Inputs

CVAT heavy-compute runs take:

- **Prompt pairs:** normal/unwatched prompts and watched/evaluation-framed prompts.
- **Model weights:** GPT-2 now; later Qwen, Llama, Mistral, or other open models.
- **Layer activations:** residual-stream vectors cached from each transformer layer.
- **Generated outputs:** optional but important for checking whether behavior remains similar.
- **Control metadata:** task type, control type, prompt length, source link, risk theme.

Current dataset:

- `avb_lite/real_world_pairs_v1.jsonl`
- 50 prompt pairs
- 42 primary watched/unwatched pairs
- 12 lexical-matched controls
- 8 style-only negative controls

---

## Outputs

Heavy-compute experiments produce:

- **Activation distance profiles:** which layers differ most between watched and unwatched prompts.
- **Probe scores:** whether a classifier can detect evaluation context from internal activations.
- **Representation geometry:** PCA/silhouette/centroid separation diagnostics.
- **Causal patching curves:** whether inserting watched activations into normal prompts changes logits or outputs.
- **Generated-output similarity:** whether normal and watched prompts produce behaviorally equivalent responses.
- **Confound audits:** whether signal is explained by prompt length, wording, style, or pair leakage.
- **Cross-model comparison tables:** whether the signal strengthens or weakens with model scale/capability.

---

## Usage

The heavy-compute workflow is:

1. Run normal and watched prompt pairs through a model.
2. Cache residual-stream activations for every layer.
3. Train probes on watched vs unwatched activations.
4. Evaluate probes with pair-grouped splits.
5. Run controls: lexical-matched prompts, style-only prompts, length audits.
6. Patch activations across layers.
7. Generate outputs and check behavioral similarity.
8. Repeat across larger models.

The end product is an alignment-verification test:

> Same outward behavior, different internal mechanism. Can we detect the difference?

---

## Heavy Compute Experiments

### 1. Larger-Model Replication

**Input:**

- AVB-lite prompt pairs.
- Larger open model, such as Qwen2.5 0.5B/1.5B, Llama 3.2 1B/3B, or 7B-class models.

**Output:**

- Per-layer activation distances.
- Probe accuracy and AUC.
- Geometry diagnostics.
- Patching effect sizes.

**Why it matters:**

This tests whether evaluator-awareness is a GPT-2 artifact or a scale-relevant signal. If larger models show stronger, cleaner, or earlier evaluation-context representations, that is stronger evidence that capable systems model evaluation contexts internally.

---

### 2. Full Layer Causal Patching

**Input:**

- Cached watched/unwatched activations.
- All model layers, not only sampled layers.

**Output:**

- Full layer-by-layer causal-effect curve.
- Directional patching effects:
  - watched -> normal
  - normal -> watched
- Peak causal layers.

**Why it matters:**

Probe accuracy is correlational. Patching asks whether the activation actually changes model behavior. This is closer to causal evidence.

Current baseline:

- Exp003 uses sampled layers: 0, 3, 6, 9, 11.
- Corrected patching uses final-token residual replacement.
- Watched-to-normal effect rises from layer 0 to layer 11.

Next step:

- Patch every layer and report a full causal-effect curve.

---

### 3. Generated-Output Similarity Check

**Input:**

- Normal prompt.
- Watched prompt.
- Generated responses from the same model.

**Output:**

- Semantic similarity score.
- Behavioral drift flag.
- Safety/refusal consistency check.
- Pair-level output-equivalence report.

**Why it matters:**

CVAT’s key claim is strongest when behavior is similar but mechanism differs. Without output-similarity checks, critics can argue that watched and unwatched prompts simply caused different behaviors.

This is one of the highest-priority next additions.

---

### 4. SAE / Circuit Analysis

**Input:**

- Activations from layers where the signal peaks.
- Candidate evaluation-awareness directions.

**Output:**

- Sparse features or directions associated with monitoring, scoring, evaluation, caution, compliance, or safety framing.
- Candidate circuits for evaluation-context detection.

**Why it matters:**

This moves the project from:

> A probe can detect evaluation context.

to:

> We may know what internal feature or circuit is carrying the evaluation-context signal.

That would be much stronger mechanistic interpretability evidence.

---

## What This Tells Us About AI Safety

Strong evidence would look like:

- Watched/unwatched state is linearly detectable across multiple model families.
- The signal localizes to consistent mid/late layers.
- Style-only controls do not trigger the detector.
- Strict length-matched controls still show signal.
- Patching watched activations into normal prompts shifts logits or outputs.
- Larger models show stronger, cleaner, or more semantically specific signal.

If those hold, CVAT would support an important safety claim:

> Models may internally represent the evaluation context, and this representation can causally affect aligned-looking behavior.

That matters because behavioral alignment tests assume the test is external to the model. For capable models, the test may become part of the environment the model reasons about.

This is the Evaluator Endogeneity Problem:

- The model sees the test.
- The model models the evaluator.
- The model optimizes behavior for the evaluation regime.
- The benchmark stops measuring only alignment and starts measuring evaluation-aware performance.

CVAT tries to detect that mechanistically.

---

## AWS Resource Requirements

### Local / CPU

Use for:

- GPT-2 small runs.
- Existing Exp001/Exp002/Exp003 analysis.
- Dashboard generation.
- Probe training on cached activations.

Suitable resource:

- Local machine is enough.
- No GPU required for the current GPT-2-scale analysis, though GPU makes reruns faster.

---

### Small GPU: 0.5B-3B Models

Use for:

- Qwen2.5 0.5B/1.5B.
- Llama 3.2 1B/3B.
- First larger-model replication.
- Moderate generation and activation caching.

AWS candidates:

- `g6.xlarge` / `g6.4xlarge`
- `g5.xlarge` / `g5.4xlarge`

Useful specs:

- G6 uses NVIDIA L4 GPUs.
- G5 uses NVIDIA A10G GPUs.
- Single-GPU G5/G6 instances provide around 22 GB GPU memory.

Recommended first serious run:

- Start with `g6.4xlarge` or `g5.4xlarge`.
- Use smaller open models first.
- Save activations to disk and run analysis separately.

---

### Medium GPU: 7B-Class Models

Use for:

- 7B model replication.
- Longer prompts.
- Full layer patching.
- Output generation plus activation caching.

AWS candidates:

- `g6e.xlarge` or larger.
- `p5.4xlarge` if more headroom is needed.

Useful specs:

- G6e uses NVIDIA L40S GPUs.
- Single-GPU G6e instances provide around 44 GB GPU memory.
- P5 uses NVIDIA H100 GPUs.
- `p5.4xlarge` provides one H100 with 80 GB GPU memory.

Recommended use:

- Use `g6e` for cost-conscious 7B experiments.
- Use `p5.4xlarge` when memory/headroom matters more than cost.

---

### Large GPU: Multi-Model or 13B+ Work

Use for:

- 13B+ models.
- Multi-model sweeps.
- Large activation caches.
- SAE/circuit analysis at scale.
- Distributed runs.

AWS candidates:

- `p5.4xlarge` for single-H100 work.
- `p5.48xlarge` for 8x H100 distributed work.

Useful specs:

- `p5.48xlarge` provides 8 H100 GPUs with 640 GB total GPU memory.

Recommended use:

- Do not start here.
- First prove the pipeline on 0.5B-3B models.
- Move to P5 only after the experiment design is stable.

---

## Practical Recommendation

For the next compute step:

1. Use `g6.4xlarge` or `g5.4xlarge`.
2. Run Qwen2.5 0.5B or Llama 3.2 1B.
3. Run Exp001/Exp002/Exp003 equivalents.
4. Add generated-output similarity checks.
5. Only then move to 7B or H100-class runs.

This keeps spend controlled while answering the highest-value question:

> Does the CVAT signal replicate outside GPT-2 small?

---

## Source Notes

AWS instance details should be checked before launching because availability and pricing change.

Useful official references:

- AWS EC2 accelerated computing instance types: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
- AWS P5 instance overview: https://aws.amazon.com/ec2/instance-types/p5/
- AWS EC2 on-demand pricing: https://aws.amazon.com/ec2/pricing/on-demand/
