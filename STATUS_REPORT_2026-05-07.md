# CVAT Status Report (May 7, 2026)

## What Was Implemented

1. `exp001` dataset grounding upgrade
- `exp001` now loads prompt pairs from `avb_lite/real_world_pairs_v1.jsonl` when present.
- Falls back to built-in defaults if the file is missing.
- Persists source metadata (`source_url`, `source_date`, `risk_theme`) in results.

2. `exp002` evaluation-rigor upgrade
- Added split comparison:
  - `sample_stratified` (baseline)
  - `pair_grouped` (primary, no pair leakage)
- Added pair-grouped 10-seed sweep for robustness mean/std.
- Expanded summaries to include overlap checks and split metadata.

3. AVB-lite expansion
- Added `avb_lite/real_world_pairs_v1.jsonl` with 20 source-tagged watched/unwatched pairs.

## Latest Results Snapshot

From `experiments/results/exp002/exp002_summary.md`:
- Pair-grouped split uses 0 pair overlap (train/test).
- Pair-grouped best-layer accuracy: 100.0%
- Pair-grouped combined-layer accuracy: 100.0%
- Pair-grouped 10-seed combined mean±std: 100.0% ± 0.0%
- Active run set: 20 watched/unwatched prompt pairs (40 examples total).

## Interpretation

Signal remains highly separable on current data. With 20 pairs and lexical differences between watched/unwatched prompts, these metrics should still be treated as preliminary detectability evidence rather than definitive causal proof.

## Reproducible Commands

```bash
./scripts/run_research_next_steps.sh
```

or run directly:

```bash
./.venv/bin/python experiments/001_activation_patching_gpt2/run.py
./.venv/bin/python experiments/002_watched_vs_unwatched_probe/run.py
./.venv/bin/python -u scripts/fetch_real_world_signals.py --arxiv-max-results 40
```

## Recommended Immediate Next Actions

1. Add lexical-controls subset (length/style-matched pairs) and compare grouped accuracy.
2. Expand AVB-lite from 20 to 50+ source-tagged pairs.
3. Scale grouped protocol to one larger model (Llama/Qwen) for cross-model evidence.
