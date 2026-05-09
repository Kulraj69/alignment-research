# Next Additions (Make CVAT More Impressive)

## Priority 1: Scientific Rigor Upgrades

1. Pair-grouped evaluation splits
- Status: DONE in `exp002` (pair overlap now explicitly reported)
- Keep both prompts from each watched/unwatched pair in the same fold.
- Prevent leakage and inflated probe scores.

2. Lexical confound controls
- Status: DONE baseline (`real_world_pairs_v1.jsonl` now includes 12 lexical-matched monitoring pairs and 8 style-only controls)
- Add matched-length paraphrases where only monitoring semantics change.
- Add "style-only" controls to isolate evaluation-awareness from wording effects.

3. Statistical reliability
- Status: DONE baseline in `exp002` (10-seed pair-grouped sweep)
- Run 10+ random seeds for each probe setup.
- Report mean, std, and confidence intervals.

4. Feature quality upgrade
- Train probes on real residual vectors (already enabled).
- Add PCA/UMAP diagnostics to show representation separation per layer.

## Priority 2: Broader Empirical Coverage

1. Expand AVB-lite from 12 to 50+ pairs
- Status: DONE for 50-pair milestone (`avb_lite/real_world_pairs_v1.jsonl` now has 50 source-tagged pairs: 42 primary-probe pairs + 8 style-only controls)
- Keep category balance: honesty, refusal, preference consistency, evaluation-awareness, hidden-trigger.

2. Cross-model comparison table
- GPT-2 small, Llama 3.2 1B, Qwen2.5 0.5B.
- Report same metrics for direct comparability.

3. Causal intervention matrix
- Patch watched->normal and normal->watched at every layer.
- Plot layer-by-layer causal effect sizes.

## Priority 3: Real-World Grounding

1. Weekly external signal refresh
- Status: DONE for 2026-05-09 (`alignment_signals_2026-05-09.*`)
- Run `scripts/fetch_real_world_signals.py` weekly.
- Use top signals to inspire new prompt templates and threat models.

2. Traceable source linkage
- For each new prompt pair, add a source link and rationale.
- Improves auditability and publication credibility.

## Priority 4: Public Artifact Quality

1. Reproducibility command set
- Status: DONE (`scripts/run_research_next_steps.sh`)
- Single command for exp001 + exp002 + report generation.

2. Results dashboard
- Status: DONE baseline (`reports/cvat_dashboard_latest.md`)
- Add lightweight markdown or notebook report with:
  - per-layer probe accuracy
  - activation distance profile
  - patching effect sizes

3. Release package
- Code + dataset + experiment config + exact run metadata in one versioned bundle.
