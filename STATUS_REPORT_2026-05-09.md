# CVAT Status Report - 2026-05-09

## Snapshot

The project has moved from a 20-pair pilot into a 50-pair AVB-lite benchmark with explicit confound controls, validation, fresh real-world safety signals, and a generated research dashboard.

## Completed Today

- Expanded `avb_lite/real_world_pairs_v1.jsonl` to 50 source-tagged prompt pairs.
- Added 12 lexical-matched monitoring controls.
- Added 8 style-only controls and excluded them from the primary probe via `include_in_probe=false`.
- Added `scripts/validate_avb_lite.py`.
- Added `scripts/build_results_dashboard.py`.
- Updated `scripts/run_research_next_steps.sh` to validate, run Exp001, run Exp002, refresh signals, and rebuild the dashboard.
- Refreshed real-world safety/alignment signal artifacts for 2026-05-09.
- Reran Exp001 and Exp002 on the expanded dataset.

## Latest Results

### Dataset

- Total prompt pairs: 50
- Primary-probe pairs: 42
- Style/control-only pairs excluded from primary probe: 8
- Source-tagged rows: 50/50

### Exp001

- Model: GPT-2 small
- Prompt pairs: 50
- Mean logit diff: 0.6651
- Std logit diff: 0.5425
- Signal layers: 3-11

Control-type activation shift:

- standard_monitoring: mean max activation distance 130.1837
- lexical_matched: mean max activation distance 22.5155
- style_only: mean max activation distance 62.2399

### Exp002

- Data mode: real Exp001 residual activations
- Examples: 84
- Primary pairs: 42
- Pair-grouped split pair overlap: 0
- Primary pair-grouped combined accuracy: 100.0%
- Pair-grouped seed sweep combined mean +/- std: 91.5% +/- 4.5%
- Best-layer seed sweep mean +/- std: 95.5% +/- 3.5%

## Interpretation

The evaluation-awareness signal remains strong after expansion, but the 10-seed estimate is no longer artificially perfect. That is good news: the project now has a stronger signal and a more credible uncertainty story.

The controls show that lexical and style shifts matter. Standard monitoring prompts produce the largest activation movement, lexical-matched controls produce a smaller movement, and style-only controls produce a nontrivial baseline. This supports the next methodological step: separate semantic monitoring signal from general prompt-form signal.

## Current Artifacts

- Dashboard: `reports/cvat_dashboard_latest.md`
- Dataset: `avb_lite/real_world_pairs_v1.jsonl`
- Exp001 summary: `experiments/results/exp001/exp001_summary.md`
- Exp002 summary: `experiments/results/exp002/exp002_summary.md`
- Latest real-world signals: `data/real_world/alignment_signals_2026-05-09.md`

## Next Steps

1. Add generated-output similarity checks for watched/unwatched pairs.
2. Report probe performance by control type, not just overall primary probe accuracy.
3. Add a causal-patching matrix to the dashboard.
4. Start larger-model replication once the GPT-2 control protocol is stable.
