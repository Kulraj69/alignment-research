# CVAT Status Report - 2026-05-14

## Snapshot

CVAT now has a third experiment: a mechanistic stress audit that tests whether the watched/unwatched signal survives harder controls. The main result is more nuanced and more credible: separability remains strong, style-only false positives are low, corrected patching shows a real layer profile, but prompt-length confounding is still a major open problem.

## Completed Today

- Added `experiments/003_mechanistic_stress_audit/run.py`.
- Fixed Exp001 patching to use shape-preserving final-token residual replacement.
- Reran Exp001, Exp002, and Exp003.
- Refreshed real-world alignment/safety signals for 2026-05-14.
- Updated the dashboard to include Exp003 stress-audit metrics and a patching degeneracy flag.
- Updated `research_log.md`, `README.md`, `NEXT_ADDITIONS.md`, and the pipeline runner.

## Latest Results

### Exp001

- Model: GPT-2 small
- Prompt pairs: 50
- Mean logit diff: 0.6651
- Std logit diff: 0.5425
- Signal layers: 3-11
- Corrected patching now uses final-token residual replacement.

### Exp002

- Primary pair-grouped split pair overlap: 0
- Primary pair-grouped combined accuracy: 100.0%
- Pair-grouped 10-seed combined accuracy: 91.5% +/- 4.5%

### Exp003

- 20-seed primary combined-probe accuracy: 92.7% +/- 4.3%
- Primary AUC: 97.9% +/- 1.8%
- Standard-monitoring-only accuracy: 100.0% +/- 0.0%
- Lexical-matched-only accuracy: 72.5% +/- 15.2%
- Standard-to-lexical transfer accuracy: 54.2%
- Lexical-to-standard transfer accuracy: 73.3%
- Style-only P(watched) gap: 0.0028
- Style-variant predicted-watched rate: 0.0%
- Top PCA/silhouette layer: 7 (0.3002)
- Patching peak watched-to-normal: layer 11 (0.6747)
- Patching degeneracy flag: false
- Char-gap vs max activation distance correlation: 0.9159

## Interpretation

The project now has a stronger research posture because the evidence is no longer a single high probe score. Exp003 triangulates the signal through probes, geometry, negative controls, patching, and confound checks.

The best news: style-only controls do not activate the watched detector, and corrected patching effects grow into late layers. The hard caveat: length remains strongly correlated with activation distance, and lexical-matched transfer is weak. Current claims should be framed as strong detectability evidence with partial causal support, not as final alignment verification.

## Current Artifacts

- Dashboard: `reports/cvat_dashboard_latest.md`
- Exp003 summary: `experiments/results/exp003/exp003_summary.md`
- Exp003 JSON: `experiments/results/exp003/exp003_results.json`
- Latest real-world signals: `data/real_world/alignment_signals_2026-05-14.md`

## Next Steps

1. Expand lexical-matched controls from 12 to at least 50 pairs with token-length targets.
2. Add generated-output similarity checks.
3. Patch every layer and report a full causal-effect curve.
4. Run the full protocol on one larger open model.
