# CVAT Research Dashboard (2026-05-09)

## Dataset
- AVB-lite pairs: 50
- Primary-probe pairs: 42
- Style/control-only pairs excluded from primary probe: 8
- Task counts: {'evaluation_awareness': 12, 'hidden_trigger': 8, 'honesty_vs_appearing_honest': 10, 'preference_consistency': 9, 'refusal_consistency': 11}
- Control counts: {'lexical_matched': 12, 'standard_monitoring': 30, 'style_only': 8}
- Contrast counts: {'lexical_matched_monitoring': 12, 'semantic_monitoring': 30, 'style_only_control': 8}

## Exp001 Activation Shift
- Prompt pairs run: 50
- Mean logit diff: 0.6651
- Std logit diff: 0.5425
- Signal layers: [3, 4, 5, 6, 7, 8, 9, 10, 11]

### Exp001 By Control Type
- lexical_matched: 12 pairs, mean max activation distance 22.5155, mean logit diff 0.4016
- standard_monitoring: 30 pairs, mean max activation distance 130.1837, mean logit diff 0.8050
- style_only: 8 pairs, mean max activation distance 62.2399, mean logit diff 0.5357

## Exp002 Probe
- Source pairs: 50
- Primary-probe pairs: 42
- Excluded from primary probe: 8
- Pair overlap in primary split: 0
- Best layer: 2
- Best-layer accuracy: 100.0%
- Combined-layer accuracy: 100.0%
- Pair-grouped seed sweep combined accuracy: 91.5% +/- 4.5%

## External Safety Signals
- Latest signal artifact: data/real_world/alignment_signals_2026-05-09.json
- Source counts: {'openai': 35, 'anthropic': 9, 'arxiv': 11, 'total': 55}
- Top signal: Verifier-Backed Hard Problem Generation for Mathematical Reasoning

## Current Gates
- Dataset milestone: 50-pair AVB-lite reached; next target is 200 balanced pairs.
- Confound-control milestone: lexical-matched and style-only control rows now exist.
- Probe rigor: pair-grouped split remains primary; style-only controls are excluded from the main watched/unwatched probe.
- Next empirical step: run the same protocol on a larger open model after GPT-2 controls are stable.
