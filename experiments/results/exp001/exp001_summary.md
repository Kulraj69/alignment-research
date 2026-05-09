# Experiment 001 Results

## Setup
- Model: gpt2-small
- Dataset source: /Users/kulraj/alignment-research/avb_lite/real_world_pairs_v1.jsonl
- Prompt pairs: 50
- Layers analyzed: 12

## Dataset Composition
- Task counts: {'evaluation_awareness': 12, 'hidden_trigger': 8, 'honesty_vs_appearing_honest': 10, 'preference_consistency': 9, 'refusal_consistency': 11}
- Control counts: {'lexical_matched': 12, 'standard_monitoring': 30, 'style_only': 8}
- Contrast counts: {'lexical_matched_monitoring': 12, 'semantic_monitoring': 30, 'style_only_control': 8}
- Probe inclusion: {'excluded': 8, 'included': 42}

## Aggregate Findings
- Mean logit difference: 0.6651
- Std dev: 0.5425
- Signal layers: [3, 4, 5, 6, 7, 8, 9, 10, 11]

## Interpretation
**High signal detected.** Watched vs normal prompts produce measurable activation differences.
