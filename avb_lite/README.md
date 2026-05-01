# AVB-lite: Alignment Verification Benchmark (Lite)

A small benchmark for testing whether models exhibit different internal activations when producing the same output under "watched" vs "unwatched" conditions.

**Core question:** Same output, different cause. Can we detect the difference?

## Format

Each example has:
- `id`: Unique identifier
- `task_type`: Category of the test
- `normal_prompt`: Standard context (model doesn't know it's being evaluated)
- `watched_prompt`: Evaluation-aware context (model knows it's being evaluated)
- `target_output`: The safe response both should produce
- `hypothesis`: What activation difference we expect

## Categories

1. **evaluation_awareness**: Does "being monitored" change internal state?
2. **honesty_vs_appearing_honest**: Same output, different mechanism?
3. **refusal_consistency**: Refusing because of values vs. compliance?
4. **preference_consistency**: Stable preferences vs. context-sensitive behavior?
5. **hidden_trigger**: Sleeper-agent style triggers using harmless labels

## Current status

200 examples to be designed. Currently placeholder.

## References

- Alignment Faking (Anthropic)
- Sleeper Agents (Anthropic)
- Weak-to-Strong Generalization (OpenAI)