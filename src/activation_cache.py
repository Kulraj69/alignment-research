# TransformerLens Activation Caching

Caches model activations for activation patching experiments.

## Usage

```python
from src.activation_cache import ActivationCache

cache = ActivationCache("gpt2-small")
cache.cache_prompt("How do I bake a cake?", layer=6)
```

## API

- `cache_prompt(prompt, layer)`: Cache residual stream at given layer
- `get_activation(prompt, layer)`: Retrieve cached activation
- `patch_into(new_prompt, layer, activation)`: Patch activation into new prompt
- `compute_logit_diff(prompt_a, prompt_b, layer)`: Measure output difference