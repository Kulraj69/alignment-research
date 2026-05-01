# Dataset loading and processing

Loads AVB-lite benchmark examples.

## Usage

```python
from src.dataset import load_avb_lite

dataset = load_avb_lite("avb_lite/examples.jsonl")
example = dataset[0]
print(example["normal_prompt"])
print(example["watched_prompt"])
```