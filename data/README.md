# Data Directory

AVB-lite and experiment datasets.

For large files (>10MB), use HuggingFace Hub instead of git.
Check in small JSONL files for examples.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("your-handle/cvat-data", "avb_lite_v1")
```