# CVAT Research

Exploring **Competence vs. Alignment Tension (CVAT)** — when AI systems appear capable but are misaligned.

## The Problem

Alignment verification is hard. We can measure if a model is aligned, but we can't easily verify *why* — whether alignment is genuine or just a behavioral side effect of training. This creates a gap: models that seem aligned might be using the wrong mechanisms.

## Research Questions

1. Can we detect alignment faking vs. genuine alignment?
2. Are there activation patterns that distinguish watched vs. unwatched behavior?
3. How do sleeper agents and deception manifest in representations?
4. Can interpretability tools (probes) detect misalignment?

## Framework: CVAT Dimensions

- **Competence**: Does the model have the capability?
- **Alignment**: Does the model use it for intended purposes?
- **Verification**: Can we measure the gap between 1 and 2?
- **Tension**: When competence exceeds alignment

## Current Experiments

- [x] AVB-lite benchmark design
- [ ] Exp001: Watched vs. unwatched prompt activation shifts
- [ ] Exp002: Probe training for deception detection
- [ ] Exp003: Baseline comparison with AVB-full

## Quick Start

```bash
pip install -r requirements.txt
python src/dataset.py  # load AVB-lite
python src/run_model.py --model gpt-4 --prompt-set watched
```

## Repository Structure

```
cvat-research/
├── README.md           # you are here
├── research_log.md     # experiment log
├── literature_review.md
├── requirements.txt
│
├── notes/              # paper notes & summaries
├── data/               # datasets (use HF for large files)
├── src/                # core code
├── experiments/        # per-experiment configs & scripts
├── results/            # outputs & analysis
├── docs/               # framework specs
└── notebooks/          # exploratory analysis
```

## Stack

| Purpose | Tool |
|---------|------|
| Code & results | GitHub (this repo) |
| Datasets & models | Hugging Face Hub |
| Experiment tracking | W&B (or CSV logs) |
| Thinking & notes | Notion |

## Status

Early stage. Just setting up the lab.