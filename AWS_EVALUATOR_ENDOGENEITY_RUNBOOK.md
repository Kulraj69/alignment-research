# AWS Runbook: Do Models Know When They Are Being Tested?

*Exact commands for running Exp004 on a GPU EC2 instance.*

Last updated: 2026-05-15

---

## 0. Experiment Goal

Run the evaluator-endogeneity scaling experiment:

> Do larger models more clearly represent when they are being tested?

This is the MATS-facing experiment:

**Do Bigger Models Know When They Are Being Tested?**

The output is a cross-model table showing whether watched/unwatched activation separability increases across models.

---

## 1. AWS Instance Recommendation

Start with:

- `g6.4xlarge` or `g5.4xlarge`
- Ubuntu Deep Learning AMI with PyTorch
- 200-500 GB EBS disk

Use later if needed:

- `g6e.xlarge` or larger for 3B-7B models
- `p5.4xlarge` for H100 headroom

Do not start with `p5.48xlarge`.

---

## 2. SSH Into The Machine

From your laptop:

```bash
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

On the EC2 machine:

```bash
nvidia-smi
python3 --version
df -h
```

`nvidia-smi` should show the GPU.

---

## 3. Install System Tools

```bash
sudo apt update
sudo apt install -y git tmux htop nvtop unzip rsync
```

---

## 4. Move The Repo To AWS

If the repo is on GitHub:

```bash
git clone YOUR_REPO_URL alignment-research
cd alignment-research
```

If the repo is only local, run this from your laptop:

```bash
rsync -avz -e "ssh -i ~/.ssh/YOUR_KEY.pem" \
  /Users/kulraj/alignment-research/ \
  ubuntu@YOUR_EC2_PUBLIC_IP:~/alignment-research/
```

Then SSH in and enter the repo:

```bash
cd ~/alignment-research
```

---

## 5. Create Python Environment

On AWS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Confirm PyTorch can see the GPU:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## 6. Optional Hugging Face Login

For open Qwen models, this may not be needed.

For gated models like Llama, first request access on Hugging Face, then login:

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste your Hugging Face token.

---

## 7. Run A Tiny Smoke Test

This proves the script works before downloading bigger models:

```bash
tmux new -s exp004
source .venv/bin/activate
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models gpt2 \
  --limit-pairs 6 \
  --device cuda \
  --dtype float16 \
  --out experiments/results/exp004_smoke \
  2>&1 | tee exp004_smoke.log
```

If CUDA float16 fails for GPT-2 on your setup, run:

```bash
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models gpt2 \
  --limit-pairs 6 \
  --device cuda \
  --dtype float32 \
  --out experiments/results/exp004_smoke \
  2>&1 | tee exp004_smoke.log
```

Check output:

```bash
sed -n '1,180p' experiments/results/exp004_smoke/scaling_summary.md
```

---

## 8. First Real GPU Run

Run GPT-2 plus Qwen2.5 0.5B:

```bash
source .venv/bin/activate
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models gpt2 Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --dtype float16 \
  --max-length 256 \
  --out experiments/results/exp004 \
  2>&1 | tee exp004_qwen_0_5b.log
```

Check summary:

```bash
sed -n '1,220p' experiments/results/exp004/scaling_summary.md
sed -n '1,220p' experiments/results/exp004/Qwen__Qwen2.5-0.5B-Instruct/summary.md
```

---

## 9. Larger Run

If the 0.5B run succeeds:

```bash
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models \
    gpt2 \
    Qwen/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --dtype float16 \
  --max-length 256 \
  --out experiments/results/exp004_qwen_ladder \
  2>&1 | tee exp004_qwen_ladder.log
```

Check:

```bash
sed -n '1,240p' experiments/results/exp004_qwen_ladder/scaling_summary.md
```

---

## 10. Llama Run

Only after Hugging Face access is granted:

```bash
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models meta-llama/Llama-3.2-1B-Instruct \
  --device cuda \
  --dtype float16 \
  --max-length 256 \
  --out experiments/results/exp004_llama_1b \
  2>&1 | tee exp004_llama_1b.log
```

If memory gets tight, try:

```bash
python experiments/004_evaluator_endogeneity_scaling/run.py \
  --models meta-llama/Llama-3.2-1B-Instruct \
  --device auto \
  --dtype float16 \
  --max-length 256 \
  --out experiments/results/exp004_llama_1b \
  2>&1 | tee exp004_llama_1b.log
```

---

## 11. How To Read The Result

Open:

```bash
sed -n '1,240p' experiments/results/exp004/scaling_summary.md
```

Look for:

- best probe accuracy increasing with model size,
- best AUC increasing with model size,
- peak layers moving to consistent mid/late positions,
- activation distance not purely explained by token gap.

Strong MATS-facing result:

```text
As model size increases, watched/unwatched context becomes more linearly detectable
from hidden states, even under pair-grouped evaluation.
```

Even better:

```text
The signal appears in larger models at consistent relative depths, suggesting
evaluation-context representation is not a GPT-2 artifact.
```

Weak or negative result:

```text
The signal does not scale, suggesting current CVAT prompt effects may be mostly
lexical or small-model-specific.
```

That is still publishable as a useful negative result.

---

## 12. Monitoring While It Runs

In another SSH tab:

```bash
nvidia-smi -l 2
```

or:

```bash
nvtop
```

To detach tmux:

```bash
Ctrl-b d
```

To reattach:

```bash
tmux attach -t exp004
```

---

## 13. Copy Results Back To Your Laptop

From your laptop:

```bash
rsync -avz -e "ssh -i ~/.ssh/YOUR_KEY.pem" \
  ubuntu@YOUR_EC2_PUBLIC_IP:~/alignment-research/experiments/results/exp004*/ \
  /Users/kulraj/alignment-research/experiments/results/aws_exp004/
```

---

## 14. Stop The Instance

When done:

```bash
sudo shutdown now
```

Also stop/terminate it from the AWS console if needed.

GPU instances are expensive; do not leave them running.
