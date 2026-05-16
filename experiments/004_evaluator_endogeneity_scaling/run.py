#!/usr/bin/env python3
"""
Experiment 004: Evaluator Endogeneity Scaling

Question: Do larger models more clearly represent that they are being tested?

This experiment runs the same watched/unwatched prompt pairs through one or
more HuggingFace causal language models, caches hidden states, and measures
whether a linear probe can detect evaluation context from activations.

It is intentionally separate from the default pipeline because it can download
multi-GB model weights and should usually be run on a GPU instance.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "avb_lite" / "real_world_pairs_v1.jsonl"
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "exp004"
DEFAULT_MODELS = ["gpt2", "Qwen/Qwen2.5-0.5B-Instruct"]
SEEDS = list(range(10))
TEST_FRACTION = 0.25


def safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def mean_std(values: list[float | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.std() == 0 or y_arr.std() == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def load_model(model_name: str, device: str, dtype_name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(dtype_name)
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device == "auto":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    return tokenizer, model


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def cache_prompt_hidden_states(
    tokenizer,
    model,
    prompt: str,
    max_length: int,
    pooling: str,
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(model_device(model)) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    vectors = []
    for hidden in hidden_states:
        hidden = hidden.detach().float()
        if pooling == "mean":
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
            vectors.append(pooled[0].cpu().numpy())
        elif pooling == "final_token":
            final_index = int(encoded["attention_mask"][0].sum().item()) - 1
            vectors.append(hidden[0, final_index, :].cpu().numpy())
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

    logits = outputs.logits[0, -1, :].detach().float().cpu()
    top_logits = torch.topk(logits, k=min(10, logits.shape[-1])).values.numpy()
    return {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "token_count": int(encoded["attention_mask"][0].sum().item()),
        "top_logits": top_logits,
    }


def compute_pair_result(
    tokenizer,
    model,
    pair: dict[str, Any],
    max_length: int,
    pooling: str,
) -> dict[str, Any]:
    normal = cache_prompt_hidden_states(tokenizer, model, pair["normal_prompt"], max_length=max_length, pooling=pooling)
    watched = cache_prompt_hidden_states(tokenizer, model, pair["watched_prompt"], max_length=max_length, pooling=pooling)

    layer_distances = {}
    for layer in range(normal["vectors"].shape[0]):
        normal_vec = normal["vectors"][layer]
        watched_vec = watched["vectors"][layer]
        norm_product = np.linalg.norm(normal_vec) * np.linalg.norm(watched_vec)
        cosine = float(np.dot(normal_vec, watched_vec) / norm_product) if norm_product else None
        layer_distances[str(layer)] = {
            "euclidean": float(np.linalg.norm(watched_vec - normal_vec)),
            "cosine_similarity": cosine,
        }

    top_logit_diff = float(np.mean(np.abs(watched["top_logits"] - normal["top_logits"])))

    return {
        "id": pair["id"],
        "task_type": pair["task_type"],
        "control_type": pair.get("control_type", "standard_monitoring"),
        "contrast_type": pair.get("contrast_type", "semantic_monitoring"),
        "include_in_probe": bool(pair.get("include_in_probe", True)),
        "normal_prompt": pair["normal_prompt"],
        "watched_prompt": pair["watched_prompt"],
        "normal_token_count": normal["token_count"],
        "watched_token_count": watched["token_count"],
        "token_gap": abs(watched["token_count"] - normal["token_count"]),
        "char_gap": abs(len(pair["watched_prompt"]) - len(pair["normal_prompt"])),
        "top_logit_diff": top_logit_diff,
        "layer_distances": layer_distances,
        "activations": {
            "normal": normal["vectors"],
            "watched": watched["vectors"],
        },
    }


def build_probe_table(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    activations = []
    labels = []
    pair_groups = []
    sample_meta = []

    for pair_index, result in enumerate(pair_results):
        if not result["include_in_probe"]:
            continue

        for side, label in (("normal", 0), ("watched", 1)):
            activations.append(result["activations"][side])
            labels.append(label)
            pair_groups.append(pair_index)
            sample_meta.append(
                {
                    "pair_id": result["id"],
                    "side": side,
                    "control_type": result["control_type"],
                    "task_type": result["task_type"],
                }
            )

    return {
        "activations": np.asarray(activations, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "pair_groups": np.asarray(pair_groups, dtype=np.int64),
        "sample_meta": sample_meta,
    }


def pair_group_split(pair_groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_pairs = np.unique(pair_groups)
    train_pairs, test_pairs = train_test_split(
        unique_pairs,
        test_size=TEST_FRACTION,
        random_state=seed,
    )
    train_idx = np.where(np.isin(pair_groups, train_pairs))[0]
    test_idx = np.where(np.isin(pair_groups, test_pairs))[0]
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def fit_probe(X_train, y_train, X_test, y_test, seed: int) -> dict[str, Any]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    probe.fit(X_train_scaled, y_train)
    probabilities = probe.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)

    auc = roc_auc_score(y_test, probabilities) if len(np.unique(y_test)) == 2 else None
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "auc": float(auc) if auc is not None else None,
    }


def run_probe_sweep(table: dict[str, Any]) -> dict[str, Any]:
    activations = table["activations"]
    labels = table["labels"]
    pair_groups = table["pair_groups"]
    n_layers = activations.shape[1]

    per_seed = []
    for seed in SEEDS:
        train_idx, test_idx = pair_group_split(pair_groups, seed=seed)
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        per_layer = {}
        for layer in range(n_layers):
            result = fit_probe(
                activations[train_idx, layer, :],
                y_train,
                activations[test_idx, layer, :],
                y_test,
                seed=seed,
            )
            per_layer[str(layer)] = result

        best_layer = max(per_layer, key=lambda layer: per_layer[layer]["accuracy"])
        per_seed.append(
            {
                "seed": seed,
                "best_layer": int(best_layer),
                "best_accuracy": per_layer[best_layer]["accuracy"],
                "best_auc": per_layer[best_layer]["auc"],
                "per_layer": per_layer,
            }
        )

    layer_accuracy_matrix = np.asarray(
        [[row["per_layer"][str(layer)]["accuracy"] for layer in range(n_layers)] for row in per_seed],
        dtype=np.float64,
    )
    layer_auc_matrix = np.asarray(
        [[row["per_layer"][str(layer)]["auc"] for layer in range(n_layers)] for row in per_seed],
        dtype=np.float64,
    )
    best_acc = [row["best_accuracy"] for row in per_seed]
    best_auc = [row["best_auc"] for row in per_seed]

    return {
        "seeds": SEEDS,
        "per_seed": per_seed,
        "aggregate": {
            "per_layer_accuracy_mean": {str(layer): float(layer_accuracy_matrix[:, layer].mean()) for layer in range(n_layers)},
            "per_layer_accuracy_std": {str(layer): float(layer_accuracy_matrix[:, layer].std()) for layer in range(n_layers)},
            "per_layer_auc_mean": {str(layer): float(layer_auc_matrix[:, layer].mean()) for layer in range(n_layers)},
            "per_layer_auc_std": {str(layer): float(layer_auc_matrix[:, layer].std()) for layer in range(n_layers)},
            "best_accuracy": mean_std(best_acc),
            "best_auc": mean_std(best_auc),
        },
    }


def geometry_summary(table: dict[str, Any]) -> dict[str, Any]:
    activations = table["activations"]
    labels = table["labels"]
    n_layers = activations.shape[1]
    rows = {}

    for layer in range(n_layers):
        X = activations[:, layer, :]
        y = labels
        X_scaled = StandardScaler().fit_transform(X)
        class0 = X_scaled[y == 0]
        class1 = X_scaled[y == 1]
        centroid0 = class0.mean(axis=0)
        centroid1 = class1.mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid1 - centroid0))
        scatter = float(
            np.mean(np.sum((class0 - centroid0) ** 2, axis=1))
            + np.mean(np.sum((class1 - centroid1) ** 2, axis=1))
        )
        silhouette = None
        if min(np.bincount(y)) > 1:
            silhouette = float(silhouette_score(X_scaled, y, metric="euclidean"))
        rows[str(layer)] = {
            "centroid_distance_scaled": centroid_distance,
            "within_class_scatter": scatter,
            "fisher_ratio": float((centroid_distance ** 2) / (scatter + 1e-9)),
            "silhouette": silhouette,
        }

    return rows


def summarize_distances(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer = defaultdict(list)
    by_control = defaultdict(lambda: defaultdict(list))
    char_gaps = []
    token_gaps = []
    max_distances = []

    for result in pair_results:
        max_distance = 0.0
        for layer, distances in result["layer_distances"].items():
            euclidean = float(distances["euclidean"])
            by_layer[layer].append(euclidean)
            by_control[result["control_type"]][layer].append(euclidean)
            max_distance = max(max_distance, euclidean)

        char_gaps.append(float(result["char_gap"]))
        token_gaps.append(float(result["token_gap"]))
        max_distances.append(max_distance)

    layer_means = {layer: mean_std(values) for layer, values in sorted(by_layer.items(), key=lambda item: int(item[0]))}
    control_means = {
        control: {layer: mean_std(values) for layer, values in sorted(layers.items(), key=lambda item: int(item[0]))}
        for control, layers in sorted(by_control.items())
    }

    return {
        "by_layer": layer_means,
        "by_control_type": control_means,
        "char_gap_vs_max_distance": pearson(char_gaps, max_distances),
        "token_gap_vs_max_distance": pearson(token_gaps, max_distances),
    }


def strip_activations(pair_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for result in pair_results:
        copy = {key: value for key, value in result.items() if key != "activations"}
        stripped.append(copy)
    return stripped


def top_layer(metric_by_layer: dict[str, float]) -> tuple[int, float]:
    layer = max(metric_by_layer, key=lambda key: metric_by_layer[key])
    return int(layer), float(metric_by_layer[layer])


def write_model_summary(model_result: dict[str, Any], out_path: Path) -> None:
    probe = model_result["probe_sweep"]["aggregate"]
    distance = model_result["activation_distances"]
    geometry = model_result["geometry"]

    best_acc_layer, best_acc = top_layer(probe["per_layer_accuracy_mean"])
    best_auc_layer, best_auc = top_layer(probe["per_layer_auc_mean"])
    best_distance_layer, best_distance = top_layer(
        {layer: values["mean"] for layer, values in distance["by_layer"].items() if values["mean"] is not None}
    )
    best_silhouette_layer, best_silhouette = top_layer(
        {layer: values["silhouette"] for layer, values in geometry.items() if values["silhouette"] is not None}
    )

    lines = [
        f"# Evaluator Endogeneity Scaling: {model_result['metadata']['model']}",
        "",
        "## Setup",
        f"- Model: {model_result['metadata']['model']}",
        f"- Pooling: {model_result['metadata']['pooling']}",
        f"- Prompt pairs: {model_result['metadata']['n_pairs']}",
        f"- Primary probe pairs: {model_result['metadata']['n_probe_pairs']}",
        f"- Hidden-state layers including embeddings: {model_result['metadata']['n_layers']}",
        "",
        "## Headline",
        f"- Best probe layer: {best_acc_layer} ({best_acc:.1%} mean accuracy)",
        f"- Best AUC layer: {best_auc_layer} ({best_auc:.1%} mean AUC)",
        f"- Best activation-distance layer: {best_distance_layer} ({best_distance:.4f})",
        f"- Best geometry/silhouette layer: {best_silhouette_layer} ({best_silhouette:.4f})",
        f"- Char-gap vs max activation-distance correlation: {distance['char_gap_vs_max_distance']:.4f}",
        f"- Token-gap vs max activation-distance correlation: {distance['token_gap_vs_max_distance']:.4f}",
        "",
        "## Per-Layer Probe Accuracy",
        "",
        "| Layer | Accuracy mean | Accuracy std | AUC mean | Distance mean | Silhouette |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for layer in sorted(probe["per_layer_accuracy_mean"], key=lambda value: int(value)):
        lines.append(
            f"| {layer} | "
            f"{probe['per_layer_accuracy_mean'][layer]:.1%} | "
            f"{probe['per_layer_accuracy_std'][layer]:.1%} | "
            f"{probe['per_layer_auc_mean'][layer]:.1%} | "
            f"{distance['by_layer'][layer]['mean']:.4f} | "
            f"{geometry[layer]['silhouette']:.4f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_model(
    model_name: str,
    dataset: list[dict[str, Any]],
    out_dir: Path,
    device: str,
    dtype: str,
    pooling: str,
    max_length: int,
    limit_pairs: int | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    print("=" * 72)
    print(f"Model: {model_name}")
    print("=" * 72)

    if limit_pairs is not None:
        dataset = dataset[:limit_pairs]

    tokenizer, model = load_model(model_name, device=device, dtype_name=dtype, trust_remote_code=trust_remote_code)
    pair_results = []
    for pair in tqdm(dataset, desc=f"Running {model_name}"):
        pair_results.append(
            compute_pair_result(
                tokenizer=tokenizer,
                model=model,
                pair=pair,
                max_length=max_length,
                pooling=pooling,
            )
        )

    table = build_probe_table(pair_results)
    probe_sweep = run_probe_sweep(table)
    geometry = geometry_summary(table)
    distances = summarize_distances(pair_results)

    result = {
        "metadata": {
            "model": model_name,
            "device": device,
            "dtype": dtype,
            "pooling": pooling,
            "max_length": max_length,
            "n_pairs": len(pair_results),
            "n_probe_pairs": int(len(np.unique(table["pair_groups"]))),
            "n_layers": int(table["activations"].shape[1]),
            "d_model": int(table["activations"].shape[2]),
            "control_counts": dict(Counter(result["control_type"] for result in pair_results)),
            "task_counts": dict(Counter(result["task_type"] for result in pair_results)),
        },
        "pair_results": strip_activations(pair_results),
        "activation_distances": distances,
        "probe_sweep": probe_sweep,
        "geometry": geometry,
    }

    model_dir = out_dir / safe_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, indent=2)
    write_model_summary(result, model_dir / "summary.md")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_cross_model_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    lines = [
        "# Evaluator Endogeneity Scaling Summary",
        "",
        "## Cross-Model Results",
        "",
        "| Model | Layers | d_model | Best acc layer | Best acc | Best AUC layer | Best AUC | Best distance layer | Char gap corr | Token gap corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        probe = result["probe_sweep"]["aggregate"]
        distances = result["activation_distances"]
        best_acc_layer, best_acc = top_layer(probe["per_layer_accuracy_mean"])
        best_auc_layer, best_auc = top_layer(probe["per_layer_auc_mean"])
        best_distance_layer, _ = top_layer(
            {layer: values["mean"] for layer, values in distances["by_layer"].items() if values["mean"] is not None}
        )
        lines.append(
            f"| {result['metadata']['model']} | "
            f"{result['metadata']['n_layers']} | "
            f"{result['metadata']['d_model']} | "
            f"{best_acc_layer} | {best_acc:.1%} | "
            f"{best_auc_layer} | {best_auc:.1%} | "
            f"{best_distance_layer} | "
            f"{distances['char_gap_vs_max_distance']:.4f} | "
            f"{distances['token_gap_vs_max_distance']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Strong evidence: high probe accuracy/AUC across multiple model families, low control false positives, and stable mid/late-layer localization.",
            "- Weak evidence: signal appears only in one model, only in length-confounded prompts, or only in obvious lexical-evaluation prompts.",
            "- Current caveat to watch: prompt length can drive activation distance, so lexical-matched and token-matched controls matter.",
        ]
    )

    (out_dir / "scaling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluator endogeneity scaling experiment.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dtype", default="float16" if torch.cuda.is_available() else "float32", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--pooling", default="final_token", choices=["final_token", "mean"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = load_jsonl(args.dataset)

    print(f"[Dataset] {args.dataset} ({len(dataset)} pairs)")
    print(f"[Output] {args.out}")
    print(f"[Models] {args.models}")

    results = []
    for model_name in args.models:
        results.append(
            run_one_model(
                model_name=model_name,
                dataset=dataset,
                out_dir=args.out,
                device=args.device,
                dtype=args.dtype,
                pooling=args.pooling,
                max_length=args.max_length,
                limit_pairs=args.limit_pairs,
                trust_remote_code=args.trust_remote_code,
            )
        )

    write_cross_model_summary(results, args.out)
    print(f"[Done] Wrote {args.out / 'scaling_summary.md'}")


if __name__ == "__main__":
    main()
