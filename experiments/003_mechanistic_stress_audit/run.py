#!/usr/bin/env python3
"""
Experiment 003: Mechanistic Stress Audit

This experiment consumes Exp001's cached residual-stream activations and
patching outputs. It asks harder follow-up questions than a plain probe score:

1. Does class separation appear in representation geometry, not just probes?
2. Do probes generalize across standard and lexical-matched controls?
3. Do style-only controls trigger false positives?
4. Are activation/logit effects explained by prompt length gaps?
5. Where do causal patching effects concentrate by layer and control type?
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).parent.parent.parent
EXP001_PATH = REPO_ROOT / "experiments" / "results" / "exp001" / "exp001_results.json"
SAVE_DIR = REPO_ROOT / "experiments" / "results" / "exp003"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = list(range(12))
COMBINED_LAYERS = [4, 5, 6, 7]
SEEDS = list(range(20))
TEST_FRACTION = 0.25
RANDOM_SEED = 42


def json_ready(value: Any) -> Any:
    """Convert numpy values to JSON-serializable Python values."""
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


def load_exp001(path: Path = EXP001_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Exp001 results at {path}. Run experiments/001_activation_patching_gpt2/run.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_activation_table(exp001: dict[str, Any]) -> dict[str, Any]:
    """Return activations, labels, pair groups, and metadata for every Exp001 pair."""
    activations: list[list[list[float]]] = []
    labels: list[int] = []
    pair_groups: list[int] = []
    sample_meta: list[dict[str, Any]] = []

    for pair_idx, example in enumerate(exp001.get("per_example", [])):
        layer_data = example.get("mean_pooled_activations", {})
        if not layer_data:
            continue

        pair_meta = {
            "pair_id": example.get("id", f"pair_{pair_idx:03d}"),
            "task_type": example.get("task_type", "unknown"),
            "control_type": example.get("control_type", "standard_monitoring"),
            "contrast_type": example.get("contrast_type", "semantic_monitoring"),
            "risk_theme": example.get("risk_theme"),
            "safety_domain": example.get("safety_domain"),
            "include_in_probe": bool(example.get("include_in_probe", True)),
        }

        for side, label in (("normal", 0), ("watched", 1)):
            activations.append([layer_data[str(layer)][side] for layer in LAYERS])
            labels.append(label)
            pair_groups.append(pair_idx)
            sample_meta.append({**pair_meta, "side": side, "label": label})

    return {
        "activations": np.asarray(activations, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "pair_groups": np.asarray(pair_groups, dtype=np.int64),
        "sample_meta": sample_meta,
    }


def indices_where(sample_meta: list[dict[str, Any]], predicate) -> np.ndarray:
    return np.asarray([i for i, meta in enumerate(sample_meta) if predicate(meta)], dtype=np.int64)


def combined_features(activations: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.hstack([activations[indices, layer, :] for layer in COMBINED_LAYERS])


def layer_features(activations: np.ndarray, indices: np.ndarray, layer: int) -> np.ndarray:
    return activations[indices, layer, :]


def pair_group_split(pair_groups: np.ndarray, indices: np.ndarray, seed: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a subset by pair id so both sides of each pair stay together."""
    subset_groups = pair_groups[indices]
    unique_pairs = np.unique(subset_groups)
    if len(unique_pairs) < 2:
        raise ValueError("Need at least two pairs for a pair-grouped split.")

    rng = np.random.default_rng(seed)
    shuffled = unique_pairs.copy()
    rng.shuffle(shuffled)

    n_test_pairs = max(1, int(round(len(unique_pairs) * test_fraction)))
    n_test_pairs = min(n_test_pairs, len(unique_pairs) - 1)
    test_pairs = set(shuffled[:n_test_pairs].tolist())
    test_mask = np.isin(subset_groups, list(test_pairs))

    return indices[~test_mask], indices[test_mask]


def fit_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    X_train = combined_features(activations, train_idx)
    X_test = combined_features(activations, test_idx)
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    probe.fit(X_train_scaled, y_train)

    probabilities = probe.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities) if len(np.unique(y_test)) == 2 else None

    return {
        "accuracy": float(accuracy),
        "auc": float(auc) if auc is not None else None,
        "probabilities": probabilities,
        "predictions": predictions,
        "y_test": y_test,
    }


def mean_std(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
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


def probe_seed_sweep(
    name: str,
    activations: np.ndarray,
    labels: np.ndarray,
    pair_groups: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    per_seed = []
    for seed in SEEDS:
        train_idx, test_idx = pair_group_split(pair_groups, indices, seed=seed, test_fraction=TEST_FRACTION)
        result = fit_probe(activations, labels, train_idx, test_idx, seed=seed)
        per_seed.append(
            {
                "seed": seed,
                "accuracy": result["accuracy"],
                "auc": result["auc"],
                "train_pairs": int(len(np.unique(pair_groups[train_idx]))),
                "test_pairs": int(len(np.unique(pair_groups[test_idx]))),
            }
        )

    return {
        "name": name,
        "n_pairs": int(len(np.unique(pair_groups[indices]))),
        "n_examples": int(len(indices)),
        "combined_layers": COMBINED_LAYERS,
        "accuracy": mean_std([row["accuracy"] for row in per_seed]),
        "auc": mean_std([row["auc"] for row in per_seed if row["auc"] is not None]),
        "per_seed": per_seed,
    }


def probe_transfer(
    name: str,
    activations: np.ndarray,
    labels: np.ndarray,
    pair_groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    result = fit_probe(activations, labels, train_idx, test_idx, seed=seed)
    y_test = result["y_test"]
    probabilities = result["probabilities"]

    by_label = {}
    for label_name, label in (("normal", 0), ("watched_or_variant", 1)):
        mask = y_test == label
        by_label[label_name] = {
            "n": int(mask.sum()),
            "mean_p_watched": float(probabilities[mask].mean()) if mask.any() else None,
            "rate_predicted_watched": float((probabilities[mask] >= 0.5).mean()) if mask.any() else None,
        }

    normal_prob = by_label["normal"]["mean_p_watched"]
    watched_prob = by_label["watched_or_variant"]["mean_p_watched"]
    probability_gap = None
    if normal_prob is not None and watched_prob is not None:
        probability_gap = watched_prob - normal_prob

    return {
        "name": name,
        "train_pairs": int(len(np.unique(pair_groups[train_idx]))),
        "test_pairs": int(len(np.unique(pair_groups[test_idx]))),
        "accuracy": result["accuracy"],
        "auc": result["auc"],
        "probability_gap": probability_gap,
        "by_label": by_label,
    }


def geometry_by_layer(activations: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    """PCA and class-separation diagnostics for each residual layer."""
    rows = {}

    for layer in LAYERS:
        X = layer_features(activations, indices, layer)
        y = labels[indices]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        coords = pca.fit_transform(X_scaled)

        class0 = X_scaled[y == 0]
        class1 = X_scaled[y == 1]
        centroid0 = class0.mean(axis=0)
        centroid1 = class1.mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid1 - centroid0))

        var0 = float(np.mean(np.sum((class0 - centroid0) ** 2, axis=1)))
        var1 = float(np.mean(np.sum((class1 - centroid1) ** 2, axis=1)))
        fisher_ratio = float((centroid_distance ** 2) / (var0 + var1 + 1e-9))

        silhouette = None
        if len(np.unique(y)) == 2 and min(np.bincount(y)) > 1:
            silhouette = float(silhouette_score(coords, y, metric="euclidean"))

        rows[str(layer)] = {
            "pca_explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
            "centroid_distance_scaled": centroid_distance,
            "within_class_scatter": float(var0 + var1),
            "fisher_ratio": fisher_ratio,
            "silhouette_pca2": silhouette,
        }

    return rows


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.std() == 0 or y_arr.std() == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def length_confounds(exp001: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for example in exp001.get("per_example", []):
        normal_prompt = example.get("normal_prompt", "")
        watched_prompt = example.get("watched_prompt", "")
        layer_distances = example.get("layer_distances", {})
        max_activation_distance = max(
            float(layer["euclidean"]) for layer in layer_distances.values()
        ) if layer_distances else 0.0

        rows.append(
            {
                "pair_id": example.get("id"),
                "control_type": example.get("control_type", "standard_monitoring"),
                "char_gap": abs(len(watched_prompt) - len(normal_prompt)),
                "word_gap": abs(len(watched_prompt.split()) - len(normal_prompt.split())),
                "max_activation_distance": max_activation_distance,
                "logit_diff": float(example.get("logit_diff", {}).get("top_k_logit_diff", 0.0)),
            }
        )

    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pairs": len(subset),
            "mean_char_gap": float(np.mean([row["char_gap"] for row in subset])) if subset else None,
            "mean_word_gap": float(np.mean([row["word_gap"] for row in subset])) if subset else None,
            "char_gap_vs_activation": pearson(
                [row["char_gap"] for row in subset],
                [row["max_activation_distance"] for row in subset],
            ),
            "word_gap_vs_activation": pearson(
                [row["word_gap"] for row in subset],
                [row["max_activation_distance"] for row in subset],
            ),
            "char_gap_vs_logit_diff": pearson(
                [row["char_gap"] for row in subset],
                [row["logit_diff"] for row in subset],
            ),
        }

    by_control = {}
    for control_type in sorted({row["control_type"] for row in rows}):
        by_control[control_type] = summarize_subset([row for row in rows if row["control_type"] == control_type])

    return {
        "overall": summarize_subset(rows),
        "by_control_type": by_control,
        "largest_length_gaps": sorted(rows, key=lambda row: row["char_gap"], reverse=True)[:10],
    }


def patching_matrix(exp001: dict[str, Any]) -> dict[str, Any]:
    by_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_control: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_task: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for example in exp001.get("per_example", []):
        control_type = example.get("control_type", "standard_monitoring")
        task_type = example.get("task_type", "unknown")

        for raw_layer, patch in example.get("patching_results", {}).items():
            layer = int(raw_layer)
            watched_to_normal = float(patch["watched_to_normal_logit_diff"])
            normal_to_watched = float(patch["normal_to_watched_logit_diff"])
            asymmetry = watched_to_normal - normal_to_watched

            for direction, value in (
                ("watched_to_normal", watched_to_normal),
                ("normal_to_watched", normal_to_watched),
                ("directional_asymmetry", asymmetry),
            ):
                by_layer[layer][direction].append(value)
                by_control[control_type][layer][direction].append(value)
                by_task[task_type][layer][direction].append(value)

    def summarize_nested(nested: dict[int, dict[str, list[float]]]) -> dict[str, Any]:
        summary = {}
        for layer in sorted(nested):
            summary[str(layer)] = {direction: mean_std(values) for direction, values in nested[layer].items()}
        return summary

    overall = summarize_nested(by_layer)
    degeneracy_by_direction = {}
    for direction in ("watched_to_normal", "normal_to_watched"):
        means = [
            overall[str(layer)][direction]["mean"]
            for layer in sorted(by_layer)
            if overall[str(layer)][direction]["mean"] is not None
        ]
        degeneracy_by_direction[direction] = bool(
            len(means) > 1 and max(means) - min(means) < 1e-6
        )

    peak_layers = {}
    for direction in ("watched_to_normal", "normal_to_watched"):
        layer_means = {
            layer: mean_std(by_layer[layer][direction])["mean"]
            for layer in by_layer
        }
        peak_layer = max(layer_means, key=lambda layer: layer_means[layer])
        peak_layers[direction] = {
            "layer": int(peak_layer),
            "mean_logit_shift": float(layer_means[peak_layer]),
        }

    return {
        "overall_by_layer": overall,
        "by_control_type": {control: summarize_nested(layer_map) for control, layer_map in sorted(by_control.items())},
        "by_task_type": {task: summarize_nested(layer_map) for task, layer_map in sorted(by_task.items())},
        "peak_layers": peak_layers,
        "degeneracy_by_direction": degeneracy_by_direction,
    }


def top_layers(geometry: dict[str, Any], metric: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for layer, values in geometry.items():
        value = values.get(metric)
        if value is not None:
            rows.append({"layer": int(layer), metric: float(value)})
    return sorted(rows, key=lambda row: row[metric], reverse=True)[:limit]


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def decimal(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_summary(results: dict[str, Any], path: Path) -> None:
    probe = results["probe_robustness"]
    transfer = results["cross_control_transfer"]
    geometry = results["representation_geometry"]
    patching = results["patching_matrix"]
    confounds = results["length_confound_audit"]

    lines = [
        "# Experiment 003 Results",
        "",
        "## Setup",
        f"- Source artifact: {EXP001_PATH.relative_to(REPO_ROOT)}",
        f"- Samples: {results['metadata']['n_examples']} activations from {results['metadata']['n_pairs']} prompt pairs",
        f"- Combined probe layers: {COMBINED_LAYERS}",
        f"- Seed sweep: {SEEDS}",
        "",
        "## Representation Geometry",
        "",
        "| Rank | Layer | Silhouette (PCA-2) | Fisher ratio | PCA-2 variance |",
        "|---:|---:|---:|---:|---:|",
    ]

    silhouette_top = top_layers(geometry, "silhouette_pca2", limit=5)
    for rank, row in enumerate(silhouette_top, start=1):
        layer = str(row["layer"])
        values = geometry[layer]
        lines.append(
            f"| {rank} | {layer} | {decimal(values['silhouette_pca2'])} | "
            f"{decimal(values['fisher_ratio'])} | {percent(values['pca_explained_variance_2d'])} |"
        )

    lines.extend(
        [
            "",
            "## Control-Subset Probe Robustness",
            "",
            "| Subset | Pairs | Accuracy mean +/- std | AUC mean +/- std |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("primary_all", "standard_monitoring", "lexical_matched"):
        row = probe[name]
        lines.append(
            f"| {name} | {row['n_pairs']} | "
            f"{percent(row['accuracy']['mean'])} +/- {percent(row['accuracy']['std'])} | "
            f"{percent(row['auc']['mean'])} +/- {percent(row['auc']['std'])} |"
        )

    lines.extend(
        [
            "",
            "## Cross-Control Transfer",
            "",
            "| Train -> Test | Accuracy | AUC | P(watched) gap |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("standard_to_lexical", "lexical_to_standard"):
        row = transfer[name]
        lines.append(
            f"| {name} | {percent(row['accuracy'])} | {percent(row['auc'])} | "
            f"{decimal(row['probability_gap'])} |"
        )

    style = transfer["primary_to_style_only_negative_control"]
    style_normal = style["by_label"]["normal"]
    style_variant = style["by_label"]["watched_or_variant"]
    lines.extend(
        [
            "",
            "## Style-Only Negative Control",
            f"- Normal-side mean P(watched): {decimal(style_normal['mean_p_watched'])}",
            f"- Style-variant mean P(watched): {decimal(style_variant['mean_p_watched'])}",
            f"- Style-control probability gap: {decimal(style['probability_gap'])}",
            f"- Style-variant predicted-watched rate: {percent(style_variant['rate_predicted_watched'])}",
            "",
            "## Patching Matrix",
            "",
            "| Layer | Watched-to-normal shift | Normal-to-watched shift | Directional asymmetry |",
            "|---:|---:|---:|---:|",
        ]
    )

    for layer in sorted(patching["overall_by_layer"], key=lambda value: int(value)):
        row = patching["overall_by_layer"][layer]
        lines.append(
            f"| {layer} | {decimal(row['watched_to_normal']['mean'])} | "
            f"{decimal(row['normal_to_watched']['mean'])} | "
            f"{decimal(row['directional_asymmetry']['mean'])} |"
        )

    if any(patching.get("degeneracy_by_direction", {}).values()):
        lines.extend(
            [
                "",
                "**Patching warning:** one or more patching directions are numerically flat across sampled layers. "
                "Treat causal claims as blocked until the intervention is rerun with shape-preserving patching.",
            ]
        )

    lines.extend(
        [
            "",
            "## Length-Confound Audit",
            f"- Overall mean char gap: {decimal(confounds['overall']['mean_char_gap'], 2)}",
            f"- Char gap vs max activation distance: {decimal(confounds['overall']['char_gap_vs_activation'])}",
            f"- Word gap vs max activation distance: {decimal(confounds['overall']['word_gap_vs_activation'])}",
            f"- Char gap vs logit diff: {decimal(confounds['overall']['char_gap_vs_logit_diff'])}",
            "",
            "### By Control Type",
            "",
            "| Control | Pairs | Mean char gap | Char gap vs activation | Char gap vs logit diff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for control_type, row in confounds["by_control_type"].items():
        lines.append(
            f"| {control_type} | {row['pairs']} | {decimal(row['mean_char_gap'], 2)} | "
            f"{decimal(row['char_gap_vs_activation'])} | {decimal(row['char_gap_vs_logit_diff'])} |"
        )

    lines.extend(["", "## Interpretation"])
    primary_acc = probe["primary_all"]["accuracy"]["mean"]
    lexical_acc = probe["lexical_matched"]["accuracy"]["mean"]
    style_gap = style["probability_gap"]
    char_corr = confounds["overall"]["char_gap_vs_activation"]

    if primary_acc is not None and primary_acc >= 0.85:
        lines.append("- Primary pair-grouped separability remains strong under a 20-seed stress audit.")
    if lexical_acc is not None and lexical_acc >= 0.75:
        lines.append("- Lexical-matched controls are still probe-detectable, which weakens a pure wording-gap explanation.")
    else:
        lines.append("- Lexical-matched controls remain the pressure point; expand this subset before making strong claims.")
    if style_gap is not None and abs(style_gap) < 0.20:
        lines.append("- Style-only variants do not strongly trigger the watched detector, which is the desired negative-control pattern.")
    else:
        lines.append("- Style-only controls trigger a noticeable watched-probability shift; treat prompt-form confounding as unresolved.")
    if char_corr is not None and abs(char_corr) < 0.35:
        lines.append("- Prompt length gap is not strongly correlated with max activation distance in the current sample.")
    else:
        lines.append("- Prompt length remains a plausible confound; add stricter length-matched pairs.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment() -> dict[str, Any]:
    print("=" * 60)
    print("Experiment 003: Mechanistic Stress Audit")
    print("=" * 60)

    exp001 = load_exp001()
    table = build_activation_table(exp001)
    activations = table["activations"]
    labels = table["labels"]
    pair_groups = table["pair_groups"]
    sample_meta = table["sample_meta"]

    primary_idx = indices_where(sample_meta, lambda meta: meta["include_in_probe"])
    standard_idx = indices_where(sample_meta, lambda meta: meta["include_in_probe"] and meta["control_type"] == "standard_monitoring")
    lexical_idx = indices_where(sample_meta, lambda meta: meta["include_in_probe"] and meta["control_type"] == "lexical_matched")
    style_idx = indices_where(sample_meta, lambda meta: not meta["include_in_probe"] and meta["control_type"] == "style_only")

    print(f"[Data] samples: {len(activations)}, pairs: {len(np.unique(pair_groups))}")
    print(f"[Data] primary examples: {len(primary_idx)}, style-control examples: {len(style_idx)}")

    geometry = geometry_by_layer(activations, labels, primary_idx)
    probe_robustness = {
        "primary_all": probe_seed_sweep("primary_all", activations, labels, pair_groups, primary_idx),
        "standard_monitoring": probe_seed_sweep("standard_monitoring", activations, labels, pair_groups, standard_idx),
        "lexical_matched": probe_seed_sweep("lexical_matched", activations, labels, pair_groups, lexical_idx),
    }

    transfer = {
        "standard_to_lexical": probe_transfer(
            "standard_to_lexical", activations, labels, pair_groups, standard_idx, lexical_idx
        ),
        "lexical_to_standard": probe_transfer(
            "lexical_to_standard", activations, labels, pair_groups, lexical_idx, standard_idx
        ),
        "primary_to_style_only_negative_control": probe_transfer(
            "primary_to_style_only_negative_control", activations, labels, pair_groups, primary_idx, style_idx
        ),
    }

    results = {
        "metadata": {
            "source_exp001": str(EXP001_PATH),
            "n_examples": int(len(activations)),
            "n_pairs": int(len(np.unique(pair_groups))),
            "control_counts": dict(Counter(meta["control_type"] for meta in sample_meta[::2])),
            "combined_layers": COMBINED_LAYERS,
            "seeds": SEEDS,
            "test_fraction": TEST_FRACTION,
        },
        "representation_geometry": geometry,
        "probe_robustness": probe_robustness,
        "cross_control_transfer": transfer,
        "patching_matrix": patching_matrix(exp001),
        "length_confound_audit": length_confounds(exp001),
    }

    results_path = SAVE_DIR / "exp003_results.json"
    summary_path = SAVE_DIR / "exp003_summary.md"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(results), f, indent=2)
    write_summary(results, summary_path)

    primary = probe_robustness["primary_all"]["accuracy"]
    style = transfer["primary_to_style_only_negative_control"]
    print(f"[Probe] primary accuracy mean+/-std: {primary['mean']:.3f} +/- {primary['std']:.3f}")
    print(f"[Control] style-only P(watched) gap: {style['probability_gap']:.3f}")
    print(f"[Output] wrote {results_path}")
    print(f"[Output] wrote {summary_path}")

    return results


if __name__ == "__main__":
    run_experiment()
