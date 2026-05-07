#!/usr/bin/env python3
"""
Experiment 002: Probe Training — Watched vs Unwatched Classification

Core question: Can a linear probe classify whether a prompt is watched or
unwatched from internal activations alone?

This version reports two split protocols:
1) sample_stratified: random train/test over individual samples
2) pair_grouped: keeps both prompts from each pair in the same split

The pair-grouped split is the primary metric because it better controls for
pair leakage.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy and tensor values."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path(__file__).parent.parent.parent
SAVE_DIR = REPO_ROOT / "experiments" / "results" / "exp002"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LAYERS_TO_PROBE = list(range(12))
COMBINED_LAYERS = [4, 5, 6, 7]
RANDOM_SEED = 42
TEST_FRACTION = 0.25
SEED_SWEEP = list(range(10))

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def generate_synthetic_dataset(n_pairs=100, n_layers=12, d_model=768, seed=42):
    """Generate synthetic paired activations when real data is unavailable."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    activations = []
    labels = []
    pair_groups = []
    sample_meta = []

    for pair_idx in range(n_pairs):
        normal = torch.randn(n_layers, d_model) * 0.8
        normal[4:8] *= 1.2

        watched = torch.randn(n_layers, d_model) * 0.8
        watched[4:8] *= 1.5
        watched[5:7] += torch.randn(d_model) * 0.3

        activations.append(normal)
        labels.append(0)
        pair_groups.append(pair_idx)
        sample_meta.append({"pair_id": f"synthetic_{pair_idx:03d}", "label": "normal"})

        activations.append(watched)
        labels.append(1)
        pair_groups.append(pair_idx)
        sample_meta.append({"pair_id": f"synthetic_{pair_idx:03d}", "label": "watched"})

    return {
        "activations": torch.stack(activations),
        "labels": torch.tensor(labels, dtype=torch.long),
        "pair_groups": np.array(pair_groups, dtype=np.int64),
        "sample_meta": sample_meta,
        "data_mode": "synthetic",
    }


def load_or_generate_data():
    """Load real activations from Exp001, or generate synthetic data."""
    exp001_results = REPO_ROOT / "experiments" / "results" / "exp001" / "exp001_results.json"

    print(f"[Data] Exp001 path: {exp001_results}")
    print(f"[Data] Exists: {exp001_results.exists()}")

    if exp001_results.exists():
        print("[Data] Loading cached activations from Exp001")
        with open(exp001_results, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_example = data.get("per_example", [])
        has_real_activations = bool(per_example) and "mean_pooled_activations" in per_example[0]

        if has_real_activations:
            all_activations = []
            all_labels = []
            pair_groups = []
            sample_meta = []

            for pair_idx, example in enumerate(per_example):
                pair_id = example.get("id", f"pair_{pair_idx:03d}")
                normal_layers = []
                watched_layers = []

                for layer in LAYERS_TO_PROBE:
                    layer_key = str(layer)
                    layer_data = example["mean_pooled_activations"][layer_key]
                    normal_layers.append(layer_data["normal"])
                    watched_layers.append(layer_data["watched"])

                all_activations.append(normal_layers)
                all_labels.append(0)
                pair_groups.append(pair_idx)
                sample_meta.append({"pair_id": pair_id, "label": "normal"})

                all_activations.append(watched_layers)
                all_labels.append(1)
                pair_groups.append(pair_idx)
                sample_meta.append({"pair_id": pair_id, "label": "watched"})

            activations = torch.tensor(all_activations, dtype=torch.float32)
            labels = torch.tensor(all_labels, dtype=torch.long)

            print(f"[Data] Loaded {len(per_example)} prompt pairs from Exp001")
            print(f"[Data] Expanded to {len(activations)} examples (real normal + real watched)")
            print(f"[Data] Watched: {labels.sum().item()}, Normal: {(~labels.bool()).sum().item()}")

            return {
                "activations": activations,
                "labels": labels,
                "pair_groups": np.array(pair_groups, dtype=np.int64),
                "sample_meta": sample_meta,
                "data_mode": "real",
            }

        print("[Data] Exp001 found, but no saved real activations. Falling back to synthetic data.")
    else:
        print("[Data] No cached activations found, generating synthetic data")

    dataset = generate_synthetic_dataset()
    labels = dataset["labels"]
    print(
        f"[Data] Generated {len(dataset['activations'])} examples: "
        f"{labels.sum().item()} watched, {(~labels.bool()).sum().item()} normal"
    )
    return dataset


def extract_layer_features(activations, layer):
    """Extract [n_examples, d_model] features from one layer."""
    layer_activations = activations[:, layer, :]
    if hasattr(layer_activations, "numpy"):
        return layer_activations.numpy()
    return layer_activations


def fit_probe(X_train, y_train, X_test, y_test, seed, return_details=True):
    """Fit logistic probe and return accuracy (+details optionally)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe.fit(X_train_scaled, y_train)

    y_pred = probe.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    if not return_details:
        return {"accuracy": float(accuracy)}

    return {
        "accuracy": float(accuracy),
        "coefficients": probe.coef_[0],
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.scale_,
    }


def train_all_layers(activations, labels, train_idx, test_idx, seed, verbose=True, with_details=True):
    """Train one probe per layer for a fixed split."""
    results = {}

    X_all = activations.numpy()
    y_all = labels.numpy()

    X_train_full = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    if verbose:
        print(f"[Probe] Train examples: {len(train_idx)}, Test examples: {len(test_idx)}")
        print(f"[Probe] Class balance — train: {int(y_train.sum())}/{len(y_train)}, test: {int(y_test.sum())}/{len(y_test)}")

    best_layer = None
    best_accuracy = -1.0

    iterator = tqdm(LAYERS_TO_PROBE, desc="Training probes") if verbose else LAYERS_TO_PROBE
    for layer in iterator:
        X_train = extract_layer_features(X_train_full, layer)
        X_test_layer = extract_layer_features(X_test, layer)

        result = fit_probe(X_train, y_train, X_test_layer, y_test, seed=seed, return_details=with_details)
        result["layer"] = layer
        results[layer] = result

        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_layer = layer

        if verbose:
            marker = " ← BEST" if layer == best_layer else ""
            print(f"  Layer {layer:2d}: accuracy = {result['accuracy']:.3f}{marker}")

    return results, best_layer


def analyze_probe_coefficients(layer_results, best_layer, top_k=20):
    """Summarize top positive/negative coefficients for the best layer."""
    best_result = layer_results[best_layer]
    coefs = np.array(best_result["coefficients"])

    top_positive_idx = np.argsort(coefs)[-top_k:][::-1]
    top_negative_idx = np.argsort(coefs)[:top_k]

    return {
        "best_layer": int(best_layer),
        "top_watched_features": top_positive_idx.tolist(),
        "top_normal_features": top_negative_idx.tolist(),
        "coef_magnitude_mean": float(np.abs(coefs).mean()),
        "coef_magnitude_std": float(np.abs(coefs).std()),
    }


def train_combined_probe(activations, labels, train_idx, test_idx, seed, with_details=True):
    """Train probe on concatenated features from COMBINED_LAYERS."""
    X_all = activations.numpy()
    y_all = labels.numpy()

    X_train_list = [X_all[train_idx][:, layer, :] for layer in COMBINED_LAYERS]
    X_test_list = [X_all[test_idx][:, layer, :] for layer in COMBINED_LAYERS]
    X_train = np.hstack(X_train_list)
    X_test = np.hstack(X_test_list)

    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    result = fit_probe(X_train, y_train, X_test, y_test, seed=seed, return_details=with_details)
    result["layers"] = COMBINED_LAYERS
    return result


def build_sample_stratified_split(labels, seed, test_fraction):
    """Random stratified split over individual samples."""
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def build_pair_group_split(pair_groups, seed, test_fraction):
    """Split by pair id so both samples from a pair stay together."""
    unique_pairs = np.unique(pair_groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_pairs.copy()
    rng.shuffle(shuffled)

    n_test_pairs = max(1, int(round(len(unique_pairs) * test_fraction)))
    n_test_pairs = min(n_test_pairs, len(unique_pairs) - 1)

    test_pairs = set(shuffled[:n_test_pairs].tolist())
    test_mask = np.isin(pair_groups, list(test_pairs))

    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def summarize_split(labels_np, pair_groups, train_idx, test_idx):
    """Return split metadata and sanity checks."""
    train_pairs = set(pair_groups[train_idx].tolist())
    test_pairs = set(pair_groups[test_idx].tolist())
    overlap = train_pairs.intersection(test_pairs)

    y_train = labels_np[train_idx]
    y_test = labels_np[test_idx]

    return {
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "train_watched": int(y_train.sum()),
        "test_watched": int(y_test.sum()),
        "train_pairs": int(len(train_pairs)),
        "test_pairs": int(len(test_pairs)),
        "pair_overlap": int(len(overlap)),
    }


def evaluate_split(split_name, activations, labels, pair_groups, train_idx, test_idx, seed, verbose=True):
    """Run full per-layer + combined evaluation for one split."""
    if verbose:
        print("\n" + "-" * 60)
        print(f"Split: {split_name}")
        print("-" * 60)

    labels_np = labels.numpy()
    split_meta = summarize_split(labels_np, pair_groups, train_idx, test_idx)
    if verbose:
        print(
            f"[Split] pairs train/test: {split_meta['train_pairs']}/{split_meta['test_pairs']}, "
            f"overlap: {split_meta['pair_overlap']}"
        )

    layer_results, best_layer = train_all_layers(
        activations,
        labels,
        train_idx,
        test_idx,
        seed=seed,
        verbose=verbose,
        with_details=True,
    )

    combined = train_combined_probe(
        activations,
        labels,
        train_idx,
        test_idx,
        seed=seed,
        with_details=True,
    )

    interpretation = analyze_probe_coefficients(layer_results, best_layer)
    if verbose:
        print(f"\n[Split {split_name}] best layer: {best_layer} ({layer_results[best_layer]['accuracy']:.3f})")
        print(f"[Split {split_name}] combined ({COMBINED_LAYERS}) accuracy: {combined['accuracy']:.3f}")

    return {
        "split_meta": split_meta,
        "best_layer": int(best_layer),
        "best_layer_accuracy": float(layer_results[best_layer]["accuracy"]),
        "per_layer": {str(layer): layer_results[layer] for layer in LAYERS_TO_PROBE},
        "combined_probe": combined,
        "interpretation": interpretation,
    }


def pair_group_seed_sweep(activations, labels, pair_groups, seeds, test_fraction):
    """Run pair-grouped evaluation across many seeds for robustness stats."""
    per_seed = []

    for seed in seeds:
        train_idx, test_idx = build_pair_group_split(pair_groups, seed=seed, test_fraction=test_fraction)

        layer_results, best_layer = train_all_layers(
            activations,
            labels,
            train_idx,
            test_idx,
            seed=seed,
            verbose=False,
            with_details=False,
        )
        combined = train_combined_probe(
            activations,
            labels,
            train_idx,
            test_idx,
            seed=seed,
            with_details=False,
        )

        per_seed.append(
            {
                "seed": int(seed),
                "best_layer": int(best_layer),
                "best_layer_accuracy": float(layer_results[best_layer]["accuracy"]),
                "combined_accuracy": float(combined["accuracy"]),
                "per_layer": {str(layer): float(layer_results[layer]["accuracy"]) for layer in LAYERS_TO_PROBE},
            }
        )

    layer_matrix = np.array(
        [[seed_result["per_layer"][str(layer)] for layer in LAYERS_TO_PROBE] for seed_result in per_seed],
        dtype=np.float64,
    )
    combined_values = np.array([seed_result["combined_accuracy"] for seed_result in per_seed], dtype=np.float64)
    best_values = np.array([seed_result["best_layer_accuracy"] for seed_result in per_seed], dtype=np.float64)

    return {
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "aggregate": {
            "per_layer_mean": {str(layer): float(layer_matrix[:, i].mean()) for i, layer in enumerate(LAYERS_TO_PROBE)},
            "per_layer_std": {str(layer): float(layer_matrix[:, i].std()) for i, layer in enumerate(LAYERS_TO_PROBE)},
            "combined_mean": float(combined_values.mean()),
            "combined_std": float(combined_values.std()),
            "best_layer_acc_mean": float(best_values.mean()),
            "best_layer_acc_std": float(best_values.std()),
        },
    }


def run_experiment():
    """Run full experiment with split comparisons and grouped seed sweep."""
    print("=" * 60)
    print("Experiment 002: Probe Training — Watched vs Unwatched")
    print("=" * 60)

    dataset = load_or_generate_data()
    activations = dataset["activations"]
    labels = dataset["labels"]
    pair_groups = dataset["pair_groups"]
    data_mode = dataset["data_mode"]

    sample_train_idx, sample_test_idx = build_sample_stratified_split(
        labels.numpy(), seed=RANDOM_SEED, test_fraction=TEST_FRACTION
    )
    pair_train_idx, pair_test_idx = build_pair_group_split(
        pair_groups, seed=RANDOM_SEED, test_fraction=TEST_FRACTION
    )

    split_results = {}
    split_results["sample_stratified"] = evaluate_split(
        "sample_stratified",
        activations,
        labels,
        pair_groups,
        sample_train_idx,
        sample_test_idx,
        seed=RANDOM_SEED,
        verbose=True,
    )
    split_results["pair_grouped"] = evaluate_split(
        "pair_grouped",
        activations,
        labels,
        pair_groups,
        pair_train_idx,
        pair_test_idx,
        seed=RANDOM_SEED,
        verbose=True,
    )

    print("\n" + "-" * 60)
    print("Phase: Pair-grouped seed sweep")
    print("-" * 60)
    seed_sweep = pair_group_seed_sweep(
        activations,
        labels,
        pair_groups,
        seeds=SEED_SWEEP,
        test_fraction=TEST_FRACTION,
    )
    print(
        f"[Seed Sweep] combined mean±std: "
        f"{seed_sweep['aggregate']['combined_mean']:.3f} ± {seed_sweep['aggregate']['combined_std']:.3f}"
    )

    primary = split_results["pair_grouped"]

    results = {
        "metadata": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "data_mode": data_mode,
            "n_examples": int(len(activations)),
            "n_pairs": int(len(np.unique(pair_groups))),
            "n_layers": int(len(LAYERS_TO_PROBE)),
            "test_fraction": TEST_FRACTION,
            "random_seed": RANDOM_SEED,
            "seed_sweep": SEED_SWEEP,
            "combined_layers": COMBINED_LAYERS,
        },
        "splits": split_results,
        "seed_sweep_pair_grouped": seed_sweep,
        "aggregate": {
            "primary_split": "pair_grouped",
            "best_layer": primary["best_layer"],
            "best_layer_accuracy": primary["best_layer_accuracy"],
            "combined_layer_accuracy": primary["combined_probe"]["accuracy"],
            "seed_sweep_combined_mean": seed_sweep["aggregate"]["combined_mean"],
            "seed_sweep_combined_std": seed_sweep["aggregate"]["combined_std"],
        },
    }

    results_path = SAVE_DIR / "exp002_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    summary_path = SAVE_DIR / "exp002_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 002 Results\n\n")
        f.write("## Setup\n")
        f.write(f"- Model: {MODEL_NAME}\n")
        f.write(f"- Data mode: {data_mode}\n")
        f.write(f"- Examples: {len(activations)}\n")
        f.write(f"- Pairs: {len(np.unique(pair_groups))}\n")
        f.write(f"- Layers probed: {LAYERS_TO_PROBE}\n")
        f.write(f"- Test fraction: {TEST_FRACTION}\n")
        f.write(f"- Combined layers: {COMBINED_LAYERS}\n\n")

        f.write("## Split Comparison\n\n")
        for split_name in ["sample_stratified", "pair_grouped"]:
            split = split_results[split_name]
            meta = split["split_meta"]
            f.write(f"### {split_name}\n")
            f.write(
                f"- Train/Test examples: {meta['train_examples']}/{meta['test_examples']}\n"
                f"- Train/Test pairs: {meta['train_pairs']}/{meta['test_pairs']}\n"
                f"- Pair overlap: {meta['pair_overlap']}\n"
                f"- Best layer: {split['best_layer']} ({split['best_layer_accuracy']:.1%})\n"
                f"- Combined probe accuracy ({COMBINED_LAYERS}): {split['combined_probe']['accuracy']:.1%}\n\n"
            )

        f.write("## Pair-Grouped Seed Sweep\n\n")
        f.write(f"- Seeds: {SEED_SWEEP}\n")
        f.write(
            f"- Combined probe mean±std: "
            f"**{seed_sweep['aggregate']['combined_mean']:.1%} ± {seed_sweep['aggregate']['combined_std']:.1%}**\n"
        )
        f.write(
            f"- Best-layer accuracy mean±std: "
            f"**{seed_sweep['aggregate']['best_layer_acc_mean']:.1%} ± {seed_sweep['aggregate']['best_layer_acc_std']:.1%}**\n\n"
        )

        f.write("## Interpretation\n")
        f.write(
            "Pair-grouped metrics are the primary estimate because they prevent "
            "same-pair leakage across train/test.\n"
        )
        f.write(
            "Use seed-sweep variability as the uncertainty signal until AVB-lite is expanded.\n"
        )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"Primary (pair_grouped) best layer: {primary['best_layer']} "
        f"({primary['best_layer_accuracy']:.1%})"
    )
    print(
        f"Primary (pair_grouped) combined accuracy: {primary['combined_probe']['accuracy']:.1%}"
    )
    print(
        f"Seed sweep combined mean±std: "
        f"{seed_sweep['aggregate']['combined_mean']:.1%} ± {seed_sweep['aggregate']['combined_std']:.1%}"
    )
    print(f"Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
