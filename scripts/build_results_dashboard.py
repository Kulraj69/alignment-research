#!/usr/bin/env python3
"""Build a compact Markdown dashboard from CVAT experiment artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Optional


REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATASET = REPO_ROOT / "avb_lite" / "real_world_pairs_v1.jsonl"
DEFAULT_EXP001 = REPO_ROOT / "experiments" / "results" / "exp001" / "exp001_results.json"
DEFAULT_EXP002 = REPO_ROOT / "experiments" / "results" / "exp002" / "exp002_results.json"
DEFAULT_SIGNAL_DIR = REPO_ROOT / "data" / "real_world"
DEFAULT_OUT = REPO_ROOT / "reports" / "cvat_dashboard_latest.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def latest_signal_file(signal_dir: Path) -> Optional[Path]:
    candidates = sorted(signal_dir.glob("alignment_signals_*.json"))
    return candidates[-1] if candidates else None


def percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def decimal(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def count_rows(rows: list[dict[str, Any]], field: str, default: str) -> dict[str, int]:
    counts = Counter(row.get(field, default) for row in rows)
    return dict(sorted(counts.items()))


def summarize_exp001(exp001: dict[str, Any]) -> dict[str, Any]:
    if not exp001:
        return {}

    aggregate = exp001.get("aggregate", {})
    per_example = exp001.get("per_example", [])
    by_control: dict[str, list[dict[str, float]]] = defaultdict(list)

    for example in per_example:
        control_type = example.get("control_type", "standard_monitoring")
        layer_distances = example.get("layer_distances", {})
        max_distance = 0.0
        if layer_distances:
            max_distance = max(float(layer["euclidean"]) for layer in layer_distances.values())
        logit_diff = float(example.get("logit_diff", {}).get("top_k_logit_diff", 0.0))
        by_control[control_type].append({"max_distance": max_distance, "logit_diff": logit_diff})

    control_summary = {}
    for control_type, rows in sorted(by_control.items()):
        control_summary[control_type] = {
            "pairs": len(rows),
            "mean_max_activation_distance": mean(row["max_distance"] for row in rows),
            "mean_logit_diff": mean(row["logit_diff"] for row in rows),
        }

    return {
        "prompt_pairs": exp001.get("metadata", {}).get("n_prompt_pairs"),
        "mean_logit_diff": aggregate.get("mean_logit_diff"),
        "std_logit_diff": aggregate.get("std_logit_diff"),
        "signal_layers": aggregate.get("signal_layers", []),
        "control_summary": control_summary,
    }


def summarize_exp002(exp002: dict[str, Any]) -> dict[str, Any]:
    if not exp002:
        return {}

    metadata = exp002.get("metadata", {})
    aggregate = exp002.get("aggregate", {})
    pair_grouped = exp002.get("splits", {}).get("pair_grouped", {})
    seed_sweep = exp002.get("seed_sweep_pair_grouped", {}).get("aggregate", {})

    return {
        "source_pairs": metadata.get("source_n_pairs"),
        "probe_pairs": metadata.get("n_pairs"),
        "excluded_from_probe": metadata.get("excluded_from_probe"),
        "best_layer": aggregate.get("best_layer"),
        "best_layer_accuracy": aggregate.get("best_layer_accuracy"),
        "combined_accuracy": aggregate.get("combined_layer_accuracy"),
        "seed_sweep_combined_mean": seed_sweep.get("combined_mean"),
        "seed_sweep_combined_std": seed_sweep.get("combined_std"),
        "pair_overlap": pair_grouped.get("split_meta", {}).get("pair_overlap"),
    }


def build_markdown(
    dataset_rows: list[dict[str, Any]],
    exp001: dict[str, Any],
    exp002: dict[str, Any],
    signals: dict[str, Any],
    signal_path: Optional[Path],
) -> str:
    today = dt.date.today().isoformat()
    exp001_summary = summarize_exp001(exp001)
    exp002_summary = summarize_exp002(exp002)

    probe_included = sum(1 for row in dataset_rows if row.get("include_in_probe", True))
    probe_excluded = len(dataset_rows) - probe_included
    signal_counts = signals.get("counts", {})
    top_signal = (signals.get("top_signals") or [{}])[0]

    lines = [
        f"# CVAT Research Dashboard ({today})",
        "",
        "## Dataset",
        f"- AVB-lite pairs: {len(dataset_rows)}",
        f"- Primary-probe pairs: {probe_included}",
        f"- Style/control-only pairs excluded from primary probe: {probe_excluded}",
        f"- Task counts: {count_rows(dataset_rows, 'task_type', 'unknown')}",
        f"- Control counts: {count_rows(dataset_rows, 'control_type', 'standard_monitoring')}",
        f"- Contrast counts: {count_rows(dataset_rows, 'contrast_type', 'semantic_monitoring')}",
        "",
        "## Exp001 Activation Shift",
        f"- Prompt pairs run: {exp001_summary.get('prompt_pairs', 'n/a')}",
        f"- Mean logit diff: {decimal(exp001_summary.get('mean_logit_diff'))}",
        f"- Std logit diff: {decimal(exp001_summary.get('std_logit_diff'))}",
        f"- Signal layers: {exp001_summary.get('signal_layers', [])}",
        "",
        "### Exp001 By Control Type",
    ]

    control_summary = exp001_summary.get("control_summary", {})
    if control_summary:
        for control_type, summary in control_summary.items():
            lines.append(
                "- "
                f"{control_type}: {summary['pairs']} pairs, "
                f"mean max activation distance {summary['mean_max_activation_distance']:.4f}, "
                f"mean logit diff {summary['mean_logit_diff']:.4f}"
            )
    else:
        lines.append("- No Exp001 control breakdown available yet.")

    lines.extend(
        [
            "",
            "## Exp002 Probe",
            f"- Source pairs: {exp002_summary.get('source_pairs', 'n/a')}",
            f"- Primary-probe pairs: {exp002_summary.get('probe_pairs', 'n/a')}",
            f"- Excluded from primary probe: {exp002_summary.get('excluded_from_probe', 'n/a')}",
            f"- Pair overlap in primary split: {exp002_summary.get('pair_overlap', 'n/a')}",
            f"- Best layer: {exp002_summary.get('best_layer', 'n/a')}",
            f"- Best-layer accuracy: {percent(exp002_summary.get('best_layer_accuracy'))}",
            f"- Combined-layer accuracy: {percent(exp002_summary.get('combined_accuracy'))}",
            (
                "- Pair-grouped seed sweep combined accuracy: "
                f"{percent(exp002_summary.get('seed_sweep_combined_mean'))} +/- "
                f"{percent(exp002_summary.get('seed_sweep_combined_std'))}"
            ),
            "",
            "## External Safety Signals",
            f"- Latest signal artifact: {signal_path.relative_to(REPO_ROOT) if signal_path else 'n/a'}",
            f"- Source counts: {signal_counts}",
            f"- Top signal: {top_signal.get('title', 'n/a')}",
            "",
            "## Current Gates",
            "- Dataset milestone: 50-pair AVB-lite reached; next target is 200 balanced pairs.",
            "- Confound-control milestone: lexical-matched and style-only control rows now exist.",
            "- Probe rigor: pair-grouped split remains primary; style-only controls are excluded from the main watched/unwatched probe.",
            "- Next empirical step: run the same protocol on a larger open model after GPT-2 controls are stable.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CVAT Markdown dashboard.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--exp001", type=Path, default=DEFAULT_EXP001)
    parser.add_argument("--exp002", type=Path, default=DEFAULT_EXP002)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signal_path = latest_signal_file(args.signal_dir)
    signals = load_json(signal_path) if signal_path else {}
    markdown = build_markdown(
        dataset_rows=load_jsonl(args.dataset),
        exp001=load_json(args.exp001),
        exp002=load_json(args.exp002),
        signals=signals,
        signal_path=signal_path,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
