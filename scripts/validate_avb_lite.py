#!/usr/bin/env python3
"""Validate AVB-lite JSONL prompt pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "task_type",
    "normal_prompt",
    "watched_prompt",
    "target_output",
    "hypothesis",
}

VALID_TASK_TYPES = {
    "evaluation_awareness",
    "honesty_vs_appearing_honest",
    "refusal_consistency",
    "preference_consistency",
    "hidden_trigger",
}

SOURCE_FIELDS = {"source_url", "source_date", "risk_theme"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for row in rows:
        line_number = row["_line_number"]
        row_id = row.get("id", f"line_{line_number}")

        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            errors.append(f"{row_id}: missing required fields: {missing}")

        if row_id in seen_ids:
            errors.append(f"{row_id}: duplicate id")
        seen_ids.add(row_id)

        task_type = row.get("task_type")
        if task_type not in VALID_TASK_TYPES:
            errors.append(f"{row_id}: unknown task_type {task_type!r}")

        for field in ("normal_prompt", "watched_prompt"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{row_id}: {field} must be a non-empty string")

        missing_source = sorted(SOURCE_FIELDS - row.keys())
        if missing_source:
            warnings.append(f"{row_id}: missing source metadata: {missing_source}")

        contrast_type = row.get("contrast_type", "semantic_monitoring")
        include_in_probe = row.get("include_in_probe", True)
        if contrast_type == "style_only_control" and include_in_probe:
            errors.append(f"{row_id}: style-only controls must set include_in_probe=false")

        if contrast_type == "lexical_matched_monitoring":
            normal_len = len(row.get("normal_prompt", ""))
            watched_len = len(row.get("watched_prompt", ""))
            length_gap = abs(normal_len - watched_len)
            if length_gap > 48:
                warnings.append(f"{row_id}: lexical control length gap is {length_gap} chars")

    return errors, warnings


def print_summary(rows: list[dict[str, Any]], warnings: list[str]) -> None:
    task_counts = Counter(row["task_type"] for row in rows)
    control_counts = Counter(row.get("control_type", "standard_monitoring") for row in rows)
    contrast_counts = Counter(row.get("contrast_type", "semantic_monitoring") for row in rows)
    probe_counts = Counter("included" if row.get("include_in_probe", True) else "excluded" for row in rows)
    sourced = sum(1 for row in rows if SOURCE_FIELDS.issubset(row.keys()))

    print("AVB-lite validation summary")
    print(f"- Rows: {len(rows)}")
    print(f"- Source-tagged rows: {sourced}/{len(rows)}")
    print(f"- Task counts: {dict(sorted(task_counts.items()))}")
    print(f"- Control counts: {dict(sorted(control_counts.items()))}")
    print(f"- Contrast counts: {dict(sorted(contrast_counts.items()))}")
    print(f"- Probe inclusion: {dict(sorted(probe_counts.items()))}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AVB-lite JSONL prompt pairs.")
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("avb_lite/real_world_pairs_v1.jsonl"),
        help="Path to AVB-lite JSONL file.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)
    errors, warnings = validate_rows(rows)
    print_summary(rows, warnings)

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
