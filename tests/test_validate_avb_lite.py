"""Tests for scripts/validate_avb_lite.py (stdlib only: python -m unittest tests.test_validate_avb_lite)."""

import io
import importlib.util
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_avb_lite.py"
spec = importlib.util.spec_from_file_location("validate_avb_lite", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def valid_row(**overrides):
    row = {
        "id": "row-1",
        "task_type": "evaluation_awareness",
        "normal_prompt": "normal",
        "watched_prompt": "watched",
        "target_output": "target",
        "hypothesis": "hypothesis",
    }
    row.update(overrides)
    return row


class ValidateRowsTest(unittest.TestCase):
    def test_valid_row_has_no_errors(self):
        errors, _ = mod.validate_rows([valid_row(_line_number=1)])
        self.assertEqual(errors, [])

    def test_missing_task_type_is_reported_not_crash(self):
        row = valid_row(_line_number=1)
        del row["task_type"]
        errors, _ = mod.validate_rows([row])
        self.assertTrue(any("missing required fields" in e for e in errors))
        # print_summary must not raise KeyError on the invalid row.
        with redirect_stdout(io.StringIO()):
            mod.print_summary([row], [])

    def test_style_only_control_must_be_excluded_from_probe(self):
        errors, _ = mod.validate_rows(
            [valid_row(_line_number=1, contrast_type="style_only_control", include_in_probe=True)]
        )
        self.assertTrue(any("include_in_probe=false" in e for e in errors))

    def test_duplicate_ids_flagged(self):
        errors, _ = mod.validate_rows([valid_row(_line_number=1), valid_row(_line_number=2)])
        self.assertTrue(any("duplicate id" in e for e in errors))


class RealDatasetTest(unittest.TestCase):
    def test_shipped_dataset_is_valid(self):
        dataset = SCRIPT.parent.parent / "avb_lite" / "real_world_pairs_v1.jsonl"
        rows = mod.load_jsonl(dataset)
        errors, _ = mod.validate_rows(rows)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
