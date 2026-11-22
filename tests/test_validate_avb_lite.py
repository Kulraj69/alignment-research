"""Tests for scripts/validate_avb_lite.py (stdlib only: python -m unittest tests.test_validate_avb_lite)."""

import io
import json
import importlib.util
import unittest
import subprocess
import sys
import tempfile
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


class InputErrorsTest(unittest.TestCase):
    def test_nonobject_json_lines(self):
        for text in ("[]", "null", "1", '"text"', "true"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rows.jsonl"
                path.write_text(text + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Line 1: expected a JSON object"):
                    mod.load_jsonl(path)

    def test_invalid_field_types_report_errors_and_allow_summary(self):
        for field in ("id", "task_type", "normal_prompt", "watched_prompt",
                      "control_type", "contrast_type"):
            for value in (None, 12, [], {}, "", False):
                with self.subTest(field=field, value=value):
                    row = valid_row(_line_number=1)
                    row[field] = value
                    errors, warnings = mod.validate_rows([row])
                    self.assertTrue(errors)
                    with redirect_stdout(io.StringIO()):
                        mod.print_summary([row, valid_row(_line_number=2)], warnings)

    def test_invalid_lexical_prompts_report_errors_without_length_crash(self):
        for field in ("normal_prompt", "watched_prompt"):
            for value in (None, 123):
                with self.subTest(field=field, value=value):
                    row = valid_row(_line_number=1, contrast_type="lexical_matched_monitoring")
                    row[field] = value
                    errors, _ = mod.validate_rows([row])
                    self.assertTrue(any(field in error for error in errors))

    def test_cli_bad_json_and_nonobjects_have_no_traceback(self):
        for text in ("{bad", "[]", "null", "1", '"text"'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "rows.jsonl"
                path.write_text(text + "\n", encoding="utf-8")
                result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("Line 1", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_cli_invalid_field_types_have_no_traceback(self):
        for field in ("id", "task_type", "control_type", "contrast_type"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                row = valid_row()
                row[field] = []
                path = Path(directory) / "rows.jsonl"
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("Errors:", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_cli_missing_file_has_no_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent.jsonl"
            result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_blank_lines_keep_physical_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("\n\n[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Line 3:"):
                mod.load_jsonl(path)


class IdDiagnosticsTest(unittest.TestCase):
    def test_invalid_ids_use_line_label_for_every_diagnostic(self):
        for value in (None, 12, [], {}, "", "   ", False):
            with self.subTest(value=value):
                row = valid_row(id=value, _line_number=23, task_type="unsupported_task",
                                normal_prompt=[], watched_prompt=None,
                                control_type=False, contrast_type={})
                del row["hypothesis"]
                errors, warnings = mod.validate_rows([row])
                self.assertIn("line_23: id must be a non-empty string", errors)
                self.assertTrue(warnings)
                self.assertTrue(all(message.startswith("line_23:")
                                    for message in errors + warnings))

    def test_missing_ids_do_not_reserve_line_labels(self):
        for missing_first in (True, False):
            with self.subTest(missing_first=missing_first):
                missing = valid_row(_line_number=23)
                del missing["id"]
                named = valid_row(id="line_23", _line_number=24)
                rows = [missing, named] if missing_first else [named, missing]
                errors, _ = mod.validate_rows(rows)
                self.assertEqual(errors, ["line_23: missing required fields: ['id']"])

    def test_cli_invalid_id_diagnostics_use_physical_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = valid_row(id=[], task_type="unsupported_task")
            path.write_text("\n\n" + json.dumps(row) + "\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("line_3: id must be a non-empty string", result.stdout)
            for section in ("Warnings:", "Errors:"):
                messages = result.stdout.split(section, 1)[1].split("\n\n", 1)[0]
                self.assertTrue(messages.strip())
                for message in messages.strip().splitlines():
                    self.assertTrue(message.startswith("- line_3:"), message)


class RealDatasetTest(unittest.TestCase):
    def test_shipped_dataset_is_valid(self):
        dataset = SCRIPT.parent.parent / "avb_lite" / "real_world_pairs_v1.jsonl"
        rows = mod.load_jsonl(dataset)
        errors, _ = mod.validate_rows(rows)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
