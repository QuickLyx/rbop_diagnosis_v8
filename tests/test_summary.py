# -*- coding: utf-8 -*-
"""固定144格清单及空结果不虚报完成。"""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

from oneclass import load_config

SPEC = importlib.util.spec_from_file_location("summary_stage3", Path(__file__).resolve().parents[1] / "scripts/summarize_stage3.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummaryTests(unittest.TestCase):
    def test_empty_matrix_is_not_complete(self):
        config = load_config()
        result = MODULE.summarize(config, {})
        self.assertEqual(result["main_completed"], 0)
        self.assertEqual(len(result["main_missing"]), 144)
        self.assertEqual(len({(r["arm"], r["fold"], r["seed"]) for r in result["main_missing"]}), 144)
        self.assertEqual(config["counts"]["gradient_runs"], 96)
        self.assertEqual(config["counts"]["fusion_ablation_outputs"], 48)

    def test_partial_97_units_never_count_as_complete(self):
        import evaluate
        rows = [{"dataset": "paderborn", "unit_key": f"K{i}/c", "label": 0,
                 "probabilities": [1.0, 0, 0, 0]} for i in range(97)]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(evaluate, "read_predictions", return_value=rows):
                with self.assertRaisesRegex(ValueError, "116"):
                    MODULE.validate_report(Path(tmp), {"evaluation_splits": {"all_primary": [r["unit_key"] for r in rows]}},
                                           {"D3": {"n": 97, "E1": 1, "E2": 1, "passed": None}})

    def test_real_csv_full_116_coverage(self):
        import evaluate
        from bearing_facts import BY_ID, CONDITIONS, PRIMARY_BEARINGS
        records = []
        for bearing in PRIMARY_BEARINGS:
            fact = BY_ID[bearing]
            for condition in CONDITIONS:
                unit = f"{bearing}/{condition}"
                n = 19 if unit == "KA08/N15_M01_F10" else 20
                for rep in range(n):
                    filename = f"{condition}_{bearing}_{rep+1}.mat"
                    p = [0.0]*4
                    p[fact.label] = 1.0
                    offset = len(records)
                    records.append({"unit_key": unit, "record_key": f"{bearing}/{filename}::vibration_1",
                        "path": filename, "channel": "vibration_1", "dataset": "paderborn", "source": "paderborn",
                        "bearing_id": bearing, "condition": condition, "origin": fact.origin, "severity": fact.severity,
                        "src_fs": 64000, "label": fact.label, "start": offset, "stop": offset+1,
                        "window_count": 1, "probabilities": p})
        units = evaluate.aggregate_units(records)
        info = {"evaluation_splits": {"all_primary": [r["unit_key"] for r in units]}}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            evaluate._write_predictions(records, folder/"predictions.csv")
            evaluate._write_predictions(units, folder/"unit_predictions.csv", unit=True)
            MODULE.validate_report(folder, info, {"D3": {"n": 116, "E1": 1, "E2": 1, "passed": True}})

    def test_missing_auxiliary_tasks_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(MODULE, "ROOT", Path(tmp)), patch.object(MODULE, "project_path", side_effect=lambda p: Path(tmp)/p):
                tasks = MODULE.auxiliary_status(load_config())
        self.assertEqual(len(tasks), 6)
        self.assertTrue(all(not task["completed"] for task in tasks.values()))

    def test_training_and_eval_completion_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(MODULE, "ROOT", Path(tmp)):
                tasks = MODULE.training_status(load_config(), {})
        self.assertEqual(len(tasks["cells"]), 120)
        self.assertEqual(tasks["gradient_completed"], 0)
        self.assertEqual(tasks["adabn_completed"], 0)
        self.assertEqual(tasks["shallow_completed"], 0)

    def test_sample_std(self):
        result = MODULE.statistics([1, 3])
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["std_ddof1"], 2**0.5)
        self.assertIsNone(MODULE.statistics([])["mean"])


if __name__ == "__main__":
    unittest.main()
