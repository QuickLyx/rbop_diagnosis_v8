# -*- coding: utf-8 -*-
"""IES硬阈值、未映射分析窗计数与独立自然留出回归。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import evaluate
import physical_baseline as physical
from oneclass import load_config
from tests.test_selection import pool_payload
import selection


class PhysicalTests(unittest.TestCase):
    def test_threshold_and_positive_soft_evidence(self):
        R = np.zeros((5, 16))
        R[1, :4], R[1, 4:8] = 3, 2
        R[2, :4], R[2, 4:8] = 2, 3
        R[3, :4], R[3, 4:8] = 3.01, 2
        R[4, :4], R[4, 4:8] = 2, 3.01
        hard, q, _, _ = physical.ies_evidence(R)
        self.assertEqual(hard.argmax(axis=1).tolist(), [0, 0, 0, 3, 2])
        self.assertTrue((q > 0).all())
        np.testing.assert_allclose(q.sum(axis=1), 1)
        np.testing.assert_array_equal(q[:, 0], q[:, 1])

    def test_natural_holdouts_may_include_other_pool(self):
        from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
        keys = sorted(f"{b}/{c}" for b in PRIMARY_BEARINGS for c in CONDITIONS)
        old, gg = pool_payload(keys[:19]), pool_payload(keys[19:38])
        paired = selection.paired_splits(old, gg)
        with patch("selection.load_selection", side_effect=[old, gg]), patch.object(evaluate, "_read_json", return_value=paired):
            scopes = physical._paired_lists(load_config())
        self.assertEqual(len(scopes["T_old"]), 97)
        self.assertEqual(len(scopes["T_pair"]), 78)

    def test_real_subset_preserves_nonzero_analysis_ranges(self):
        from types import SimpleNamespace
        from features import feature_subset
        records = []
        analysis = np.zeros((4, 16), dtype=np.float32)
        for i, (filename, channel) in enumerate((
                ("B007_0.mat", "DE"), ("B007_0.mat", "FE"),
                ("B007_1.mat", "DE"), ("B007_1.mat", "FE"))):
            unit = f"12k_DE/{filename}"
            records.append({"dataset": "cwru", "source": "12k_DE", "unit_key": unit,
                "record_key": f"{unit}::{channel}", "path": filename, "channel": channel,
                "label": 1, "bearing_id": "", "condition": "", "origin": "", "severity": 0,
                "src_fs": 12000, "start": i*4, "stop": (i+1)*4,
                "analysis_start": i, "analysis_stop": i+1})
            analysis[i, :4] = i*10
        cache = SimpleNamespace(X=np.zeros((16, 1, 256), np.float32),
            phys25=np.zeros((16, 25), np.float32), ies_analysis=analysis, records=records)
        subset = feature_subset(cache, ["12k_DE/B007_1.mat"], labeled=False)
        rows = evaluate._q_phys_record_rows(subset, subset.records)
        expected = physical.ies_evidence(analysis[2:])[1]
        np.testing.assert_allclose([row["probabilities"] for row in rows], expected)
        self.assertEqual([row["window_count"] for row in rows], [1, 1])

    def test_ies_record_and_unit_csv_roundtrip(self):
        from bearing_facts import BY_ID
        fact = BY_ID["K001"]
        rows = []
        for i in range(20):
            row = {"unit_key": "K001/N15_M01_F10", "dataset": "paderborn", "label": 0,
                "source": "paderborn", "bearing_id": "K001", "condition": "N15_M01_F10",
                "origin": fact.origin, "severity": 0, "src_fs": 64000,
                "path": f"measure{i}.mat", "record_key": f"measure{i}::vibration_1", "channel": "vibration_1",
                "start": 45*i, "stop": 45*(i+1), "window_count": 45, "logspec_window_count": 93,
                "probabilities": [1, 0, 0, 0], "q_phys": [.3, .3, .2, .2]}
            rows.append(row)
        units = physical.aggregate_by_unit(rows)
        self.assertEqual(units[0]["window_count"], 900)
        self.assertEqual(units[0]["logspec_window_count"], 1860)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"predictions.csv"
            evaluate._write_predictions(rows, path)
            restored = evaluate.read_predictions(path)
        self.assertEqual(restored[0]["window_count"], 45)
        self.assertEqual(restored[0]["logspec_window_count"], 93)
        self.assertEqual(evaluate.aggregate_units(restored)[0]["probabilities"], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            physical.aggregate_by_unit(rows[:-1])
        with self.assertRaises(ValueError):
            physical.aggregate_by_unit(rows[:-1]+[rows[0]])


if __name__ == "__main__":
    unittest.main()
