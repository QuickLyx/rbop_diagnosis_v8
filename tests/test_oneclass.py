# -*- coding: utf-8 -*-
"""单类异常方向、分位口径、软证据及非有限输入检查。"""
import unittest
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from oneclass import fit_normal, load_config, normal_evidence, peak_memory_bytes


class NormalTests(unittest.TestCase):
    def test_direction_and_normalization(self):
        q = normal_evidence([-1e20, 0, 1e20], 0, 1)
        np.testing.assert_allclose(q.sum(axis=1), 1)
        np.testing.assert_allclose(q[1], [0.5, 1/6, 1/6, 1/6])
        self.assertEqual(q[0, 0], 1)
        self.assertEqual(q[-1, 0], 0)
        self.assertGreater(q[0, 0], q[-1, 0])
        self.assertTrue(np.isfinite(q).all())

    def test_quantile_linear(self):
        self.assertAlmostEqual(np.quantile(np.arange(100), 0.99, method="linear"), 98.01)

    def test_production_normal_fit_uses_validation_normal_threshold(self):
        train = SimpleNamespace(phys25=np.array([[1.0]*25, [3.0]*25, [999.0]*25]),
            y=np.array([0, 0, 1]), records=[{"unit_key": "train_normal", "label": 0},
                                           {"unit_key": "train_ball", "label": 1}])
        val = SimpleNamespace(phys25=np.array([[10.0]*25, [20.0]*25, [888.0]*25]),
            y=np.array([0, 0, 2]), records=[{"unit_key": "val_normal", "label": 0},
                                           {"unit_key": "val_ir", "label": 2}])
        captured = []
        def scaler(values):
            captured.append(np.asarray(values).copy())
            return {"mean": [0.0]*25, "std": [1.0]*25}
        fake_features = SimpleNamespace(feature_subset=lambda cache, units: train if units == ["train"] else val,
            fit_scaler=scaler, scale_phys=lambda x, _: np.asarray(x))
        class FakeOCSVM:
            def __init__(self, **kwargs):
                self.fit_status_ = 0
            def fit(self, x):
                captured.append(x.copy())
                return self
            def decision_function(self, x):
                return -x[:, 0]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"features": fake_features}), \
                 patch("oneclass.source_split", return_value={"train": ["train"], "val": ["val"]}), \
                 patch("oneclass.new_output", return_value=Path(tmp)), \
                 patch("oneclass.OneClassSVM", FakeOCSVM), patch("oneclass.joblib.dump"):
                fit_normal(None, 0, tmp, load_config())
            calibration = json.loads((Path(tmp)/"calibration.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(captured[0], train.phys25)
        np.testing.assert_array_equal(captured[1][:, 0], [1, 3])
        self.assertAlmostEqual(calibration["t"], 19.9)
        self.assertAlmostEqual(calibration["s"], 1)
        self.assertEqual(calibration["train_windows"], 2)
        self.assertEqual(calibration["validation_windows"], 2)

    def test_process_memory_measurement(self):
        self.assertGreater(peak_memory_bytes(), 0)

    def test_invalid_scores(self):
        for scores, scale in (([np.nan], 1), ([np.inf], 1), ([0], 0)):
            with self.assertRaises(ValueError):
                normal_evidence(scores, 0, scale)


if __name__ == "__main__":
    unittest.main()
