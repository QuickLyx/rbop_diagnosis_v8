# -*- coding: utf-8 -*-
"""配额、域权重、单元分组、独立校准及粗糙集的合成检查。"""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from shallow import (allocate_quota, calibration_weights, candidate_score,
                     class_probabilities, rough_dependency, rough_set_analysis,
                     sample_units, training_weights, fit_shallow)


class ShallowTests(unittest.TestCase):
    def test_quota_unequal_records(self):
        np.testing.assert_array_equal(allocate_quota([1, 100, 100]), [1, 32, 31])
        np.testing.assert_array_equal(allocate_quota([100, 100]), [32, 32])
        np.testing.assert_array_equal(allocate_quota([2, 3]), [2, 3])
        self.assertEqual(allocate_quota([93]*19).sum(), 64)
        self.assertTrue((allocate_quota([93]*20) > 0).all())

    def test_sample_no_duplicates(self):
        records = [{"unit_key": "Normal/a.mat", "record_key": f"a::{ch}", "dataset": "cwru",
                    "channel": ch, "label": 0, "start": i*93, "stop": (i+1)*93}
                   for i, ch in enumerate(("DE", "FE"))]
        cache = SimpleNamespace(records=records)
        sampled = sample_units(cache, ["Normal/a.mat"])
        self.assertEqual(len(sampled["indices"]), 64)
        self.assertEqual(len(set(sampled["indices"])), 64)
        self.assertEqual([r["quota"] for r in sampled["quotas"]], [32, 32])

    def test_equal_domain_class_unit_mass(self):
        y = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2])
        groups = np.array(["s0", "s0", "s1", "s2", "s2", "t0", "t2", "t2", "t2"])
        domains = np.array(["cwru"]*5 + ["paderborn"]*4)
        w = training_weights(y, groups, domains)
        self.assertAlmostEqual(w.mean(), 1)
        self.assertAlmostEqual(w[:5].sum(), w[5:].sum())
        self.assertAlmostEqual(w[:3].sum(), w[3:5].sum())
        self.assertAlmostEqual(w[:2].sum(), w[2])
        self.assertAlmostEqual(candidate_score(y, y, groups, domains), 1)

    def test_grouped_cv(self):
        y = np.repeat(np.tile(np.arange(4), 6), 3)
        groups = np.repeat(np.arange(24), 3)
        split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
        for train, val in split.split(np.zeros((72, 25)), y, groups):
            self.assertFalse(set(groups[train]) & set(groups[val]))
            self.assertEqual(set(y[train]), {0, 1, 2, 3})

    def test_frozen_does_not_refit(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(64, 25))
        y = np.tile(np.arange(4), 16)
        svc = SVC().fit(x[:32], y[:32])
        with patch.object(svc, "fit", side_effect=AssertionError("禁止重拟合")):
            cal = CalibratedClassifierCV(FrozenEstimator(svc), method="sigmoid").fit(x[32:], y[32:])
        self.assertEqual(class_probabilities(cal, x).shape, (64, 4))
        class Fake:
            classes_ = np.array([3, 1, 0, 2])
            def predict_proba(self, values):
                return np.tile([0.4, 0.2, 0.1, 0.3], (len(values), 1))
        np.testing.assert_allclose(class_probabilities(Fake(), x[:1]), [[0.1, 0.2, 0.3, 0.4]])

    def test_calibration_hierarchy(self):
        records, offset = [], 0
        for label in range(4):
            for unit in range(label+1):
                for channel, n in (("DE", 2), ("FE", 4)):
                    records.append({"label": label, "unit_key": f"{label}/{unit}",
                                    "channel": channel, "start": offset, "stop": offset+n})
                    offset += n
        w = calibration_weights(SimpleNamespace(records=records, phys25=np.zeros((offset, 25))))
        class_masses = [sum(w[r["start"]:r["stop"]].sum() for r in records if r["label"] == label) for label in range(4)]
        np.testing.assert_allclose(class_masses, [offset/4]*4)

    def test_internal_scaler_uses_all_source_training_windows_only(self):
        from oneclass import load_config
        source_records, target_records = [], []
        for label in range(4):
            for unit in range(4):
                for channel in ("DE", "FE"):
                    start = len(source_records)*40
                    source_records.append({"dataset": "cwru", "unit_key": f"L{label}/u{unit}",
                        "record_key": f"L{label}/u{unit}::{channel}", "channel": channel,
                        "label": label, "start": start, "stop": start+40})
        for unit in range(19):
            start = unit*80
            target_records.append({"dataset": "paderborn", "unit_key": f"K{unit}/c",
                "record_key": f"K{unit}/c::vibration_1", "channel": "vibration_1",
                "label": (0, 2, 3)[unit % 3], "start": start, "stop": start+80})
        source_raw = np.zeros((1280, 25))
        for i, row in enumerate(source_records):
            source_raw[row["start"]:row["stop"], 0] = i // 2
        target_raw = np.full((1520, 25), 100.0)
        source = SimpleNamespace(records=source_records, phys25=source_raw)
        target = SimpleNamespace(records=target_records, phys25=target_raw)
        train = [f"L{label}/u{unit}" for label in range(4) for unit in range(3)]
        val = [f"L{label}/u3" for label in range(4)]
        captured = []
        def subset(cache, units):
            rows, raw, y, offset = [], [], [], 0
            for row in cache.records:
                if row["unit_key"] in units:
                    n = row["stop"]-row["start"]
                    raw.append(cache.phys25[row["start"]:row["stop"]])
                    y.extend([row["label"]]*n)
                    rows.append({**row, "start": offset, "stop": offset+n})
                    offset += n
            return SimpleNamespace(records=rows, phys25=np.concatenate(raw), y=np.array(y))
        def scaler(values):
            captured.append(np.asarray(values).copy())
            return {"mean": [0.0]*25, "std": [1.0]*25}
        fake_features = SimpleNamespace(feature_subset=subset, fit_scaler=scaler, scale_phys=lambda x, _: np.asarray(x))
        class FakeSVC:
            def __init__(self, **kwargs):
                self.fit_status_ = 0
                self.support_ = np.arange(4)
                self.n_support_ = np.ones(4, dtype=int)
            def fit(self, x, y, sample_weight=None):
                return self
            def predict(self, x):
                return np.zeros(len(x), dtype=int)
        class FakeCalibration:
            def __init__(self, estimator, method):
                pass
            def fit(self, *args, **kwargs):
                return self
        pool = [r["unit_key"] for r in target_records]
        test = [f"Unused{i}/c" for i in range(97)]
        selection = {"all": pool+test, "budgets": {"K19": {"pool": pool, "test": test}}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"features": fake_features}), \
                 patch("shallow.source_split", return_value={"train": train, "val": val}), \
                 patch("shallow.new_output", return_value=Path(tmp)), \
                 patch("shallow.SVC", FakeSVC), patch("shallow.FrozenEstimator", lambda x: x), \
                 patch("shallow.CalibratedClassifierCV", FakeCalibration), patch("shallow.joblib.dump"):
                fit_shallow(source, target, selection, 0, 0, tmp, load_config())
            saved = json.loads((Path(tmp)/"split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(captured), 4)
        unit_ids = {f"cwru/L{label}/u{unit}": label*4+unit for label in range(4) for unit in range(3)}
        for values, split in zip(captured[:3], saved["internal"]):
            expected = {unit_ids[key] for key in split["source_scaler_units"]}
            self.assertEqual(set(values[:, 0]), expected)
            self.assertEqual(len(values), len(expected)*80)
        self.assertEqual(len(captured[-1]), 12*80)
        self.assertTrue(all(np.max(values[:, 0]) < 100 for values in captured))

    def test_rough_set(self):
        self.assertEqual(rough_dependency([[0], [0], [1]], [0, 1, 1]), (1/3, 2, 1))
        rows = [{"unit_key": f"K{i}/c", "label": i % 3, "start": i, "stop": i+1} for i in range(19)]
        cache = SimpleNamespace(records=rows, phys25=np.ones((19, 25), dtype=np.float32))
        scaler = {"mean": [0.0]*25, "std": [1.0]*25}
        result = rough_set_analysis(cache, [r["unit_key"] for r in rows], scaler)
        self.assertEqual(result["gamma"], 0)
        self.assertEqual(len(result["importance"]), 25)
        self.assertTrue(all(edge == [] for edge in result["boundaries"]))


if __name__ == "__main__":
    unittest.main()
