# -*- coding: utf-8 -*-
"""冻结纯比值软证据与独立导出回归；只使用合成缓存。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from physical_baseline_v2 import (
    CACHE_VERSION_V2, q_phys_ratio, run_phys_ies_v2, summarize_evidence,
)


class PhysicalV2Tests(unittest.TestCase):
    def test_equal_and_zero_energy_are_uniform(self):
        np.testing.assert_array_equal(q_phys_ratio(np.zeros((2, 16))), np.full((2, 4), 0.25))
        np.testing.assert_array_equal(q_phys_ratio(np.ones((2, 16))), np.full((2, 4), 0.25))

    def test_type_direction_and_scale_invariance(self):
        R = np.zeros((3, 16))
        R[:, :4] = [[1], [3], [0]]
        R[:, 4:8] = [[3], [1], [2]]
        q = q_phys_ratio(R)
        np.testing.assert_allclose(q, [[.25, .25, .375, .125], [.25, .25, .125, .375],
                                      [.25, .25, .5, 0]])
        np.testing.assert_allclose(q_phys_ratio(R * 1e-100), q)
        np.testing.assert_allclose(q_phys_ratio(R * 1e100), q)
        np.testing.assert_allclose(q.sum(axis=1), 1)
        self.assertEqual(q.argmax(axis=1).tolist(), [2, 3, 2])

    def test_large_finite_energy_does_not_overflow(self):
        R = np.full((1, 16), np.finfo(np.float64).max)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            np.testing.assert_array_equal(q_phys_ratio(R), [[.25, .25, .25, .25]])

    def test_tiny_nonzero_and_unequal_shares_preserved(self):
        R = np.zeros((2, 16))
        R[0, 4:8] = np.nextafter(0.0, 1.0)
        R[1, :4], R[1, 4:8] = 1, 1e20
        q = q_phys_ratio(R)
        np.testing.assert_array_equal(q[0], [.25, .25, .5, 0])
        self.assertGreater(q[1, 3], 0)
        self.assertAlmostEqual(q[1, 3] / 5e-21, 1)

    def test_output_outside_exploration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "forbidden"
            with self.assertRaises(ValueError):
                run_phys_ies_v2({"phys_ies_v2_output": str(output)})
            self.assertFalse(output.exists())

    def test_invalid_input_rejected(self):
        for R in (np.zeros((0, 16)), np.zeros((2, 15)), np.zeros(16),
                  np.full((1, 16), -1), np.full((1, 16), np.nan),
                  np.full((1, 16), np.inf), np.ones((1, 16), dtype=complex)):
            with self.subTest(shape=R.shape), self.assertRaises(ValueError):
                q_phys_ratio(R)

    def test_normal_push_is_not_push_of_mean(self):
        rows = [{"label": 0, "probabilities": [.25, .25, .4, .1]},
                {"label": 0, "probabilities": [.25, .25, .1, .4]},
                {"label": None, "probabilities": [.25, .25, .25, .25]}]
        summary = summarize_evidence(rows)
        self.assertEqual(summary["unlabeled_count"], 1)
        self.assertEqual(summary["classes"]["Ball"]["support"], 0)
        self.assertIsNone(summary["classes"]["Ball"]["recall"])
        self.assertAlmostEqual(summary["normal_push"]["mean"], .15)
        self.assertAlmostEqual(summary["normal_push"]["push_of_class_mean"], 0)

    def test_export_aggregation_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch("physical_baseline_v2.ROOT", Path(tmp)):
            root = Path(tmp)
            output = root / "outputs" / "analysis_ies_diag" / "out"
            records = []
            R = np.zeros((3, 16))
            R[0, :4], R[0, 4:8] = 1, 3
            R[1, :4], R[1, 4:8] = 9, 1
            R[2, :4], R[2, 4:8] = 1, 1
            for channel, start, stop in (("DE", 0, 2), ("FE", 2, 3)):
                records.append({"record_key": f"12k_DE/B007_0.mat::{channel}",
                                "unit_key": "12k_DE/B007_0.mat", "dataset": "cwru",
                                "source": "12k_DE", "path": "B007_0.mat", "label": 1,
                                "channel": channel, "analysis_start": start, "analysis_stop": stop})
            meta = {"version": CACHE_VERSION_V2, "dataset": "cwru", "records": records}
            (root / "cwru.json").write_text(json.dumps(meta), encoding="utf-8")
            np.save(root / "cwru_ies_analysis.npy", R)
            config = {"cache_dir_v2": str(root), "phys_ies_v2_output": str(output)}
            result = run_phys_ies_v2(config, execute=True, dataset="cwru")
            self.assertEqual((result["record_count"], result["unit_count"]), (2, 1))
            payload = json.loads((output / "q_phys.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["run_kind"], "exploratory")
            self.assertTrue(payload["target_labels_seen"])
            self.assertEqual(payload["formula_version"], "q_phys_ratio_c025_v1")
            np.testing.assert_allclose(payload["records"][0]["q_phys"], [.25, .25, .2125, .2875])
            np.testing.assert_allclose(payload["units"][0]["q_phys"], [.25, .25, .23125, .26875])
            self.assertEqual(result["diagnostics"]["unit_level"]["classes"]["Ball"]["support"], 1)
            before = (output / "q_phys.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_phys_ies_v2(config, execute=True, dataset="cwru")
            self.assertEqual((output / "q_phys.json").read_bytes(), before)
            bad_output = output.parent / "bad"
            config["phys_ies_v2_output"] = str(bad_output)
            meta["version"] = "旧版本"
            (root / "cwru.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_phys_ies_v2(config, execute=True, dataset="cwru")
            self.assertFalse(bad_output.exists())
            meta["version"] = CACHE_VERSION_V2
            meta["records"][1]["analysis_start"] = 1
            (root / "cwru.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_phys_ies_v2(config, execute=True, dataset="cwru")


if __name__ == "__main__":
    unittest.main()
