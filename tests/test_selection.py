# -*- coding: utf-8 -*-
"""GG 配方、无效k、固定预算与新旧池共同留出的合成回归。"""
import unittest
from unittest.mock import patch
import numpy as np

from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
import selection


def pool_payload(pool):
    keys = sorted(f"{b}/{c}" for b in PRIMARY_BEARINGS for c in CONDITIONS)
    return {"all": keys, "budgets": {"K19": {"pool": sorted(pool), "test": sorted(set(keys)-set(pool))},
            "K116": {"pool": keys, "test": []}}}


class SelectionTests(unittest.TestCase):
    def test_covariance_regularization_exact_and_singular(self):
        z = np.zeros((4, 3))
        u = np.array([[.9, .1]]*4)
        _, covs, _ = selection._weighted_moments(z, u, selection.GG_DEFAULTS)
        np.testing.assert_allclose(covs, np.repeat((0.1*np.eye(3))[None], 2, axis=0))
        self.assertTrue(np.isfinite(np.linalg.cholesky(covs)).all())

    def test_gath_geva_singular_or_nonconverged_invalid(self):
        with patch.object(selection, "_gg_initialization", return_value=np.full((20, 6), 1/6)):
            result = selection.gath_geva(np.zeros((20, 4)), 6, selection.GG_DEFAULTS)
        self.assertFalse(result["valid"])
        self.assertIn(result["reason"], ("empty_cluster", "center_distance"))
        with patch.object(selection, "_gath_geva_single", side_effect=selection._InvalidK("max_iter")):
            result = selection.gath_geva(np.zeros((20, 4)), 6, selection.GG_DEFAULTS)
        self.assertEqual(result["reason"], "max_iter")

    def test_fixed19_reproducible_tie_and_labels_not_input(self):
        keys = sorted(f"{b}/{c}" for b in PRIMARY_BEARINGS for c in CONDITIONS)
        matrix = np.random.default_rng(4).normal(size=(116, 256))
        def synthetic(z, k, cfg):
            if k not in (6, 7):
                return {"valid": False, "reason": "max_iter"}
            hard = np.arange(116) % k
            u = np.eye(k)[hard]
            return {"valid": True, "xb": 0.1, "u": u,
                    "centers": np.zeros((k, 256)), "covs": np.repeat(np.eye(256)[None], k, axis=0),
                    "min_center_distance_squared": 1.0}
        with patch.object(selection, "gath_geva", side_effect=synthetic):
            a = selection.select_gg(keys, matrix)
            b = selection.select_gg(keys, matrix)
        self.assertEqual(a["k_selected"], 6)
        self.assertEqual(len(a["medoids"]), 6)
        self.assertEqual(len(a["budgets"]["K19"]["pool"]), 19)
        self.assertEqual(len(a["ranked_units"]), 116)
        self.assertEqual(a["ranked_units"], b["ranked_units"])
        self.assertFalse(a["selection_metadata_used"])
        with patch.object(selection, "gath_geva", return_value={"valid": False, "reason": "max_iter"}):
            with self.assertRaisesRegex(ValueError, "全部"):
                selection.select_gg(keys, matrix)

    def test_all_pool_intersections(self):
        keys = sorted(f"{b}/{c}" for b in PRIMARY_BEARINGS for c in CONDITIONS)
        for overlap in range(20):
            old = pool_payload(keys[:19])
            new = pool_payload(keys[:overlap]+keys[19:38-overlap])
            result = selection.paired_splits(old, new)
            self.assertEqual(len(result["T_pair"]), 78+overlap)
            self.assertFalse(set(result["T_pair"]) & (set(result["P_old"]) | set(result["P_gg"])))
            self.assertEqual(set(result["T_old"]) | set(result["P_old"]), set(keys))
            self.assertEqual(set(result["T_gg"]) | set(result["P_gg"]), set(keys))


if __name__ == "__main__":
    unittest.main()
