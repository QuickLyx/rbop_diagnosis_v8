# -*- coding: utf-8 -*-
"""晚期融合小测试：六权重选择/平票/置零消融/记录-单元线性一致/tau/熵/端到端 CLI。

只用合成数据：物理锚函数、深度评估目录与 Normal 模型目录全部合成注入；
`physical_baseline.ies_evidence` 尚由物理锚代理实现，这里 monkeypatch 冻结公式。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate
import oneclass
import physical_baseline
from evaluate import (_align_units, _entropy, _fuse_records, _q_phys_record_rows,
                      _renormalize, _select_weights, _uncertainty_report, aggregate_units)


def eye(index):
    return np.eye(4)[index].tolist()


def unit_rows(unit_key, label, probabilities):
    """合成单元行，只喂给 _select_weights/_align_units，不走 aggregate_units 校验。"""
    return {"unit_key": unit_key, "dataset": "cwru", "label": label, "source": "12k_DE",
            "bearing_id": None, "condition": None, "origin": None, "severity": None,
            "src_fs": 12000, "channels": ["DE", "FE"], "probabilities": probabilities,
            "record_count": 2, "window_count": 2}


def record_row(record_key, unit_key, label, probabilities, dataset="cwru", channel="DE",
               start=0, window_count=1):
    """合成记录行；cwru 单元键沿用 v7 约定含文件名（如 12k_DE/b.mat）。"""
    return {"record_key": record_key, "unit_key": unit_key,
            "path": (unit_key.split("/", 1)[-1] if dataset == "cwru"
                     else f"{unit_key}/{record_key}.mat"),
            "channel": channel, "start": start, "stop": start + window_count,
            "dataset": dataset, "source": "12k_DE" if dataset == "cwru" else "paderborn",
            "bearing_id": None if dataset == "cwru" else unit_key.split("/")[0],
            "condition": None if dataset == "cwru" else unit_key.split("/")[1],
            "origin": None, "severity": None, "src_fs": 12000 if dataset == "cwru" else 64000,
            "label": label, "probabilities": list(probabilities), "window_count": window_count}


def frozen_ies_evidence(analysis):
    """计划 §7.2/§7.3 的冻结公式；正式实现落在 physical_baseline（接口约定）。"""
    analysis = np.asarray(analysis, dtype=np.float64)
    e_or = analysis[:, :4].mean(axis=1)
    e_ir = analysis[:, 4:8].mean(axis=1)
    hard = np.where((e_or > 1.5 * e_ir)[:, None], [0, 0, 0, 1],
                    np.where((e_ir > 1.5 * e_or)[:, None], [0, 0, 1, 0], [1, 0, 0, 0]))
    v = np.column_stack([np.ones(len(analysis)), np.ones(len(analysis)),
                         (1 + e_ir) / (1.5 * (1 + e_or)),
                         (1 + e_or) / (1.5 * (1 + e_ir))])
    return hard, v / v.sum(axis=1, keepdims=True), e_ir, e_or


def source_val_units():
    """四个源验证单元：deep 全部判下一类、normal 全判 Normal、q_phys 全对。"""
    keys = ["U0/c0", "U1/c0", "U2/c0", "U3/c0"]
    labels = [0, 1, 2, 3]
    deep = [unit_rows(k, l, eye((l + 1) % 4)) for k, l in zip(keys, labels)]
    normal = [unit_rows(k, l, eye(0)) for k, l in zip(keys, labels)]
    phys = [unit_rows(k, l, eye(l)) for k, l in zip(keys, labels)]
    return keys, labels, deep, normal, phys


class SelectWeightsTests(unittest.TestCase):
    def test_candidates_scored_on_source_val_e1_phys_wins(self):
        keys, _, deep, normal, phys = source_val_units()
        report, (rkeys, rlabels, _) = _select_weights(
            deep, normal, phys, [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)])
        self.assertEqual(rkeys, keys)
        self.assertEqual(rlabels, [0, 1, 2, 3])
        # (1,0,0) 全错 0 分；(0.5,0.5,0) 只有 Normal 对（0.25）；(0.5,0,0.5) 三类对（0.75）。
        self.assertEqual([c["source_val_macro_recall_E1"] for c in report["candidates"]],
                         [0.0, 0.25, 0.75])
        self.assertEqual(report["chosen_weights"], [0.5, 0.0, 0.5])

    def test_exact_tie_prefers_larger_deep_fewer_auxiliary_larger_normal(self):
        keys = ["A", "B"]
        deep = [unit_rows(k, 0, eye(0)) for k in keys]
        normal = [unit_rows(k, 0, eye(0)) for k in keys]
        phys = [unit_rows(k, 0, eye(0)) for k in keys]
        candidates = [(1.0, 0.0, 0.0), (0.75, 0.25, 0.0), (0.75, 0.0, 0.25),
                      (0.5, 0.5, 0.0), (0.5, 0.25, 0.25), (0.5, 0.0, 0.5)]
        report, _ = _select_weights(deep, normal, phys, candidates)
        # 全对时所有候选同分：先 w_D 大，再辅助支路少，再 w_N 大。
        self.assertEqual(report["chosen_weights"], [1.0, 0.0, 0.0])

    def test_zero_auxiliary_weight_is_formal_result(self):
        keys = ["A", "B"]
        deep = [unit_rows(k, 0, eye(0)) for k in keys]
        noise = [unit_rows(k, 0, eye(2)) for k in keys]  # 单类证据全错
        phys = [unit_rows(k, 0, eye(0)) for k in keys]
        report, _ = _select_weights(deep, noise, phys, [(1.0, 0.0, 0.0), (0.75, 0.25, 0.0)])
        # 两者同分 1.0 → 平票规则选 w_D 较大者；辅助权重为零是正式结果。
        self.assertEqual(report["chosen_weights"], [1.0, 0.0, 0.0])

    def test_align_rejects_missing_duplicate_and_label_conflict(self):
        _, _, deep, normal, phys = source_val_units()
        with self.assertRaisesRegex(ValueError, "不一致"):
            _align_units([deep, normal[:-1], phys], ("d", "n", "p"))
        with self.assertRaisesRegex(ValueError, "重复"):
            _align_units([deep + deep[:1], normal, phys], ("d", "n", "p"))
        flipped = [dict(row, label=3) for row in normal[:2]] + normal[2:]
        with self.assertRaisesRegex(ValueError, "真值"):
            _align_units([deep, flipped, phys], ("d", "n", "p"))


class AblationTests(unittest.TestCase):
    def test_renormalize_two_path_versions(self):
        np.testing.assert_allclose(_renormalize([0.5, 0.25, 0.25], "without_phys"),
                                   [0.5 / 0.75, 0.25 / 0.75, 0.0])
        # 置零支路本已为零 → 归一化不改变权重。
        self.assertEqual(_renormalize([0.75, 0.25, 0.0], "without_phys"), [0.75, 0.25, 0.0])
        # 置零后仅剩深度 → 等于深度原始。
        self.assertEqual(_renormalize([0.75, 0.25, 0.0], "without_normal"), [1.0, 0.0, 0.0])
        self.assertEqual(_renormalize([0.5, 0.0, 0.5], "without_normal"), [0.5, 0.0, 0.5])
        self.assertEqual(_renormalize([0.75, 0.0, 0.25], "without_phys"), [1.0, 0.0, 0.0])

    def test_deep_only_full_fusion_equals_original(self):
        rows = [record_row(f"r{i}", f"u{i}", 0, eye(0)) for i in range(3)]
        fused = _fuse_records([1.0, 0.0, 0.0], rows, [dict(r) for r in rows], [dict(r) for r in rows])
        for original, row in zip(rows, fused):
            self.assertEqual(row["probabilities"], original["probabilities"])

    def test_fuse_rejects_missing_rows_for_positive_weight(self):
        deep = [record_row("r", "u", 0, eye(0))]
        with self.assertRaisesRegex(ValueError, "辅助支路"):
            _fuse_records([0.5, 0.5, 0.0], deep)  # 权重 0.5 但缺 normal 行
        fused = _fuse_records([0.5, 0.5, 0.0], deep, [record_row("r", "u", 0, eye(2))])
        np.testing.assert_allclose(fused[0]["probabilities"], [0.5, 0, 0.5, 0], atol=1e-12)


class LinearConsistencyTests(unittest.TestCase):
    def test_record_then_unit_equals_unit_level_fusion(self):
        """记录级融合后等权聚合 == 单元级概率直接加权（线性等权两路径一致）。"""
        rng = np.random.default_rng(7)
        deep_records, normal_records, phys_records = [], [], []
        for unit in range(3):
            unit_key, label = f"12k_DE/u{unit}", unit % 4
            for channel in ("DE", "FE"):
                key = f"{unit_key}::{channel}"
                start = len(deep_records)
                for rows in (deep_records, normal_records, phys_records):
                    rows.append(record_row(key, unit_key, label, rng.dirichlet(np.ones(4)),
                                           channel=channel, start=start))
        weights = [0.5, 0.25, 0.25]
        fused_records = _fuse_records(weights, deep_records, normal_records, phys_records)
        path_a = aggregate_units(fused_records)
        deep_units = aggregate_units(deep_records)
        normal_units = aggregate_units(normal_records)
        phys_units = aggregate_units(phys_records)
        keys, _, (d, n, p) = _align_units([deep_units, normal_units, phys_units], ("d", "n", "p"))
        self.assertEqual(len(keys), 3)
        for key in keys:
            fused = (weights[0] * np.asarray(d[key]["probabilities"])
                     + weights[1] * np.asarray(n[key]["probabilities"])
                     + weights[2] * np.asarray(p[key]["probabilities"]))
            unit_a = next(row for row in path_a if row["unit_key"] == key)
            np.testing.assert_allclose(unit_a["probabilities"], fused, atol=1e-12)


class UncertaintyTests(unittest.TestCase):
    def test_tau_from_source_val_quantile_marks_low_confidence_only(self):
        # 源验证 max p = [0.05, 0.10, ..., 1.00]；tau = Q_0.05 = 0.0975，覆盖率 95%。
        maxp = np.arange(1, 21, dtype=float) / 20
        val = np.column_stack([maxp, 1 - maxp, np.zeros(20), np.zeros(20)])
        target = np.asarray([[0.6, 0.2, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
        target_units = [{"unit_key": "A", "label": 2}, {"unit_key": "B", "label": None}]
        before = target.copy()
        report = _uncertainty_report({"v": (val, target)}, target_units, quantile=0.05)
        # 源验证 max p 排序后 Q_0.05（线性）落在 0.5 与 0.55 之间。
        self.assertAlmostEqual(report["v"]["tau"], 0.5 + 0.95 * 0.05, places=12)
        self.assertAlmostEqual(report["v"]["source_val_coverage"], 0.95)
        # 主PU恰 A 高置信、KB 的 B 低置信；只标记名单，不改类、不改分母。
        self.assertEqual(report["v"]["scopes"]["main_pu"]["unit_count"], 1)
        self.assertEqual(report["v"]["scopes"]["main_pu"]["low_confidence_units"], [])
        self.assertAlmostEqual(report["v"]["scopes"]["main_pu"]["coverage"], 1.0)
        self.assertEqual(report["v"]["scopes"]["kb"]["low_confidence_units"], ["B"])
        self.assertAlmostEqual(report["v"]["scopes"]["kb"]["coverage"], 0.0)
        self.assertIn("normalized_entropy_by_unit", report["v"]["scopes"]["kb"])
        np.testing.assert_array_equal(target, before)

    def test_entropy_zero_log_zero_and_rejects_empty(self):
        self.assertEqual(_entropy([[1.0, 0.0, 0.0, 0.0]])[0], 0.0)
        np.testing.assert_allclose(_entropy([[0.25, 0.25, 0.25, 0.25]]), [1.0], atol=1e-12)
        with self.assertRaises(ValueError):
            _uncertainty_report({"v": (np.empty((0, 4)), np.empty((0, 4)))}, [])


class QPhysTests(unittest.TestCase):
    def _fake_cache(self, ranges=None):
        rng = np.random.default_rng(0)
        total = 0 if ranges is None else sum(b - a for a, b in ranges)
        return type("C", (), {"ies_analysis": rng.random((total, 16)) + 0.1,
                              "analysis_ranges": ranges})()

    def setUp(self):
        # ies_evidence 正式实现由物理锚模块提供；测试注入冻结公式。
        patcher = patch.object(physical_baseline, "ies_evidence", frozen_ies_evidence, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_q_phys_rows_counts_analysis_windows_not_logspec(self):
        cache = self._fake_cache([(0, 3), (3, 5)])
        records = [{"record_key": "r1", "start": 0, "stop": 10,
                    "analysis_start": 0, "analysis_stop": 3},
                   {"record_key": "r2", "start": 10, "stop": 15,
                    "analysis_start": 3, "analysis_stop": 5}]
        rows = _q_phys_record_rows(cache, records)
        # window_count 是物理分析窗数（3、2），不是 logspec 窗数（10、5）。
        self.assertEqual([row["window_count"] for row in rows], [3, 2])
        for row in rows:
            self.assertTrue(all(v > 0 for v in row["probabilities"]))
            self.assertAlmostEqual(sum(row["probabilities"]), 1, places=6)

    def test_missing_analysis_interval_rejected(self):
        # 无 analysis_ranges：必须读记录自身字段，None 区间报错。
        cache = self._fake_cache()
        with self.assertRaisesRegex(ValueError, "区间"):
            _q_phys_record_rows(cache, [{"record_key": "r", "start": 0, "stop": 2,
                                         "analysis_start": None, "analysis_stop": 3}])

    def test_frozen_evidence_formula_positive_and_boundary_normal(self):
        hard, q, e_ir, e_or = frozen_ies_evidence(
            np.asarray([[2.0, 1.0, 0.5, 0.2, 4.0, 2.0, 1.0, 0.5] + [0.0] * 8,
                        [4.0, 2.0, 1.0, 0.5, 4.0, 2.0, 1.0, 0.5] + [0.0] * 8]))
        self.assertAlmostEqual(e_ir[0], 1.875)
        self.assertAlmostEqual(e_or[0], 0.925)
        weights = [1, 1, (1 + 1.875) / (1.5 * (1 + 0.925)), (1 + 0.925) / (1.5 * (1 + 1.875))]
        np.testing.assert_allclose(q[0], np.asarray(weights) / sum(weights), atol=1e-12)
        self.assertTrue((q > 0).all())
        # e_IR == 1.5*e_OR 边界与全零都判 Normal（硬 one-hot 第一位）。
        self.assertEqual(hard[1].tolist(), [1, 0, 0, 0])  # 1.5*2.375==2.625? e_OR 不占优
        hard2, _, _, _ = frozen_ies_evidence(np.zeros((1, 16)))
        self.assertEqual(hard2[0].tolist(), [1, 0, 0, 0])


class EndToEndFusionTests(unittest.TestCase):
    """合成 deep cw/PU 评估目录与 Normal 模型目录，跑完整融合并验三份产物。"""

    CW_VAL_UNITS = ["12k_DE/b.mat", "12k_DE/c.mat", "12k_DE/d.mat", "12k_DE/e.mat"]

    def _prepare(self, folder, cache_version):
        folder = Path(folder)
        eval_dir = {}
        provenance = {"checkpoint": "outputs/train/joint_f0_s0/last.pt", "checkpoint_kind": "last",
                      "epoch": 30, "last_epoch": 30, "repr_config": {},
                      "scaler": {"mean": [0.0] * 25, "std": [1.0] * 25},
                      "source_split": {"train": ["12k_DE/a"], "val": self.CW_VAL_UNITS},
                      "supervised_pool_units": []}
        for dataset in ("cwru", "paderborn"):
            eval_dir[dataset] = folder / "eval" / f"joint_f0_s0_{dataset}"
            eval_dir[dataset].mkdir(parents=True)
            info = {"protocol": "v8_stage3", "arm": "joint", "fold": 0, "seed": 0,
                    "dataset": dataset, "cache_version": cache_version, "run_kind": "formal",
                    "provenance": provenance}
            (eval_dir[dataset] / "eval_info.json").write_text(json.dumps(info), encoding="utf-8")
            (eval_dir[dataset] / "metrics.json").write_text("{}", encoding="utf-8")
        # PU 深度记录：2 个主单元 + 1 个 KB 单元（各 20 条，合成轴承不适用 KA08 例外），全判 IR。
        pu_rows, offset = [], 0
        for bearing in ("U000", "U015", "KB23"):
            for index in range(20):
                key = f"{bearing}/C/{bearing}_{index + 1}.mat::vibration_1"
                pu_rows.append(record_row(key, f"{bearing}/C", 2 if bearing != "KB23" else None,
                                          [0.1, 0.1, 0.7, 0.1],
                                          dataset="paderborn", channel="vibration_1",
                                          start=offset))
                offset += 1
        # CW 深度记录：4 个验证单元 × DE/FE，全判 Normal（权重选择用）。
        cw_rows, offset = [], 0
        for unit in self.CW_VAL_UNITS:
            for channel in ("DE", "FE"):
                cw_rows.append(record_row(f"{unit}::{channel}", unit, 0, [0.7, 0.1, 0.1, 0.1],
                                          channel=channel, start=offset))
                offset += 1
        from evaluate import _write_predictions
        _write_predictions(pu_rows, eval_dir["paderborn"] / "predictions.csv")
        _write_predictions(cw_rows, eval_dir["cwru"] / "predictions.csv")
        from evaluate import aggregate_units
        _write_predictions(aggregate_units(pu_rows),
                           eval_dir["paderborn"] / "unit_predictions.csv", unit=True)
        _write_predictions(aggregate_units(cw_rows),
                           eval_dir["cwru"] / "unit_predictions.csv", unit=True)
        # phys_ies 硬判单元目录：错误互补表用硬 one-hot，不是 q_phys 软分。
        phys_dir = folder / "eval" / "phys_ies"
        phys_dir.mkdir()
        (phys_dir / "eval_info.json").write_text(json.dumps(
            {"protocol": "v8_stage3", "arm": "phys_ies", "dataset": "paderborn",
             "cache_version": cache_version, "run_kind": "formal"}), encoding="utf-8")
        phys_units = aggregate_units(pu_rows)
        for row, guess in zip(phys_units, ([1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0])):
            row["probabilities"] = guess
        _write_predictions(phys_units, phys_dir / "unit_predictions.csv", unit=True)
        normal_dir = folder / "train" / "normal_f0"
        normal_dir.mkdir(parents=True)
        for name in ("config.json", "split.json", "model.joblib", "scaler.json", "calibration.json"):
            (normal_dir / name).write_text("stub", encoding="utf-8")
        return pu_rows, cw_rows

    def test_full_fusion_cli_writes_three_versions_and_calibration(self):
        config = json.loads(Path("configs/stage3.json").read_text(encoding="utf-8"))
        cache_version = config["cache_version"]
        normal_prob = [0.6, 0.2, 0.1, 0.1]

        def fake_predict_records(cache, model_dir):
            return [{**record, "probabilities": list(normal_prob),
                     "window_count": record["stop"] - record["start"],
                     "anomaly_mean": 0.5, "normal_threshold_fraction": 1.0,
                     "anomaly_quantiles": [0.4, 0.5, 0.6]}
                    for record in cache.records]

        def fake_read_json(path):
            path = Path(path)
            payload = {
                "paired_splits.json": {"U": [f"U{i:03}/C" for i in range(116)],
                                       "P_old": [f"U{i:03}/C" for i in range(19)],
                                       "P_gg": [f"U{i:03}/C" for i in range(15, 34)],
                                       "T_old": [f"U{i:03}/C" for i in range(19, 116)],
                                       "T_gg": [f"U{i:03}/C" for i in list(range(15)) + list(range(34, 116))],
                                       "T_pair": [f"U{i:03}/C" for i in range(34, 116)],
                                       "common_unseen_bearings": ["U050"],
                                       "common_unseen_units": ["U050/C"]},
                "selection.json": {"protocol": "v8_stage3", "k_selected": 19,
                                   "all": [f"U{i:03}/C" for i in range(116)],
                                   "budgets": {"K19": {"pool": [f"U{i:03}/C" for i in range(19)],
                                                       "test": [f"U{i:03}/C" for i in range(19, 116)]},
                                               "K116": {"pool": [f"U{i:03}/C" for i in range(116)],
                                                        "test": []}}},
                "selection_gg.json": {"protocol": "v8_stage3", "k_selected": 19,
                                      "all": [f"U{i:03}/C" for i in range(116)],
                                      "budgets": {"K19": {"pool": [f"U{i:03}/C" for i in range(15, 34)],
                                                          "test": [f"U{i:03}/C" for i in list(range(15)) + list(range(34, 116))]},
                                                  "K116": {"pool": [f"U{i:03}/C" for i in range(116)], "test": []}}},
                "split.json": {"train": ["12k_DE/a"], "val": self.CW_VAL_UNITS},
                "config.json": {"protocol": "v8_stage3", "arm": "normal", "fold": 0,
                                "seed": 0, "cache_version": cache_version,
                                "run_kind": "formal"},
            }
            if path.name in payload:
                return payload[path.name]
            return original_read_json(path)  # 其余（eval_info 等）读真实文件

        def fake_load_cache(dataset, cache_dir):
            # 惰性构造：依赖当前测试已准备的合成记录。
            return self._caches[dataset]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            pu_rows, cw_rows = self._prepare(root, cache_version)
            original_read_json = evaluate._read_json  # mock 内不能递归引用被替换的名字
            analysis = np.asarray([[2.0, 1.0, 0.5, 0.2, 4.0, 2.0, 1.0, 0.5] + [0.0] * 8]
                                  * len(pu_rows))
            analysis[10:12] = [4.0, 2.0, 1.0, 0.5, 2.0, 1.0, 0.5, 0.2] + [0.0] * 8
            target = type("C", (), {"records": pu_rows, "phys25": None,
                                    "ies_analysis": analysis,
                                    "analysis_ranges": [(i, i + 1) for i in range(len(pu_rows))],
                                    "version": cache_version, "repr_config": {}})()
            val_analysis = np.asarray([[2.0, 1.0, 0.5, 0.2, 4.0, 2.0, 1.0, 0.5] + [0.0] * 8]
                                      * len(cw_rows))
            val_cache = type("C", (), {"records": cw_rows,
                                       "phys25": np.zeros((len(cw_rows), 25), dtype=np.float32),
                                       "ies_analysis": val_analysis,
                                       "analysis_ranges": [(i, i + 1) for i in range(len(cw_rows))],
                                       "version": cache_version, "repr_config": {}})()
            source = type("C", (), {"records": cw_rows, "version": cache_version,
                                    "repr_config": {}})()
            self._caches = {"paderborn": target, "cwru": source}
            output = root / "outputs" / "eval" / "fusion_joint_f0_s0"
            args = type("A", (), {"arm": "fusion_joint", "fold": 0, "seed": 0, "dataset": None,
                                  "output": str(output),
                                  "deep_eval_root": str(root / "eval"),
                                  "normal_root": str(root / "train")})()
            paired = {"all_primary": ["U000/C", "U015/C"], "T_old": ["U000/C"],
                      "T_gg": [], "T_pair": ["U000/C"], "common_unseen_units": []}
            with patch.object(evaluate, "ROOT", root), \
                    patch.object(evaluate, "_paired_splits", return_value=paired), \
                    patch.object(evaluate, "_read_json", side_effect=fake_read_json), \
                    patch.object(evaluate, "_source_split",
                                 return_value={"train": ["12k_DE/a"], "val": self.CW_VAL_UNITS}), \
                    patch("features.load_cache", side_effect=fake_load_cache), \
                    patch("features.feature_subset", return_value=val_cache), \
                    patch.object(physical_baseline, "ies_evidence", frozen_ies_evidence, create=True), \
                    patch.object(oneclass, "predict_records", side_effect=fake_predict_records):
                evaluate._run_fusion(config, args, execute=True)
            full = output
            for suffix, arm in (("", "fusion_joint"),
                                ("_without_phys", "fusion_joint_without_phys"),
                                ("_without_normal", "fusion_joint_without_normal")):
                merged = full if not suffix else full.parent / f"{full.name}{suffix}"
                for artifact in ("metrics.json", "predictions.csv", "unit_predictions.csv",
                                 "eval_info.json"):
                    self.assertTrue((merged / artifact).is_file(), f"{merged}/{artifact}")
                version_info = json.loads((merged / "eval_info.json").read_text(encoding="utf-8"))
                # 三版本编号不同 arm，summary 不会把消融当同格重复报告。
                self.assertEqual(version_info["arm"], arm)
                self.assertEqual(version_info["fusion_version"],
                                 "fusion_full" if not suffix else f"fusion{suffix}")
            calibration = json.loads((full / "calibration.json").read_text(encoding="utf-8"))
            self.assertIn(calibration["chosen_weights"], [[1.0, 0.0, 0.0], [0.75, 0.25, 0.0],
                                                          [0.75, 0.0, 0.25], [0.5, 0.5, 0.0],
                                                          [0.5, 0.25, 0.25], [0.5, 0.0, 0.5]])
            self.assertEqual(calibration["selection_scope"], "source_val_unit_E1_macro_recall")
            self.assertEqual(calibration["source_val_unit_count"], 4)
            info = json.loads((full / "eval_info.json").read_text(encoding="utf-8"))
            self.assertEqual((info["dataset"], info["fold"], info["seed"], info["cache_version"]),
                             ("paderborn", 0, 0, cache_version))
            self.assertEqual(info["fusion"]["selection"]["unit_count"], 4)
            self.assertIn("deep_original", info["fusion"]["uncertainty"])
            self.assertEqual(set(info["fusion"]["versions"]),
                             {"fusion_full", "fusion_without_phys", "fusion_without_normal"})
            # 不确定报告覆盖四个版本，主PU/KB分开报覆盖与低置信名单。
            self.assertEqual(set(info["fusion"]["uncertainty"]),
                             {"fusion_full", "fusion_without_phys", "fusion_without_normal",
                              "deep_original"})
            deep_scopes = info["fusion"]["uncertainty"]["deep_original"]["scopes"]
            self.assertEqual(deep_scopes["main_pu"]["unit_count"], 2)
            self.assertEqual(deep_scopes["kb"]["unit_count"], 1)
            # Normal 证据分布与 a<=t 辅助判定单独落 info，不参与融合 argmax。
            evidence = info["fusion"]["normal_evidence_distribution"]
            self.assertEqual((evidence["main_pu"]["record_count"], evidence["kb"]["record_count"]),
                             (40, 20))
            self.assertEqual(evidence["main_pu"]["mean_window_fraction_at_or_below_threshold"], 1.0)
            # 错误互补表按冻结清单统计 deep 与 phys_ies 硬判（非 q_phys）。
            complement = info["fusion"]["error_complementarity"]
            self.assertEqual(complement["all_primary"]["unit_count"], 2)
            self.assertEqual(sum(complement["all_primary"]["cells"].values()), 2)
            self.assertEqual(complement["all_primary"]["P_ball"], {"deep": 0.0, "phys_ies": 0.0})
            # 消融目录记录自己的版本与权重（物理位置零）。
            ablated = json.loads((full.parent / f"{full.name}_without_phys" / "eval_info.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(ablated["fusion_version"], "fusion_without_phys")
            self.assertEqual(ablated["weights_used"][2], 0.0)
            # 完整融合目录恰有五份产物，无额外训练产物。
            self.assertEqual(sorted(p.name for p in full.iterdir()),
                             ["calibration.json", "eval_info.json", "metrics.json",
                              "predictions.csv", "unit_predictions.csv"])
            # 记录级与单元级 CSV 覆盖一致：60 条 PU 记录、3 个单元。
            records_back = evaluate.read_predictions(full / "predictions.csv")
            units_back = evaluate.read_predictions(full / "unit_predictions.csv", unit=True)
            self.assertEqual(len(records_back), 60)
            self.assertEqual({row["unit_key"] for row in units_back}, {"U000/C", "U015/C", "KB23/C"})


if __name__ == "__main__":
    unittest.main()
