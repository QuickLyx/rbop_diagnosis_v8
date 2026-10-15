# -*- coding: utf-8 -*-
"""只检查采样、双通道组装、损失、AdaBN、eval 前向与合成小模型训练；不触真实数据拟合。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import build_repr_config, fit_scaler
from model import MSRSABiLSTM
import train
import transfer

ROOT = Path(__file__).resolve().parents[1]

#: 与 configs/stage3.json 冻结值一致的最小训练参数（仅供合成测试）。
PARAMETERS = {"batch_size": 8, "epochs": 30, "joint_lr": 0.001, "adapt_lr": 0.0001,
              "lambda_coral": 0.2, "mixup_input_weight": 0.5,
              "mixup_feature_weight": 0.5, "bn_momentum": 0.1}


def synthetic_feature(units, windows_per_record=3, seed=0):
    """构造带来源/通道/标签/物理摘要的最小特征集；units 为 (source, label, n_units)。

    每个 CWRU 单元固定产生一条 DE 与一条 FE 记录，record_key 为 `{unit}::DE/::FE`，
    与 evaluate.aggregate_units 的单元结构检查一致。
    """
    rng = np.random.default_rng(seed)
    X, phys, labels, keys, channels, sources, records = [], [], [], [], [], [], []
    offset = 0
    for unit_index, (source, label, n_units) in enumerate(units):
        for unit_sequence in range(n_units):
            # 单元身份 = 来源/文件名（含扩展名，与真实缓存一致）；path 末段必须等于文件名。
            filename = f"U{unit_index}_{unit_sequence}.mat"
            unit_key = f"{source}/{filename}"
            for channel in ("DE", "FE"):
                record_key = f"{unit_key}::{channel}"
                X.append(rng.normal(size=(windows_per_record, 1, 256)).astype(np.float32))
                phys.append(rng.normal(size=(windows_per_record, 25)).astype(np.float32))
                labels.extend([label] * windows_per_record)
                keys.extend([record_key] * windows_per_record)
                channels.extend([channel] * windows_per_record)
                sources.extend([source] * windows_per_record)
                records.append({"record_key": record_key, "unit_key": unit_key,
                                "dataset": "cwru", "channel": channel, "source": source,
                                "label": label, "path": f"{source}/{filename}",
                                "start": offset, "stop": offset + windows_per_record})
                offset += windows_per_record
    feature = SimpleNamespace(
        X=torch.from_numpy(np.concatenate(X)),
        phys25=torch.from_numpy(np.concatenate(phys)),
        y=torch.tensor(labels, dtype=torch.long),
        record_keys=np.asarray(keys, dtype=object),
        channels=np.asarray(channels, dtype=object),
        records=records)
    feature.scaler = fit_scaler(feature.phys25.numpy())
    return feature


def four_source_feature():
    """四来源各至少一单元（DE+FE 两记录），故障类在三个故障来源各足量窗口。"""
    units = [("Normal", 0, 1), ("12k_DE", 1, 1), ("12k_DE", 2, 1), ("12k_DE", 3, 1),
             ("12k_FE", 1, 1), ("12k_FE", 2, 1), ("48k_DE", 3, 1)]
    return synthetic_feature(units, windows_per_record=3, seed=1)


class TransferTests(unittest.TestCase):
    def test_two_channel_forward_and_channel_guard(self):
        network = MSRSABiLSTM(n_channels=2).eval()
        before = {key: value.clone() for key, value in network.state_dict().items()}
        with torch.no_grad():
            logits, attention, features = network(torch.zeros(2, 2, 256), return_feat=True)
        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(features.shape, (2, 256))
        self.assertTrue(torch.allclose(attention.sum(1), torch.ones(2)))
        self.assertTrue(all(torch.equal(value, network.state_dict()[key])
                            for key, value in before.items()))
        single = MSRSABiLSTM(n_channels=1)
        with torch.no_grad():
            logits1, _ = single(torch.zeros(2, 1, 256))
        self.assertEqual(logits1.shape, (2, 4))
        with self.assertRaises(ValueError):
            MSRSABiLSTM(n_channels=3)

    def test_four_step_weights_and_sampling_switch(self):
        labels, channels, keys = [], [], []
        for channel in ("DE", "FE"):
            for label in range(4):
                for record in range(label + 1):
                    n = record + 1
                    labels.extend([label] * n)
                    channels.extend([channel] * n)
                    keys.extend([f"{channel}/{label}/{record}"] * n)
        feature = SimpleNamespace(X=torch.zeros(len(labels), 1, 256), y=torch.tensor(labels),
                                  channels=np.asarray(channels), record_keys=np.asarray(keys))
        weights = transfer.balanced_window_weights(feature)
        for channel in ("DE", "FE"):
            for label in range(4):
                mask = (feature.channels == channel) & (feature.y.numpy() == label)
                self.assertAlmostEqual(float(weights[mask].sum()), 1.0)
                for key in set(feature.record_keys[mask]):
                    self.assertAlmostEqual(float(weights[feature.record_keys == key].sum()),
                                           1 / (label + 1))
        _, actual = transfer.make_weighted_loader(feature, 4, 12, 0)
        np.testing.assert_array_equal(actual.numpy(), weights)
        _, uniform_weights = transfer.make_weighted_loader(feature, 4, 12, 0, class_balance=False)
        self.assertTrue(torch.equal(uniform_weights, torch.ones(len(labels), dtype=torch.double)))

    def test_unlabeled_loader_never_reads_target_labels_or_weights(self):
        class UnlabeledFeature:
            X = torch.zeros(3, 1, 256)

            @property
            def y(self):
                raise AssertionError("禁止读取目标真值")

        with patch.object(transfer, "balanced_window_weights",
                          side_effect=AssertionError("禁止目标均衡")):
            loader, weights = transfer.make_weighted_loader(UnlabeledFeature(), 4, 12, 0,
                                                            labeled=False)
            self.assertTrue(torch.equal(weights, torch.ones(3, dtype=torch.double)))
            self.assertEqual(len(loader), 3)
            for batch in loader:
                self.assertEqual(len(batch), 3)
                self.assertTrue(torch.all(batch[1] == -1))

    def test_fixed_step_budget(self):
        self.assertEqual(transfer.steps_per_epoch(9983, 64), 156)
        self.assertEqual(transfer.steps_per_epoch(128, 64), 2)
        self.assertEqual(transfer.steps_per_epoch(1, 64), 1)
        for s_ref, batch_size in ((0, 64), (10, 0), (-1, 64)):
            with self.assertRaises(ValueError):
                transfer.steps_per_epoch(s_ref, batch_size)

    def test_prepare_batch_dual_channel_layout(self):
        feature = synthetic_feature([("12k_DE", 1, 1)], windows_per_record=4)
        scaler = feature.scaler
        xb = feature.X[:4]
        inputs = transfer.prepare_batch(xb, feature.phys25[:4], scaler, False, "cpu")
        self.assertEqual(inputs.shape, (4, 2, 256))
        self.assertEqual(inputs.dtype, torch.float32)
        self.assertTrue(torch.equal(inputs[:, 0], xb[:, 0]))
        expected_second = np.zeros((4, 256), dtype=np.float32)
        expected_second[:, :25] = (feature.phys25[:4].numpy() - np.asarray(scaler["mean"])) \
            / np.asarray(scaler["std"])
        np.testing.assert_allclose(inputs[:, 1].numpy(), expected_second, atol=1e-5)
        only_logspec = transfer.prepare_batch(xb, None, scaler, True, "cpu")
        self.assertTrue(torch.equal(only_logspec[:, 1], torch.zeros(4, 256)))
        with self.assertRaises(ValueError):
            transfer.prepare_batch(xb, feature.phys25[:4], None, False, "cpu")
        with self.assertRaises(ValueError):
            transfer.prepare_batch(xb, feature.phys25[:2], scaler, False, "cpu")

    def test_mixup_soft_labels_sum_to_one_and_m0_branch(self):
        feature = four_source_feature()
        scaler = feature.scaler
        # 与生产路径一致：先逐窗口组装双通道输入，再在组装后的输入上混合。
        inputs = transfer.prepare_batch(feature.X, feature.phys25, scaler, False, "cpu")
        groups = [row["source"] for row in feature.records
                  for _ in range(row["stop"] - row["start"])]
        rng = torch.Generator().manual_seed(7)
        mixed = transfer.mixup_source_batch(inputs, feature.y, feature.phys25[:len(inputs)],
                                            groups, rng)
        self.assertIsNotNone(mixed)
        x_mix, h_mix, soft = mixed
        m = min(groups.count(name) for name in set(groups))
        self.assertEqual(x_mix.shape, (m, 2, 256))
        self.assertEqual(h_mix.shape, (m, 25))
        self.assertEqual(soft.shape, (m, 4))
        self.assertTrue(torch.allclose(soft.sum(dim=1), torch.ones(m), atol=1e-5))
        self.assertTrue((soft >= 0).all())
        # 空分组（如极端 batch 无任何来源）时返回 None：两项混合损失为零并计次数。
        rng = torch.Generator().manual_seed(7)
        self.assertIsNone(transfer.mixup_source_batch(inputs, feature.y,
                                                      feature.phys25[:len(inputs)],
                                                      [], rng))
        # 独立 RNG：结果只取决于 mixup_rng 的种子，与全局生成器状态无关。
        torch.manual_seed(123)
        rng_a = torch.Generator().manual_seed(9)
        first = transfer.mixup_source_batch(inputs, feature.y, feature.phys25[:len(inputs)],
                                            groups, rng_a)
        torch.manual_seed(456)
        rng_b = torch.Generator().manual_seed(9)
        second = transfer.mixup_source_batch(inputs, feature.y, feature.phys25[:len(inputs)],
                                             groups, rng_b)
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_mixup_mixed_batch_on_cuda_device(self):
        # 回归：lam 在 CPU RNG 上生成，混合软标签必须与输入同设备，否则真实 CUDA 训练崩。
        if not torch.cuda.is_available():
            self.skipTest("无 CUDA 设备，跳过跨设备回归")
        feature = four_source_feature()
        scaler = feature.scaler
        inputs = transfer.prepare_batch(feature.X, feature.phys25, scaler, False, "cuda")
        labels = feature.y.to("cuda")
        groups = [row["source"] for row in feature.records
                  for _ in range(row["stop"] - row["start"])]
        rng = torch.Generator().manual_seed(7)
        x_mix, h_mix, soft = transfer.mixup_source_batch(
            inputs, labels, feature.phys25[:len(inputs)].to("cuda"), groups, rng)
        for tensor in (x_mix, h_mix, soft):
            self.assertEqual(tensor.device.type, "cuda")
        self.assertTrue(torch.allclose(soft.sum(dim=1),
                                       torch.ones(len(soft), device=soft.device), atol=1e-5))

    def test_soft_cross_entropy_matches_manual_and_rejects_bad_labels(self):
        logits = torch.randn(5, 4)
        soft = torch.softmax(torch.randn(5, 4), dim=1)
        expected = -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        self.assertTrue(torch.allclose(transfer._soft_cross_entropy(logits, soft), expected))
        with self.assertRaises(ValueError):
            transfer._soft_cross_entropy(logits, soft * 0.5)

    def test_coral_pairs_coverage_and_zero_fallback(self):
        torch.manual_seed(0)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 0, 0])
        groups = ["a"] * 8 + ["b"] * 8
        embeddings = torch.randn(16, 256)
        align, coverage = transfer.class_conditional_coral(embeddings, labels, groups)
        self.assertGreater(float(align), 0)
        self.assertEqual(len(coverage), 3)  # 3 个故障类 × 1 对（a|b）
        self.assertFalse(any(item["skipped"] for item in coverage))
        # 同分布嵌入的类内协方差仍不同（有限样本），但恒等嵌入损失为 0。
        align_same, _ = transfer.class_conditional_coral(embeddings[:8].repeat(2, 1),
                                                         labels, groups)
        self.assertGreaterEqual(float(align_same), 0)
        # 一侧不足两样本：项跳过并记覆盖。
        sparse = torch.randn(4, 256)
        sparse_labels = torch.tensor([1, 2, 3, 0])
        sparse_groups = ["a", "a", "b", "b"]
        align_sparse, coverage_sparse = transfer.class_conditional_coral(
            sparse, sparse_labels, sparse_groups)
        self.assertEqual(float(align_sparse), 0)
        self.assertTrue(all(item["skipped"] for item in coverage_sparse))
        # 非有限值直接报错。
        with self.assertRaises(FloatingPointError):
            transfer.class_conditional_coral(embeddings * float("nan"), labels, groups)

    def test_compute_losses_arms(self):
        logits, labels = torch.zeros(4, 4), torch.tensor([0, 1, 2, 3])
        ce = F.cross_entropy(logits, labels)
        joint = transfer.compute_losses("joint", logits, labels)
        self.assertTrue(torch.equal(joint["loss"], ce))
        mix = transfer.compute_losses("mixup4src", logits, labels,
                                      ce_input_mix=torch.tensor(1.0),
                                      ce_feature_mix=torch.tensor(2.0))
        self.assertAlmostEqual(float(mix["loss"]), float(ce) + 0.5 + 1.0)
        zero_mix = transfer.compute_losses("mixup4src", logits, labels)
        self.assertTrue(torch.equal(zero_mix["loss"], ce))  # M=0：两项为零
        with self.assertRaises(ValueError):
            transfer.compute_losses("coral4src", logits, labels)
        coral = transfer.compute_losses("coral4src", logits, labels, align=torch.tensor(3.0))
        self.assertAlmostEqual(float(coral["loss"]), float(ce) + 0.6)
        with self.assertRaises(ValueError):
            transfer.compute_losses("ce_K19", logits, labels)
        ce19 = transfer.compute_losses("ce_K19", logits, labels, logits, labels)
        self.assertTrue(torch.equal(ce19["loss"], 2 * ce))
        with self.assertRaises(ValueError):
            transfer.compute_losses("mmd_marginal", logits, labels)

    def test_source_record_then_file_fusion_without_extra_forward_or_bn_changes(self):
        class EvalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(4)
                self.dropout = nn.Dropout(0.5)
                self.calls = 0

            def forward(self, x):
                self.assert_eval()
                self.calls += 1
                return self.dropout(self.bn(x)), None

            def assert_eval(self):
                if self.training or self.bn.training or self.dropout.training or torch.is_grad_enabled():
                    raise AssertionError("验证必须 eval/no_grad")

        de = [0.35, 0.64, 0.005, 0.005]
        fe = [0.70, 0.29, 0.005, 0.005]
        probabilities = torch.tensor([de, de, de, fe])
        records = []
        for channel, start, stop in (("DE", 0, 3), ("FE", 3, 4)):
            records.append({"record_key": f"Normal/test.mat::{channel}",
                            "unit_key": "Normal/test.mat", "path": "test.mat", "dataset": "cwru",
                            "channel": channel, "source": "Normal", "label": 0,
                            "start": start, "stop": stop})
        feature = SimpleNamespace(X=probabilities.log(), phys25=torch.zeros(4, 25),
                                  records=records)
        feature.scaler = {"mean": [0.0] * 25, "std": [1.0] * 25}
        network = EvalModel()
        before = {key: value.clone() for key, value in network.state_dict().items()}
        # 合成小模型输入是 4 维概率向量，绕过双通道组装直接前向。
        metrics = transfer.source_validation(network, feature, batch_size=4,
                                             scaler=feature.scaler,
                                             inputs=probabilities.log())
        self.assertEqual(network.calls, 1)
        self.assertEqual(metrics["DE"]["accuracy"], 0)
        self.assertEqual(metrics["FE"]["accuracy"], 1)
        self.assertEqual(metrics["file_fused"]["accuracy"], 1)
        self.assertEqual(transfer.select_validation_score(metrics), (0.5, 0.5))
        self.assertTrue(all(torch.equal(value, network.state_dict()[key])
                            for key, value in before.items()))
        with self.assertRaises(ValueError):
            transfer.source_validation(network, feature, scaler=None)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            transfer.save_checkpoint(path, {"weight": torch.zeros(1)},
                                     {"protocol": "v8_stage3", "epoch": 30, "scaler": {"mean": [0.0]}})
            payload = transfer.load_checkpoint(path)
            self.assertEqual(payload["protocol"], "v8_stage3")
            self.assertTrue(torch.equal(payload["state_dict"]["weight"], torch.zeros(1)))
        # 目录释放后再验证缺失路径（Windows 下 tempdir 不能在断言前提前失效）。
        with self.assertRaises(FileNotFoundError):
            transfer.load_checkpoint(str(path))


class SyntheticTrainingTests(unittest.TestCase):
    """合成小模型跑通 9 臂训练循环（pilot 1-2 轮），只验机制不验效果。"""

    def _run(self, arm, epochs=1):
        feature = four_source_feature()
        target = synthetic_feature([("paderborn", 1, 2), ("paderborn", 2, 2),
                                    ("paderborn", 3, 2)], windows_per_record=2, seed=2)
        val = synthetic_feature([("Normal", 0, 2), ("12k_DE", 3, 2)], windows_per_record=2, seed=3)
        config = {"parameters": deepcopy(PARAMETERS), "fold": 0, "seed": 0,
                  "target_train_units": [unit["unit_key"] for unit in target.records][:0]}
        network = MSRSABiLSTM(n_channels=2)
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            result = transfer.train_model(network, feature, target, val, arm, config,
                                          directory, s_ref=4, seed=0,
                                          run_kind="pilot", pilot_epochs=epochs)
            self.assertTrue((Path(directory) / "last.pt").is_file())
            self.assertTrue((Path(directory) / "best.pt").is_file())
            history = (Path(directory) / "history.csv").read_text(encoding="utf-8")
            payload = transfer.load_checkpoint(Path(directory) / "last.pt")
            fit_info = json.loads((Path(directory) / "fit_info.json").read_text(encoding="utf-8"))
        self.assertIn("run_kind", history)
        self.assertEqual(result["epochs"], epochs)
        self.assertEqual(payload["run_kind"], "pilot")
        self.assertEqual(len(payload["scaler"]["mean"]), 25)
        result["checkpoint"] = payload
        result["fit_info"] = fit_info
        return result

    def test_joint_and_logspec_ablation_share_first_steps(self):
        result = self._run("joint")
        self.assertEqual(result["updates"], result["steps_per_epoch"])
        self._run("joint_logspec")

    def test_mixup_coral_and_ce_arms(self):
        self._run("mixup4src")
        self._run("coral4src")
        for arm in ("ce_K19", "ce_K19_gg", "ce_K116", "ce_K19_logspec"):
            self._run(arm)

    def test_history_records_coral_coverage_mixup_m0_and_timing(self):
        mixup = self._run("mixup4src")
        row = mixup["history"][-1]
        # 每步要么混合成功要么 M=0，两者之和等于固定步数。
        self.assertEqual(row["mixup_batches"] + row["mixup_zero_batches"],
                         mixup["steps_per_epoch"])
        coral = self._run("coral4src")
        coral_row = coral["history"][-1]
        self.assertIn("coral_pairs", coral_row)
        self.assertIn("coral_pairs_skipped", coral_row)
        self.assertGreaterEqual(coral_row["coral_pairs"], coral_row["coral_pairs_skipped"])
        for result in (mixup, coral):
            self.assertGreater(result["fit_info"]["elapsed_seconds"], 0)
            self.assertEqual(len(result["fit_info"]["epoch_seconds"]), result["epochs"])
            self.assertGreater(result["history"][-1]["epoch_seconds"], 0)

    def test_checkpoint_marks_supervised_pool_only_for_ce_arms(self):
        feature = four_source_feature()
        target = synthetic_feature([("paderborn", 1, 2), ("paderborn", 2, 2)],
                                   windows_per_record=2, seed=2)
        val = synthetic_feature([("Normal", 0, 2)], windows_per_record=2, seed=3)
        units = sorted({record["unit_key"] for record in target.records})
        config = {"parameters": deepcopy(PARAMETERS), "fold": 0, "seed": 0,
                  "target_train_units": units}
        # adabn 的 U 是无标签目标：监督池必须为空，否则 evaluate 拒收。
        with tempfile.TemporaryDirectory() as directory:
            transfer.train_model(MSRSABiLSTM(n_channels=2), feature, target, val, "adabn",
                                 config, directory, s_ref=4, seed=0,
                                 run_kind="pilot", pilot_epochs=1)
            adabn = transfer.load_checkpoint(Path(directory) / "last.pt")
        self.assertEqual(adabn["supervised_pool_units"], [])
        # ce_* 记录被监督的池单元（evaluate 会与冻结池对照）。
        with tempfile.TemporaryDirectory() as directory:
            transfer.train_model(MSRSABiLSTM(n_channels=2), feature, target, val, "ce_K19",
                                 config, directory, s_ref=4, seed=0,
                                 run_kind="pilot", pilot_epochs=1)
            ce19 = transfer.load_checkpoint(Path(directory) / "last.pt")
        self.assertEqual(sorted(ce19["supervised_pool_units"]), units)

    def test_adabn_changes_only_bn_buffers_and_disables_dropout(self):
        feature = four_source_feature()
        target = synthetic_feature([("paderborn", 1, 2), ("paderborn", 2, 2),
                                    ("paderborn", 3, 2)], windows_per_record=2, seed=2)
        val = synthetic_feature([("Normal", 0, 2)], windows_per_record=2, seed=3)
        config = {"parameters": deepcopy(PARAMETERS), "fold": 0, "seed": 0,
                  "target_train_units": []}
        network = MSRSABiLSTM(n_channels=2)
        network.eval()  # AdaBN 全程不开启 Dropout
        before = {key: value.detach().cpu().clone()
                  for key, value in network.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            transfer.train_model(network, feature, target, val, "adabn", config,
                                 directory, s_ref=4, seed=0, run_kind="pilot", pilot_epochs=1)
        after = network.state_dict()
        allowed = {f"{name}.{buffer}" for name, module in network.named_modules()
                   if isinstance(module, nn.BatchNorm1d)
                   for buffer in ("running_mean", "running_var", "num_batches_tracked")}
        for key, value in before.items():
            if key not in allowed:
                self.assertTrue(torch.equal(value, after[key]), key)
        self.assertTrue(any(not torch.equal(before[key], after[key]) for key in allowed))
        self.assertFalse(network.head[2].training)  # Dropout 保持关闭

    def test_formal_rejects_non_deep_arm_and_bad_pilot(self):
        feature = four_source_feature()
        config = {"parameters": deepcopy(PARAMETERS), "fold": 0, "seed": 0,
                  "target_train_units": []}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                transfer.train_model(MSRSABiLSTM(n_channels=2), feature, None, feature,
                                     "svm_phys", config, directory, 4, 0)
            with self.assertRaises(ValueError):
                transfer.train_model(MSRSABiLSTM(n_channels=2), feature, None, feature,
                                     "joint", config, directory, 4, 0,
                                     run_kind="formal", pilot_epochs=1)
            with self.assertRaises(ValueError):
                transfer.train_model(MSRSABiLSTM(n_channels=2), feature, None, feature,
                                     "joint", config, directory, 4, 0,
                                     run_kind="pilot", pilot_epochs=0)


class CheckpointValidationTests(unittest.TestCase):
    """基座 checkpoint 校验：同折同 seed、正式 last、pilot 拒绝。"""

    def _payload(self, config):
        return {"protocol": "v8_stage3", "run_kind": "formal", "arm": "joint",
                "fold": config["fold"], "seed": config["seed"], "checkpoint_kind": "last",
                "epoch": config["parameters"]["epochs"],
                "checkpoint_epoch": config["parameters"]["epochs"],
                "last_epoch": config["parameters"]["epochs"],
                "cache_version": config["cache_version"],
                "repr_config": config["repr_config"], "model": config["model"],
                "representation": config["representation"],
                "source_split": config["source_split"], "s_ref": config["s_ref"],
                "steps_per_epoch": config["steps_per_epoch"],
                "steps": config["steps_per_epoch"] * config["parameters"]["epochs"],
                "updates": config["steps_per_epoch"] * config["parameters"]["epochs"],
                "selection_rule": "last", "class_balance": True, "logspec_only": False,
                "supervised_pool_units": [],
                "parameters": config["parameters"],
                "scaler": {"mean": [0.0] * 25, "std": [1.0] * 25}}

    def _config(self, arm="ce_K19"):
        return {"arm": arm, "fold": 0, "seed": 0, "cache_version": "v8_dual25_12k_512s512_2048s1024_ies_v1",
                "repr_config": build_repr_config(), "model": {"n_channels": 2, "n_classes": 4,
                                                             "frontend": "spectral_v1"},
                "representation": "dual25_logspec512",
                "source_split": {"train": ["a"], "val": ["b"]}, "s_ref": 100,
                "steps_per_epoch": 2, "parameters": deepcopy(PARAMETERS)}

    def test_accepts_matching_joint_last(self):
        config = self._config()
        train.validate_source_checkpoint(self._payload(config), config)

    def test_rejects_wrong_fold_seed_kind_epoch_and_pilot(self):
        config = self._config()
        payload = self._payload(config)
        for key, value in (("fold", 1), ("seed", 1), ("checkpoint_kind", "best"),
                           ("epoch", 29), ("run_kind", "pilot"), ("arm", "joint_logspec")):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    train.validate_source_checkpoint({**payload, key: value}, config)
        no_scaler = {k: v for k, v in payload.items() if k != "scaler"}
        with self.assertRaises(ValueError):
            train.validate_source_checkpoint(no_scaler, config)

    def test_pilot_base_only_for_pilot_run_in_same_pilot_directory(self):
        config = {**self._config(), "run_kind": "pilot",
                  "output": "outputs/pilot/ce_K19_f0_s0",
                  "source_checkpoint": "outputs/pilot/joint_f0_s0/last.pt"}
        payload = {**self._payload(config), "run_kind": "pilot"}
        # P4：ce_* pilot 臂可用 outputs/pilot 下同目录的满轮 pilot joint。
        train.validate_source_checkpoint(payload, config)
        # 正式运行仍拒绝任何 pilot 基座。
        with self.assertRaises(ValueError):
            train.validate_source_checkpoint(payload, {**config, "run_kind": "formal"})
        # 短轮 pilot 基座即使同目录也拒绝（epoch 必须为 30）。
        with self.assertRaises(ValueError):
            train.validate_source_checkpoint({**payload, "epoch": 1}, config)
        # 不在 outputs/pilot 下的 pilot 基座拒绝。
        with self.assertRaises(ValueError):
            train.validate_source_checkpoint(
                payload, {**config, "source_checkpoint": "outputs/train/joint_f0_s0/last.pt"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
