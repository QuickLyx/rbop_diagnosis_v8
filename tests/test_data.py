# -*- coding: utf-8 -*-
"""数据层小样本检查：不训练、不重估 BN、不生成全量缓存。

v8 版：v6 纯函数已在本地 `dataset.py`，不再经 `_v6` 动态加载；
冻结事实、PU 枚举/丢弃凭据、缓存身份校验口径与 v7 测试一致。
"""
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly
import torch

from bearing_facts import BY_ID, CONDITIONS
from datasets import cwru, paderborn
from datasets.recording import Recording, to_windows
import dataset as v6ds
import features
from preprocessing import build_repr_config, config_from_saved, extract_logspec

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_frozen_record_and_exact_resampling(self):
        record = Recording("r", "u", Path("test.mat"), "cwru", "DE", 0, 12000)
        with self.assertRaises(FrozenInstanceError):
            record.label = 1
        signal = np.random.default_rng(0).normal(size=8193)
        for rate, up, down in ((64000, 3, 16), (48000, 1, 4), (12000, 1, 1)):
            result = to_windows(signal, rate)
            expected = signal if rate == 12000 else resample_poly(signal, up, down)
            np.testing.assert_array_equal(result, expected[:len(result) * 512].reshape(-1, 512))
        self.assertEqual(to_windows(np.arange(1025), 12000).shape, (2, 512))
        for signal in ([], [np.nan] * 512, [np.inf] * 512, np.zeros((2, 512)),
                       np.zeros(511), np.ones(512, dtype=complex)):
            with self.subTest(shape=np.asarray(signal).shape):
                with self.assertRaises(ValueError):
                    to_windows(signal, 12000)
        for rate in (0, -12000, 12000.5, True):
            with self.assertRaises(ValueError):
                to_windows(np.zeros(512), rate)
        with self.assertRaises(ValueError):
            to_windows(np.zeros(512), 64000)

    def test_record_signal_resamples_once_and_matches_to_windows(self):
        """整记录信号只重采样一次，截断后与 `to_windows` 逐位一致（§3.1）。"""
        signal = np.random.default_rng(5).normal(size=8193)
        for rate in (12000, 48000, 64000):
            record = Recording("r", "u", Path("test.mat"), "cwru", "DE", 0, rate)
            with patch.object(cwru, "read_signal", return_value=signal):
                full = features._record_signal_12k(record)
            windows = to_windows(signal, rate)
            np.testing.assert_array_equal(full[: len(windows) * 512].reshape(-1, 512), windows)
            self.assertGreaterEqual(len(full), len(windows) * 512)  # 末尾不足一窗部分被保留
        # 不足 2048 点的记录在构建时直接报错，不得发布没有分析窗的缓存
        records = [r for r in cwru.enumerate_recordings() if r.src_fs == 12000][:2]
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru={"records": 2, "units": 1, "windows": 6}), \
                patch.object(cwru, "enumerate_recordings", return_value=records), \
                patch.object(cwru, "read_signal", return_value=np.arange(1536.0)):
            with self.assertRaisesRegex(ValueError, "2048"):
                features.build_cache("cwru", directory, execute=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_logspec_config_roundtrip_and_finiteness(self):
        X = extract_logspec(np.random.default_rng(0).normal(size=(3, 512)))
        self.assertEqual(X.shape, (3, 1, 256))
        self.assertEqual(X.dtype, np.float32)
        self.assertLess(np.max(np.abs(X.mean(axis=2))), 1e-5)
        self.assertEqual(config_from_saved(build_repr_config(), True), build_repr_config())
        with self.assertRaises(ValueError):
            config_from_saved({"seg_len": 1024, "input_dim": 256, "n_channels": 1})
        with self.assertRaises(ValueError):
            extract_logspec(np.zeros((2, 512)) + np.nan)

    def test_cwru_counts_folds_and_channel_keys(self):
        records = cwru.enumerate_recordings()
        self.assertEqual(len(records), 306)
        self.assertEqual(len({r.unit_key for r in records}), 153)
        files = v6ds.enumerate_files(cwru.DEFAULT_DATA_ROOT, cwru.SOURCES, exclude_028=True)
        # v8 四折与 v6 `split_source_files` 逐键一致（同一随机数消耗与排序）
        self.assertEqual(cwru.split_source_units(records), v6ds.split_source_files(files))
        folds = cwru.split_source_units(records)
        for fold in folds:
            self.assertFalse(set(fold["train"]) & set(fold["val"]))
            self.assertEqual(len(fold["train"]) + len(fold["val"]), 153)
        self.assertEqual(len({k for fold in folds for k in fold["val"]}), 153)
        with self.assertRaises(ValueError):
            cwru.split_source_units([records[0], records[0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "B007_0.mat"
            record = Recording("12k_DE/B007_0.mat::DE", "12k_DE/B007_0.mat", path,
                               "cwru", "DE", 1, 12000, source="12k_DE")
            savemat(path, {"X001_DE_time": np.arange(1024), "X002_DE_time": np.arange(1024)})
            with self.assertRaises(ValueError):
                cwru.read_signal(record)
            savemat(path, {"X001_FE_time": np.arange(1024)})
            with self.assertRaises(ValueError):
                cwru.read_signal(record)
            savemat(path, {"X001_DE_time": np.arange(1024)})
            np.testing.assert_array_equal(cwru.read_signal(record), np.arange(1024))
            special = replace(record, path=Path(directory) / "Normal_2.mat", source="Normal",
                              unit_key="Normal/Normal_2.mat", record_key="Normal/Normal_2.mat::DE")
            savemat(special.path, {"X001_DE_time": np.arange(1024), "X099_DE_time": np.arange(1024) + 1})
            np.testing.assert_array_equal(cwru.read_signal(special), np.arange(1024) + 1)

    def test_pu_channels_unique_and_vibration_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "N15_M07_F10_K001_1.mat"
            def save(names):
                channels = np.empty(len(names), dtype=[("Name", object), ("Data", object)])
                for index, name in enumerate(names):
                    channels[index] = (name, np.arange(1024))
                savemat(path, {path.stem: {"Y": channels}})
            save(["vibration_1", "speed"])
            self.assertEqual(set(paderborn.read_channels(path)), {"vibration_1", "speed"})
            record = Recording("K001/test.mat::vibration_1", "K001/N15_M07_F10", path,
                               "paderborn", "vibration_1", 0, 64000)
            np.testing.assert_array_equal(paderborn.read_signal(record), np.arange(1024))
            for names in (["vibration_1", "vibration_1"], ["speed"]):
                save(names)
                with self.assertRaises(ValueError):
                    paderborn.read_channels(path)
            savemat(path, {"wrong": {"Y": []}})
            with self.assertRaises(ValueError):
                paderborn.read_channels(path)

    def test_pu_counts_and_non_overwriting_drop_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "extracted"
            # 这里只枚举文件名，临时空文件即可覆盖完整口径与 KA08 丢弃凭据。
            for bearing_id in BY_ID:
                folder = root / bearing_id
                folder.mkdir(parents=True)
                for condition in CONDITIONS:
                    for rep in range(1, 21):
                        (folder / f"{condition}_{bearing_id}_{rep}.mat").touch()
            dropped_path = Path(directory) / "dropped.json"
            primary = paderborn.enumerate_recordings(root, dropped_path=dropped_path)
            all_records = paderborn.enumerate_recordings(root, include_kb=True, dropped_path=dropped_path)
            self.assertEqual(len(primary), 2319)
            self.assertEqual(len(all_records), 2559)
            self.assertEqual(len({r.unit_key for r in primary}), 116)
            self.assertEqual(sum(r.label is None for r in all_records), 240)
            dropped = json.loads(dropped_path.read_text(encoding="utf-8"))
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0]["filename"], "N15_M01_F10_KA08_2.mat")
            self.assertFalse(any("N15_M01_F10_KA08_2.mat" in r.record_key for r in all_records))
            self.assertEqual(sum(r.unit_key == "KA08/N15_M01_F10" for r in all_records), 19)
            dropped_path.write_text("[]", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                paderborn.enumerate_recordings(root, dropped_path=dropped_path)
            self.assertEqual(dropped_path.read_text(encoding="utf-8"), "[]")

    def test_cache_identity_accepts_windows_and_posix_paths_without_weakening_checks(self):
        condition = next(iter(CONDITIONS))
        bearing = BY_ID["K001"]
        target = []
        for index in range(20):
            filename = f"{condition}_K001_{index + 1}.mat"
            target.append(Recording(f"K001/{filename}::vibration_1", f"K001/{condition}",
                                    Path(filename), "paderborn", "vibration_1", bearing.label,
                                    64000, "paderborn", "K001", condition, bearing.origin, bearing.severity))
        for dataset, recordings, counts in (
                ("cwru", cwru.enumerate_recordings()[:2], {"records": 2, "units": 1, "windows": 2}),
                ("paderborn", target, {"records": 20, "units": 1, "windows": 20,
                                       "primary_records": 20, "primary_units": 1, "primary_windows": 20})):
            X = np.zeros((len(recordings), 1, 256), dtype=np.float32)
            for prefix in ("E:\\冻结缓存\\", "cache/"):
                records = [{**asdict(record), "path": prefix + record.path.name,
                            "start": index, "stop": index + 1}
                           for index, record in enumerate(recordings)]
                payload = {"version": features.CACHE_VERSION, "dataset": dataset,
                           "script": "features.py", "repr_config": build_repr_config(),
                           "phys_config": features.PHYS_CONFIG,
                           "records": records, "counts": counts}
                with self.subTest(dataset=dataset, prefix=prefix), \
                        patch.dict(features.EXPECTED_COUNTS, {dataset: counts}), \
                        patch.object(features, "Path", PurePosixPath):
                    # 身份校验不需要数组内容；analysis 区间在此构造为空窗特例不可行，
                    # 因此直接调用 _validate_identity（完整校验由 roundtrip 测试覆盖）。
                    features._validate_identity(dataset, records)
                    records[0]["path"] = prefix + "wrong.mat"
                    with self.assertRaisesRegex(ValueError, "身份"):
                        features._validate_identity(dataset, records)
                    # 逐记录 RPM 数值/来源必须显式且与冻结表一致
                    records[0]["path"] = prefix + recordings[0].path.name
                    filename = records[0]["path"].replace("\\", "/").rsplit("/", 1)[-1]
                    rpm, source = ((1797.0, "field") if dataset == "cwru"
                                   else (1500.0, "condition"))
                    records[0].update(rpm=rpm, rpm_source=source)
                    features._validate_rpm(dataset, records[0], filename)
                    for rpm, source in ((0.0, "field"), (1797.0, "other"),
                                        (1799.0, "fallback")):
                        records[0].update(rpm=rpm, rpm_source=source)
                        with self.assertRaises(ValueError):
                            features._validate_rpm(dataset, records[0], filename)
                    if dataset == "cwru":  # 载荷档回退必须落在冻结表内
                        records[0].update(rpm=1797.0, rpm_source="fallback")
                        features._validate_rpm(dataset, records[0], filename)

    def test_small_cache_roundtrip_rejects_stale_or_corrupt_metadata(self):
        records = cwru.enumerate_recordings()[:2]
        # 信号 patch 为 8192 点：48k 来源重采样 1/4 后 2048 点 = 4 窗，每记录 1 个分析窗。
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru={"records": 2, "units": 1, "windows": 8}), \
                patch.object(cwru, "enumerate_recordings", return_value=records), \
                patch.object(cwru, "read_signal", return_value=np.arange(8192)) as read:
            cache = features.build_cache("cwru", directory, execute=True)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(cache.X.shape, (8, 1, 256))
            self.assertEqual(cache.phys25.shape, (8, 25))
            self.assertEqual(cache.ies_analysis.shape, (2, 16))
            self.assertFalse(cache.X.flags.writeable)
            self.assertIsInstance(cache.records[0]["path"], str)
            _close(cache)
            # 已有产物必须显式复用，不静默重建也不静默校验
            with self.assertRaises(FileExistsError):
                features.build_cache("cwru", directory, execute=True)
            self.assertEqual(read.call_count, 2)
            # 显式复用直接命中且不重建
            cache = features.build_cache("cwru", directory, execute=True, reuse_cache=True)
            self.assertEqual(read.call_count, 2)
            subset = features.feature_subset(cache, channels=["FE"])
            self.assertEqual(subset.X.shape, (4, 1, 256))
            self.assertEqual(subset.X.dtype, torch.float32)
            self.assertEqual(subset.y.dtype, torch.long)
            self.assertEqual((subset.records[0]["start"], subset.records[0]["stop"]), (0, 4))
            self.assertEqual(cache.records[1]["start"], 4)
            empty = features.feature_subset(cache, unit_keys=[])
            self.assertEqual(empty.X.shape, (0, 1, 256))
            with self.assertRaises(KeyError):
                features.feature_subset(cache, unit_keys=["unknown"])
            metadata_path = Path(directory) / "cwru.json"
            original = metadata_path.read_text(encoding="utf-8")
            for mutation in (lambda p: p.update(version="stale"),
                             lambda p: p["repr_config"].update(fs=64000),
                             lambda p: p["counts"].update(windows=3),
                             lambda p: p["records"][1].update(start=0),
                             lambda p: p["records"][0].pop("origin"),
                             lambda p: p["records"][1].update(record_key=p["records"][0]["record_key"]),
                             lambda p: p["records"][0].update(analysis_start=9),
                             lambda p: p["records"][0].update(rpm=None),
                             lambda p: p["records"][0].update(rpm_source="other"),
                             lambda p: p["records"][0].update(map_ties=1),
                             lambda p: p["records"][0].update(constant_windows=99),
                             lambda p: p["records"][0].update(
                                 analysis_stop=p["records"][0]["analysis_start"]),
                             lambda p: (p["records"][0].update(stop=3),
                                        p["records"][1].update(start=3))):
                payload = json.loads(original)
                mutation(payload)
                metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    features.load_cache("cwru", directory)
            metadata_path.write_text(original, encoding="utf-8")
            _close(cache)
            del cache
            # 数组内容损坏（形状/非有限）必须报错：直接写坏文件，不 mock np.load。
            np.save(Path(directory) / "cwru.npy", np.zeros((3, 256), np.float32))
            with self.assertRaises(ValueError):
                features.load_cache("cwru", directory)
            np.save(Path(directory) / "cwru.npy", np.full((8, 1, 256), np.nan, np.float32))
            with self.assertRaises(ValueError):
                features.load_cache("cwru", directory)

    def test_failed_read_releases_windows_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in
                     ("cwru.npy", "cwru.json", "cwru_phys25.npy", "cwru_ies_analysis.npy")]
            np.save(paths[0], np.zeros((2, 1, 256), np.float32))
            np.save(paths[2], np.zeros((2, 25), np.float32))
            np.save(paths[3], np.zeros((0, 16), np.float32))
            paths[1].write_text(json.dumps({"version": "stale"}), encoding="utf-8")
            saved_error = None
            try:
                features.load_cache("cwru", directory)
            except ValueError as error:
                saved_error = error
                # 保留异常及其栈时仍能删除文件，才证明映射已经释放。
                for path in paths:
                    path.unlink()
            self.assertIsNotNone(saved_error)
            self.assertIn("过期", str(saved_error))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unlabeled_kb_not_silently_filtered(self):
        condition = next(iter(CONDITIONS))
        records = []
        for bearing_id in ("K001", "KB23"):
            bearing = BY_ID[bearing_id]
            for rep in range(1, 21):
                filename = f"{condition}_{bearing_id}_{rep}.mat"
                records.append(Recording(f"{bearing_id}/{filename}::vibration_1",
                                         f"{bearing_id}/{condition}", Path(bearing_id) / filename,
                                         "paderborn", "vibration_1", bearing.label, 64000,
                                         "paderborn", bearing_id, condition, bearing.origin, bearing.severity))
        # 40960 点/记录，64k 重采样 3/16 后 7680 点 = 15 窗；每轴承 300 窗。
        counts = {"records": 40, "units": 2, "windows": 600,
                  "primary_records": 20, "primary_units": 1, "primary_windows": 300}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, paderborn=counts), \
                patch.object(paderborn, "enumerate_recordings", return_value=records) as enum, \
                patch.object(paderborn, "read_signal", return_value=np.arange(40960)):
            cache = features.build_cache("paderborn", directory, execute=True)
            enum.assert_called_once_with(None, include_kb=True)
            with self.assertRaises(ValueError):
                features.feature_subset(cache)
            subset = features.feature_subset(cache, labeled=False)
            self.assertEqual(len(subset.X), 600)
            self.assertTrue(torch.all(subset.y == -1))
            primary = features.feature_subset(cache, unit_keys=[f"K001/{condition}"])
            self.assertEqual(len(primary.X), 300)
            self.assertTrue(torch.all(primary.y == 0))
            # 无标签入口不访问真值字段，避免纯迁移路径意外依赖标签。
            stripped = SimpleNamespace(X=cache.X, phys25=cache.phys25,
                                       ies_analysis=cache.ies_analysis,
                                       records=[{k: v for k, v in r.items() if k != "label"}
                                                for r in cache.records])
            self.assertTrue(torch.all(features.feature_subset(stripped, labeled=False).y == -1))
            # 关闭 cache 自身 mmap 后再删目录；子集的 ies_analysis 引用同一 mmap，需一并释放。
            _close(cache, subset, primary, stripped)
            del cache, stripped, subset, primary

    def test_failed_build_and_partial_cache_not_success(self):
        records = cwru.enumerate_recordings()[:2]
        counts = {"records": 2, "units": 1, "windows": 8}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=records), \
                patch.object(cwru, "read_signal", side_effect=[np.arange(8192), ValueError("模拟坏信号")]):
            with self.assertRaises(ValueError):
                features.build_cache("cwru", directory, execute=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            np.save(Path(directory) / "cwru.npy", np.zeros((8, 1, 256), np.float32))
            with self.assertRaises(FileExistsError):
                features.build_cache("cwru", directory, execute=True)
            with self.assertRaises(FileNotFoundError):
                features.load_cache("cwru", directory)
            with self.assertRaises(FileNotFoundError):
                features.build_cache("cwru", directory, execute=True, reuse_cache=True)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=[records[0], records[0]]), \
                patch.object(cwru, "read_signal", return_value=np.arange(8192)):
            with self.assertRaisesRegex(ValueError, "重复"):
                features.build_cache("cwru", directory, execute=True)
            self.assertEqual(list(Path(directory).iterdir()), [])


def _close(*caches):
    """Windows 测试辅助：关闭缓存内全部 mmap，避免临时目录删除失败。"""
    for cache in caches:
        for name in ("X", "phys25", "ies_analysis"):
            array = getattr(cache, name, None)
            if isinstance(array, np.memmap):
                array._mmap.close()


if __name__ == "__main__":
    unittest.main()
