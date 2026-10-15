# -*- coding: utf-8 -*-
"""特征层小测试：不构建全量缓存、不训练；合成信号 + 临时目录小缓存。

覆盖计划 §12.1 第 1/2/3/4 条：
- np.arange(1025) 为 2 个 512 窗、起点 0/512；logspec 与 v7 逐位一致（静态加载纯 preprocessing）。
- 2048/1024 窗数、中心映射同分/首尾；合成调制信号频率维命中，FTF 不含 DC。
- `12k_FE::DE`、`48k_DE::FE` 来源几何正确；RPM 特殊键/回退/非法字段。
- 零/常量窗无 NaN/Inf；25 维顺序；缩放仅训练拟合；补零顺序；错误版本/区间报错。
"""
import json
from dataclasses import asdict
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

import dataset as v6ds
from bearing_facts import CWRU_DE_ORDERS, CWRU_FE_ORDERS
from datasets import cwru, paderborn
from datasets.recording import Recording, to_windows
import features
from preprocessing import build_repr_config, extract_logspec

ROOT = Path(__file__).resolve().parents[1]


def _make_cwru_mat(path, prefix, rpm=None, n=48001):
    """写一个最小合法 CWRU `.mat`：两通道 + 可选 RPM 标量。"""
    payload = {f"{prefix}_DE_time": np.arange(n, dtype=float),
               f"{prefix}_FE_time": np.arange(n, dtype=float) * 2}
    if rpm is not None:
        payload[f"{prefix}RPM"] = float(rpm)
    savemat(path, payload)


def _close_memmaps(*caches):
    """Windows 测试辅助：关闭缓存内全部 mmap，避免临时目录删除失败。"""
    for cache in caches:
        for name in ("X", "phys25", "ies_analysis"):
            array = getattr(cache, name, None)
            if isinstance(array, np.memmap):
                array._mmap.close()


def _legacy_rows(records, signals):
    """旧 v7 缓存口径的记录行：身份字段 + 连续 start/stop（无 analysis 区间）。"""
    rows, offset = [], 0
    for record, signal in zip(records, signals):
        n = signal.size // 512
        rows.append({**asdict(record), "path": str(record.path), "start": offset, "stop": offset + n})
        offset += n
    return rows


def _legacy_logspec(signals):
    return np.concatenate([extract_logspec(s[: (s.size // 512) * 512].reshape(-1, 512))
                           for s in signals])


def _write_legacy_cache(folder, rows, X, version=features.LEGACY_LOGSPEC_VERSION):
    """写一个最小合法 v7 logspec 缓存：版本、表征配置、记录索引与计数。"""
    payload = {"dataset": "cwru", "version": version, "script": "features.py",
               "repr_config": build_repr_config(),
               "counts": features._counts("cwru", rows), "records": rows}
    np.save(Path(folder) / "cwru.npy", X)
    (Path(folder) / "cwru.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FeatureTests(unittest.TestCase):
    def test_windows_and_bitwise_logspec_matches_v7(self):
        windows = to_windows(np.arange(1025), 12000)
        self.assertEqual(windows.shape, (2, 512))
        np.testing.assert_array_equal(windows[0], np.arange(512))
        np.testing.assert_array_equal(windows[1], np.arange(512, 1024))
        # 旧版纯函数只读作参照；禁写字节码，测试也不写入旧目录。
        sys.path.insert(0, str(ROOT.parent / "rbop_diagnosis_v7"))
        try:
            with patch.object(sys, "dont_write_bytecode", True):
                import _v6
                v7_extract = _v6.load_v6("preprocessing").extract_logspec
        finally:
            sys.path.pop(0)
        rng = np.random.default_rng(0)
        signals = rng.standard_normal((5, 512))
        np.testing.assert_array_equal(extract_logspec(signals), v7_extract(signals))

    def test_analysis_windows_and_center_mapping(self):
        self.assertEqual(features.analysis_window_count(3), 0)   # 1536 点不足 2048
        self.assertEqual(features.analysis_window_count(4), 1)   # 2048 点恰好一窗
        self.assertEqual(features.analysis_window_count(5), 1)   # 2560 点：j=1 需 3072 点，不完整
        self.assertEqual(features.analysis_window_count(6), 2)   # 3072 点：起点 0/1024 恰好两窗
        self.assertEqual(features.analysis_window_count(93), 45)  # 计划 §3.1 的 48001 点例
        # 单物理窗：全部 logspec 窗映射到 0
        np.testing.assert_array_equal(features.map_logspec_to_analysis(4, 1), [0, 0, 0, 0])
        # 首尾取最近：只有两端时中间窗不会越界
        mapping = features.map_logspec_to_analysis(93, 45)
        self.assertEqual(len(mapping), 93)
        self.assertTrue((mapping >= 0).all() and (mapping <= 44).all())
        # 同距取较早 j：logspec 中心 255.5 与物理中心 1023.5/2047.5 —— 构造对称点
        # n_windows=2、n_analysis=1 时窗 0/1 都映射 0；再验证 4 窗 2 分析窗的边界
        mapping = features.map_logspec_to_analysis(4, 2)
        self.assertEqual(mapping.tolist()[0], 0)
        self.assertEqual(mapping.tolist()[-1], 1)
        # 中心映射附带计数：唯一物理中心时 4 个 logspec 中心全在覆盖区外，且无同距候选
        index, ties, outside = features.mapping_detail(4, 1)
        self.assertEqual(index.tolist(), [0, 0, 0, 0])
        self.assertEqual((ties, outside), (0, 4))
        _, ties, outside = features.mapping_detail(93, 45)
        self.assertEqual(ties, 0)      # 512/1024 步长下中点不落在整窗中心
        self.assertGreater(outside, 0)  # 首尾中心在物理中心覆盖区外，计数必须报出

    def test_synthetic_modulation_hits_harmonics_without_dc(self):
        fs, rotation_hz, n = 12000, 25.0, 48001
        rng = np.random.default_rng(1)
        t_axis = np.arange(n) / fs
        bpfo_hz = CWRU_DE_ORDERS["BPFO"] * rotation_hz
        carrier = np.cos(2 * np.pi * 3000 * t_axis)
        signal = carrier * (1 + 0.8 * np.cos(2 * np.pi * bpfo_hz * t_axis)) \
            + 0.01 * rng.standard_normal(n)
        R = features.ies_features(signal, CWRU_DE_ORDERS, rotation_hz)
        self.assertEqual(R.shape, (45, 16))
        self.assertTrue(np.isfinite(R).all())
        bpfo_mean = R[:, 0:4].mean()
        others = np.concatenate([R[:, 4:12], R[:, 13:16]], axis=1).mean()
        self.assertGreater(bpfo_mean, 2 * others)
        # 0 号 bin（DC）永不积分：全部行有限且 R>0 由 eps 保证，且 FTF 维不含 DC 假峰
        self.assertTrue((R[:, 12:16] < bpfo_mean).all())
        # 合成全零信号：包络与噪声底全零，R 必须严格为 0（eps 不引入假能量）
        zeros = features.ies_features(np.zeros(n), CWRU_DE_ORDERS, rotation_hz)
        self.assertTrue((zeros == 0).all())

    def test_ies_input_boundaries_reject_complex_and_invalid_rotation(self):
        """§3.2 边界：不静默丢虚部，转频/阶次必须正有限。"""
        signal = np.cos(2 * np.pi * 3000 * np.arange(4096) / 12000)
        with self.assertRaisesRegex(ValueError, "复数"):
            features.ies_features(signal.astype(np.complex128), CWRU_DE_ORDERS, 25.0)
        with self.assertRaisesRegex(ValueError, "实数"):
            features.ies_features(signal, CWRU_DE_ORDERS, 25.0 + 1j)
        with self.assertRaisesRegex(ValueError, "正有限"):
            features.ies_features(signal, CWRU_DE_ORDERS, 25.0 - 25.0)
        with self.assertRaisesRegex(ValueError, "正有限"):
            features.ies_features(signal, CWRU_DE_ORDERS, np.nan)
        with self.assertRaisesRegex(ValueError, "正有限"):
            features.ies_features(signal, CWRU_DE_ORDERS, np.inf)
        with self.assertRaisesRegex(ValueError, "实数"):
            features.ies_features(signal, CWRU_DE_ORDERS, "25")
        for bad in ({**CWRU_DE_ORDERS, "BSF": -1.0}, {**CWRU_DE_ORDERS, "FTF": 0.0},
                    {**CWRU_DE_ORDERS, "BPFO": complex(3.58, 0)}, {k: v for k, v in CWRU_DE_ORDERS.items() if k != "FTF"}):
            with self.subTest(orders=bad):
                with self.assertRaises(ValueError):
                    features.ies_features(signal, bad, 25.0)
        with self.assertRaisesRegex(ValueError, "非有限"):
            features.ies_features(np.full(2048, np.nan), CWRU_DE_ORDERS, 25.0)
        # 物理摘要与时域同一条边界：复数不降为实部
        analysis = features.ies_features(signal, CWRU_DE_ORDERS, 25.0)
        with self.assertRaisesRegex(ValueError, "复数"):
            features.physical_features(signal.astype(np.complex128), analysis)
        with self.assertRaisesRegex(ValueError, "复数"):
            features._time_features(np.zeros((2, 512), dtype=np.complex128))

    def test_source_geometry_independent_of_labels_and_severity(self):
        # 12k_FE 的两传感器都用 FE 几何；48k_DE/12k_DE 都用 DE 几何；Normal 按测点。
        self.assertEqual(cwru.fault_orders("12k_FE", "DE")["BPFO"], 3.0530)
        self.assertEqual(cwru.fault_orders("12k_FE", "FE")["BPFO"], 3.0530)
        self.assertEqual(cwru.fault_orders("48k_DE", "FE")["BPFO"], 3.5848)
        self.assertEqual(cwru.fault_orders("12k_DE", "DE")["BPFO"], 3.5848)
        self.assertEqual(cwru.fault_orders("Normal", "DE")["BPFO"], 3.5848)
        self.assertEqual(cwru.fault_orders("Normal", "FE")["BPFO"], 3.0530)
        # BSF 不是 2BSF：FE BSF≈1.993557418，历史 3.9874 不再混用
        self.assertAlmostEqual(cwru.fault_orders("12k_FE", "DE")["BSF"], 1.993557418, places=6)
        # 函数签名不接收标签：改标签不可能改提取结果
        with self.assertRaises(TypeError):
            cwru.fault_orders("12k_FE", "DE", "OR")

    def test_rpm_special_key_fallback_and_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "B007_0.mat"
            _make_cwru_mat(path, "X118", rpm=1796.0)
            self.assertEqual(cwru.read_rpm(path, "12k_DE", "B007_0.mat", "DE"), 1796.0)
            # 特殊键：X217 与 X215 并存时读 X217RPM（v6 特殊前缀规则）
            special = Path(directory) / "IR021_3.mat"
            savemat(special, {"X215_DE_time": np.arange(4096.0), "X215_FE_time": np.arange(4096.0),
                              "X217_DE_time": np.arange(4096.0), "X217_FE_time": np.arange(4096.0),
                              "X215RPM": 1752.0, "X217RPM": 1727.0})
            self.assertEqual(cwru.read_rpm(special, "48k_DE", "IR021_3.mat", "DE"), 1727.0)
            # 真实缺失回退：Normal_2.mat 无 RPM → 文件名载荷 2 → 1750
            real = ROOT.parent / "rbop_diagnosis_v6" / "Data" / "Normal" / "Normal_2.mat"
            if real.is_file():
                self.assertEqual(cwru.read_rpm(real, "Normal", "Normal_2.mat", "DE"), 1750.0)
            # 非法字段：零、负、复数、非标量、不唯一
            for bad in ({"X118RPM": 0.0}, {"X118RPM": -1.0}, {"X118RPM": 1 + 2j},
                        {"X118RPM": [[1797.0, 1798.0]]}, {"X118RPM": 1.0, "X118_RPM": 2.0},
                        {"X118RPM": np.nan}):
                _make_cwru_mat(path, "X118", n=2048)
                savemat(path, {**{"X118_DE_time": np.arange(2048.0), "X118_FE_time": np.arange(2048.0)},
                               **bad})
                with self.assertRaises(ValueError):
                    cwru.read_rpm(path, "12k_DE", "B007_0.mat", "DE")
            # 完全缺失 → 载荷档回退 0 → 1797，不借用其他前缀 RPM
            savemat(path, {"X118_DE_time": np.arange(2048.0), "X119RPM": 1234.0})
            self.assertEqual(cwru.read_rpm(path, "12k_DE", "B007_0.mat", "DE"), 1797.0)

    def test_time_features_zero_constant_and_25_layout(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((3, 512))
        t = features._time_features(x)
        self.assertEqual(t.shape, (3, 9))
        np.testing.assert_allclose(t[:, 0], x.mean(axis=1))
        np.testing.assert_allclose(t[:, 1], x.var(axis=1))
        np.testing.assert_allclose(t[:, 2], x.max(axis=1) - x.min(axis=1))
        kurt = (((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)) ** 4).mean(axis=1)
        np.testing.assert_allclose(t[:, 3], kurt)
        # 零窗：v=0 → 峭度 0；比值分母 0 → 0；无 NaN/Inf
        zero = features._time_features(np.zeros((1, 512)))
        self.assertEqual(zero[0, 3], 0)
        self.assertTrue(np.isfinite(zero).all())
        # 常量窗：v=0 → 峭度 0；有限
        const = features._time_features(np.full((1, 512), 3.5))
        self.assertEqual(const[0, 3], 0)
        self.assertTrue(np.isfinite(const).all())
        # 25 维 = 16 IES(log1p R) + 9 时域：R 由调用方一次算好，物理摘要只复用不重算
        signal = np.cos(2 * np.pi * 3000 * np.arange(48001) / 12000)
        analysis = features.ies_features(signal, CWRU_DE_ORDERS, 25.0)
        phys = features.physical_features(signal, analysis)
        self.assertEqual(phys.shape, (93, 25))
        self.assertTrue(np.isfinite(phys).all())
        log_r = np.log1p(analysis)
        np.testing.assert_array_equal(phys[:, :16], log_r[features.map_logspec_to_analysis(93, 45)])
        np.testing.assert_array_equal(
            phys[:, 16:], features._time_features(signal[: 93 * 512].reshape(93, 512)))
        with self.assertRaises(ValueError):
            features.physical_features(signal, analysis[:, :15])

    def test_scaler_fits_train_only_and_dual_padding_order(self):
        train = np.random.default_rng(3).standard_normal((20, 25))
        scaler = features.fit_scaler(train)
        self.assertEqual(len(scaler["mean"]), 25)
        scaled = features.scale_phys(train, scaler)
        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(scaled.std(axis=0), 1, atol=1e-12)
        # σ=0 维取 1：常量列缩放后为 0
        train[:, 5] = 7.0
        scaler = features.fit_scaler(train)
        self.assertEqual(scaler["std"][5], 1.0)
        self.assertTrue((features.scale_phys(train, scaler)[:, 5] == 0).all())
        # dual_input：第二通道前 25 缩放、后 231 补零；logspec_only 全零
        X = np.zeros((4, 1, 256), np.float32)
        dual = features.dual_input(X, train[:4], scaler)
        self.assertEqual(dual.shape, (4, 2, 256))
        self.assertEqual(dual.dtype, np.float32)
        self.assertTrue(np.allclose(dual[:, 1, 25:], 0))
        np.testing.assert_allclose(dual[:, 1, :25], features.scale_phys(train[:4], scaler), atol=1e-6)
        only = features.dual_input(X, train[:4], scaler, logspec_only=True)
        self.assertTrue(np.allclose(only[:, 1, :], 0))

    def _small_cache(self, directory):
        """两记录小缓存：patch 冻结计数与记录枚举，信号用真实 CWRU 文件。"""
        records = cwru.enumerate_recordings()[:2]
        counts = {"records": 2, "units": 1, "windows": sum(
            (r.stop - 0) if False else 0 for r in records)}
        # 真实记录窗口数不定，直接用真实窗数覆盖冻结事实
        windows = [to_windows(cwru.read_signal(r), r.src_fs).shape[0] for r in records]
        counts = {"records": 2, "units": 1, "windows": sum(windows)}
        with patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=records):
            yield features.build_cache("cwru", directory, execute=True)

    def test_small_cache_roundtrip_subset_and_rejections(self):
        records = cwru.enumerate_recordings()[:2]
        windows = [to_windows(cwru.read_signal(r), r.src_fs).shape[0] for r in records]
        counts = {"records": 2, "units": 1, "windows": sum(windows)}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=records):
            cache = features.build_cache("cwru", directory, execute=True)
            self.assertEqual(cache.X.shape, (sum(windows), 1, 256))
            self.assertEqual(cache.phys25.shape, (sum(windows), 25))
            expected_analysis = sum(features.analysis_window_count(w) for w in windows)
            self.assertEqual(cache.ies_analysis.shape, (expected_analysis, 16))
            self.assertTrue(np.isfinite(np.asarray(cache.phys25)).all())
            # Windows mmap 先释放后再二次加载，避免文件占用
            _close_memmaps(cache)
            del cache
            # 已有产物必须显式复用，不静默重建
            with self.assertRaises(FileExistsError):
                features.build_cache("cwru", directory, execute=True)
            again = features.build_cache("cwru", directory, execute=True, reuse_cache=True)
            self.assertEqual(len(again.records), 2)
            cache = again
            # 子集重排区间
            subset = features.feature_subset(cache, channels=["FE"])
            self.assertEqual(subset.X.shape[1:], (1, 256))
            self.assertEqual(len(subset.records), 1)
            self.assertEqual((subset.records[0]["start"], subset.records[0]["stop"]), (0, windows[1]))
            self.assertEqual(subset.records[0]["analysis_start"], 0)
            self.assertEqual(subset.phys25.shape, (windows[1], 25))
            empty = features.feature_subset(cache, unit_keys=[])
            self.assertEqual(len(empty.X), 0)
            with self.assertRaises(KeyError):
                features.feature_subset(cache, unit_keys=["unknown"])
            # 版本过期 / 区间错误必须报错
            metadata_path = Path(directory) / "cwru.json"
            original = metadata_path.read_text(encoding="utf-8")
            payload = json.loads(original)
            self.assertEqual(payload["phys_spec"]["envelope_dim_names"],
                             [f"{name}x{h}" for name in ("BPFO", "BPFI", "BSF", "FTF")
                              for h in (1, 2, 3, 4)])
            self.assertEqual(payload["phys_spec"]["script_version"], features.CACHE_VERSION)
            for mutation in (lambda p: p.update(version="stale"),
                             lambda p: p["counts"].update(windows=3),
                             lambda p: p["records"][1].update(start=0),
                             lambda p: p["records"][0].update(analysis_start=5),
                             lambda p: p["records"][0].pop("analysis_stop"),
                             lambda p: p.update(phys_config={"analysis_window": 1024}),
                             lambda p: p.pop("phys_spec"),
                             lambda p: p["phys_spec"].update(mapping="stale")):
                payload = json.loads(original)
                mutation(payload)
                metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    features.load_cache("cwru", directory)
            metadata_path.write_text(original, encoding="utf-8")
            _close_memmaps(cache)
            del cache

    def test_full_record_envelope_rpm_source_and_mapping_metadata(self):
        """整记录滤波与逐记录元数据：RPM 来源/回退、映射计数、常量窗（§2.1/§3.1/§3.3）。"""
        records = [r for r in cwru.enumerate_recordings()
                   if r.unit_key == "Normal/Normal_2.mat"]
        self.assertEqual([r.channel for r in records], ["DE", "FE"])
        rng = np.random.default_rng(11)
        signal = np.cos(2 * np.pi * 900 * np.arange(20001) / 48000) \
            + 0.05 * rng.standard_normal(20001)
        signal[-500:] += 8.0  # 尾部强非平稳：先截断再滤波拿不到这段
        resolved = resample_poly(signal, 1, 4)
        windows = len(resolved) // 512
        counts = {"records": 2, "units": 1, "windows": 2 * windows}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=records), \
                patch.object(cwru, "read_signal", return_value=signal):
            cache = features.build_cache("cwru", directory, execute=True)
            geometry = [features._geometry_for(record) for record in records]
            self.assertEqual({(rpm, source) for _, _, rpm, source in geometry},
                             {(1750.0, "fallback")})  # Normal_2 无 RPM 字段
            rows = [features.ies_features(resolved, orders, rotation_hz)
                    for orders, rotation_hz, _, _ in geometry]
            expected = np.concatenate(rows)
            np.testing.assert_array_equal(np.asarray(cache.ies_analysis), expected.astype(np.float32))
            # 先按 512 截断再带通/Hilbert 会改变尾部：两者必须有差异
            truncated = features.ies_features(resolved[: windows * 512], *geometry[0][:2])
            self.assertEqual(truncated.shape, rows[0].shape)
            self.assertFalse(np.array_equal(truncated.astype(np.float32),
                                            rows[0].astype(np.float32)))
            np.testing.assert_array_equal(
                np.asarray(cache.phys25),
                np.concatenate([features.physical_features(resolved, row) for row in rows])
                .astype(np.float32))
            for record, row in zip(cache.records, rows):
                self.assertEqual((record["rpm"], record["rpm_source"]), (1750.0, "fallback"))
                _, ties, outside = features.mapping_detail(windows, len(row))
                self.assertEqual((record["map_ties"], record["map_outside"]), (ties, outside))
                self.assertEqual(record["constant_windows"], 0)
            _close_memmaps(cache)
            del cache

    def test_legacy_logspec_reuse_verifies_keys_windows_and_bits(self):
        """旧 v7 缓存显式复用：逐记录核对 record_key/窗数，并逐位比对重算 logspec。"""
        source = cwru.enumerate_recordings()
        records = [r for r in source if r.src_fs == 12000][:2]  # 同一单元的 DE+FE
        other = [r for r in source if r.src_fs == 12000 and r.unit_key != records[0].unit_key][:2]
        self.assertEqual([r.channel for r in records], ["DE", "FE"])
        rng = np.random.default_rng(21)
        signals = [rng.standard_normal(3000 + 512 * index) for index in range(2)]  # 5/6 窗
        counts = {"records": 2, "units": 1, "windows": 11}
        by_key = {record.record_key: signal for record, signal in zip(records, signals)}
        rows, X = _legacy_rows(records, signals), _legacy_logspec(signals)
        self.assertEqual(len(X), counts["windows"])
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(features.EXPECTED_COUNTS, cwru=counts), \
                patch.object(cwru, "enumerate_recordings", return_value=records), \
                patch.object(cwru, "read_signal", side_effect=lambda r: by_key[r.record_key]):
            root = Path(directory)
            legacy = root / "legacy"
            legacy.mkdir()
            _write_legacy_cache(legacy, rows, X)
            cache = features.build_cache("cwru", root / "out0", reuse_logspec=legacy, execute=True)
            np.testing.assert_array_equal(np.asarray(cache.X), X)  # 复用的就是旧数组
            self.assertTrue(all(r["rpm"] > 0 and r["rpm_source"] in ("field", "fallback")
                                for r in cache.records))
            _close_memmaps(cache)
            del cache
            # 逐位不一致：拒绝复用
            broken = X.copy()
            broken[0, 0, 0] += 1.0
            _write_legacy_cache(legacy, rows, broken)
            with self.assertRaisesRegex(ValueError, "逐位"):
                features.build_cache("cwru", root / "out1", reuse_logspec=legacy, execute=True)
            # 窗数不一致（区间仍连续、总数不变）：拒绝复用
            _write_legacy_cache(legacy, [dict(rows[0], stop=6), dict(rows[1], start=6, stop=11)], X)
            with self.assertRaisesRegex(ValueError, "窗数"):
                features.build_cache("cwru", root / "out2", reuse_logspec=legacy, execute=True)
            # 记录键集合与本次枚举不一致（另一完整同计数单元）：拒绝复用
            _write_legacy_cache(legacy, _legacy_rows(other, signals), X)
            with self.assertRaisesRegex(ValueError, "记录键集合"):
                features.build_cache("cwru", root / "out3", reuse_logspec=legacy, execute=True)
            # 只有成功的那次构建发布了产物，失败路径不得留下半成品
            self.assertEqual(sorted(p.parent.name for p in root.rglob("cwru_phys25.npy")), ["out0"])
            self.assertEqual(sorted(p.parent.name for p in root.rglob("cwru.json")), ["legacy", "out0"])

    def test_load_config_reads_frozen_blocks_and_main_uses_data_roots(self):
        config = features.load_config("configs/stage3.json")
        self.assertEqual(config["cache_version"], features.CACHE_VERSION)
        self.assertEqual(config["data_roots"]["cwru"], "../rbop_diagnosis_v6/Data")
        self.assertEqual({name: config["physical"][name] for name in features.PHYS_CONFIG},
                         features.PHYS_CONFIG)
        base = json.loads((ROOT / "configs" / "stage3.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            for mutation in (lambda p: p.update(cache_version="stale"),
                             lambda p: p["physical"].update(logspec_window=1024),
                             lambda p: p["data_roots"].pop("paderborn"),
                             lambda p: p.pop("cache_dir")):
                payload = json.loads(json.dumps(base))
                mutation(payload)
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    features.load_config(path)
        with patch.object(features, "build_cache", return_value=None) as build, \
                patch("sys.argv", ["features.py", "--config", "configs/stage3.json",
                                   "--dataset", "cwru"]):
            features.main()
        self.assertEqual(build.call_args.kwargs["data_root"],
                         (ROOT / "../rbop_diagnosis_v6/Data").resolve())
        self.assertEqual(build.call_args.kwargs["cache_dir"], ROOT / "outputs" / "features")
        self.assertFalse(build.call_args.kwargs["execute"])

    def test_legacy_reuse_rejects_wrong_version(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.save(folder / "cwru.npy", np.zeros((2, 1, 256), np.float32))
            (folder / "cwru.json").write_text(json.dumps({"version": "wrong"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                features._reuse_legacy_logspec("cwru", folder)
            np.save(folder / "cwru.npy", np.full((2, 1, 256), np.nan, np.float32))
            (folder / "cwru.json").write_text(json.dumps(
                {"version": features.LEGACY_LOGSPEC_VERSION}), encoding="utf-8")
            with self.assertRaises(ValueError):
                features._reuse_legacy_logspec("cwru", folder)


if __name__ == "__main__":
    unittest.main()
