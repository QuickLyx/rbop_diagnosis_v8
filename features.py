# -*- coding: utf-8 -*-
"""v8 冻结特征：logspec 与物理摘要分开缓存，读取时拒绝过期或不完整产物。

三个缓存文件（计划 §3.3）：`<dataset>.npy`/`.json` 为 (N,1,256) logspec；
`<dataset>_phys25.npy` 为 (N,25) 原始物理摘要（16 维包络 IES + 9 维时域）；
`<dataset>_ies_analysis.npy` 为 (A,16) 逐分析窗原始 R（物理锚与 q_phys 复用）。
记录元数据存 JSON，含 logspec 区间 start/stop 与分析窗区间
analysis_start/analysis_stop，两套区间各自连续完整覆盖。

每记录只重采样一次并保留完整记录：带通/Hilbert 在整记录上做（不先截断再滤波），
logspec 仍按 512/512 独立切窗；16 维 IES 一次算全记录，25 维摘要复用同一 R。

logspec 数值与 v7 逐位一致（同一冻结函数）。旧 v7 logspec 缓存可显式只读复用
（`--reuse-logspec`），但逐记录核对 record_key、窗数并逐位比较重算结果，物理数组本地重建。
已有 v8 缓存必须显式 `--reuse-cache` 才复用，不静默重建；逐记录 RPM 数值与来源
（字段/载荷档回退/工况表）落缓存 JSON（计划 §2.1）。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
from math import gcd
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from bearing_facts import CWRU_DE_ORDERS, CWRU_FE_ORDERS
import dataset as v6ds  # noqa: E402  v6 本地副本（RAW_FS 等常量唯一定义）
from datasets import cwru, paderborn
from datasets.recording import Recording, _as_signal
from preprocessing import TARGET_FS, build_repr_config, config_from_saved, extract_logspec

ROOT = Path(__file__).resolve().parent
CACHE_VERSION = "v8_dual25_12k_512s512_2048s1024_ies_v1"
#: 物理块配置（与 configs/stage3.json physical 块一致；features 只按冻结值执行）。
PHYS_CONFIG = {
    "target_fs": 12000, "logspec_window": 512, "logspec_stride": 512,
    "analysis_window": 2048, "analysis_stride": 1024,
    "bandpass_order": 4, "bandpass_hz": [2000, 5000],
    "harmonics": [1, 2, 3, 4], "order_names": ["BPFO", "BPFI", "BSF", "FTF"],
    "neighbor_bins": 2, "noise_max_hz": 500, "eps_absolute": 1e-30, "eps_relative": 1e-12,
}
EXPECTED_COUNTS = {
    "cwru": {"records": 306, "units": 153, "windows": 68516},
    "paderborn": {"records": 2559, "units": 128, "windows": 238752,
                  "primary_records": 2319, "primary_units": 116, "primary_windows": 216350},
}
LOADERS = {"cwru": cwru, "paderborn": paderborn}
#: 旧 v7 logspec 缓存可显式复用所必须匹配的版本。
LEGACY_LOGSPEC_VERSION = "v7_logspec512_12k_drop_ka08_v1"
#: 缓存元数据里的公式/维序/映射说明（计划 §3.3）：普通 JSON 字符串与维名，不建 schema。
PHYS_SPEC = {
    "script": "features.py",
    "script_version": CACHE_VERSION,
    "target_fs": TARGET_FS,
    "envelope_steps": "整记录 4 阶 Butterworth 2000–5000 Hz 带通（sos/sosfiltfilt）→"
                      " 整记录 Hilbert 包络 → 2048/1024 完整分析窗去均值 → np.hanning(2048)"
                      " → rFFT 功率 P=abs(FFT)**2",
    "envelope_ratio": "b=floor(f/df+0.5)，df=12000/2048；J=[max(1,b-2),min(1024,b+2)]；"
                      "noise=median(P[0<j*df<=500Hz])；epsP=max(1e-30,1e-12*mean(P[1:]))；"
                      "R(f)=sum(P[J])/(len(J)*(noise+epsP))",
    "envelope_dim_names": [f"{name}x{harmonic}" for name in PHYS_CONFIG["order_names"]
                           for harmonic in PHYS_CONFIG["harmonics"]],
    "time_dim_names": ["mean", "variance", "peak_to_peak", "pearson_kurtosis", "rms",
                       "crest_factor", "impulse_factor", "shape_factor", "clearance_factor"],
    "dim_layout": "phys25 = 16 维 IES 原始 R 经 log1p 后按中心映射展开（维序同上） + 9 维时域",
    "mapping": "样本索引为时间轴：logspec 中心 512*i+255.5，物理中心 1024*j+1023.5；"
               "取最近中心，同距取较早 j，首尾取最近完整物理窗；不跨记录、不新增 logspec 窗",
    "logspec": "512/512 无重叠，末尾不足一窗丢弃，数值等于冻结 v7 logspec512",
    "window_counts": "每记录 analysis_stop-analysis_start=analysis_window_count(logspec 窗数)；"
                     "记录 map_ties（同距取较早 j 的窗数）与 map_outside（中心覆盖区外的窗数）",
    "constant_windows": "记录 512 点窗方差为 0 的数量，不删除窗",
}


@dataclass
class FeatureCache:
    X: np.ndarray
    phys25: np.ndarray
    ies_analysis: np.ndarray
    records: list[dict]
    version: str
    repr_config: dict


def _cache_paths(dataset, cache_dir):
    if dataset not in LOADERS:
        raise ValueError(f"未知数据集：{dataset}")
    folder = Path(cache_dir) if cache_dir is not None else ROOT / "outputs" / "features"
    return (folder / f"{dataset}.npy", folder / f"{dataset}.json",
            folder / f"{dataset}_phys25.npy", folder / f"{dataset}_ies_analysis.npy")


def _counts(dataset, records):
    result = {"records": len(records), "units": len({r["unit_key"] for r in records}),
              "windows": sum(r["stop"] - r["start"] for r in records)}
    if dataset == "paderborn":
        primary = [r for r in records if r["label"] is not None]
        result.update(primary_records=len(primary),
                      primary_units=len({r["unit_key"] for r in primary}),
                      primary_windows=sum(r["stop"] - r["start"] for r in primary))
    return result


def analysis_window_count(n_windows):
    """logspec 窗数（512/512）对应的物理分析窗数（2048/1024，仅完整窗）。

    分析窗 j 覆盖样本 [1024*j, 1024*j+2048)，共需 1024*j+2048 <= 512*n_windows。
    """
    if n_windows < 0:
        raise ValueError("窗数不能为负")
    return max(0, (512 * n_windows - 2048) // 1024 + 1)


def mapping_detail(n_windows, n_analysis):
    """中心映射与附带计数（计划 §3.1）：返回 (index, map_ties, map_outside)。

    以样本索引为时间轴：logspec 中心 512*i+255.5，物理中心 1024*j+1023.5；最近中心、
    同距取较早 j；`map_ties` 为同距窗数，`map_outside` 为落在首个/末个物理中心之外的窗数。
    """
    if n_windows < 1 or n_analysis < 1:
        raise ValueError("映射要求两类窗口数均不少于 1")
    centers = 512 * np.arange(n_windows) + 255.5
    physical = 1024 * np.arange(n_analysis) + 1023.5
    # |c-p| 相同取较早 j：searchsorted 对右端点取左侧；对左端点 np.argmin 天然取前。
    index = np.searchsorted(physical, centers, side="right") - 1
    index = np.clip(index, 0, n_analysis - 1)
    next_index = np.minimum(index + 1, n_analysis - 1)
    left = np.abs(centers - physical[index])
    right = np.abs(physical[next_index] - centers)
    outside = (centers < physical[0]) | (centers > physical[-1])
    # 裁剪后的端点窗 index==next_index，不构成同距候选，不计入 map_ties。
    ties = (next_index != index) & (left == right)
    return np.where(right < left, next_index, index), int(ties.sum()), int(outside.sum())


def map_logspec_to_analysis(n_windows, n_analysis):
    """logspec 窗 i → 最近分析窗中心 j；同距取较早 j；首尾取最近完整物理窗。

    映射不跨记录、不新增 logspec 窗。返回 (n_windows,) int 数组。
    """
    return mapping_detail(n_windows, n_analysis)[0]


def _real_signal(values, name):
    """转为 float64 实数数组；显式拒绝复数，`asarray(dtype=float64)` 会静默丢掉虚部。"""
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise ValueError(f"{name} 不能为复数")
    return values.astype(np.float64, copy=False)


def _time_features(windows):
    """九维时域特征（计划 §3.3 顺序与公式），从原始重采样 512 点窗计算。

    v=0 时峭度=0；其他比值分母=0 时该比值=0。幅值维依赖传感器单位，非不变量。
    """
    x = _real_signal(windows, "时域窗")
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError("时域特征要求 (N,L) 非空二维输入")
    mean = x.mean(axis=1)
    var = ((x - mean[:, None]) ** 2).mean(axis=1)
    peak = np.abs(x).max(axis=1)
    ptp = x.max(axis=1) - x.min(axis=1)
    a = np.abs(x).mean(axis=1)
    rms = np.sqrt((x * x).mean(axis=1))
    c = (np.sqrt(np.abs(x)).mean(axis=1)) ** 2
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt = np.where(var > 0, (((x - mean[:, None]) / np.where(std[:, None] > 0, std[:, None], 1.0)) ** 4).mean(axis=1), 0.0)
        crest = np.where(rms > 0, peak / rms, 0.0)
        impulse = np.where(a > 0, peak / a, 0.0)
        shape = np.where(a > 0, rms / a, 0.0)
        clearance = np.where(c > 0, peak / c, 0.0)
    return np.column_stack([mean, var, ptp, kurt, rms, crest, impulse, shape, clearance])


def _bandpass_envelope(signal_12k):
    """整记录 4 阶 Butterworth 2000–5000 Hz 带通（sos）→ Hilbert 包络。"""
    from scipy.signal import hilbert
    sos = butter(4, [2000, 5000], btype="bandpass", fs=12000, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(signal_12k, dtype=np.float64))
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    if not np.isfinite(envelope).all():
        raise ValueError("带通或包络产生非有限值")
    return envelope


def _validate_rotation(orders, rotation_hz):
    """转频与阶次必须为正有限实数（计划 §3.2 的输入边界）。"""
    if isinstance(rotation_hz, bool) or np.iscomplexobj(rotation_hz) \
            or not isinstance(rotation_hz, (int, float, np.integer, np.floating)):
        raise ValueError(f"转频必须是实数标量：{rotation_hz}")
    rotation_hz = float(rotation_hz)
    if not np.isfinite(rotation_hz) or rotation_hz <= 0:
        raise ValueError(f"转频必须是正有限数：{rotation_hz}")
    if not isinstance(orders, dict):
        raise ValueError(f"orders 必须是 BPFO/BPFI/BSF/FTF 字典：{type(orders)}")
    for name in PHYS_CONFIG["order_names"]:
        value = orders.get(name)
        if isinstance(value, bool) or np.iscomplexobj(value) \
                or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"orders[{name}] 必须是实数标量：{value}")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"orders[{name}] 必须是正有限数：{value}")
    return rotation_hz


def ies_features(signal_12k, orders, rotation_hz):
    """计划 §3.2 的 16 维包络 IES：分析窗 (A,2048) 逐窗算原始 R。

    `orders` 为含 BPFO/BPFI/BSF/FTF 的阶次字典（乘转频得目标频率）；
    流程：带通（整记录）→ Hilbert 包络 → 2048/1024 完整窗 → 去均值 →
    Hann → rFFT 功率 → 谐波邻带积分/噪声底。返回 (A,16) float64。
    """
    signal_12k = _real_signal(signal_12k, "IES 输入")
    rotation_hz = _validate_rotation(orders, rotation_hz)
    if signal_12k.ndim != 1 or signal_12k.size < 2048:
        raise ValueError("IES 输入必须是不少于 2048 点的一维 12 kHz 信号")
    if not np.isfinite(signal_12k).all():
        raise ValueError("IES 输入包含非有限值")
    envelope = _bandpass_envelope(signal_12k)
    n_analysis = (signal_12k.size - 2048) // 1024 + 1
    if n_analysis < 1:
        raise ValueError("信号不足一个 2048 点分析窗")
    windows = np.stack([envelope[1024 * j: 1024 * j + 2048] for j in range(n_analysis)])
    windowed = (windows - windows.mean(axis=1, keepdims=True)) * np.hanning(2048)
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    df = 12000 / 2048
    bins = np.arange(power.shape[1])
    noise_mask = (bins > 0) & (bins * df <= 500)
    noise = np.median(power[:, noise_mask], axis=1)
    mean_power = power[:, 1:].mean(axis=1)
    eps_p = np.maximum(1e-30, 1e-12 * mean_power)
    result = np.empty((n_analysis, 16), dtype=np.float64)
    column = 0
    for name in ("BPFO", "BPFI", "BSF", "FTF"):
        for harmonic in (1, 2, 3, 4):
            center = orders[name] * harmonic * rotation_hz
            b = int(np.floor(center / df + 0.5))
            if b < 1:
                # DC 永不积分：目标中心落入 0 号 bin 时邻带从 1 开始。
                low, high = 1, b + 2
            else:
                low, high = b - 2, b + 2
            low, high = max(1, low), min(1024, high)
            if low > high:
                raise ValueError(f"谐波 {name}x{harmonic} 的邻带为空：{center} Hz")
            columns = power[:, low:high + 1]
            ratio = columns.sum(axis=1) / (columns.shape[1] * (noise + eps_p))
            result[:, column] = ratio
            column += 1
    if not np.isfinite(result).all():
        raise ValueError("IES 特征包含非有限值")
    return result


def physical_features(signal_12k, analysis):
    """25 维原始物理摘要 = 16 维 IES(log1p(R)) 中心映射 + 9 维时域（计划 §3.3）。

    `analysis` 为整记录 `ies_features` 的 (A,16) 原始 R：复用同一份 R，不重复
    带通/Hilbert；时域直接从对应 512 点窗计算。返回 (n_windows, 25)。
    """
    signal_12k = _real_signal(signal_12k, "物理摘要输入")
    if signal_12k.ndim != 1:
        raise ValueError("物理摘要要求一维 12 kHz 信号")
    analysis = np.asarray(analysis, dtype=np.float64)
    if analysis.ndim != 2 or analysis.shape[1] != 16 or not len(analysis):
        raise ValueError(f"analysis 必须是 (A,16) 非空原始 R，实际 {analysis.shape}")
    n_windows = signal_12k.size // 512
    if n_windows < 1:
        raise ValueError("物理摘要要求不少于 512 点")
    mapped = map_logspec_to_analysis(n_windows, len(analysis))
    envelope_summary = np.log1p(analysis)[mapped]
    windows = signal_12k[: n_windows * 512].reshape(n_windows, 512)
    return np.column_stack([envelope_summary, _time_features(windows)])


def fit_scaler(phys25):
    """逐维均值/总体标准差（ddof=0）；σ=0 取 1。返回普通字典，JSON 可存。"""
    values = np.asarray(phys25, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 25 or not len(values):
        raise ValueError("fit_scaler 要求非空 (N,25) 输入")
    if not np.isfinite(values).all():
        raise ValueError("fit_scaler 输入包含非有限值")
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    return {"mean": mean.tolist(), "std": np.where(std > 0, std, 1.0).tolist()}


def scale_phys(phys25, scaler):
    """用 fit_scaler 的字典缩放 (N,25)；float64 数值计算。"""
    values = np.asarray(phys25, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 25:
        raise ValueError("scale_phys 要求 (N,25) 输入")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if mean.shape != (25,) or std.shape != (25,) or not len(values):
        raise ValueError("缩放器必须是 25 维且输入非空")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("缩放器均值/标准差必须有限且标准差为正")
    scaled = (values - mean) / std
    if not np.isfinite(scaled).all():
        raise ValueError("缩放后包含非有限值（输入含非有限或缩放器无效）")
    return scaled


def dual_input(X, phys25, scaler, logspec_only=False):
    """组装 (N,2,256) float32：第一通道 logspec，第二通道缩放 phys25 后补零到 231。

    补零在缩放之后；logspec_only 时第二通道全零（对照臂）。不永久缓存补零通道。
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (1, 256):
        raise ValueError(f"logspec 必须是 (N,1,256)，实际 {X.shape}")
    phys = np.asarray(phys25, dtype=np.float64)
    if phys.ndim != 2 or phys.shape[0] != len(X) or phys.shape[1] != 25:
        raise ValueError(f"phys25 必须是 ({len(X)},25)，实际 {phys.shape}")
    second = np.zeros((len(X), 256), dtype=np.float32)
    if not logspec_only:
        second[:, :25] = scale_phys(phys, scaler).astype(np.float32)
    return np.stack([X[:, 0, :], second], axis=1).astype(np.float32)


def _require_finite(array, name):
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限值")
    return array


def _loader_read_signal(recording):
    if recording.dataset == "cwru":
        return cwru.read_signal(recording)
    if recording.dataset == "paderborn":
        return paderborn.read_signal(recording)
    raise ValueError(f"未知数据集：{recording.dataset}")


def _close_memmap(array):
    if isinstance(array, np.memmap):
        array._mmap.close()


def _enumerate_recordings(dataset, data_root):
    loader = LOADERS[dataset]
    return (loader.enumerate_recordings(data_root, include_kb=True) if dataset == "paderborn"
            else loader.enumerate_recordings(data_root))


def _record_signal_12k(recording):
    """整记录 12 kHz 信号：精确有理比重采样一次，保留末尾不足一窗部分。

    计划 §3.1 要求带通/Hilbert 在整记录上做：`to_windows` 先截掉末尾再滤波会改变
    Hilbert 尾部，所以这里保留完整记录，只与 `to_windows` 共用同一有理比重采样。
    """
    signal = _as_signal(_loader_read_signal(recording))
    src_fs = recording.src_fs
    for rate in (src_fs, TARGET_FS):
        if (isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, np.integer))
                or rate <= 0):
            raise ValueError(f"采样率必须是正整数：{rate}")
    if src_fs != TARGET_FS:
        g = gcd(int(TARGET_FS), int(src_fs))
        signal = _as_signal(resample_poly(signal, int(TARGET_FS) // g, int(src_fs) // g))
    if signal.size // 512 < 1:
        raise ValueError("重采样后的信号不足一个 512 点窗口")
    return signal


def _validate_identity(dataset, records):
    """记录身份逐字段校验（v7 口径不变）；返回单元→通道计数与逐记录检查。"""
    seen, unit_values, unit_channels = set(), {}, {}
    required = {f.name for f in fields(Recording)} | {"start", "stop"}
    # RPM 数值/来源由 `_validate_rpm` 单独校验：旧 v7 缓存没有这两个字段。
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("缓存记录元数据字段不完整")
        key, unit = record["record_key"], record["unit_key"]
        if not isinstance(key, str) or not isinstance(unit, str) or key in seen:
            raise ValueError("缓存记录键无效或重复")
        seen.add(key)
        if record["dataset"] != dataset or not isinstance(record["path"], str):
            raise ValueError(f"缓存记录数据集或路径无效：{key}")
        # 冻结缓存生成于 Windows，身份校验只取跨平台文件名，不访问原路径。
        filename = record["path"].replace("\\", "/").rsplit("/", 1)[-1]
        label = record["label"]
        if label is not None and (type(label) is not int or label not in range(4)):
            raise ValueError(f"缓存记录标签无效：{key}")
        identity = tuple(record[name] for name in
                         ("label", "source", "bearing_id", "condition", "origin", "severity", "src_fs"))
        if unit in unit_values and identity != unit_values[unit]:
            raise ValueError(f"同单元的记录元数据不一致：{unit}")
        unit_values[unit] = identity
        unit_channels.setdefault(unit, []).append(record["channel"])
        if dataset == "cwru":
            source, channel = record["source"], record["channel"]
            if (source not in cwru.SOURCES or channel not in cwru.ALL_CHANNELS
                    or label is None or record["src_fs"] != v6ds.RAW_FS[source]
                    or unit != f"{source}/{filename}"
                    or key != f"{unit}::{channel}"):
                raise ValueError(f"CWRU 缓存记录身份无效：{key}")
        else:
            from bearing_facts import BY_ID, CONDITIONS
            bearing_id, condition = record["bearing_id"], record["condition"]
            bearing = BY_ID.get(bearing_id)
            if (bearing is None or condition not in CONDITIONS or record["src_fs"] != 64000
                    or record["source"] != "paderborn" or record["channel"] != "vibration_1"
                    or unit != f"{bearing_id}/{condition}"
                    or key != f"{bearing_id}/{filename}::vibration_1"
                    or label != bearing.label or record["origin"] != bearing.origin
                    or record["severity"] != bearing.severity):
                raise ValueError(f"PU 缓存记录身份或真值无效：{key}")
    for unit, channels in unit_channels.items():
        if dataset == "cwru":
            expected = Counter(cwru.ALL_CHANNELS)
        else:
            expected = Counter({"vibration_1": 19 if unit == "KA08/N15_M01_F10" else 20})
        if Counter(channels) != expected:
            raise ValueError(f"缓存单元的记录数量或通道不完整：{unit}")
    return records


def _validate_cache(dataset, X, phys25, ies_analysis, payload):
    """整体校验：版本、口径、区间连续性、两套区间各自覆盖、身份与有限性。"""
    if payload.get("version") != CACHE_VERSION:
        raise ValueError("缓存口径版本过期或缺失，拒绝使用；请在新目录重新生成")
    if payload.get("dataset") != dataset or payload.get("script") != "features.py":
        raise ValueError("缓存数据集或生成脚本不匹配")
    config = config_from_saved(payload.get("repr_config"), require_complete=True)
    if config != build_repr_config():
        raise ValueError("缓存表征配置与冻结口径不一致")
    if payload.get("phys_config") != PHYS_CONFIG:
        raise ValueError("缓存物理配置与冻结口径不一致")
    if payload.get("phys_spec") != PHYS_SPEC:
        raise ValueError("缓存缺少公式/维序/映射说明或与冻结口径不一致")
    if X.ndim != 3 or X.shape[1:] != (1, 256) or X.dtype != np.float32:
        raise ValueError(f"logspec 必须是 (N,1,256) float32，实际 {X.shape} {X.dtype}")
    if phys25.ndim != 2 or phys25.shape[0] != len(X) or phys25.shape[1] != 25 \
            or phys25.dtype != np.float32:
        raise ValueError(f"phys25 必须是 (N,25) float32，实际 {phys25.shape} {phys25.dtype}")
    if ies_analysis.ndim != 2 or ies_analysis.shape[1] != 16 or ies_analysis.dtype != np.float32:
        raise ValueError(f"ies_analysis 必须是 (A,16) float32，实际 {ies_analysis.shape} {ies_analysis.dtype}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("缓存缺少记录元数据")
    offset, analysis_offset = 0, 0
    total_analysis = 0
    for record in records:
        start, stop = record["start"], record["stop"]
        if (type(start) is not int or type(stop) is not int or start != offset
                or stop <= start or stop > len(X)):
            raise ValueError("缓存记录区间不连续、为空或超出特征行数")
        expected_analysis = analysis_window_count(stop - start)
        if expected_analysis < 1:
            raise ValueError(f"记录不足一个 2048 点分析窗：{record['record_key']}")
        try:
            a_start, a_stop = record["analysis_start"], record["analysis_stop"]
        except (KeyError, TypeError):
            raise ValueError("缓存记录缺少分析窗区间：{key}".format(key=record.get("record_key")))
        if (type(a_start) is not int or type(a_stop) is not int or a_start != analysis_offset
                or a_stop <= a_start or a_stop > len(ies_analysis)):
            raise ValueError("缓存分析窗区间不连续、为空或超出分析行数")
        if a_stop - a_start != expected_analysis:
            raise ValueError(f"记录分析窗数与冻结映射不一致：{record['record_key']}")
        _, expected_ties, expected_outside = mapping_detail(stop - start, expected_analysis)
        constant_windows = record.get("constant_windows")
        if (record.get("map_ties") != expected_ties
                or record.get("map_outside") != expected_outside
                or type(constant_windows) is not int
                or not 0 <= constant_windows <= stop - start):
            raise ValueError(f"缓存记录映射或常量窗元数据无效：{record['record_key']}")
        offset, analysis_offset = stop, a_stop
        total_analysis += a_stop - a_start
    if offset != len(X) or analysis_offset != len(ies_analysis):
        raise ValueError("缓存区间未覆盖全部特征或分析窗")
    _validate_identity(dataset, records)
    for record in records:
        _validate_rpm(dataset, record,
                      record["path"].replace("\\", "/").rsplit("/", 1)[-1])
    counts = _counts(dataset, records)
    if counts != EXPECTED_COUNTS[dataset] or payload.get("counts") != counts:
        raise ValueError(f"缓存计数与冻结事实不一致：{counts}")
    for array, name in ((X, "logspec"), (phys25, "phys25"), (ies_analysis, "ies_analysis")):
        _require_finite(array, name)
    return FeatureCache(X, phys25, ies_analysis, records, CACHE_VERSION, config)


def _validate_rpm(dataset, record, filename):
    """逐记录 RPM 数值与来源（计划 §2.1）：来源必须显式，回退值必须等于冻结表。"""
    rpm, source = record.get("rpm"), record.get("rpm_source")
    if (isinstance(rpm, bool) or not isinstance(rpm, (int, float)) or not np.isfinite(rpm)
            or rpm <= 0 or source not in ("field", "fallback", "condition")):
        raise ValueError(f"缓存记录 RPM 数值或来源无效：{record.get('record_key')}")
    if dataset == "cwru":
        if source not in ("field", "fallback") or (
                source == "fallback" and rpm != cwru.RPM_FALLBACK[cwru.load_digit(filename)]):
            raise ValueError(f"CWRU 缓存记录 RPM 来源或回退值无效：{record.get('record_key')}")
    else:
        from bearing_facts import CONDITIONS
        if (source != "condition" or record.get("condition") not in CONDITIONS
                or rpm != CONDITIONS[record["condition"]][1]):
            raise ValueError(f"PU 缓存记录 RPM 来源或工况转速无效：{record.get('record_key')}")


def _geometry_for(recording):
    """来源几何、转频与逐记录 RPM 来源（计划 §2.1）：source 决定几何，RPM/工况定转频。

    只接收 Recording 的元数据字段，不读诊断标签；改标签不改提取公式。返回
    `(orders, rotation_hz, rpm, rpm_source)`，`rpm_source` 取 field/fallback/condition。
    """
    if recording.dataset == "cwru":
        orders = cwru.fault_orders(recording.source, recording.channel)
        rpm, rpm_source = cwru.read_rpm_detail(
            recording.path, recording.source, Path(recording.path).name, recording.channel)
    else:
        from bearing_facts import BY_ID, CONDITIONS
        bearing = BY_ID[recording.bearing_id]
        bpfo, bpfi = bearing.fault_orders
        ratio = 1 - 2 * bpfo / 8
        orders = {"BPFO": bpfo, "BPFI": bpfi,
                  "BSF": (1 - ratio * ratio) / (2 * ratio), "FTF": (1 - ratio) / 2}
        rpm, rpm_source = float(CONDITIONS[recording.condition][1]), "condition"
    return orders, float(rpm) / 60.0, float(rpm), rpm_source


def _reuse_legacy_logspec(dataset, cache_dir):
    """只读复用旧 v7 logspec 缓存：校验旧版本/记录索引/身份后返回 (X, 行区间表)。

    返回的行区间表为 record_key → {"start": int, "stop": int}，供构建时逐记录
    核对窗数与逐位数值；旧缓存没有 RPM 字段，RPM 由本次枚举重新读取。
    """
    folder = Path(cache_dir) if cache_dir is not None else ROOT.parent / "rbop_diagnosis_v7" / "outputs" / "features"
    array_path, metadata_path = folder / f"{dataset}.npy", folder / f"{dataset}.json"
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"旧 logspec 缓存缺失：{array_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("version") != LEGACY_LOGSPEC_VERSION:
        raise ValueError(f"旧缓存版本 {payload.get('version')} 不是可复用的 {LEGACY_LOGSPEC_VERSION}")
    X = np.load(array_path, mmap_mode="r", allow_pickle=False)
    try:
        if X.shape[1:] != (1, 256) or X.dtype != np.float32:
            raise ValueError("旧缓存形状/类型不符")
        config = config_from_saved(payload.get("repr_config"), require_complete=True)
        if config != build_repr_config():
            raise ValueError("旧缓存表征配置与冻结口径不一致")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("旧缓存缺少记录元数据")
        offset = 0
        for record in records:
            start, stop = record["start"], record["stop"]
            if (type(start) is not int or type(stop) is not int or start != offset
                    or stop <= start or stop > len(X)):
                raise ValueError("旧缓存记录区间不连续、为空或越界")
            offset = stop
        if offset != len(X):
            raise ValueError("旧缓存区间未覆盖全部特征")
        _validate_identity(dataset, records)
        counts = _counts(dataset, records)
        if counts != EXPECTED_COUNTS[dataset] or payload.get("counts") != counts:
            raise ValueError(f"旧缓存计数与冻结事实不一致：{counts}")
        _require_finite(np.asarray(X[:8]), "旧 logspec 抽样")  # mmap 抽样有限性检查
        # 全量有限性检查（mmap 分块读取）
        for start in range(0, len(X), 65536):
            _require_finite(np.asarray(X[start:start + 65536]), "旧 logspec")
        return X, {record["record_key"]: {"start": record["start"], "stop": record["stop"]}
                   for record in records}
    except BaseException:
        _close_memmap(X)
        raise


def build_cache(dataset, cache_dir=None, data_root=None, reuse_logspec=None,
                reuse_cache=False, execute=False):
    """全量特征构建；临时数组验证成功后才发布，不覆盖已有产物。

    `execute=False` 只做 dry-run 枚举/旧缓存校验，不写文件。已有完整缓存必须显式
    `reuse_cache=True` 才复用，否则报错，不静默重建。`reuse_logspec` 给出旧 v7
    logspec 缓存目录时只读复用该数组，但逐记录核对 record_key、窗数并逐位比较重算
    结果；物理数组（整记录 IES 与 25 维摘要）仍本地重建。
    """
    if dataset not in LOADERS:
        raise ValueError(f"未知数据集：{dataset}")
    array_path, metadata_path, phys_path, ies_path = _cache_paths(dataset, cache_dir)
    legacy_X, legacy_rows = None, None
    if reuse_logspec is not None:
        legacy_X, legacy_rows = _reuse_legacy_logspec(dataset, reuse_logspec)
    if not execute:
        if legacy_rows is not None:
            _close_memmap(legacy_X)
            print(f"[dry-run] {dataset}: 旧 logspec 复用校验通过，{len(legacy_rows)} 条记录，"
                  f"目录 {reuse_logspec}")
        else:
            recordings = _enumerate_recordings(dataset, data_root)
            print(f"[dry-run] {dataset}: {len(recordings)} 条记录，缓存目录 {array_path.parent}")
        return None
    existing = [str(path) for path in (array_path, metadata_path, phys_path, ies_path)
                if path.exists()]
    if existing:
        _close_memmap(legacy_X)
        if not reuse_cache:
            raise FileExistsError(
                f"已有特征产物且未显式指定 --reuse-cache，拒绝覆盖或静默复用：{existing}")
        return load_cache(dataset, cache_dir)
    recordings = _enumerate_recordings(dataset, data_root)
    if legacy_rows is not None and sorted(legacy_rows) != sorted(r.record_key for r in recordings):
        _close_memmap(legacy_X)
        raise ValueError("旧 logspec 缓存的记录键集合与本次枚举不一致")
    config = build_repr_config()
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporaries = []
    X = phys = ies = None
    try:
        for suffix in (".npy", "_phys25.npy", "_ies_analysis.npy", ".json"):
            with tempfile.NamedTemporaryFile(prefix=f".{dataset}_", suffix=suffix,
                                             dir=array_path.parent, delete=False) as handle:
                temporaries.append(Path(handle.name))
        total_windows = EXPECTED_COUNTS[dataset]["windows"]
        X = np.lib.format.open_memmap(temporaries[0], mode="w+", dtype=np.float32,
                                      shape=(total_windows, 1, 256))
        phys = np.lib.format.open_memmap(temporaries[1], mode="w+", dtype=np.float32,
                                         shape=(total_windows, 25))
        metadata, offset = [], 0
        for recording in recordings:
            signal = _record_signal_12k(recording)
            n = signal.size // 512
            if offset + n > total_windows:
                raise ValueError("实际窗口数超出冻结事实，拒绝发布缓存")
            windows = signal[: n * 512].reshape(n, 512)
            features = extract_logspec(windows, config)
            if legacy_rows is not None:
                row = legacy_rows.get(recording.record_key)
                if row is None:
                    raise ValueError(f"旧 logspec 缓存缺少记录：{recording.record_key}")
                if row["stop"] - row["start"] != n:
                    raise ValueError(
                        f"旧 logspec 缓存窗数与重采样结果不一致：{recording.record_key}")
                cached = np.asarray(legacy_X[row["start"]:row["stop"]])
                if not np.array_equal(cached, features):
                    raise ValueError(f"旧 logspec 缓存与逐位重算不一致：{recording.record_key}")
                features = cached
            orders, rotation_hz, rpm, rpm_source = _geometry_for(recording)
            analysis = ies_features(signal, orders, rotation_hz)
            _, map_ties, map_outside = mapping_detail(n, len(analysis))
            X[offset:offset + n] = features
            phys[offset:offset + n] = physical_features(signal, analysis)
            metadata.append({"recording": recording, "start": offset, "stop": offset + n,
                             "rpm": rpm, "rpm_source": rpm_source,
                             "map_ties": map_ties, "map_outside": map_outside,
                             "constant_windows": int((windows.var(axis=1) == 0).sum()),
                             "analysis_rows": analysis})
            offset += n
        if offset != total_windows:
            raise ValueError(f"实际窗口数与冻结事实不一致：{offset}，预期 {total_windows}")
        analysis_total = sum(len(row["analysis_rows"]) for row in metadata)
        ies = np.lib.format.open_memmap(temporaries[2], mode="w+", dtype=np.float32,
                                        shape=(analysis_total, 16))
        records, analysis_offset = [], 0
        for row in metadata:
            recording = row["recording"]
            rows = row["analysis_rows"]
            ies[analysis_offset:analysis_offset + len(rows)] = rows
            record = asdict(recording)
            record.update(path=str(recording.path), start=row["start"], stop=row["stop"],
                          rpm=row["rpm"], rpm_source=row["rpm_source"],
                          map_ties=row["map_ties"], map_outside=row["map_outside"],
                          constant_windows=row["constant_windows"],
                          analysis_start=analysis_offset, analysis_stop=analysis_offset + len(rows))
            records.append(record)
            analysis_offset += len(rows)
        payload = {"dataset": dataset, "version": CACHE_VERSION, "script": "features.py",
                   "repr_config": config, "phys_config": PHYS_CONFIG, "phys_spec": PHYS_SPEC,
                   "counts": _counts(dataset, records), "records": records}
        _validate_cache(dataset, X, phys, ies, payload)
        for array in (X, phys, ies):
            array.flush()
            array._mmap.close()
        X = phys = ies = None
        temporaries[3].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _publish(temporaries, array_path, metadata_path, phys_path, ies_path)
    finally:
        _close_memmap(legacy_X)
        for array in (X, phys, ies):
            if array is not None:
                array._mmap.close()
        for path in temporaries:
            path.unlink(missing_ok=True)
    return load_cache(dataset, cache_dir)


def _publish(temporaries, array_path, metadata_path, phys_path, ies_path):
    """临时文件全部验证后按序改名发布；出现同名即报错，不覆盖。"""
    for target in (array_path, metadata_path, phys_path, ies_path):
        if target.exists():
            raise FileExistsError(f"缓存生成期间出现已有产物，拒绝覆盖：{target}")
    for source, target in zip(temporaries, (array_path, phys_path, ies_path, metadata_path)):
        source.rename(target)


def load_cache(dataset, cache_dir=None):
    """只读加载并校验缓存；任何缺失或过期直接报错。"""
    array_path, metadata_path, phys_path, ies_path = _cache_paths(dataset, cache_dir)
    missing = [str(p) for p in (array_path, metadata_path, phys_path, ies_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"缓存缺失或未完成：{missing}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("缓存元数据必须是字典")
    X = np.load(array_path, mmap_mode="r", allow_pickle=False)
    phys = np.load(phys_path, mmap_mode="r", allow_pickle=False)
    ies = np.load(ies_path, mmap_mode="r", allow_pickle=False)
    try:
        return _validate_cache(dataset, X, phys, ies, payload)
    except BaseException:
        for array in (X, phys, ies):
            if isinstance(array, np.memmap):
                array._mmap.close()
        raise


def feature_subset(cache, unit_keys=None, channels=None, labeled=True):
    """按单元与通道显式选择；保留 X/phys25/ies_analysis/records 并重排区间。"""
    units = None if unit_keys is None else set(unit_keys)
    selected_channels = None if channels is None else set(channels)
    if units is not None and units - {r["unit_key"] for r in cache.records}:
        raise KeyError(f"缓存中不存在的单元：{sorted(units - {r['unit_key'] for r in cache.records})[:3]}")
    if selected_channels is not None and selected_channels - {r["channel"] for r in cache.records}:
        raise KeyError(f"缓存中不存在的通道：{sorted(selected_channels)}")
    parts_x, parts_phys, labels, record_keys, row_channels, unit_rows, records = [], [], [], [], [], [], []
    analysis_ranges = []
    offset, analysis_offset = 0, 0
    for record in cache.records:
        if units is not None and record["unit_key"] not in units:
            continue
        if selected_channels is not None and record["channel"] not in selected_channels:
            continue
        if labeled and record["label"] is None:
            raise ValueError(f"标注子集含无真值 KB 记录，请显式排除：{record['record_key']}")
        n = record["stop"] - record["start"]
        a_n = record["analysis_stop"] - record["analysis_start"]
        parts_x.append(cache.X[record["start"]:record["stop"]])
        parts_phys.append(cache.phys25[record["start"]:record["stop"]])
        labels.extend([record["label"] if labeled else -1] * n)
        record_keys.extend([record["record_key"]] * n)
        row_channels.extend([record["channel"]] * n)
        unit_rows.extend([record["unit_key"]] * n)
        analysis_ranges.append((record["analysis_start"], record["analysis_stop"]))
        records.append({**record, "start": offset, "stop": offset + n,
                        "analysis_start": analysis_offset, "analysis_stop": analysis_offset + a_n})
        offset += n
        analysis_offset += a_n
    X = np.concatenate(parts_x) if parts_x else np.empty((0, 1, 256), dtype=np.float32)
    phys = (np.concatenate(parts_phys) if parts_phys
            else np.empty((0, 25), dtype=np.float32))
    subset = SimpleNamespace(
        X=torch.from_numpy(np.ascontiguousarray(X)),
        phys25=torch.from_numpy(np.ascontiguousarray(phys)),
        y=torch.tensor(labels, dtype=torch.long),
        record_keys=np.asarray(record_keys, dtype=object),
        channels=np.asarray(row_channels, dtype=object),
        unit_keys=np.asarray(unit_rows, dtype=object),
        records=records,
        # ies_analysis 只在物理锚按记录区间复用；子集视图不复制大数组。
        ies_analysis=cache.ies_analysis,
        analysis_ranges=analysis_ranges,
    )
    return subset


import torch  # noqa: E402  与 v7 相同用途：subset 的张量视图


def _self_check():
    """合成/少量只读检查：窗数映射、IES 合成信号命中、时域公式、缩放与补零。"""
    rng = np.random.default_rng(0)
    # 统一 12 kHz：与 preprocessing 目标采样率及冻结 physical 块一致
    assert TARGET_FS == PHYS_CONFIG["target_fs"] == 12000
    # 16+9 维序与说明：元数据里的维名数量必须与代码实际布局一致
    assert len(PHYS_SPEC["envelope_dim_names"]) == 16 and len(PHYS_SPEC["time_dim_names"]) == 9
    assert PHYS_SPEC["envelope_dim_names"][0] == "BPFOx1"
    # 窗数映射：1025 点 = 2 logspec 窗、0 分析窗；2048 点 = 4 logspec 窗、1 分析窗
    assert analysis_window_count(2) == 0
    assert analysis_window_count(4) == 1
    assert analysis_window_count(93) == 45  # 48001 点 ≈ 93.75 窗 → 93 窗 → 45 分析窗
    mapping = map_logspec_to_analysis(4, 1)
    assert mapping.tolist() == [0, 0, 0, 0]
    mapping = map_logspec_to_analysis(93, 45)
    assert len(mapping) == 93 and mapping.min() >= 0 and mapping.max() <= 44
    # 中心映射附带计数：唯一物理中心时 4 个 logspec 中心全在覆盖区外
    index, ties, outside = mapping_detail(4, 1)
    assert index.tolist() == [0, 0, 0, 0] and ties == 0 and outside == 4
    _, _, edges = mapping_detail(93, 45)
    assert edges >= 1  # 首端中心在首个物理中心左侧，必然有覆盖区外映射
    # 时域公式
    x = rng.standard_normal((3, 512))
    t = _time_features(x)
    assert t.shape == (3, 9)
    assert np.allclose(t[:, 0], x.mean(axis=1))
    assert np.allclose(t[:, 1], x.var(axis=1))
    kurt_expect = (((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)) ** 4).mean(axis=1)
    np.testing.assert_allclose(t[:, 3], kurt_expect)
    # 常量窗：v=0，峭度=0；rms/p、rms/a 等比值在常量下为 1，非零且有限即可
    const = _time_features(np.full((1, 512), 3.5))
    assert const[0, 3] == 0 and np.isfinite(const).all()
    # 缩放与补零
    phys = rng.standard_normal((5, 25))
    scaler = fit_scaler(phys)
    assert len(scaler["mean"]) == 25 and len(scaler["std"]) == 25
    scaled = scale_phys(phys, scaler)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), 1, atol=1e-12)
    dual = dual_input(np.zeros((5, 1, 256), np.float32), phys, scaler)
    assert dual.shape == (5, 2, 256) and dual.dtype == np.float32
    assert np.allclose(dual[:, 1, 25:], 0)
    only = dual_input(np.zeros((5, 1, 256), np.float32), phys, scaler, logspec_only=True)
    assert np.allclose(only[:, 1, :], 0)
    # IES 合成信号：外圈故障特征频率处调制应命中 BPFO 谐波维
    fs = 12000
    rotation_hz = 25.0
    n = 48001
    t_axis = np.arange(n) / fs
    bpfo_hz = 3.5848 * rotation_hz
    carrier = np.cos(2 * np.pi * 3000 * t_axis)
    modulator = 1 + 0.8 * np.cos(2 * np.pi * bpfo_hz * t_axis)
    signal = carrier * modulator + 0.01 * rng.standard_normal(n)
    R = ies_features(signal, CWRU_DE_ORDERS, rotation_hz)
    assert R.shape == (45, 16)
    bpfo_mean = R[:, 0:4].mean()
    other_mean = np.concatenate([R[:, 4:12], R[:, 13:16]], axis=1).mean()
    assert bpfo_mean > 2 * other_mean, (bpfo_mean, other_mean)
    ftf_mean = R[:, 12:16].mean()
    # FTF 频率极低，调制信号不应命中
    assert ftf_mean < bpfo_mean
    print("特征自检通过（合成信号，未构建全量缓存）")


def _project(value):
    """配置里的相对路径相对 v8 根解析；resolve 归一化 `..`，保持数据根字符串稳定。"""
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(path="configs/stage3.json"):
    """读取主配置：校验 cache_version 与 physical 冻结块，返回 cache_dir/data_roots。"""
    payload = json.loads(_project(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != "v8_stage3":
        raise ValueError("必须使用阶段三冻结配置 configs/stage3.json")
    if payload.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"配置 cache_version 与冻结口径不一致：{payload.get('cache_version')}")
    physical = payload.get("physical")
    if not isinstance(physical, dict):
        raise ValueError("配置缺少 physical 块")
    mismatched = {name: (physical.get(name), value) for name, value in PHYS_CONFIG.items()
                  if physical.get(name) != value}
    if mismatched:
        raise ValueError(f"配置 physical 与冻结口径不一致：{mismatched}")
    roots = payload.get("data_roots")
    if not isinstance(roots, dict) or sorted(roots) != sorted(LOADERS):
        raise ValueError(f"配置 data_roots 必须覆盖全部数据集：{roots}")
    if not isinstance(payload.get("cache_dir"), str):
        raise ValueError("配置缺少 cache_dir")
    return payload


def main():
    parser = argparse.ArgumentParser(description="v8 冻结特征缓存（logspec + phys25 + ies_analysis）")
    parser.add_argument("--dataset", choices=("cwru", "paderborn", "all"), default="all")
    parser.add_argument("--config", default="configs/stage3.json",
                        help="主配置（cache_version、physical 与数据根）")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--reuse-logspec", default=None,
                        help="显式只读复用旧 v7 logspec 缓存的目录（逐记录核窗数与逐位数值）")
    parser.add_argument("--reuse-cache", action="store_true", help="已有完整 v8 缓存时显式复用")
    parser.add_argument("--self-check", action="store_true", help="仅合成/少量只读检查")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只枚举检查，不写文件（默认）")
    mode.add_argument("--execute", action="store_true", help="执行全量特征构建（P2 授权后）")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    config = load_config(args.config)
    datasets = tuple(LOADERS) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        cache_dir = args.cache_dir or _project(config["cache_dir"])
        data_root = args.data_root or _project(config["data_roots"][dataset])
        if args.execute:
            cache = build_cache(dataset, cache_dir=cache_dir, data_root=data_root,
                                reuse_logspec=args.reuse_logspec, reuse_cache=args.reuse_cache,
                                execute=True)
            print(f"{dataset} 缓存就绪：{len(cache.records)} 条记录，{len(cache.X)} 窗，版本 {cache.version}")
        else:
            build_cache(dataset, cache_dir=cache_dir, data_root=data_root,
                        reuse_logspec=args.reuse_logspec, reuse_cache=args.reuse_cache, execute=False)


if __name__ == "__main__":
    main()
