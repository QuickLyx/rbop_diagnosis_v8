# -*- coding: utf-8 -*-
"""v8 特征 v2：修订 IES 规则（分频段噪底、边带扩展、源拟合 alpha）。

修订来源：docs/IES_审定_v1.md。v1 保持不变，v2 另建缓存目录。
核心改动：
1. 分频段噪底：BPFO/BPFI 各用本频段的噪声底，消除 1/f 背景偏置。
2. BPFI 边带扩展：±(2 bin, ±1·rot, ±2·rot) 并集，覆盖转频调制。
3. q_phys alpha：从源域 Normal 拟合 e_IR/e_OR 几何平均，替代硬编码 1。

与 v1 的接口兼容：ies_features_v2 返回 (A,16)、physical_baseline_v2 的
ies_evidence_v2 同签名；缓存版本为 v8_dual25_12k_512s512_2048s1024_ies_v2。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

# 复用 v1 的基础设施与常量
from features import (
    TARGET_FS, PHYS_CONFIG, _real_signal, _validate_rotation,
    _bandpass_envelope, _time_features, map_logspec_to_analysis,
    fit_scaler, scale_phys, _record_signal_12k, _geometry_for,
    _loader_read_signal, LOADERS, FeatureCache,
)
from bearing_facts import LABEL_NAMES

__all__ = ["ies_features_v2", "physical_features_v2", "CACHE_VERSION_V2",
           "alpha_from_source_normal", "ies_evidence_v2"]

CACHE_VERSION_V2 = "v8_dual25_12k_512s512_2048s1024_ies_v2"

#: 修订后的 IES 规则说明（落盘到 rule.json）
IES_RULE_V2 = {
    "version": "v2",
    "analysis_window": 2048, "analysis_stride": 1024,
    "bandpass_hz": [2000.0, 5000.0], "bandpass_order": 4,
    "harmonics": [1, 2, 3, 4], "order_names": ["BPFO", "BPFI", "BSF", "FTF"],
    "hard_rule": "e_OR > 1.5*e_IR 判 OR；e_IR > 1.5*e_OR 判 IR；否则 Normal；Ball 恒零",
    "ies_ratio_threshold": 1.5,
    "revision_v2": {
        "segmented_noise": "BPFO 用 50–350 Hz 中位，BPFI 用 100–550 Hz 中位（N15 基准，N09 等比例缩放）",
        "sideband_aware": "BPFI 邻带扩展为 ±(2 bin, ±1·rot, ±2·rot) 并集；BPFO 保持 ±2 bin",
        "alpha_calibration": "q_phys 伪计数 alpha = 源域 Normal 记录 sqrt(e_IR·e_OR) 几何平均",
    },
    "training": "零拟合：alpha 在 CWRU 源训练/验证 Normal 上拟合，不用 PU 数据",
}


def _noise_bands_v2(rotation_hz):
    """分频段噪底的频率范围（Hz）：[BPFO 段, BPFI 段]。
    
    N15 基准（25 Hz）：BPFO 四谐波 77–307 Hz 用 50–350 Hz 噪底；
    BPFI 四谐波 123–493 Hz 用 100–550 Hz 噪底。
    其他转频等比例缩放，保持相对带宽。
    """
    ratio = rotation_hz / 25.0
    return [(50 * ratio, 350 * ratio), (100 * ratio, 550 * ratio)]


def ies_features_v2(signal_12k, orders, rotation_hz):
    """修订版 IES：分频段噪底 + BPFI 边带扩展。返回 (A,16) 原始 R。
    
    与 v1 接口兼容，内部实现三项修订（审定 v1）。
    """
    signal_12k = _real_signal(signal_12k, "IES v2 输入")
    rotation_hz = _validate_rotation(orders, rotation_hz)
    if signal_12k.ndim != 1 or signal_12k.size < 2048:
        raise ValueError("IES v2 要求不少于 2048 点的一维 12 kHz 信号")
    if not np.isfinite(signal_12k).all():
        raise ValueError("IES v2 输入包含非有限值")
    
    envelope = _bandpass_envelope(signal_12k)
    n_analysis = (signal_12k.size - 2048) // 1024 + 1
    if n_analysis < 1:
        raise ValueError("信号不足一个 2048 点分析窗")
    
    windows = np.stack([envelope[1024 * j: 1024 * j + 2048] for j in range(n_analysis)])
    windowed = (windows - windows.mean(axis=1, keepdims=True)) * np.hanning(2048)
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    
    df = 12000 / 2048
    bins = np.arange(power.shape[1])
    
    # 分频段噪底
    noise_bands = _noise_bands_v2(rotation_hz)
    noise_bpfo_mask = (bins * df >= noise_bands[0][0]) & (bins * df <= noise_bands[0][1])
    noise_bpfi_mask = (bins * df >= noise_bands[1][0]) & (bins * df <= noise_bands[1][1])
    noise_bpfo = np.median(power[:, noise_bpfo_mask], axis=1)
    noise_bpfi = np.median(power[:, noise_bpfi_mask], axis=1)
    
    mean_power = power[:, 1:].mean(axis=1)
    eps_p = np.maximum(1e-30, 1e-12 * mean_power)
    
    result = np.empty((n_analysis, 16), dtype=np.float64)
    column = 0
    
    for name in ("BPFO", "BPFI", "BSF", "FTF"):
        # BPFI 边带扩展，其余保持 ±2 bin
        use_sideband = (name == "BPFI")
        noise = noise_bpfi if use_sideband else noise_bpfo
        
        for harmonic in (1, 2, 3, 4):
            center = orders[name] * harmonic * rotation_hz
            b = int(np.floor(center / df + 0.5))
            
            if use_sideband:
                # BPFI: ±(2 bin, ±1·rot, ±2·rot) 并集
                candidates = {b - 2, b - 1, b, b + 1, b + 2}
                for k in (1, 2):
                    sb_bins = int(np.floor(k * rotation_hz / df + 0.5))
                    candidates.update({b - sb_bins, b + sb_bins})
                candidates = sorted([x for x in candidates if 1 <= x <= 1024])
                if not candidates:
                    raise ValueError(f"BPFI {name}x{harmonic} 边带为空")
                J = candidates
            else:
                # BPFO/BSF/FTF: ±2 bin
                if b < 1:
                    low, high = 1, b + 2
                else:
                    low, high = b - 2, b + 2
                low, high = max(1, low), min(1024, high)
                if low > high:
                    raise ValueError(f"{name}x{harmonic} 邻带为空：{center} Hz")
                J = list(range(low, high + 1))
            
            # 按实际邻带长度归一化
            columns = power[:, J]
            ratio = columns.sum(axis=1) / (len(J) * (noise + eps_p))
            result[:, column] = ratio
            column += 1
    
    if not np.isfinite(result).all():
        raise ValueError("IES v2 特征包含非有限值")
    return result


def physical_features_v2(signal_12k, analysis):
    """25 维原始物理摘要 v2 = 16 维 IES_v2(log1p(R)) 映射 + 9 维时域。
    
    与 v1 接口兼容：analysis 为 ies_features_v2 的 (A,16)。
    """
    signal_12k = _real_signal(signal_12k, "物理摘要 v2 输入")
    if signal_12k.ndim != 1:
        raise ValueError("物理摘要 v2 要求一维 12 kHz 信号")
    analysis = np.asarray(analysis, dtype=np.float64)
    if analysis.ndim != 2 or analysis.shape[1] != 16 or not len(analysis):
        raise ValueError(f"analysis 必须是 (A,16) 非空原始 R，实际 {analysis.shape}")
    
    n_windows = signal_12k.size // 512
    if n_windows < 1:
        raise ValueError("物理摘要 v2 要求不少于 512 点")
    
    mapped = map_logspec_to_analysis(n_windows, len(analysis))
    envelope_summary = np.log1p(analysis)[mapped]
    windows = signal_12k[: n_windows * 512].reshape(n_windows, 512)
    return np.column_stack([envelope_summary, _time_features(windows)])


def alpha_from_source_normal(normal_records, cache_ies_analysis):
    """从源域 Normal 记录拟合 alpha = sqrt(e_IR · e_OR) 几何平均。
    
    Args:
        normal_records: 源域 Normal 记录列表（带 analysis_start/stop）
        cache_ies_analysis: 整个缓存的 ies_analysis 数组（可以是 v1 或 v2）
    
    Returns:
        float: alpha 伪计数（>0）
    """
    if not normal_records:
        raise ValueError("源域 Normal 记录为空，无法拟合 alpha")
    
    e_ir_all, e_or_all = [], []
    for rec in normal_records:
        R = np.asarray(cache_ies_analysis[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        if not len(R):
            continue
        e_or = R[:, 0:4].mean(axis=1)
        e_ir = R[:, 4:8].mean(axis=1)
        e_ir_all.append(e_ir)
        e_or_all.append(e_or)
    
    if not e_ir_all:
        raise ValueError("所有源域 Normal 记录均无有效分析窗")
    
    e_ir = np.concatenate(e_ir_all)
    e_or = np.concatenate(e_or_all)
    
    # 几何平均：sqrt(e_IR · e_OR) 对所有 Normal 窗口取平均
    geom = np.sqrt(e_ir * e_or)
    alpha = float(geom.mean())
    
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"拟合的 alpha 无效：{alpha}")
    
    return alpha


def ies_evidence_v2(R, alpha):
    """修订版 IES 证据：分频段噪底已在 R 中体现，这里只改 q_phys 公式。
    
    Args:
        R: (A,16) 原始 R，来自 ies_features_v2
        alpha: 源域拟合的伪计数
    
    Returns:
        (hard_onehot, q_phys, e_ir, e_or)
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != 16 or not len(R):
        raise ValueError("ies_evidence_v2 要求非空 (A,16) 原始 R")
    if not np.isfinite(R).all() or (R < 0).any():
        raise ValueError("原始 R 必须有限且非负")
    if not isinstance(alpha, (int, float, np.number)) or alpha <= 0:
        raise ValueError(f"alpha 必须是正数：{alpha}")
    
    e_or = R[:, 0:4].mean(axis=1)
    e_ir = R[:, 4:8].mean(axis=1)
    
    # 硬判规则不变
    hard = np.zeros(len(R), dtype=np.int64)
    hard[e_or > 1.5 * e_ir] = 3
    hard[e_ir > 1.5 * e_or] = 2
    onehot = np.zeros((len(R), 4), dtype=np.float64)
    onehot[np.arange(len(R)), hard] = 1.0
    
    # 修订 q_phys：用拟合的 alpha 替代硬编码 1
    alpha_arr = np.full(len(R), alpha, dtype=np.float64)
    v = np.column_stack([
        alpha_arr,
        alpha_arr,
        (alpha_arr + e_ir) / (1.5 * (alpha_arr + e_or)),
        (alpha_arr + e_or) / (1.5 * (alpha_arr + e_ir)),
    ])
    q_phys = v / v.sum(axis=1, keepdims=True)
    
    if not np.isfinite(q_phys).all() or (q_phys <= 0).any():
        raise ValueError("q_phys_v2 必须有限且恒正")
    
    return onehot, q_phys, e_ir, e_or


def _self_check_v2():
    """v2 规则自检：合成信号、边界、alpha 范围。"""
    # 合成 N15 工况：BPFO=76.8 Hz, BPFI=123.2 Hz
    from bearing_facts import BY_ID
    bearing = BY_ID["KI01"]
    bpfo, bpfi = bearing.fault_orders
    rotation_hz = 25.0
    orders = {"BPFO": bpfo, "BPFI": bpfi, "BSF": 2.0, "FTF": 0.4}
    
    # 4 秒信号，单频 BPFI
    t = np.linspace(0, 4, 12000 * 4, endpoint=False)
    sig = np.sin(2 * np.pi * bpfi * rotation_hz * t)
    
    R = ies_features_v2(sig, orders, rotation_hz)
    assert R.shape == (45, 16), f"窗数错误：{R.shape}"
    assert np.isfinite(R).all() and (R >= 0).all(), "R 非有限或负"
    
    # BPFI 通道应显著高于 BPFO
    e_ir_mean = R[:, 4:8].mean()
    e_or_mean = R[:, 0:4].mean()
    assert e_ir_mean > 2 * e_or_mean, f"BPFI 单频信号 e_IR 未占优：{e_ir_mean} vs {e_or_mean}"
    
    # alpha 边界
    alpha = 1.5
    onehot, q, _, _ = ies_evidence_v2(R, alpha)
    assert q.shape == (45, 4) and np.allclose(q.sum(axis=1), 1.0), "q_phys 不归一"
    assert (q > 0).all(), "q_phys 有非正分量"
    
    # 全零应判 Normal
    zero_R = np.zeros((1, 16))
    onehot_z, qz, _, _ = ies_evidence_v2(zero_R, alpha)
    assert onehot_z[0, 0] == 1.0, "全零未判 Normal"
    
    print("IES v2 规则自检通过")
    print(f"  边带扩展：BPFI 单频信号 e_IR/e_OR = {e_ir_mean/e_or_mean:.2f}")
    print(f"  alpha 测试：{alpha}（源域实际值需从 CWRU Normal 拟合）")


if __name__ == "__main__":
    _self_check_v2()
