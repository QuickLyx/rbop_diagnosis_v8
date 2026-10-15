# -*- coding: utf-8 -*-
"""共享记录身份与精确重采样切窗。"""
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class Recording:
    record_key: str
    unit_key: str
    path: Path
    dataset: str
    channel: str
    label: int | None
    src_fs: int
    source: str = ""
    bearing_id: str = ""
    condition: str = ""
    origin: str = ""
    severity: int = 0


def _as_signal(values):
    """拒绝空、非有限、多维或复数输入，不填充或修复信号。"""
    values = np.asarray(values)
    if values.ndim not in (1, 2) or (values.ndim == 2 and min(values.shape) > 1):
        raise ValueError(f"信号必须为一维向量，实际形状为 {values.shape}")
    if np.iscomplexobj(values):
        raise ValueError("信号不能为复数")
    signal = values.reshape(-1).astype(np.float64, copy=False)
    if not signal.size:
        raise ValueError("信号为空")
    if not np.isfinite(signal).all():
        raise ValueError("信号包含非有限值")
    return signal


def to_windows(signal, src_fs, target_fs=12000):
    """按精确有理比重采样，切成 512 点不重叠窗，丢弃末尾不足一窗部分。"""
    signal = _as_signal(signal)
    for rate in (src_fs, target_fs):
        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, np.integer)) or rate <= 0:
            raise ValueError(f"采样率必须是正整数：{rate}")
    if src_fs != target_fs:
        g = gcd(int(target_fs), int(src_fs))
        signal = resample_poly(signal, int(target_fs) // g, int(src_fs) // g)
    signal = _as_signal(signal)
    n = signal.size // 512
    if not n:
        raise ValueError("重采样后的信号不足一个 512 点窗口")
    return signal[:n * 512].reshape(n, 512)
