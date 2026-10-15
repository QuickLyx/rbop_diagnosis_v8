# -*- coding: utf-8 -*-
"""冻结 logspec512 口径的纯本地实现；v8 不再经 `_v6` 动态加载 v6。

由 v6 `preprocessing.py` 原样本地化（计划 §11：实际用到的旧纯函数直接本地复制，
保持校验/异常行为）。logspec 数值必须与 v7 逐位一致（测试用静态加载本模块比对）。

口径（与 v6/v7 完全相同，冻结）：512 点窗口 → 乘 Hanning → rFFT 取幅度 + 1e-9 →
在 0～6000 Hz 全频段用 `linspace(0, 末索引, 256).astype(int)` 等距取 256 点 →
取 log → 逐窗口跨频标准化（eps=1e-9）→ (N,1,256) float32。

报错约定：空信号、非有限值、不足一个窗口、形状不符均直接报错，不填充修复。
"""
from dataclasses import dataclass

import numpy as np

#: 统一采样率（Hz）；48k 来源在读取时已按精确有理比降到该采样率。
TARGET_FS = 12000
#: v8 单表征协议：只允许 logspec512。
REPRESENTATIONS = ("logspec512",)


@dataclass
class FeatureSet:
    X: np.ndarray
    y: np.ndarray
    record_keys: np.ndarray
    channels: np.ndarray


def build_repr_config(seg_len=512, nbin=256):
    """构造 logspec512 表征配置；窗长只允许 512，分箱固定 256。

    返回可 JSON 序列化的配置字典，`input_dim=256`、`n_channels=1` 供模型构造输入层。
    """
    seg_len, nbin = int(seg_len), int(nbin)
    if seg_len != 512 or nbin != 256:
        raise ValueError("v8 表征固定 logspec512：seg_len=512、nbin=256")
    df = TARGET_FS / seg_len
    return {
        "representation": "logspec512",
        "repr_kind": "logspec",
        "seg_len": seg_len,
        "window": "hanning",
        "nbin": 256,
        "log_eps": 1e-9,
        "std_eps": 1e-9,
        "seg_std": True,
        "n_channels": 1,
        "input_dim": 256,
        "band": None,
        "filter": None,
        "scipy_version": None,
        "fs": TARGET_FS,
        "freq": {"df": float(df), "bins": 256, "mode": "sparse_fullband", "end_hz": float(TARGET_FS / 2)},
    }


def config_from_saved(saved, require_complete=False):
    """恢复表征配置；旧 512 简写格式允许补默认，新配置逐字段校验。

    v8 只保留 logspec512 一条路径：`require_complete` 供正式 checkpoint 拒绝缺配置的产物。
    """
    if saved is None:
        saved = {}
    if not isinstance(saved, dict):
        raise ValueError("保存的表征配置必须是字典")
    saved = dict(saved)
    legacy = set(saved) <= {"seg_len", "input_dim", "n_channels"}
    if legacy:
        if require_complete:
            raise ValueError("新 checkpoint 缺少完整预处理配置")
        expected = {"seg_len": 512, "input_dim": 256, "n_channels": 1}
        if any(value != expected[key] for key, value in saved.items()):
            raise ValueError("旧格式只兼容 512 点、256 维、单通道表征")
        return build_repr_config()
    if "representation" not in saved:
        raise ValueError("新表征配置缺少 representation，不能按旧格式补默认")
    if "seg_len" not in saved or "nbin" not in saved:
        raise ValueError("新表征配置缺少窗长或频点数量")
    restored = build_repr_config(saved["seg_len"], saved["nbin"])
    for name, value in restored.items():
        if name not in saved or saved[name] != value:
            raise ValueError(f"保存的表征配置缺失或与当前实现、环境不一致：{name}")
    return {**saved, **restored}


def representation_of(config):
    """返回配置的表征名；缺省按 logspec512 处理。"""
    if config is None:
        return "logspec512"
    name = config.get("representation")
    if name is None:
        return "logspec512"
    if name not in REPRESENTATIONS:
        raise ValueError(f"未知表征：{name}")
    return name


def _shape_std(values, eps=1e-9):
    """逐窗口跨频带标准化：(N,D)→(N,D)，每个窗口独立做 z-score。

    按窗口做而不跨窗口共享均值方差：每个窗口的能量量级不同，跨窗口统计会把
    大能量窗口的幅值差异带进所有样本；窗口内标准化只保留窗内相对谱形。
    """
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + eps)


def extract_logspec(signals, config=None):
    """logspec 特征提取：分段信号 (N,512) → (N,1,256) float32。

    Hanning → rFFT 幅度 + log_eps → 全频段等距取 256 点（含端点、int 截断）→
    log → 窗口内标准化。形状不符或含非有限值直接报错。
    """
    cfg = config or build_repr_config()
    representation_of(cfg)
    seg_len = int(cfg["seg_len"])
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != seg_len:
        raise ValueError(f"signals 必须为 (N,{seg_len})")
    if signals.size and not np.isfinite(signals).all():
        raise ValueError("signals 包含非有限值")
    amp = np.abs(np.fft.rfft(signals * np.hanning(seg_len), axis=1)) + float(cfg["log_eps"])
    pick = np.linspace(0, amp.shape[1] - 1, int(cfg["nbin"])).astype(int)
    selected = amp[:, pick]
    features = _shape_std(np.log(selected), float(cfg["std_eps"])).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("对数谱计算产生非有限特征，拒绝预测或训练")
    return features[:, None, :]


def _self_check():
    """模块自检：校验 logspec 形状与窗口内标准化均值≈0，不依赖外部数据。"""
    rng = np.random.default_rng(0)
    features = extract_logspec(rng.standard_normal((4, 512)))
    assert features.shape == (4, 1, 256)
    assert np.max(np.abs(features.mean(axis=2))) < 1e-5
    assert config_from_saved(build_repr_config(), require_complete=True) == build_repr_config()
    print("ALL PREPROCESSING CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
