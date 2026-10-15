# -*- coding: utf-8 -*-
"""决定性实验：窗长是否是 IES 硬判 IR 失败的主因？

对同一条重采样信号，用不同分析窗长计算 e_ir/e_or 硬判，比较 IR 召回。
窗长：2048/4096/8192/16384/全记录。仅抽样记录，不改正式产物。

注意：本实验已看过 PU 标签，只能作为迭代探索证据，不能当盲测。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))

from features import load_cache, _bandpass_envelope, _enumerate_recordings, _record_signal_12k, _geometry_for, _validate_rotation  # noqa: E402
from bearing_facts import LABEL_NAMES  # noqa: E402

import random
random.seed(0)

cache = load_cache("paderborn", ROOT / "outputs" / "features")
recordings = {r.record_key: r for r in _enumerate_recordings(
    "paderborn", ROOT.parent / "rbop_diagnosis_v7" / "Data" / "paderborn" / "extracted")}

# 抽样：每类每轴承 2 条
by_bearing = {}
for rec in cache.records:
    if rec["label"] in (0, 2, 3):
        by_bearing.setdefault(rec["bearing_id"], []).append(rec)
picked = []
for bearing, recs in by_bearing.items():
    picked.extend(random.sample(recs, min(2, len(recs))))
print(f"抽样 {len(picked)} 条记录")

df = 12000 / 2048


def band_ratio(power, noise, center, rot, use_sideband):
    """单个中心频率的邻带积分比。power 形状 (n_freq,)。"""
    b = int(np.floor(center / df + 0.5))
    if use_sideband:
        cands = {b - 2, b - 1, b, b + 1, b + 2}
        for k in (1, 2):
            sb = int(np.floor(k * rot / df + 0.5))
            cands.update({b - sb, b + sb})
        J = sorted([x for x in cands if 1 <= x <= len(power) - 1])
    else:
        low, high = (1, b + 2) if b < 1 else (b - 2, b + 2)
        low, high = max(1, low), min(len(power) - 1, high)
        J = list(range(low, high + 1))
    if not J:
        return 0.0
    total = power[J].sum()
    return total / (len(J) * noise)


def evaluate_window(sig, orders, rot, window, stride, use_segmented, use_sideband):
    """给定窗长，返回 (e_ir 数组, e_or 数组)。"""
    envelope = _bandpass_envelope(np.asarray(sig, dtype=np.float64))
    if window >= len(envelope):
        windows = envelope[None, :]
    else:
        n = (len(envelope) - window) // stride + 1
        windows = np.stack([envelope[stride * j: stride * j + window] for j in range(n)])

    windowed = (windows - windows.mean(axis=1, keepdims=True)) * np.hanning(windows.shape[1])
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    n_freq = power.shape[1]
    bins = np.arange(n_freq)

    if use_segmented:
        scale = rot / 25.0
        m_bpfo = (bins * df >= 50 * scale) & (bins * df <= 350 * scale)
        m_bpfi = (bins * df >= 100 * scale) & (bins * df <= 550 * scale)
        noise_bpfo = np.median(power[:, m_bpfo], axis=1)
        noise_bpfi = np.median(power[:, m_bpfi], axis=1)
    else:
        m = (bins > 0) & (bins * df <= 500)
        noise_g = np.median(power[:, m], axis=1)

    mean_power = power[:, 1:].mean(axis=1)
    eps = np.maximum(1e-30, 1e-12 * mean_power)

    e_or = np.zeros(len(windows))
    e_ir = np.zeros(len(windows))
    for i in range(len(windows)):
        noise_o = (noise_bpfo[i] if use_segmented else noise_g[i]) + eps[i]
        noise_i = (noise_bpfi[i] if use_segmented else noise_g[i]) + eps[i]
        e_or[i] = np.mean([band_ratio(power[i], noise_o, orders["BPFO"] * h * rot, rot, False)
                           for h in (1, 2, 3, 4)])
        e_ir[i] = np.mean([band_ratio(power[i], noise_i, orders["BPFI"] * h * rot, rot, use_sideband)
                           for h in (1, 2, 3, 4)])
    return e_ir, e_or


configs = [
    ("2048/1024 分频段+边带", 2048, 1024, True, True),
    ("4096/2048 分频段+边带", 4096, 2048, True, True),
    ("8192/4096 分频段+边带", 8192, 4096, True, True),
    ("16384/8192 分频段+边带", 16384, 8192, True, True),
    ("整记录 分频段+边带", 10**9, 0, True, True),
    ("整记录 全局噪底+无带", 10**9, 0, False, False),
    ("2048/1024 全局噪底+无带", 2048, 1024, False, False),
]

stats = {name: {0: [0, 0], 2: [0, 0], 3: [0, 0]} for name, *_ in configs}
ratio_store = {name: {0: [], 2: [], 3: []} for name, *_ in configs}

for k, rec in enumerate(picked, 1):
    r = recordings[rec["record_key"]]
    sig = _record_signal_12k(r)
    orders, rot, _, _ = _geometry_for(r)
    label = rec["label"]
    for name, window, stride, seg, sb in configs:
        e_ir, e_or = evaluate_window(sig, orders, rot, window, stride, seg, sb)
        e_ir_m = float(e_ir.mean())
        e_or_m = float(e_or.mean())
        stats[name][label][1] += 1
        ratio = e_ir_m / max(e_or_m, 1e-30)
        ratio_store[name][label].append(ratio)
        pred = 0
        if e_or_m > 1.5 * e_ir_m:
            pred = 3
        elif e_ir_m > 1.5 * e_or_m:
            pred = 2
        stats[name][label][0] += int(pred == label)
    if k % 10 == 0:
        print(f"  {k}/{len(picked)}")

print("\n===== 窗长对硬判召回的影响（记录级，% ）=====")
print(f"{'配置':>26} {'Normal':>9} {'IR':>9} {'OR':>9} {'宏召回':>9}")
for name, *_ in configs:
    vals = []
    for lab in (0, 2, 3):
        ok, tot = stats[name][lab]
        vals.append(100 * ok / max(tot, 1))
    macro = np.mean(vals)
    print(f"{name:>26} {vals[0]:>8.1f}% {vals[1]:>8.1f}% {vals[2]:>8.1f}% {macro:>8.1f}%")

print("\n===== e_ir/e_or 中位数（区分度看这个）=====")
print(f"{'配置':>26} {'Normal':>9} {'IR':>9} {'OR':>9}")
for name, *_ in configs:
    vals = []
    for lab in (0, 2, 3):
        arr = ratio_store[name][lab]
        vals.append(np.median(arr) if arr else float("nan"))
    print(f"{name:>26} {vals[0]:>9.3f} {vals[1]:>9.3f} {vals[2]:>9.3f}")
