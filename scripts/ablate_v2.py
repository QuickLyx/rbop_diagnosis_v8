# -*- coding: utf-8 -*-
"""v2 消融诊断：分离"分频段噪底"与"边带扩展"两项修订各自的贡献。

方法：用重采样后的信号重算 R（信号不重新读取，只用缓存 logspec 的窗口数推算），
分别测试：
  A. v2 全量（分频段 + 边带）
  B. 只分频段噪底（BPFI 保持 ±2 bin）
  C. 只边带扩展（全局噪底）
  D. v1 原版
只看硬判召回，不写入正式产物。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))

from features import load_cache, _bandpass_envelope, _enumerate_recordings, _record_signal_12k, _geometry_for  # noqa: E402
from bearing_facts import LABEL_NAMES  # noqa: E402

# 抽样：每类每轴承选 2 条记录
import random
random.seed(0)

cache_v1 = load_cache("paderborn", ROOT / "outputs" / "features")
recs_by_bearing = {}
for rec in cache_v1.records:
    if rec["label"] not in (0, 2, 3):
        continue
    recs_by_bearing.setdefault(rec["bearing_id"], []).append(rec)

picked = []
for bearing, recs in recs_by_bearing.items():
    picked.extend(random.sample(recs, min(2, len(recs))))
print(f"抽样记录数: {len(picked)}")

# 读取原始信号
recordings = {r.record_key: r for r in _enumerate_recordings("paderborn", ROOT.parent / "rbop_diagnosis_v7" / "Data" / "paderborn" / "extracted")}

df = 12000 / 2048
bins = np.arange(1025)


def compute_R(signal, orders, rot, use_segmented_noise, use_sideband):
    """按开关组合计算 R（逐列）。
    
    返回 (A,16) 或 (A,4)（只算 BPFO+BPFI 用于诊断）。
    """
    envelope = _bandpass_envelope(np.asarray(signal, dtype=np.float64))
    n_analysis = (len(signal) - 2048) // 1024 + 1
    windows = np.stack([envelope[1024 * j: 1024 * j + 2048] for j in range(n_analysis)])
    windowed = (windows - windows.mean(axis=1, keepdims=True)) * np.hanning(2048)
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2

    if use_segmented_noise:
        ratio_scale = rot / 25.0
        b_bpfo = (bins * df >= 50 * ratio_scale) & (bins * df <= 350 * ratio_scale)
        b_bpfi = (bins * df >= 100 * ratio_scale) & (bins * df <= 550 * ratio_scale)
        noise_bpfo = np.median(power[:, b_bpfo], axis=1)
        noise_bpfi = np.median(power[:, b_bpfi], axis=1)
    else:
        mask = (bins > 0) & (bins * df <= 500)
        noise_global = np.median(power[:, mask], axis=1)

    mean_power = power[:, 1:].mean(axis=1)
    eps_p = np.maximum(1e-30, 1e-12 * mean_power)

    result = np.empty((n_analysis, 8), dtype=np.float64)
    col = 0
    for name in ("BPFO", "BPFI"):
        use_sb = use_sideband and (name == "BPFI")
        noise = (noise_bpfi if use_sb else noise_bpfo) if use_segmented_noise else noise_global
        for harmonic in (1, 2, 3, 4):
            center = orders[name] * harmonic * rot
            b = int(np.floor(center / df + 0.5))
            if use_sb:
                cands = {b - 2, b - 1, b, b + 1, b + 2}
                for k in (1, 2):
                    sb = int(np.floor(k * rot / df + 0.5))
                    cands.update({b - sb, b + sb})
                J = sorted([x for x in cands if 1 <= x <= 1024])
            else:
                low, high = (1, b + 2) if b < 1 else (b - 2, b + 2)
                low, high = max(1, low), min(1024, high)
                J = list(range(low, high + 1))
            cols = power[:, J]
            result[:, col] = cols.sum(axis=1) / (len(J) * (noise + eps_p))
            col += 1
    return result


def hard_recall(configs):
    """对每个配置，返回各类硬判召回（记录级）。"""
    stats = {name: {0: [0, 0], 2: [0, 0], 3: [0, 0]} for name in configs}
    for rec in picked:
        r = recordings[rec["record_key"]]
        sig = _record_signal_12k(r)
        orders, rot, _, _ = _geometry_for(r)
        for name, (seg, sb) in configs.items():
            R = compute_R(sig, orders, rot, seg, sb)
            e_or = R[:, 0:4].mean(axis=1)
            e_ir = R[:, 4:8].mean(axis=1)
            hard = np.zeros(len(R), dtype=int)
            hard[e_or > 1.5 * e_ir] = 3
            hard[e_ir > 1.5 * e_or] = 2
            pred = int(np.bincount(hard, minlength=4).argmax())
            label = rec["label"]
            stats[name][label][1] += 1
            stats[name][label][0] += int(pred == label)
    return stats


configs = {
    "A_v2全量": (True, True),
    "B_只分频段": (True, False),
    "C_只边带": (False, True),
    "D_v1原版": (False, False),
}

stats = hard_recall(configs)

print("\n===== 消融：记录级硬判召回 =====")
print(f"{'配置':>12} {'Normal':>10} {'IR':>10} {'OR':>10}")
for name in configs:
    vals = []
    for lab in (0, 2, 3):
        ok, tot = stats[name][lab]
        vals.append(f"{100*ok/max(tot,1):>9.1f}%")
    print(f"{name:>12} {vals[0]} {vals[1]} {vals[2]}")

# 同时统计 e_ir/e_or 的分布，看噪声底改动的影响
print("\n===== e_ir/e_or 中位数（按记录类型） =====")
print(f"{'配置':>12} {'N_ratio':>10} {'IR_ratio':>10} {'OR_ratio':>10}")
ratio_med = {name: {} for name in configs}
for rec in picked:
    r = recordings[rec["record_key"]]
    sig = _record_signal_12k(r)
    orders, rot, _, _ = _geometry_for(r)
    label = rec["label"]
    for name, (seg, sb) in configs.items():
        R = compute_R(sig, orders, rot, seg, sb)
        e_or = float(R[:, 0:4].mean())
        e_ir = float(R[:, 4:8].mean())
        ratio_med[name].setdefault(label, []).append(e_ir / max(e_or, 1e-30))

for name in configs:
    vals = []
    for lab in (0, 2, 3):
        arr = ratio_med[name].get(lab, [1.0])
        vals.append(f"{np.median(arr):>10.3f}")
    print(f"{name:>12} {vals[0]} {vals[1]} {vals[2]}")

# ---- alpha 与 q_phys 的关系 ----
print("\n===== q_phys 对 IR 记录的方向（alpha 敏感性） =====")
print(f"{'alpha':>8} {'N':>8} {'B':>8} {'IR':>8} {'OR':>8}  (PU IR 记录均值)")

# 用 v2 全量的 R 在 PU IR 记录上测不同 alpha
ir_recs = [r for r in cache_v1.records if r["label"] == 2]
sample_ir = random.sample(ir_recs, min(40, len(ir_recs)))

# 用 v2 缓存里的 R（已是 v2 规则）
cache_v2_meta = json.loads((ROOT / "outputs" / "features_v2" / "paderborn.json").read_text(encoding="utf-8"))
ies_v2 = np.load(ROOT / "outputs" / "features_v2" / "paderborn_ies_analysis.npy", mmap_mode="r")

for alpha in (0.1, 0.5, 1.0, 2.317, 5.0):
    qs = []
    for rec in sample_ir:
        R = np.asarray(ies_v2[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        e_or = R[:, 0:4].mean(axis=1)
        e_ir = R[:, 4:8].mean(axis=1)
        v = np.column_stack([np.full(len(R), alpha), np.full(len(R), alpha),
                             (alpha + e_ir) / (1.5 * (alpha + e_or)),
                             (alpha + e_or) / (1.5 * (alpha + e_ir))])
        q = v / v.sum(axis=1, keepdims=True)
        qs.append(q.mean(axis=0))
    m = np.mean(qs, axis=0)
    print(f"{alpha:>8.3f} {m[0]:>8.3f} {m[1]:>8.3f} {m[2]:>8.3f} {m[3]:>8.3f}")
