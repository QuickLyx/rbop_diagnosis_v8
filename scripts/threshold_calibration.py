# -*- coding: utf-8 -*-
"""源域阈值校准：CWRU 上的最优 IR/OR 判别阈值是多少？

只读 CWRU 缓存（源域），搜索使三类平衡正确率最高的阈值对。
PU 只在最后核验，且本实验已看过 PU 分布，结论只能作为迭代探索证据。

判决形式：ratio = e_IR/e_OR；ratio > t_ir 判 IR；ratio < t_or 判 OR；否则 Normal。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))

from features import load_cache  # noqa: E402
from bearing_facts import LABEL_NAMES  # noqa: E402


def ratios(cache):
    """逐记录 ratio = e_IR/e_OR（记录级平均后的比值）。"""
    out = {}
    for rec in cache.records:
        label = rec["label"]
        if label not in (0, 2, 3):
            continue
        R = np.asarray(cache.ies_analysis[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        e_or = float(R[:, 0:4].mean())
        e_ir = float(R[:, 4:8].mean())
        out.setdefault(label, []).append(e_ir / max(e_or, 1e-30))
    return {k: np.array(v) for k, v in out.items()}


def score(r, t_ir, t_or, macro=True):
    """给定阈值对，返回三类召回与平衡正确率。"""
    n_ok = np.mean(r[0] <= t_ir) * 0 + np.mean((r[0] >= t_or) & (r[0] <= t_ir))
    ir_ok = np.mean(r[2] > t_ir)
    or_ok = np.mean(r[3] < t_or)
    if macro:
        return np.mean([n_ok, ir_ok, or_ok]), (n_ok, ir_ok, or_ok)
    return (n_ok + ir_ok + or_ok) / 3, (n_ok, ir_ok, or_ok)


cwru = load_cache("cwru", ROOT / "outputs" / "features")
r_cwru = ratios(cwru)

print("===== CWRU 源域 ratio 分布（e_IR/e_OR）=====")
for lab in (0, 2, 3):
    a = r_cwru[lab]
    print(f"  {LABEL_NAMES[lab]:>11}: n={len(a):>3} "
          f"10%={np.percentile(a,10):.3f} 中位={np.median(a):.3f} 90%={np.percentile(a,90):.3f}")

# 固定现行 OR 阈值 1/1.5=0.667，只搜 IR 阈值
print("\n===== 固定 t_or=0.667，搜索 t_ir（源域宏召回）=====")
best = None
for t_ir in np.arange(0.6, 1.65, 0.05):
    m, (n_ok, ir_ok, or_ok) = score(r_cwru, t_ir, 1 / 1.5)
    flag = ""
    if best is None or m > best[0]:
        best = (m, t_ir)
        flag = "  <-"
    print(f"  t_ir={t_ir:.2f}: N={100*n_ok:>5.1f}% IR={100*ir_ok:>5.1f}% OR={100*or_ok:>5.1f}% 宏={100*m:>5.1f}%{flag}")

print(f"\n源域最优 t_ir = {best[1]:.2f}（宏召回 {100*best[0]:.1f}%）")

# 二维搜索
print("\n===== 二维搜索 t_ir × t_or =====")
best2 = None
for t_ir in np.arange(0.6, 1.65, 0.05):
    for t_or in np.arange(0.4, 1.05, 0.05):
        if t_or > t_ir:
            continue
        m, _ = score(r_cwru, t_ir, t_or)
        if best2 is None or m > best2[0]:
            best2 = (m, t_ir, t_or)
print(f"  最优组合: t_ir={best2[1]:.2f}, t_or={best2[2]:.2f}（宏召回 {100*best2[0]:.1f}%）")

# PU 核验（事后）
print("\n===== PU 事后核验（仅探索证据）=====")
pu = load_cache("paderborn", ROOT / "outputs" / "features")
r_pu = ratios(pu)

for name, t_ir, t_or in [("现行 1.5/0.667", 1.5, 1 / 1.5),
                          (f"源域最优 {best[1]:.2f}/{1/1.5:.3f}", best[1], 1 / 1.5),
                          (f"二维最优 {best2[1]:.2f}/{best2[2]:.2f}", best2[1], best2[2])]:
    m, (n_ok, ir_ok, or_ok) = score(r_pu, t_ir, t_or)
    print(f"  {name:>22}: N={100*n_ok:>5.1f}% IR={100*ir_ok:>5.1f}% OR={100*or_ok:>5.1f}% 宏={100*m:>5.1f}%")

print("\nPU ratio 分布参考:")
for lab in (0, 2, 3):
    a = r_pu[lab]
    print(f"  {LABEL_NAMES[lab]:>11}: 中位={np.median(a):.3f} 90%={np.percentile(a,90):.3f}")
