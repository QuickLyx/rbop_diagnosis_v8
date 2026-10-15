# -*- coding: utf-8 -*-
"""排除性检查：BPFI/BPFO 的实际峰位是否偏离理论值？

三类独立方法（邻带积分、边带比、倒谱）都显示 IR 无特征。剩下的可能性：
理论故障频率与实际的偏移超出积分窗（滑移、几何误差、转速误差）。

本检查在理论值 ±20% 范围内扫描包络谱峰值位置，比较三类的偏移统计。
若 IR 记录的峰位系统性偏移 → 频率假设问题；若无系统性偏移 → 确认特征缺失。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))

from features import (  # noqa: E402
    load_cache, _bandpass_envelope, _enumerate_recordings, _record_signal_12k, _geometry_for,
)
from bearing_facts import LABEL_NAMES  # noqa: E402

random.seed(0)
cache = load_cache("paderborn", ROOT / "outputs" / "features")
recordings = {r.record_key: r for r in _enumerate_recordings(
    "paderborn", ROOT.parent / "rbop_diagnosis_v7" / "Data" / "paderborn" / "extracted")}

by_bearing = {}
for rec in cache.records:
    if rec["label"] in (0, 2, 3):
        by_bearing.setdefault(rec["bearing_id"], []).append(rec)
picked = []
for bearing, recs in by_bearing.items():
    picked.extend(random.sample(recs, min(2, len(recs))))
print(f"抽样 {len(picked)} 条记录")


def envelope_power(signal, window=8192, stride=4096):
    """整记录包络的功率谱（长窗提高频率分辨率到 1.46 Hz）。"""
    env = _bandpass_envelope(np.asarray(signal, dtype=np.float64))
    n = (len(env) - window) // stride + 1
    if n < 1:
        return None
    blocks = np.stack([env[stride * j: stride * j + window] for j in range(n)])
    blocks = (blocks - blocks.mean(axis=1, keepdims=True)) * np.hanning(window)
    power = (np.abs(np.fft.rfft(blocks, axis=1)) ** 2).mean(axis=0)
    df = 12000 / window
    return power, df


def search_peak(power, df, f_center, rel_range=0.20):
    """在 f_center ±rel_range 内找最大峰，返回（峰频/理论频， 峰高/背景）。"""
    lo = max(1, int(f_center * (1 - rel_range) / df))
    hi = min(len(power) - 1, int(f_center * (1 + rel_range) / df))
    if hi <= lo:
        return np.nan, np.nan
    seg = power[lo:hi + 1]
    k = int(np.argmax(seg))
    f_peak = (lo + k) * df
    # 背景 = 同一范围内除峰邻域外的中位
    mask = np.ones(len(seg), dtype=bool)
    mask[max(0, k - 2):k + 3] = False
    bg = np.median(seg[mask]) if mask.sum() > 3 else np.median(seg)
    return f_peak / f_center, float(seg[k] / max(bg, 1e-30))


rows = []
for rec in picked:
    r = recordings[rec["record_key"]]
    sig = _record_signal_12k(r)
    orders, rot, _, _ = _geometry_for(r)
    out = envelope_power(sig)
    if out is None:
        continue
    power, df = out
    ratio_ir, snr_ir = search_peak(power, df, orders["BPFI"] * rot)
    ratio_or, snr_or = search_peak(power, df, orders["BPFO"] * rot)
    rows.append((rec["label"], ratio_ir, snr_ir, ratio_or, snr_or))

print(f"\n有效记录 {len(rows)}")
print("\n===== BPFI 搜索结论（理论值 ±20% 内的最强峰）=====")
print(f"{'类别':>8} {'n':>4} {'峰位/理论 中位':>15} {'[10%, 90%]':>22} {'峰/背景 中位':>14}")
for lab in (0, 2, 3):
    a = np.array([(r[1], r[2]) for r in rows if r[0] == lab], dtype=float)
    if not len(a):
        continue
    print(f"{LABEL_NAMES[lab]:>8} {len(a):>4} {np.nanmedian(a[:,0]):>15.3f} "
          f"[{np.nanpercentile(a[:,0],10):.3f}, {np.nanpercentile(a[:,0],90):.3f}]".rjust(22)
          + f" {np.nanmedian(a[:,1]):>14.2f}")

print("\n===== BPFO 搜索结论 =====")
print(f"{'类别':>8} {'n':>4} {'峰位/理论 中位':>15} {'[10%, 90%]':>22} {'峰/背景 中位':>14}")
for lab in (0, 2, 3):
    a = np.array([(r[3], r[4]) for r in rows if r[0] == lab], dtype=float)
    if not len(a):
        continue
    print(f"{LABEL_NAMES[lab]:>8} {len(a):>4} {np.nanmedian(a[:,0]):>15.3f} "
          f"[{np.nanpercentile(a[:,0],10):.3f}, {np.nanpercentile(a[:,0],90):.3f}]".rjust(22)
          + f" {np.nanmedian(a[:,1]):>14.2f}")

print("\n判读：")
print("  - 峰位/理论 ≈ 1.00 且三类一致 → 理论频率正确，特征确实缺失")
print("  - IR 类系统性偏离 1.00 → 频率假设需修正（滑移/几何）")
print("  - 峰/背景 若 IR 明显高于 Normal → 该位置确有特征（长窗可见）")
