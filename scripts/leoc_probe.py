# -*- coding: utf-8 -*-
"""LEOC 可行性检查：倒谱域是否存在 IES 漏掉的 IR 信息？

背景：IES 用包络谱的功率（丢失相位），e_IR 与 e_OR 在 PU 上不可分。
LEOC（对数平方包络 + 阶次循环分量）理论上能提取"周期性"，可能在倒谱域
（quefrency = 1/f）看到功率谱看不到的结构。

本检查只做诊断，不产出配方：
  1. 对每条记录算带通包络 → 对数平方包络 → 去趋势 → 倒谱
  2. 在 quefrency = 1/BPFI 与 1/BPFO 处取峰，比较三类记录的峰值统计
  3. 若 IR 记录在 1/BPFI 处显著更高 → 值得实施 LEOC

声明：本检查已看过 PU 标签，结论只能作为迭代探索证据。
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
TARGET_FS = 12000

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


def envelope_cepstrum(signal, block=4096):
    """整记录包络 → 对数平方 → 分块去趋势 → 倒谱平均。"""
    env = _bandpass_envelope(np.asarray(signal, dtype=np.float64))
    log_env = np.log(env ** 2 + 1e-30)
    n = len(log_env) // block
    if n < 2:
        return None, None
    blocks = log_env[: n * block].reshape(n, block)
    # 去趋势：减去块内线性拟合（周期性冲击的长期起伏）
    blocks = blocks - blocks.mean(axis=1, keepdims=True)
    win = np.hanning(block)
    spec = np.abs(np.fft.rfft(blocks * win, axis=1)) ** 2
    # 倒谱：对功率谱再取对数后 FFT
    ceps = np.abs(np.fft.rfft(np.log(spec + 1e-30), axis=1)) ** 2
    return ceps.mean(axis=0), block


def peak_at(ceps, freq, block):
    """倒谱在 quefrency = 1/freq 处的峰（用邻域 ±1 bin 最大）。"""
    q = TARGET_FS / freq                       # 倒频率采样点（样本）
    if q < 2 or q > block // 2:
        return np.nan
    idx = int(round(q))
    lo, hi = max(1, idx - 1), min(len(ceps) - 1, idx + 1)
    return float(ceps[lo:hi + 1].max())


stats = {}
for k, rec in enumerate(picked, 1):
    r = recordings[rec["record_key"]]
    sig = _record_signal_12k(r)
    orders, rot, _, _ = _geometry_for(r)
    ceps, block = envelope_cepstrum(sig)
    if ceps is None:
        continue
    f_bpfi = orders["BPFI"] * rot
    f_bpfo = orders["BPFO"] * rot
    label = rec["label"]
    # 主检查：在 1/BPFI 与 1/BPFO 取峰；对照：相邻频率的峰（伪影控制）
    ir1 = peak_at(ceps, f_bpfi, block)
    or1 = peak_at(ceps, f_bpfo, block)
    ir_ref = np.nanmedian([peak_at(ceps, f_bpfi * s, block) for s in (1.03, 1.06, 0.97, 0.94)])
    or_ref = np.nanmedian([peak_at(ceps, f_bpfo * s, block) for s in (1.03, 1.06, 0.97, 0.94)])
    store = stats.setdefault(label, [])
    store.append((ir1, or1, ir_ref, or_ref))
    if k % 15 == 0:
        print(f"  {k}/{len(picked)}")

print("\n===== 倒谱峰值（类均值，中位数）=====")
print(f"{'类别':>8} {'n':>4} {'1/BPFI峰':>10} {'1/BPFO峰':>10} {'BPFI邻域参照':>12} {'BPFO邻域参照':>12}")
for lab in (0, 2, 3):
    if lab not in stats:
        continue
    a = np.array(stats[lab], dtype=float)
    print(f"{LABEL_NAMES[lab]:>8} {len(a):>4} "
          f"{np.nanmedian(a[:,0]):>10.3e} {np.nanmedian(a[:,1]):>10.3e} "
          f"{np.nanmedian(a[:,2]):>12.3e} {np.nanmedian(a[:,3]):>12.3e}")

print("\n===== 判别力检查：峰 / 邻域参照 比值 =====")
print(f"{'类别':>8} {'BPFI峰/参照':>14} {'BPFO峰/参照':>14} {'两者之比(BPFI/BPFO)':>20}")
for lab in (0, 2, 3):
    a = np.array(stats[lab], dtype=float)
    r_ir = np.nanmedian(a[:, 0] / np.maximum(a[:, 2], 1e-30))
    r_or = np.nanmedian(a[:, 1] / np.maximum(a[:, 3], 1e-30))
    ratio = np.nanmedian((a[:, 0] / np.maximum(a[:, 2], 1e-30)) /
                         np.maximum(a[:, 1] / np.maximum(a[:, 3], 1e-30), 1e-30))
    print(f"{LABEL_NAMES[lab]:>8} {r_ir:>14.3f} {r_or:>14.3f} {ratio:>20.3f}")

print("\n判读：若 IR 类的『两者之比』显著高于 Normal 与 OR，说明倒谱域有可分信息。")
