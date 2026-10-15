# -*- coding: utf-8 -*-
"""q_phys 中性基准扫描：找"不伤 Normal 又能判 IR/OR"的平衡点。

形式（线性份额族，基准 c 可调）：
    q = [c, c, (1-2c)*e_IR/(e_IR+e_OR), (1-2c)*e_OR/(e_IR+e_OR)]

c=0.1 → 接近纯比值（IR/OR 占 0.8）
c=1/3 → 四类均匀（无信息）
v1 的等效基准不是常数，但量级约在 0.28（PU Normal 实测 N 分量）

指标：记录级 argmax 命中率（逐记录算 q 均值后 argmax）。
同时在 CWRU（源域，标定应在此决定）与 PU（目标域，事后核验）上报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))
from bearing_facts import LABEL_NAMES  # noqa: E402

V2 = ROOT / "outputs" / "features_v2"
V1 = ROOT / "outputs" / "features"


def load(cache_dir, ds):
    ies = np.load(Path(cache_dir) / f"{ds}_ies_analysis.npy", mmap_mode="r")
    meta = json.loads((Path(cache_dir) / f"{ds}.json").read_text(encoding="utf-8"))
    return meta["records"], ies


def q_linear(R, c):
    e_ir = R[:, 4:8].mean(axis=1)
    e_or = R[:, 0:4].mean(axis=1)
    s = e_ir + e_or + 1e-30
    return np.column_stack([np.full(len(R), c), np.full(len(R), c),
                            (1 - 2 * c) * e_ir / s, (1 - 2 * c) * e_or / s])


def scan(records, ies, cs):
    """返回 {c: {label: (命中率, 均值软分)}}。"""
    per_rec = {c: {} for c in cs}
    for rec in records:
        lab = rec["label"]
        if lab not in (0, 2, 3):
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        if not len(R):
            continue
        for c in cs:
            q = q_linear(R, c).mean(axis=0)
            per_rec[c].setdefault(lab, []).append(q)
    out = {}
    for c in cs:
        out[c] = {}
        for lab in (0, 2, 3):
            if lab not in per_rec[c]:
                continue
            arr = np.array(per_rec[c][lab])
            hit = float((arr.argmax(axis=1) == lab).mean())
            out[c][lab] = (hit, arr.mean(axis=0))
    return out


CS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

print("=" * 78)
print("CWRU 源域（v2 缓存 R）：基准 c 扫描")
print("=" * 78)
cr, ci = load(V2, "cwru")
cwru = scan(cr, ci, CS)
print(f"{'c':>6} {'Normal':>10} {'IR':>10} {'OR':>10}   {'三类均值':>9}")
for c in CS:
    h = [cwru[c][l][0] * 100 for l in (0, 2, 3) if l in cwru[c]]
    print(f"{c:>6.2f} {h[0]:>9.1f}% {h[1]:>9.1f}% {h[2]:>9.1f}%   {np.mean(h):>8.1f}%")

print("\n" + "=" * 78)
print("PU 目标域（v2 缓存 R）：基准 c 扫描（事后核验）")
print("=" * 78)
pr, pi = load(V2, "paderborn")
pu = scan(pr, pi, CS)
print(f"{'c':>6} {'Normal':>10} {'IR':>10} {'OR':>10}   {'三类均值':>9}")
for c in CS:
    h = [pu[c][l][0] * 100 for l in (0, 2, 3) if l in pu[c]]
    print(f"{c:>6.2f} {h[0]:>9.1f}% {h[1]:>9.1f}% {h[2]:>9.1f}%   {np.mean(h):>8.1f}%")

print("\n" + "=" * 78)
print("各类软分均值（看方向而非只看命中）")
print("=" * 78)
for tag, data, name in (("CWRU", cwru, "源域"), ("PU", pu, "目标域")):
    print(f"\n{tag}（{name}）：")
    for c in (0.10, 0.20, 0.25, 0.30):
        if c not in data:
            continue
        print(f"  c={c:.2f}:")
        for lab in (0, 2, 3):
            if lab not in data[c]:
                continue
            m = data[c][lab][1]
            top = LABEL_NAMES[int(np.argmax(m))]
            mark = " ✓" if int(np.argmax(m)) == lab else ""
            print(f"    {LABEL_NAMES[lab]:>11}: [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]"
                  f" 最高={top}{mark}")

# 结果落盘
res = {"cs": CS,
       "cwru": {str(c): {LABEL_NAMES[l]: [cwru[c][l][0], cwru[c][l][1].tolist()] for l in (0, 2, 3)
                         if l in cwru[c]} for c in CS},
       "paderborn": {str(c): {LABEL_NAMES[l]: [pu[c][l][0], pu[c][l][1].tolist()] for l in (0, 2, 3)
                              if l in pu[c]} for c in CS}}
path = ROOT / "outputs" / "analysis_ies_diag" / "q_phys_balance_scan.json"
path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n结果已存：{path}")
