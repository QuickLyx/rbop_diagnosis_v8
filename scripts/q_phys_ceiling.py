# -*- coding: utf-8 -*-
"""诊断：q_phys 失败是"跨域标定"问题还是"特征无信息"问题？

上界测试（仅诊断，非可用方案）：
  A. 源域基线（v3 现状）：只看 CWRU Normal 拟合的参数
  B. 目标域基线（作弊上界）：用 PU 自身健康记录拟合 mu/b
     —— 用了目标域标签，违反零拟合原则，只用于判断特征本身有无信息
  C. 纯比值（无基线）：q_IR ∝ e_IR/(e_IR+e_OR)，q_OR ∝ e_OR/(e_IR+e_OR)，无 Normal

若 B 能分开三类而 A 不能 → 是跨域标定问题；
若 B 也分不开 → 特征本身在 PU 上无判别力。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))
from bearing_facts import LABEL_NAMES  # noqa: E402

V2 = ROOT / "outputs" / "features_v2"


def load(ds):
    ies = np.load(V2 / f"{ds}_ies_analysis.npy", mmap_mode="r")
    meta = json.loads((V2 / f"{ds}.json").read_text(encoding="utf-8"))
    return meta["records"], ies


def e_of(R):
    return R[:, 4:8].mean(axis=1), R[:, 0:4].mean(axis=1)


def fit(records, ies, label):
    li, lo = [], []
    for rec in records:
        if rec["label"] != label:
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        e_ir, e_or = e_of(R)
        li.append(np.log1p(e_ir)); lo.append(np.log1p(e_or))
    li, lo = np.concatenate(li), np.concatenate(lo)
    mu_ir, mu_or = float(li.mean()), float(lo.mean())
    md = np.maximum(li - mu_ir, lo - mu_or)
    return {"mu_ir": mu_ir, "mu_or": mu_or, "b": float(np.quantile(md, 0.95)), "k": 1.0}


def q_v3(R, p):
    e_ir, e_or = e_of(R)
    d_ir, d_or = np.log1p(e_ir) - p["mu_ir"], np.log1p(e_or) - p["mu_or"]
    md = np.maximum(d_ir, d_or)
    pf = expit(p["k"] * (md - p["b"]))
    ei, eo = np.exp(d_ir - md), np.exp(d_or - md)
    qn = (1 - pf) / 2
    return np.column_stack([qn, qn, pf * ei / (ei + eo), pf * eo / (ei + eo)])


def q_ratio(R):
    """纯比值：仅 IR/OR 相对份额，N/B 固定小基准。"""
    e_ir, e_or = e_of(R)
    s = e_ir + e_or + 1e-30
    return np.column_stack([0.1 * np.ones(len(R)), 0.1 * np.ones(len(R)),
                            0.8 * e_ir / s, 0.8 * e_or / s])


def summarize(name, records, ies, fn):
    store, hits = {}, []
    for rec in records:
        lab = rec["label"]
        if lab not in (0, 2, 3):
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        q = fn(R).mean(axis=0)
        store.setdefault(lab, []).append(q)
        hits.append((lab, int(q.argmax())))
    print(f"\n  [{name}]")
    for lab in (0, 2, 3):
        m = np.mean(store[lab], axis=0)
        acc = np.mean([p == lab for l, p in hits if l == lab]) * 100
        print(f"    {LABEL_NAMES[lab]:>11}: [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]"
              f"  argmax命中={acc:>5.1f}%")


def e_stats(name, records, ies):
    print(f"\n  [{name} 的 e 绝对值（逐记录平均后再取类均值）]")
    store = {}
    for rec in records:
        lab = rec["label"]
        if lab not in (0, 2, 3):
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        e_ir, e_or = e_of(R)
        store.setdefault(lab, []).append((float(e_ir.mean()), float(e_or.mean())))
    for lab in (0, 2, 3):
        a = np.array(store[lab])
        print(f"    {LABEL_NAMES[lab]:>11}: e_IR 中位={np.median(a[:,0]):>7.3f}  "
              f"e_OR 中位={np.median(a[:,1]):>7.3f}  e_IR+e_OR 中位={np.median(a[:,0]+a[:,1]):>7.3f}  "
              f"n={len(a)}")


cr, ci = load("cwru")
pr, pi = load("paderborn")

print("=" * 64)
print("诊断一：e 的绝对量级是否跨域可比")
print("=" * 64)
e_stats("CWRU", cr, ci)
e_stats("PU", pr, pi)

print("\n" + "=" * 64)
print("诊断二：三种基线的 q_phys 表现（PU 上）")
print("=" * 64)
p_src = fit(cr, ci, 0)
p_tgt = fit(pr, pi, 0)
print(f"  源域基线（CWRU Normal）: mu_ir={p_src['mu_ir']:.3f} mu_or={p_src['mu_or']:.3f} b={p_src['b']:.3f}")
print(f"  目标域基线（PU Normal） : mu_ir={p_tgt['mu_ir']:.3f} mu_or={p_tgt['mu_or']:.3f} b={p_tgt['b']:.3f}")

summarize("A. 源域基线 (v3 现状)", pr, pi, lambda R: q_v3(R, p_src))
summarize("B. 目标域基线 (作弊上界)", pr, pi, lambda R: q_v3(R, p_tgt))
summarize("C. 纯比值 (无基线)", pr, pi, q_ratio)

print("\n" + "=" * 64)
print("诊断三：源域同步对照（CWRU 上三种基线）")
print("=" * 64)
summarize("A. 源域基线 (CWRU 自身)", cr, ci, lambda R: q_v3(R, p_src))
summarize("C. 纯比值", cr, ci, q_ratio)
