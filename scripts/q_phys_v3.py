# -*- coding: utf-8 -*-
"""q_phys 重构（v3 公式）：用相对健康基线的对数偏离替代绝对值公式。

原公式缺陷：v = [α, α, (α+e_IR)/(1.5(α+e_OR)), (α+e_OR)/(1.5(α+e_IR))]
  —— 混合了"证据绝对强度"与"IR/OR 相对比值"，当 e 与 α 同量级时退化为常数分布。

重构设计：
  1. 源域健康基线：mu_ir = mean(log1p(e_IR) | 源域 Normal)，mu_or 同理
  2. 相对偏离：dev_ir = log1p(e_IR) - mu_ir，dev_or = log1p(e_OR) - mu_or
  3. 故障强度：p_fault = sigmoid(k*(max(dev_ir,dev_or) - b))，b = 源域健康 max_dev 的 95 分位
  4. 类型份额：share = softmax([dev_ir, dev_or])
  5. q = [(1-p_fault)/2, (1-p_fault)/2, p_fault*share_ir, p_fault*share_or]

标定规则（预先声明，不按目标域结果调整）：
  k = 1（固定）；b = Q_0.95(max_dev | 源域健康窗)；mu 用均值。
全部参数只从 CWRU 源域 Normal 拟合，不使用 PU 标签。
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


def load_v2(dataset):
    ies = np.load(V2 / f"{dataset}_ies_analysis.npy", mmap_mode="r")
    meta = json.loads((V2 / f"{dataset}.json").read_text(encoding="utf-8"))
    return meta["records"], ies


def fit_params(records, ies, label=0):
    """从源域健康记录的逐分析窗拟合 mu_ir / mu_or / b。"""
    log_ir, log_or = [], []
    for rec in records:
        if rec["label"] != label:
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        if not len(R):
            continue
        log_ir.append(np.log1p(R[:, 4:8].mean(axis=1)))
        log_or.append(np.log1p(R[:, 0:4].mean(axis=1)))
    if not log_ir:
        raise ValueError("源域 Normal 无可用分析窗")
    log_ir = np.concatenate(log_ir)
    log_or = np.concatenate(log_or)
    mu_ir, mu_or = float(log_ir.mean()), float(log_or.mean())
    max_dev = np.maximum(log_ir - mu_ir, log_or - mu_or)
    return {"mu_ir": mu_ir, "mu_or": mu_or,
            "b": float(np.quantile(max_dev, 0.95)), "k": 1.0,
            "n_healthy_windows": int(len(log_ir))}


def q_phys_v3(R, p):
    """重构的 q_phys：返回 (A,4)，N/B 相等于 (1-p_fault)/2。"""
    e_or = R[:, 0:4].mean(axis=1)
    e_ir = R[:, 4:8].mean(axis=1)
    dev_ir = np.log1p(e_ir) - p["mu_ir"]
    dev_or = np.log1p(e_or) - p["mu_or"]
    max_dev = np.maximum(dev_ir, dev_or)
    p_fault = expit(p["k"] * (max_dev - p["b"]))
    # softmax([dev_ir, dev_or])，用 max_dev 做数值稳定
    exp_ir = np.exp(dev_ir - max_dev)
    exp_or = np.exp(dev_or - max_dev)
    norm = exp_ir + exp_or
    q_n = (1.0 - p_fault) / 2.0
    return np.column_stack([q_n, q_n, p_fault * exp_ir / norm, p_fault * exp_or / norm])


def evaluate(dataset, records, ies, p, tag):
    """逐类 q_phys 均值 + argmax 是否正确。"""
    store = {}
    rec_level = {}
    for rec in records:
        lab = rec["label"]
        if lab not in (0, 1, 2, 3):
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        if not len(R):
            continue
        q = q_phys_v3(R, p)
        store.setdefault(lab, []).append(q.mean(axis=0))
        rec_level.setdefault(lab, []).append(int(q.mean(axis=0).argmax()))
    print(f"\n===== {dataset.upper()} {tag}：q_phys v3 逐类均值 [N, B, IR, OR] =====")
    out = {}
    for lab in (0, 1, 2, 3):
        if lab not in store:
            continue
        m = np.mean(store[lab], axis=0)
        preds = np.array(rec_level[lab])
        acc = float((preds == lab).mean())
        note = ""
        if lab == 0:
            note = "（Ball 列为 N/B 中性，不计）"
        print(f"  {LABEL_NAMES[lab]:>11}: [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]"
              f"  记录级argmax命中={100*acc:>5.1f}%{note}")
        out[LABEL_NAMES[lab]] = {"mean": m.tolist(), "argmax_hit": acc, "n": len(preds)}
    return out


def main():
    print("=" * 62)
    print("q_phys 重构 v3：相对健康基线偏离")
    print("=" * 62)

    cwru_records, cwru_ies = load_v2("cwru")
    pu_records, pu_ies = load_v2("paderborn")
    print(f"CWRU 记录 {len(cwru_records)}；PU 记录 {len(pu_records)}")

    params = fit_params(cwru_records, cwru_ies, label=0)
    print(f"\n源域标定参数（只用 CWRU Normal {params['n_healthy_windows']} 个分析窗，k=1 固定）：")
    print(f"  mu_ir = {params['mu_ir']:.4f}   mu_or = {params['mu_or']:.4f}   b = {params['b']:.4f}")

    cwru_out = evaluate("cwru", cwru_records, cwru_ies, params, "源域")
    pu_out = evaluate("paderborn", pu_records, pu_ies, params, "目标域事后核验")

    # 与原/新公式对比：v1 的 q_phys 在 PU 上的类均值
    print("\n===== 对比：各版本对 PU 真 IR / 真 OR / 真 Normal 的 q_phys =====")
    print("（v1/v2 数据来自此前实验，v3 为本次重构）")
    for lab, name in ((0, "Normal"), (2, "IR"), (3, "OR")):
        v3 = pu_out[name]["mean"]
        print(f"  {name:>7}: v3 = [{v3[0]:.3f}, {v3[1]:.3f}, {v3[2]:.3f}, {v3[3]:.3f}]"
              f"  最高={LABEL_NAMES[int(np.argmax(v3))]}")

    result = {"params": params, "cwru": cwru_out, "paderborn": pu_out,
              "calibration_rule": "k=1 固定；mu=源域健康 log1p 均值；b=源域健康 max_dev 的 95 分位",
              "note": "参数全部来自 CWRU 源域 Normal，未使用 PU 标签"}
    path = ROOT / "outputs" / "analysis_ies_diag" / "q_phys_v3.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存：{path}")


if __name__ == "__main__":
    main()
