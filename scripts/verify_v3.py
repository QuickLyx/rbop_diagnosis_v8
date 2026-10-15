# -*- coding: utf-8 -*-
"""v3 最小改动方案验证：R 用 v1 规则（不改缓存），只改 q_phys 的 alpha。

对比：
  v1:  R=v1, alpha=1（隐含）
  v3:  R=v1, alpha=2.3170（从 CWRU Normal 拟合）
检验 q_phys 对 IR 记录的方向是否修正，硬判保持不变。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))

from features import load_cache  # noqa: E402
from bearing_facts import LABEL_NAMES  # noqa: E402

ALPHA = 2.316955

cache = load_cache("paderborn", ROOT / "outputs" / "features")
print(f"缓存: {cache.version}")

q_store = {}
hard_store = {0: [0, 0], 2: [0, 0], 3: [0, 0]}
unit_store = {}

for rec in cache.records:
    label = rec["label"]
    if label not in (0, 2, 3):
        continue
    R = np.asarray(cache.ies_analysis[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
    e_or = R[:, 0:4].mean(axis=1)
    e_ir = R[:, 4:8].mean(axis=1)

    # 硬判（与 v1 一致）
    hard = np.zeros(len(R), dtype=int)
    hard[e_or > 1.5 * e_ir] = 3
    hard[e_ir > 1.5 * e_or] = 2
    pred = int(np.bincount(hard, minlength=4).argmax())
    hard_store[label][1] += 1
    hard_store[label][0] += int(pred == label)

    # q_phys 各 alpha
    for a in (1.0, ALPHA):
        v = np.column_stack([np.full(len(R), a), np.full(len(R), a),
                             (a + e_ir) / (1.5 * (a + e_or)),
                             (a + e_or) / (1.5 * (a + e_ir))])
        q = v / v.sum(axis=1, keepdims=True)
        q_store.setdefault(a, {}).setdefault(label, []).append(q.mean(axis=0))
        unit_store.setdefault(a, {}).setdefault(rec["unit_key"], []).append(q.mean(axis=0))

print("\n===== 硬判（应与 v1 一致，未改 R） =====")
for lab in (0, 2, 3):
    ok, tot = hard_store[lab]
    print(f"  {LABEL_NAMES[lab]:>11}: {ok}/{tot} = {100*ok/tot:.1f}%")

print("\n===== q_phys 记录级平均软分 [N, B, IR, OR] =====")
for a in (1.0, ALPHA):
    print(f"  alpha={a}:")
    for lab in (0, 2, 3):
        m = np.mean(q_store[a][lab], axis=0)
        print(f"    {LABEL_NAMES[lab]:>11}: [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]")

print("\n===== q_phys 区分度检验（IR 记录的 IR 分量是否排名第一）=====")
for a in (1.0, ALPHA):
    ir_q = np.array(q_store[a][2])
    rank = (ir_q[:, 2] >= ir_q[:, 3]).mean()   # IR 分量 >= OR 分量
    ir_top = (ir_q.argmax(axis=1) == 2).mean()
    print(f"  alpha={a}: IR分量≥OR分量 {100*rank:.1f}% | IR分量排名第一 {100*ir_top:.1f}%")

# 单元级
print("\n===== q_phys 单元级平均软分（IR 单元）=====")
for a in (1.0, ALPHA):
    ir_units = [k for k, v in unit_store[a].items()
                if any(True for rec in cache.records
                       if rec["unit_key"] == k and rec["label"] == 2)]
    # 简化：直接用记录的 label 映射
    unit_label = {}
    for rec in cache.records:
        if rec["label"] in (0, 1, 2, 3):
            unit_label[rec["unit_key"]] = rec["label"]
    ir_units = [k for k in unit_store[a] if unit_label.get(k) == 2]
    m = np.mean([np.mean(unit_store[a][k], axis=0) for k in ir_units], axis=0)
    print(f"  alpha={a}: IR 单元平均 [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]")

# 保存
out = ROOT / "outputs" / "analysis_ies_diag"
result = {
    "scheme": "v3 (R=v1, alpha=2.3170)",
    "hard_recall": {LABEL_NAMES[l]: [hard_store[l][0], hard_store[l][1]] for l in (0, 2, 3)},
    "q_phys": {str(a): {LABEL_NAMES[l]: np.mean(q_store[a][l], axis=0).tolist() for l in (0, 2, 3)}
               for a in (1.0, ALPHA)},
}
with (out / "v3_minimal_change.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n已保存: {out/'v3_minimal_change.json'}")
