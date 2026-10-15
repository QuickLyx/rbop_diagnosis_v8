# -*- coding: utf-8 -*-
"""v2 评估：在新缓存上跑修订规则，对比 v1 的 IR 召回与 q_phys。

核心指标：
1. 记录级硬判召回（N/IR/OR）
2. 单元级硬判召回
3. q_phys 每类平均软分（检验 IR 分量是否还低于 OR）
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))

from features import load_cache, FeatureCache  # noqa: E402
from features_v2 import ies_evidence_v2  # noqa: E402
from bearing_facts import BY_ID, LABEL_NAMES  # noqa: E402


def load_cache_v2(dataset, cache_dir):
    """读取 v2 缓存（版本号不同，跳过 v1 的版本校验）。"""
    cache_dir = Path(cache_dir)
    folder = cache_dir
    X = np.load(folder / f"{dataset}.npy", mmap_mode="r")
    phys25 = np.load(folder / f"{dataset}_phys25.npy", mmap_mode="r")
    ies = np.load(folder / f"{dataset}_ies_analysis.npy", mmap_mode="r")
    meta = json.loads((folder / f"{dataset}.json").read_text(encoding="utf-8"))
    return FeatureCache(X=X, phys25=phys25, ies_analysis=ies,
                        records=meta["records"], version=meta["version"],
                        repr_config=meta["repr_config"])


ALPHA = 2.316955

cache = load_cache_v2("paderborn", ROOT / "outputs" / "features_v2")
print(f"缓存版本: {cache.version}")
print(f"记录数: {len(cache.records)}")

# ---- 记录级硬判 + q_phys ----
per_label_hard = {}
q_sum = {}
unit_records = {}

for rec in cache.records:
    label = rec["label"]
    if label not in (0, 2, 3):
        continue
    R = np.asarray(cache.ies_analysis[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
    onehot, q_phys, e_ir, e_or = ies_evidence_v2(R, ALPHA)
    hard = int(onehot.mean(axis=0).argmax())
    hit = per_label_hard.setdefault(label, [0, 0])
    hit[1] += 1
    hit[0] += int(hard == label)
    q_sum.setdefault(label, []).append(q_phys.mean(axis=0))
    unit_records.setdefault(rec["unit_key"], {"label": label, "onehot": [], "q": []})
    unit_records[rec["unit_key"]]["onehot"].append(onehot.mean(axis=0))
    unit_records[rec["unit_key"]]["q"].append(q_phys.mean(axis=0))

print("\n===== PU v2 记录级硬判召回 =====")
for lab in (0, 2, 3):
    ok, tot = per_label_hard[lab]
    print(f"  {LABEL_NAMES[lab]:>11}: {ok:>4}/{tot:<4} = {100*ok/tot:>5.1f}%")

print("\n===== PU v2 q_phys 每类平均软分 [N, B, IR, OR] =====")
for lab in (0, 2, 3):
    m = np.mean(q_sum[lab], axis=0)
    print(f"  {LABEL_NAMES[lab]:>11}: [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]")

# ---- 单元级硬判 ----
print("\n===== PU v2 单元级硬判召回 =====")
unit_conf = np.zeros((4, 4), dtype=int)
for unit, data in unit_records.items():
    lab = data["label"]
    prob = np.mean(data["onehot"], axis=0)
    pred = int(prob.argmax())
    unit_conf[lab, pred] += 1

for lab in (0, 2, 3):
    total = unit_conf[lab].sum()
    ok = unit_conf[lab, lab]
    print(f"  {LABEL_NAMES[lab]:>11}: {ok:>3}/{total:<3} = {100*ok/total:>5.1f}%")

macro = np.mean([unit_conf[l, l] / max(unit_conf[l].sum(), 1) for l in (0, 2, 3)])
print(f"  三类宏召回: {100*macro:.2f}%")

print("\n单元级混淆（真实\\预测）:")
print("         pred N   B  IR  OR")
for i, name in enumerate(("N", "B", "IR", "OR")):
    print(f"  true {name:>3}: {unit_conf[i].tolist()}")

# ---- v1 对比 ----
print("\n===== v1 vs v2 对比（单元级） =====")
print(f"{'指标':>12} {'v1':>10} {'v2':>10}")
print(f"{'Normal 召回':>12} {'8/12=66.7%':>10} {f'{unit_conf[0,0]}/{unit_conf[0].sum()}={100*unit_conf[0,0]/unit_conf[0].sum():.1f}%':>10}")
print(f"{'IR 召回':>12} {'0/44=0.0%':>10} {f'{unit_conf[2,2]}/{unit_conf[2].sum()}={100*unit_conf[2,2]/unit_conf[2].sum():.1f}%':>10}")
print(f"{'OR 召回':>12} {'39/48=81.3%':>10} {f'{unit_conf[3,3]}/{unit_conf[3].sum()}={100*unit_conf[3,3]/unit_conf[3].sum():.1f}%':>10}")
print(f"{'三类宏召回':>12} {'49.31%':>10} {f'{100*macro:.2f}%':>10}")

# 保存结果
out = ROOT / "outputs" / "analysis_ies_diag"
out.mkdir(parents=True, exist_ok=True)
result = {
    "version": "v2",
    "alpha": ALPHA,
    "record_level": {LABEL_NAMES[l]: {"hit": per_label_hard[l][0], "total": per_label_hard[l][1],
                                       "recall": per_label_hard[l][0] / per_label_hard[l][1]}
                     for l in (0, 2, 3)},
    "q_phys_mean": {LABEL_NAMES[l]: np.mean(q_sum[l], axis=0).tolist() for l in (0, 2, 3)},
    "unit_level": {LABEL_NAMES[l]: {"hit": int(unit_conf[l, l]), "total": int(unit_conf[l].sum()),
                                     "recall": float(unit_conf[l, l] / max(unit_conf[l].sum(), 1))}
                   for l in (0, 2, 3)},
    "unit_macro_recall": float(macro),
    "unit_confusion": unit_conf.tolist(),
}
with (out / "v2_evaluation.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {out/'v2_evaluation.json'}")
