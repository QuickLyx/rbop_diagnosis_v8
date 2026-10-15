#!/usr/bin/env bash
# v2 源域验证：在 CWRU 上测试修订规则的 IR 召回改善

set -euo pipefail

PYTHON="${PYTHON:-E:/tool/anaconda/envs/fault-detect/python.exe}"
ROOT="E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8"
cd "$ROOT"

echo "=== 阶段 1: 从 v1 缓存拟合 alpha ==="
alpha=$($PYTHON -c "
import sys; sys.path.insert(0,'.')
from features import load_cache
from features_v2 import alpha_from_source_normal
cache = load_cache('cwru', 'outputs/features')
normal_recs = [r for r in cache.records if r['label'] == 0]
alpha = alpha_from_source_normal(normal_recs, cache.ies_analysis)
print(f'{alpha:.6f}')
")
echo "alpha = $alpha (基于 CWRU 8 条 Normal 记录)"

echo ""
echo "=== 阶段 2: 用 v1 缓存 + v2 规则评估 CWRU ==="
$PYTHON -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from features import load_cache
from features_v2 import ies_evidence_v2

cache = load_cache('cwru', 'outputs/features')
alpha = $alpha

per_label = {}
for rec in cache.records:
    label = rec['label']
    if label not in (0, 1, 2, 3):
        continue
    R = np.asarray(cache.ies_analysis[rec['analysis_start']:rec['analysis_stop']])
    onehot, q_phys, _, _ = ies_evidence_v2(R, alpha)
    hard = int(onehot.mean(axis=0).argmax())
    hit = per_label.setdefault(label, [0, 0])
    hit[1] += 1
    hit[0] += int(hard == label)

print('CWRU 源域 v2 硬判召回（记录级）：')
for lab, name in [(0, 'Normal'), (1, 'Ball'), (2, 'IR'), (3, 'OR')]:
    if lab in per_label:
        ok, tot = per_label[lab]
        print(f'  {name:>7}: {ok:>3}/{tot:<3} = {100*ok/tot:>5.1f}%')
"

echo ""
echo "=== 阶段 3: 对比 v1 vs v2 在 CWRU 的召回 ==="
echo "v1 (原规则, alpha=1 隐含):"
$PYTHON -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from features import load_cache
from physical_baseline import ies_evidence
cache = load_cache('cwru', 'outputs/features')
per_label = {}
for rec in cache.records:
    label = rec['label']
    if label not in (0, 1, 2, 3):
        continue
    R = np.asarray(cache.ies_analysis[rec['analysis_start']:rec['analysis_stop']])
    onehot, _, _, _ = ies_evidence(R)
    hard = int(onehot.mean(axis=0).argmax())
    hit = per_label.setdefault(label, [0, 0])
    hit[1] += 1
    hit[0] += int(hard == label)
for lab, name in [(0, 'Normal'), (1, 'Ball'), (2, 'IR'), (3, 'OR')]:
    if lab in per_label:
        ok, tot = per_label[lab]
        print(f'  {name:>7}: {ok:>3}/{tot:<3} = {100*ok/tot:>5.1f}%')
"

echo ""
echo "=== 完成：源域 v2 验证通过 ==="
echo "alpha = $alpha 可用于 PU 缓存 v2 重算"
echo "若 CWRU IR 召回改善，继续生成 PU v2 缓存"
