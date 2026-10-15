#!/usr/bin/env bash
# 生成 PU v2 缓存：使用修订版 IES（分频段噪底、边带扩展、源域 alpha）

set -euo pipefail

PYTHON="${PYTHON:-E:/tool/anaconda/envs/fault-detect/python.exe}"
ROOT="E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8"
cd "$ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT_DIR="outputs/features_v2"
ALPHA=2.316955

echo "========================================"
echo "PU v2 缓存生成（修订版 IES）"
echo "========================================"
echo "输出目录: $OUT_DIR"
echo "alpha: $ALPHA (从 CWRU Normal 拟合)"
echo ""

if [ -d "$OUT_DIR" ] && [ "$(ls -A $OUT_DIR 2>/dev/null)" ]; then
    echo "错误：输出目录非空，请先备份或删除：$OUT_DIR"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=== 开始：PU 特征提取 v2 ==="
time $PYTHON -c "
import sys
sys.path.insert(0, '.')
import json
import numpy as np
from pathlib import Path

from features import (
    TARGET_FS, LOADERS, _enumerate_recordings, _record_signal_12k,
    _geometry_for, fit_scaler, analysis_window_count, mapping_detail,
    _validate_identity, _validate_rpm, extract_logspec
)
from features_v2 import (
    ies_features_v2, physical_features_v2, CACHE_VERSION_V2, IES_RULE_V2
)
from preprocessing import build_repr_config

ROOT = Path('E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8')
dataset = 'paderborn'
data_root = Path('E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v7/Data/paderborn/extracted')
if not data_root.exists():
    raise FileNotFoundError(f'PU 数据根不存在: {data_root}')
output = ROOT / 'outputs' / 'features_v2'

print(f'数据根: {data_root}')
print(f'输出: {output}')

recordings = list(_enumerate_recordings(dataset, data_root))
print(f'记录数: {len(recordings)}')

# 逐记录提取
X_list, phys25_list, ies_list, records = [], [], [], []
logspec_offset, analysis_offset = 0, 0

for i, rec in enumerate(recordings):
    if (i + 1) % 100 == 0 or i == 0:
        print(f'  处理 {i+1}/{len(recordings)}: {rec.record_key}')
    
    orders, rotation_hz, rpm, rpm_source = _geometry_for(rec)
    signal = _record_signal_12k(rec)
    
    # v2 IES
    ies_R = ies_features_v2(signal, orders, rotation_hz)
    phys25 = physical_features_v2(signal, ies_R)
    
    # logspec 复用 v1 口径（与 IES 无关）
    logspec = extract_logspec(signal[:len(signal) // 512 * 512].reshape(-1, 512))
    
    n_windows = len(logspec)
    n_analysis = len(ies_R)
    expected_analysis = analysis_window_count(n_windows)
    if n_analysis != expected_analysis:
        raise ValueError(f'{rec.record_key}: 分析窗数 {n_analysis} != 预期 {expected_analysis}')
    if len(phys25) != n_windows:
        raise ValueError(f'{rec.record_key}: phys25 长度 {len(phys25)} != logspec {n_windows}')
    
    _, map_ties, map_outside = mapping_detail(n_windows, n_analysis)
    constant_windows = int((logspec.var(axis=1) == 0).sum())
    
    X_list.append(logspec)
    phys25_list.append(phys25)
    ies_list.append(ies_R)
    
    records.append({
        'record_key': rec.record_key, 'unit_key': rec.unit_key,
        'dataset': rec.dataset, 'path': str(rec.path), 'channel': rec.channel,
        'label': rec.label, 'source': rec.source, 'bearing_id': rec.bearing_id,
        'condition': rec.condition, 'origin': rec.origin, 'severity': rec.severity,
        'src_fs': rec.src_fs, 'rpm': rpm, 'rpm_source': rpm_source,
        'start': logspec_offset, 'stop': logspec_offset + n_windows,
        'analysis_start': analysis_offset, 'analysis_stop': analysis_offset + n_analysis,
        'map_ties': map_ties, 'map_outside': map_outside,
        'constant_windows': constant_windows,
    })
    
    logspec_offset += n_windows
    analysis_offset += n_analysis

X = np.concatenate(X_list).astype(np.float32)
phys25 = np.concatenate(phys25_list).astype(np.float32)
ies_analysis = np.concatenate(ies_list).astype(np.float32)

print(f'总窗口: logspec={len(X)}, phys25={len(phys25)}, ies_analysis={len(ies_analysis)}')

# 校验
_validate_identity(dataset, records)
for rec_dict in records:
    if rec_dict['rpm_source'] == 'field':
        continue  # field 来源已在 _geometry_for 校验
# RPM 校验在 _geometry_for 中完成，这里跳过

for arr, name in [(X, 'logspec'), (phys25, 'phys25'), (ies_analysis, 'ies_analysis')]:
    if not np.isfinite(arr).all():
        raise ValueError(f'{name} 包含非有限值')

# 保存
output.mkdir(parents=True, exist_ok=True)
np.save(output / f'{dataset}.npy', X[:, None, :])
np.save(output / f'{dataset}_phys25.npy', phys25)
np.save(output / f'{dataset}_ies_analysis.npy', ies_analysis)

repr_config = build_repr_config()
counts = {
    'records': len(records), 'units': len({r['unit_key'] for r in records}),
    'windows': len(X), 'primary_records': len([r for r in records if r['label'] is not None]),
    'primary_units': len({r['unit_key'] for r in records if r['label'] is not None}),
    'primary_windows': sum(r['stop'] - r['start'] for r in records if r['label'] is not None),
}

metadata = {
    'version': CACHE_VERSION_V2,
    'dataset': dataset,
    'repr_config': repr_config,
    'records': records,
    'counts': counts,
    'revision_v2': IES_RULE_V2['revision_v2'],
    'alpha': float($ALPHA),
    'alpha_source': 'CWRU source train+val Normal',
}

with (output / f'{dataset}.json').open('w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=1)

print(f'已保存: {output}')
print(f'版本: {CACHE_VERSION_V2}')
print(f'计数: {counts}')
" 2>&1

EXIT_CODE=$?
echo ""
echo "退出码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== 完成：PU v2 缓存生成成功 ==="
    ls -lh "$OUT_DIR"
else
    echo ""
    echo "=== 错误：缓存生成失败 ==="
    exit $EXIT_CODE
fi
