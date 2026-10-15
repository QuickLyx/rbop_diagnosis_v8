#!/usr/bin/env bash
# P2 全量非训练准备；不执行模型训练、SVM 拟合或 BN 重估。
set -euo pipefail
PYTHON="${PYTHON:-E:/tool/anaconda/envs/fault-detect/python.exe}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--execute" ]; }; then
  echo "用法：bash scripts/run_prepare.sh [--execute]；默认只显示准备顺序。" >&2
  exit 2
fi
if [ "${1:-}" != "--execute" ]; then
  echo "P2准备顺序：全量特征 → 旧池只读导入/GG → 共同留出 → 旧锚复用 → 新IES锚。"
  echo "此命令未执行计算；获准备授权后使用 --execute。"
  exit 0
fi
"$PYTHON" features.py --config configs/stage3.json --execute
"$PYTHON" selection.py --config configs/stage3.json --execute
"$PYTHON" physical_baseline.py --config configs/stage3.json --arm phys_envelope --execute
"$PYTHON" physical_baseline.py --config configs/stage3.json --arm phys_ies --execute
printf '%s\n' "P2准备完成。下一阶段为零拟合preflight；尚未训练。"
