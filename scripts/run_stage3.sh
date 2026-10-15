#!/usr/bin/env bash
# 固定顺序与同格依赖；默认144格零拟合检查，--execute仅供正式授权后使用。
set -euo pipefail
PYTHON="${PYTHON:-E:/tool/anaconda/envs/fault-detect/python.exe}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--execute" ]; }; then
  echo "用法：bash scripts/run_stage3.sh [--execute]；默认只做零拟合preflight。" >&2
  exit 2
fi
ARMS=(joint joint_logspec mixup4src coral4src adabn ce_K19 ce_K19_gg ce_K116 ce_K19_logspec svm_phys fusion_joint fusion_ce_K19)
DEEP_ARMS=(joint joint_logspec mixup4src coral4src adabn ce_K19 ce_K19_gg ce_K116 ce_K19_logspec)
if [ "${1:-}" != "--execute" ]; then
  for fold in 0 1 2 3; do
    for seed in 0 1 2; do
      for arm in "${ARMS[@]}"; do
        "$PYTHON" train.py --config configs/stage3.json --arm "$arm" --fold "$fold" --seed "$seed" --dry-run
      done
    done
    "$PYTHON" oneclass.py --config configs/stage3.json --fold "$fold" --dry-run
  done
  "$PYTHON" shallow.py --config configs/stage3.json --rs --dry-run
  echo "144主要格与辅助任务的零拟合检查完成；缺依赖仍为pending，不代表P3验收通过。"
  exit 0
fi
for fold in 0 1 2 3; do
  "$PYTHON" oneclass.py --config configs/stage3.json --fold "$fold" --execute
  for seed in 0 1 2; do
    for arm in "${DEEP_ARMS[@]}"; do
      "$PYTHON" train.py --config configs/stage3.json --arm "$arm" --fold "$fold" --seed "$seed" --execute
      for dataset in cwru paderborn; do
        "$PYTHON" evaluate.py --config configs/stage3.json --checkpoint "outputs/train/${arm}_f${fold}_s${seed}/last.pt" \
          --dataset "$dataset" --output "outputs/eval/${arm}_f${fold}_s${seed}_${dataset}" --execute
      done
    done
    "$PYTHON" shallow.py --config configs/stage3.json --fold "$fold" --seed "$seed" --execute
    for dataset in cwru paderborn; do
      "$PYTHON" evaluate.py --config configs/stage3.json --arm svm_phys --model-dir "outputs/train/svm_phys_f${fold}_s${seed}" \
        --fold "$fold" --seed "$seed" --dataset "$dataset" --output "outputs/eval/svm_phys_f${fold}_s${seed}_${dataset}" --execute
    done
    for arm in fusion_joint fusion_ce_K19; do
      "$PYTHON" evaluate.py --config configs/stage3.json --arm "$arm" --fold "$fold" --seed "$seed" --execute
    done
  done
done
"$PYTHON" shallow.py --config configs/stage3.json --rs --execute
"$PYTHON" scripts/summarize_stage3.py --config configs/stage3.json
echo "正式顺序执行完成。失败则停止，单格重跑须使用 _retry1，不能按PU分数删格。"
