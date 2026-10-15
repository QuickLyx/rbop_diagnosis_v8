# 阶段三故障诊断实验（v8）

**唯一执行口径：[阶段三执行计划](docs/阶段三执行计划.md)。原则：[实验完整，工程简单](agents.md)。**

CWRU→PU 固定矩阵：12 输出臂×4 源折×3 seed=144 主要结果格，其中96梯度训练、12 AdaBN、12浅层拟合及24融合评估。另有4份 Normal 单类模型、48融合两路消融、两个物理锚、一次GG和一份探索性RS。监督全116指标只作描述，泛化结论用冻结共同留出。

## 当前状态

**P2/P3/P4 已通过：144 格零拟合 preflight 与 f0/s0 预算试跑完成；待 P5 正式全批授权。** 最新验收、两锚结果、分项耗时和边界见 [P2/P3执行记录](docs/P2_P3执行记录.md)与 [P4执行记录](docs/P4执行记录.md)。P4 实测：joint 满轮 304.5 s、mixup/coral/ce 单轮 21.7/25.9/16.4 s、SVM 69.1 s、Normal 拟合秒级；正式批外推纯算法约 13.6 h、墙钟预计 15–18 h。P4 期间修复 mixup4src CUDA 跨设备 bug（CPU 单测无法覆盖），全量 137 项测试通过。正式主要结果仍为 0/144。P1实现记录见 [P1执行记录](docs/P1执行记录.md)；本轮修复计划元数据与测试隔离后，128项测试通过。v8不通过动态桥接加载旧训练脚本，旧版代码/数据/冻结产物保持只读。

探索性物理软证据已另行冻结为 [q_phys c=0.25 补丁](docs/q_phys_c025冻结补丁.md)：使用已有 v2 R，输出 `[0.25,0.25,0.5*r,0.5*(1-r)]`，两能量全零时四类均匀。仅探索，不替换 v1 正式配方，不授权训练；实现入口为 `physical_baseline_v2.py`，最终验收的两域独立产物位于 `outputs/analysis_ies_diag/q_phys_ratio_c025_retry1/`，首轮默认路径产物保留。新增8项回归后，全量136项测试通过。历史 `features_v2.py` 中带 alpha 的软分仅供旧诊断复现，不是此次定稿。

## 环境

本地解释器：`E:/tool/anaconda/envs/fault-detect/python.exe`。当前已有 PyTorch 2.6.0+cu124、NumPy 2.5.2、SciPy 1.18.1、scikit-learn 1.9.0，支持 `FrozenEstimator`，CUDA可用。无需安装EMD或新依赖。

以下命令从 `rbop_diagnosis_v8/` 执行。Git Bash 下先指定已有解释器：

```bash
export PYTHON='E:/tool/anaconda/envs/fault-detect/python.exe'
```

云端设置 `PYTHON=python`，并将同级v6/v7只读数据、旧缓存/选标/预测同步到配置要求的位置。实际依赖版本应另行记录，不以本地版本替代云端实测。

## P1：不训练检查

```bash
"$PYTHON" bearing_facts.py --self-check
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" train.py --arm joint --fold 0 --seed 0 --dry-run
```

小测试仅使用合成输入、临时文件及少量只读检查。dry-run不构造神经网络或优化器、不拟合SVM、不重估BN。尚无P2产物时应显示依赖pending，不能据144份计划文件声称P3已通过。

## P2：非训练准备

准备前先确认授权和磁盘空间。**本次P2已完成，下面命令为全新输出目录的执行说明，不可原地重跑覆盖。** 仅显示顺序不会计算；显式 `--execute` 才生成全量新特征、GG池、配对列表和物理锚：

```bash
bash scripts/run_prepare.sh
# 获全量准备授权后：
bash scripts/run_prepare.sh --execute
```

准备严格按“特征→旧池只读导入/GG→配对列表→旧锚复用→新IES锚”顺序。旧池不重选，旧锚不改规则；不以目标得分筛选规则。新旧缓存版本、区间与记录数不符即报错，不静默重建。

## P3：144格零拟合检查

```bash
bash scripts/run_stage3.sh
```

默认模式只写逐格计划及依赖状态。P2完整后检查144主要格、4份Normal及RS等辅助依赖，才可验收P3。

**本次P3已完成**：60格ready、84格仅待正式基座/Normal/深度报告；证据为`outputs/P3_preflight.log`及逐格JSON。采用单进程顺序调用上述同一dry-run入口，避免重复冷启动。默认144个输出目录现已非空，不要直接重复运行该脚本；先只读复核已有计划，确需重跑单格时指定`--output outputs/preflight/<原名>_retry1`。尚未训练不等于所有依赖均可ready。

## P4–P6：先预算，后正式授权

P4真实数据试跑也属于拟合，需单独授权；pilot产物只能写 `outputs/pilot/`，不能冒充30轮正式完成格。深度入口已支持 `--pilot-epochs`（1至30）；同格CE pilot可复用该目录已完整30轮的joint pilot，正式训练拒绝所有pilot权重。按计划实测joint30轮、Mixup/CORAL/CE各1轮、SVM完整27次内部拟合+最终重拟合与校准、Normal完整拟合的分项成本，再决定正式批次预算。没有另建pilot调度器。

```bash
# 以下只作运行说明，本轮未执行；先获P4授权：
"$PYTHON" train.py --arm joint --fold 0 --seed 0 --pilot-epochs 30 --execute
"$PYTHON" train.py --arm mixup4src --fold 0 --seed 0 --pilot-epochs 1 --execute
"$PYTHON" train.py --arm coral4src --fold 0 --seed 0 --pilot-epochs 1 --execute
"$PYTHON" train.py --arm ce_K19 --fold 0 --seed 0 --pilot-epochs 1 --execute
"$PYTHON" shallow.py --fold 0 --seed 0 --output outputs/pilot/svm_phys_f0_s0 --execute
"$PYTHON" oneclass.py --fold 0 --output outputs/pilot/normal_f0 --execute
```

```bash
# 仅在预算检查通过并再次获得正式训练授权后：
bash scripts/run_stage3.sh --execute
# 已有结果的只读汇总（无训练）：
"$PYTHON" scripts/summarize_stage3.py
```

正式脚本显式运行9个深度臂及独立浅层/Normal/fusion入口，融合不送进训练循环。评估使用同格`last.pt`，不用best按目标得分挑模型。默认输出已有内容即停止，不覆盖；失败格单独重跑，输出另取`_retry1`，保留原格。汇总如遇同格两个正式候选会停止，需明确哪份作为正式结果，不能按PU分数选高者。

## 产物与限制

所有新增产物只在v8 `outputs/`：`features/`、`selection/`、`train/`、`eval/`和`summary_<日期>/`。模型/浅层均保存实际split、scaler及拟合或校准元数据；缓存保存生成脚本与口径版本。缓存数组保留原logspec、25维摘要及未映射IES分析窗，不永久存重复的补零通道。

E1为四类argmax，E2只在聚合概率中排Ball后全员重判。低置信只标记，不拒判、不删分母。KB只报分布与官方主导指派参考，不报单类准确率；K116不加入泛化配对。未过线先读预测CSV分析健康类、Ball假阳性、N09和损伤/轴承错误，不追加模型。

本轮不做Ball拒识、LEOC/EEMD/OWES、伪标签/MMD新增臂、GAN或部署框架；RS不改变主输入。源Normal验证文件少、固定频带与N09频率分辨率限制按原计划保留。
